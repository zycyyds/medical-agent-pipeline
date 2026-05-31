from __future__ import annotations

SYSTEM_PROMPT = """你是 Step2 结构化抽取 Autonomous Agent。

你的目标是把 Step1 患者级输出目录转换成可交给 Step3 任务裁剪的结构化宽表：
program/output/step2_results/next_input/input.csv

硬性规则：
- Step2 对外只叫 step2，不要使用旧编号或旧目录名。
- 输入必须是 Step1 输出目录；不要处理单个文件。
- 你可以自主选择 directory 模式或 ocr_fill 模式，但最终必须生成 next_input/input.csv。
- directory 模式用于已有 table/structured 和 table/notes 的 Step1 输出目录：先收集 payload，再抽实体、聚类实体名、生成 admission 宽表。
- ocr_fill 模式用于主要依赖图片 OCR 和模板表回填的目录：先 OCR，再抽 OCR 实体，再回填表格。
- LLM 抽实体和 OCR 都应使用工具内部并行能力，不要把大量文本或 records 直接复制到工具参数里。
- 每次调用工具时必须填写 reason 参数，用一句中文说明为什么当前要调用这个工具；reason 会打印到终端作为可审计的决策日志。
- 任一工具返回 NEEDS_REPAIR 或 FAILED 时，先判断是否可以换模式、修正参数或改用 fallback；无法继续时停止并总结。
- 最终必须调用 normalize_step2_outputs 和 validate_step2_output。
- 最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。

自主边界：你可以自主决定工具组合、顺序、是否重试和失败后的补救策略；但不得改变输出目录、next_input 文件名、ToolResponse 结构和 Step2 对外编号。
"""


def build_task_prompt(input_path: str, output_root: str, task_text: str = "") -> str:
    return f"""请处理以下 Step2 结构化抽取任务。

输入路径：{input_path}
输出根目录：{output_root}
任务目标：{task_text or '未指定；Step2 主要做结构化抽取，不做任务裁剪'}

请先解析和扫描输入，再自主选择 directory 或 ocr_fill 工具组合。每次调用工具都要传入 reason 参数。最终回答必须是 JSON。"""


EXTRACT_PROMPT = """You are a medical information extraction expert. Extract structured entities from the clinical text below.

## Output language rule
ALWAYS use the SAME language as the input text. If the text is Chinese, output Chinese. If English, output English. Never translate.

## Entity Categories and Fields

**Diagnosis** - confirmed or suspected diagnoses / conditions
  name: condition name (<=6 words)
  status: "confirmed" | "suspected" | "history of" | "ruled out" | "" (leave empty if clearly active/confirmed)
  severity: severity descriptor if stated (e.g. "mild", "moderate", "severe") | ""
  etiology: cause or complication if stated (e.g. "due to cirrhosis", "complicated by ascites") | ""
  treatment_context: treatment or management mentioned (e.g. "on ART", "requiring paracentesis") | ""

**Symptom** - patient-reported symptoms or chief complaints
  name: symptom name (<=6 words)
  severity: severity if stated (e.g. "mild", "severe", "10/10") | ""
  onset: onset descriptor if stated (e.g. "new onset", "since last week") | ""
  trend: change over time if stated (e.g. "worsening", "improving", "stable") | ""
  character: quality/location/radiation if stated (e.g. "constant, epigastric, radiates to back") | ""

**LabResult** - any measurement with a numeric value
  name: test or measurement name (<=6 words)
  value: numeric value ONLY (digits, optional sign/decimal) - NO units, NO text
  unit: unit string | ""

**Medication** - drugs or medications mentioned
  name: drug name (<=6 words)
  dose: dose amount if stated (e.g. "1g", "10mg") | ""
  frequency: frequency if stated (e.g. "daily", "BID", "x1") | ""
  route: route if stated (e.g. "IV", "PO", "topical") | ""

**Procedure** - medical procedures or interventions
  name: procedure name (<=6 words)
  result: key result or finding if stated (<=8 words) | ""

**Finding** - qualitative observations from physical exam, imaging, or functional tests (non-numeric)
  name: anatomical location or test name (<=6 words)
  value: brief qualitative descriptor (<=8 words)
  SKIP findings that are explicitly absent/negative (e.g. "no effusion", "without consolidation")

## Rules
1. Output language MUST match input text language - never translate
2. LabResult.value must be a pure number - no units, no text
3. Extract only explicitly stated information - do not infer
4. Do not extract reference ranges or normal value intervals
5. name must be <=6 words; free-text fields must be <=10 words - never paste full sentences
6. SKIP any Finding or Symptom whose value starts with "no", "without", "absent", "negative", "not seen", "not identified"

## Output - JSON only, no other text
{{
  "entities": [
    {{"category": "Diagnosis", "name": "...", "status": "...", "severity": "...", "etiology": "...", "treatment_context": "..."}},
    {{"category": "Symptom", "name": "...", "severity": "...", "onset": "...", "trend": "...", "character": "..."}},
    {{"category": "LabResult", "name": "...", "value": "...", "unit": "..."}},
    {{"category": "Medication", "name": "...", "dose": "...", "frequency": "...", "route": "..."}},
    {{"category": "Procedure", "name": "...", "result": "..."}},
    {{"category": "Finding", "name": "...", "value": "..."}}
  ],
  "impression": "one-sentence summary of key findings or diagnoses (same language as input text)",
  "indication": "reason for study/visit if mentioned, else empty string"
}}

Clinical text:
{text}
"""


ENTITY_FIELDS = {
    "Diagnosis": ("name", "status", "severity", "etiology", "treatment_context"),
    "Symptom": ("name", "severity", "onset", "trend", "character"),
    "LabResult": ("name", "value", "unit"),
    "Medication": ("name", "dose", "frequency", "route"),
    "Procedure": ("name", "result"),
    "Finding": ("name", "value"),
}

ENTITY_SUBFIELDS = {
    "Diagnosis": ("status", "severity", "etiology", "treatment_context"),
    "Symptom": ("severity", "onset", "trend", "character"),
    "LabResult": ("value", "unit"),
    "Medication": ("dose", "frequency", "route"),
    "Procedure": ("result",),
    "Finding": ("value",),
}


def build_fill_mapping_prompt(patient_id: str, entities: list[dict], empty_cols: list[str]) -> str:
    """Build the OCR fill-table LLM prompt for mapping entities to empty columns."""
    ent_lines: list[str] = []
    for i, entity in enumerate(entities):
        category = entity.get("category", "")
        name = entity.get("canonical_name") or entity.get("name", "")
        values = []
        for field in ENTITY_FIELDS.get(str(category), ("value",)):
            if field in {"category", "name"}:
                continue
            value = str(entity.get(field) or "").strip()
            if value:
                values.append(f"{field}={value}")
        value_text = "; ".join(values) if values else str(entity.get("value") or entity.get("name") or "").strip()
        ent_lines.append(f"  [{i}] [{category}] {name}: {value_text or '(无具体值)'}")

    col_lines = [f"  [{j}] {col}" for j, col in enumerate(empty_cols)]
    return f"""你是医学数据录入专家。请将下面的"抽取实体"匹配到"目标列"中，并输出填写值。

## 患者ID
{patient_id}

## 抽取到的实体（共{len(entities)}条）
{chr(10).join(ent_lines)}

## 需要填写的空列（共{len(empty_cols)}列，仅选出你能确定匹配的列）
{chr(10).join(col_lines)}

## 匹配规则
1. 只填写你**高置信度**确定能匹配的列，不确定的跳过
2. 列名含"慢相角速度"→ 填数值（°/s）；含"增益"→ 填增益数值；含"试验结果"→ 填阴性/阳性/未见眼震等
3. 列名含"自发眼震"→ 对应实体"自发眼震"的 value；含"扫视试验"→ 对应"扫视"的 value
4. 列名含"跟踪试验"→ 对应"视跟踪-水平方向"；含"视动眼震"→ 对应"视动眼震-水平方向"
5. 列名含"Dix-hallpike"→ 对应 "Dix-Hallpike" 相关实体；含"Roll试验"→ 对应"Roll-test"
6. 实验室检查：列名含"类风湿因子"→ 实体"风湿因子"value；含"C反应蛋白"→ 实体"超敏C反应蛋白"value
7. 若同一列有多个来源（多次检查），取**最后一次**（source_file 编号最大）的值
8. 若列名含"(详细)"或"其他"，填写原始文本描述

## 输出格式（JSON，不要输出其他内容）
{{
  "填写结果": [
    {{"列名": "精确的列名字符串", "值": "填入的值"}}
  ],
  "跳过": ["无法高置信度匹配的列名"]
}}
"""


STANDARDIZE_PROMPT = """\
你是医学术语标准化专家。请对以下已抽取的医学实体进行标准化：

1. 术语标准化：将实体名称映射到标准编码（Disease→ICD-10, Drug→ATC, Test/LabValue→LOINC, Symptom/Finding→SNOMED-CT）
2. 量纲统一：如果实体有 value+unit，将其转换为标准单位（血糖→mmol/L，血红蛋白→g/L，白细胞→×10⁹/L 等）

输入实体列表：
{entities_json}

请在每个实体中添加以下字段（无法确定则省略）：
- standard_code: 标准编码
- standard_name: 标准英文名
- standard_system: 编码系统（ICD-10/ATC/LOINC/SNOMED-CT）
- normalized_value: 标准化数值（float）
- normalized_unit: 标准化单位

只返回更新后的 entities JSON 数组，不要其他文字。
"""
