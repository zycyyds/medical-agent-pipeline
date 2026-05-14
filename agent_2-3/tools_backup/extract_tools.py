# -*- coding: utf-8 -*-
"""
信息抽取工具（Extract Tools）

使用 LLM 从医学文本中提取结构化实体。
供 ReActAgent 直接调用，也可被 CSVProcessor 和目录处理流程使用。

工具列表：
  - extract_from_text          从纯文本中抽取医学实体（async）
  - extract_from_text_sync     同步版本（内部用 asyncio.run）
  - standardize_entities       对已抽取的实体进行标准化（async）
"""
import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_AS_SRC = os.path.abspath(os.path.join(_HERE, "../../src"))
if _AS_SRC not in sys.path:
    sys.path.insert(0, _AS_SRC)

_OLD_ROOT = os.path.join(os.path.dirname(_HERE), "medical_data_cleaner")
if _OLD_ROOT not in sys.path:
    sys.path.insert(0, _OLD_ROOT)

from config import get_api_config


# ---------------------------------------------------------------------------
# 内部 LLM 调用
# ---------------------------------------------------------------------------

async def _call_llm(prompt: str, model_name: Optional[str] = None) -> Optional[str]:
    """通用 LLM 调用，返回原始文本响应。"""
    try:
        import openai
        cfg = get_api_config()
        if not cfg["api_key"]:
            return None
        client = openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
        resp = await client.chat.completions.create(
            model=model_name or cfg["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip() if resp.choices else None
    except Exception:
        return None


def _parse_json_response(content: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON。"""
    if not content:
        return None
    for marker in ["```json", "```"]:
        if marker in content:
            parts = content.split(marker)
            if len(parts) >= 2:
                try:
                    return json.loads(parts[1].split("```")[0].strip())
                except json.JSONDecodeError:
                    pass
    try:
        start, end = content.find("{"), content.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# 提取 Prompt 模板
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a medical information extraction expert. Extract structured entities from the clinical text below.

## Output language rule
ALWAYS use the SAME language as the input text. If the text is Chinese, output Chinese. If English, output English. Never translate.

## Entity Categories

**Diagnosis** — confirmed or suspected diagnoses / conditions
  name: condition name | value: status qualifier if stated (e.g. "confirmed", "suspected", "history of") | unit: ""

**Symptom** — patient-reported symptoms or chief complaints
  name: symptom name | value: qualifier / severity / duration if present | unit: ""

**LabResult** — any measurement with a numeric value (lab tests, vital signs, functional scores, angles, ratios, etc.)
  name: test or measurement name | value: numeric value ONLY (digits, optional sign/decimal) | unit: unit string
  ⚠ value must be a pure number — never include units or descriptive text in value field

**Medication** — drugs or medications mentioned
  name: drug name | value: dose / frequency / route if stated | unit: ""

**Procedure** — medical procedures or interventions
  name: procedure name | value: result or key finding if stated | unit: ""

**Finding** — qualitative observations from physical exam, imaging, or functional tests (non-numeric)
  name: finding name or anatomical / test location | value: qualitative description | unit: ""

## Rules
1. Output language MUST match input text language — never translate
2. LabResult.value must be a pure number (e.g. "5.2", "-3", "0.67") — no units, no text
3. Extract only explicitly stated information — do not infer
4. Do not extract reference ranges or normal value intervals as entities
5. Keep names concise but specific; do not paste full sentences into any single field

## Output — JSON only, no other text
{{
  "entities": [
    {{"category": "Diagnosis|Symptom|LabResult|Medication|Procedure|Finding", "name": "...", "value": "...", "unit": ""}}
  ],
  "impression": "one-sentence summary of key findings or diagnoses (same language as input text)",
  "indication": "reason for study/visit if mentioned, else empty string"
}}

Clinical text:
{text}
"""


def _postprocess_entities(entities: List[Dict]) -> List[Dict]:
    """
    通用后处理：
    1. LabResult.value 强制为纯数字字符串
    2. 过滤明显的参考范围实体
    """
    import re
    cleaned = []
    for ent in entities:
        cat = ent.get("category", "")
        value = ent.get("value", "")

        # 过滤参考范围（name 或 value 中含"参考/正常/reference/normal range"且 value 含数字区间）
        if isinstance(value, str) and re.search(r'\d+\s*[~～\-]\s*\d+', value):
            name_lower = str(ent.get("name", "")).lower()
            if any(kw in name_lower for kw in ("参考", "正常值", "reference", "normal range")):
                continue

        # LabResult.value 必须是纯数字
        if cat == "LabResult" and value is not None:
            if isinstance(value, (int, float)):
                ent["value"] = str(value)
            elif isinstance(value, str) and value.strip():
                match = re.search(r'[+-]?\d+\.?\d*', value)
                ent["value"] = match.group() if match else value

        cleaned.append(ent)
    return cleaned


async def extract_from_text(text: str, max_chars: int = 4000) -> Dict[str, Any]:
    """
    使用 LLM 从医学文本中提取结构化实体（异步）。

    Args:
        text: 医学文本（自由文本，如出院记录、检查报告等）
        max_chars: 最大处理字符数（超出则截断）

    Returns:
        {
          "success": bool,
          "entities": [{"category", "name", "value", "unit", "original_text"}],
          "temporal_info": [...],
          "quantity_info": [...],
          "relations": [...],
          "impression": str,
          "indication": str,
          "entity_count": int,
          "error": str | None,
        }
    """
    if not text or not text.strip():
        return {"success": False, "entities": [], "temporal_info": [], "quantity_info": [],
                "relations": [], "impression": "", "indication": "", "entity_count": 0,
                "error": "输入文本为空"}

    prompt = _EXTRACT_PROMPT.format(text=text[:max_chars])
    content = await _call_llm(prompt)
    parsed = _parse_json_response(content) if content else None

    if parsed is None:
        return {"success": False, "entities": [], "temporal_info": [], "quantity_info": [],
                "relations": [], "impression": "", "indication": "", "entity_count": 0,
                "error": "LLM 抽取失败或 API 不可用"}

    entities = parsed.get("entities", [])
    entities = _postprocess_entities(entities)

    return {
        "success": True,
        "entities": entities,
        "temporal_info": parsed.get("temporal_info", []),
        "quantity_info": parsed.get("quantity_info", []),
        "relations": parsed.get("relations", []),
        "impression": parsed.get("impression", ""),
        "indication": parsed.get("indication", ""),
        "entity_count": len(entities),
        "error": None,
    }


def extract_from_text_sync(text: str, max_chars: int = 4000) -> Dict[str, Any]:
    """
    同步版本的 extract_from_text，供非 async 上下文调用。
    """
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, extract_from_text(text, max_chars))
                return future.result()
        return asyncio.run(extract_from_text(text, max_chars))
    except Exception as e:
        return {"success": False, "entities": [], "temporal_info": [], "quantity_info": [],
                "relations": [], "impression": "", "indication": "", "entity_count": 0,
                "error": str(e)}


# ---------------------------------------------------------------------------
# 标准化 Prompt 模板（对已抽取实体批量标准化）
# ---------------------------------------------------------------------------

_STANDARDIZE_PROMPT = """\
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


async def standardize_entities(
    entities: List[Dict[str, Any]],
    batch_size: int = 20,
) -> Dict[str, Any]:
    """
    对已抽取的实体列表进行批量标准化（术语编码 + 量纲统一）。

    Args:
        entities: extract_from_text 返回的 entities 列表
        batch_size: 每批处理的实体数（避免 token 超限）

    Returns:
        {
          "success": bool,
          "entities": [标准化后的实体列表],
          "standardized_count": int,
          "error": str | None,
        }
    """
    if not entities:
        return {"success": True, "entities": [], "standardized_count": 0, "error": None}

    results: List[Dict] = []
    for i in range(0, len(entities), batch_size):
        batch = entities[i: i + batch_size]
        prompt = _STANDARDIZE_PROMPT.format(
            entities_json=json.dumps(batch, ensure_ascii=False, indent=2)
        )
        content = await _call_llm(prompt)
        parsed = None
        if content:
            # 响应是数组
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                parsed = json.loads(content.strip())
            except json.JSONDecodeError:
                pass

        if isinstance(parsed, list):
            results.extend(parsed)
        else:
            # 标准化失败，原样保留
            results.extend(batch)

    std_count = sum(
        1 for e in results if e.get("standard_code") or e.get("normalized_value") is not None
    )
    return {
        "success": True,
        "entities": results,
        "standardized_count": std_count,
        "error": None,
    }
