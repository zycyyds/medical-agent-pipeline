# -*- coding: utf-8 -*-
"""
ReactMedicalAgent —— 基于 ReAct 框架的统一医学数据处理 Agent

## 架构
                  ┌─────────────────────────────────────┐
  用户输入路径 ──► │          ReactMedicalAgent          │
                  │                                     │
                  │  ┌─────────────────────────────┐   │
                  │  │   ReActAgent (探索+决策)    │   │
                  │  │                             │   │
                  │  │  Thought: 分析数据情况      │   │
                  │  │  Action:  调用工具探索      │   │
                  │  │  Obs:     工具返回结果      │   │
                  │  │  ...（多轮）                │   │
                  │  │  Final:   输出处理计划JSON  │   │
                  │  └─────────────────────────────┘   │
                  │              │                      │
                  │              ▼ ProcessingPlan       │
                  │  ┌─────────────────────────────┐   │
                  │  │   Executor（按计划执行）    │   │
                  │  │   _run_csv / _run_image /   │   │
                  │  │   _run_text / _run_directory │   │
                  │  └─────────────────────────────┘   │
                  └─────────────────────────────────────┘

## 注册的工具（ReAct 可调用）

### 探索类（先调用，了解数据）
  - detect_input_type     检测数据类型
  - read_csv_sample       读 CSV 样本 + 自动列分析
  - get_csv_full_info     获取 CSV 完整报告
  - scan_directory        扫描目录（患者分组）
  - list_directory        列出目录内容
  - read_text_sample      读文本样本

### OCR 类（图片处理时调用）
  - ocr_image             图片 OCR
  - preprocess_text       文本清洗
  - ocr_and_clean         OCR+清洗合一

### 抽取类（文本/图片内容抽取时调用）
  - extract_from_text     从文本抽取医学实体（async）
  - standardize_entities  对实体批量标准化（async）

### IO 类（读写时调用）
  - read_csv_all          读取 CSV 全部数据
  - save_rows_to_csv      保存 CSV
  - save_result_json      保存 JSON
  - load_excel_table      读 Excel 表格
  - build_output_dir      创建带时间戳输出目录
"""
import inspect
import json
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_AS_SRC = os.path.abspath(os.path.join(_HERE, "../../src"))
if _AS_SRC not in sys.path:
    sys.path.insert(0, _AS_SRC)

from agentscope.agent import AgentBase, ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from agentscope.tool._response import ToolResponse
from agentscope.message import TextBlock

from config import get_api_config

# 导入所有工具
from tools.explore_tools import (
    detect_input_type, get_csv_full_info, list_directory,
    read_csv_sample, read_text_sample, scan_directory, read_excel_sample,
    read_jsonl_sample,
)
from tools.ocr_tools import ocr_and_clean, ocr_image, preprocess_text
from tools.extract_tools import extract_from_text, standardize_entities
from tools.io_tools import (
    build_output_dir, get_patient_row, load_excel_table,
    read_csv_all, read_excel_all, save_result_json, save_rows_to_csv,
)


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
你是一位专业的医学数据分析专家，负责探索用户提供的医学数据并制定最优处理计划。

## 工作流程（ReAct 多轮循环）

**Phase 1：探索**（可调用多次工具，充分了解数据）
- 先用 `detect_input_type` 判断数据类型（csv/image/text/jsonl/directory）
- 如果是 CSV → 用 `read_csv_sample` 查看列名和样本，再用 `get_csv_full_info` 获取完整报告
- 如果是 Excel（.xlsx/.xls）→ 用 `read_excel_sample` 查看列名和样本
- 如果是 JSONL（.jsonl）→ 用 `read_jsonl_sample` 查看顶层字段和样本
- 如果是目录 → 用 `scan_directory` 了解患者分组和文件分布；如果需要可用 `list_directory` 细看
- 如果是文本文件 → 用 `read_text_sample` 查看前几百字
- 如果是图片 → **不要**调用 OCR 工具，直接根据文件扩展名判断为图片类型即可

## 禁止事项
- **探索阶段严禁调用 OCR 工具**（`ocr_image`、`ocr_and_clean`、`preprocess_text`）
- OCR 是执行阶段的工作，探索阶段只需判断数据类型和结构

**Phase 2：决策**（分析探索结果，输出处理计划）
输出一个 ```json ``` 代码块，格式如下：

```json
{
    "data_type": "csv | image | text | jsonl | directory",
    "input_path": "输入路径",
    "csv_plan": {
        "extraction_columns": ["需要信息抽取的长文本列名"]
    },
    "jsonl_plan": {
        "text_fields": ["包含长文本的顶层字段名，如 就诊文本、notes 等"]
    },
    "directory_plan": {
        "has_patient_groups": true,
        "has_excel_table": false
    },
    "output_format": "csv | json | both",
    "reasoning": "一句话说明决策理由"
}
```

## 列判断规则（CSV）
- **extraction_columns**：长文本列（text, note, report, description, findings 等，通常 > 100 字符）
- ID 列、时间列、日期列、数值列、术语列：不需要处理

## JSONL 判断规则
- **text_fields**：包含长文本的顶层字段（如 `就诊文本`、`notes`、`discharge_note` 等）
- 结构化字段（诊断列表、实验室检验、生命体征等）：不需要处理

## 重要原则
- 先探索再决策，不基于文件名猜测
- 输出的 JSON 必须完整，所有字段都要有
"""


# ---------------------------------------------------------------------------
# ProcessingPlan：ReAct 产出的结构化处理计划
# ---------------------------------------------------------------------------

class ProcessingPlan:
    """解析并持有 ReAct Agent 输出的 JSON 处理计划。"""

    def __init__(self, raw: Dict[str, Any], input_path: str):
        self.data_type: str = raw.get("data_type", "unknown")
        self.input_path: str = raw.get("input_path", input_path)

        csv_plan = raw.get("csv_plan", {})
        self.extraction_columns: List[str] = csv_plan.get("extraction_columns", [])

        jsonl_plan = raw.get("jsonl_plan", {})
        self.text_fields: List[str] = jsonl_plan.get("text_fields", [])

        dir_plan = raw.get("directory_plan", {})
        self.has_patient_groups: bool = dir_plan.get("has_patient_groups", True)
        self.has_excel_table: bool = dir_plan.get("has_excel_table", False)

        self.output_format: str = raw.get("output_format", "both")
        self.reasoning: str = raw.get("reasoning", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_type": self.data_type,
            "input_path": self.input_path,
            "csv_plan": {
                "extraction_columns": self.extraction_columns,
            },
            "jsonl_plan": {
                "text_fields": self.text_fields,
            },
            "directory_plan": {
                "has_patient_groups": self.has_patient_groups,
                "has_excel_table": self.has_excel_table,
            },
            "output_format": self.output_format,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# 工具包装辅助：把返回 dict/str 的函数包装成返回 ToolResponse
# ---------------------------------------------------------------------------

def _wrap_tool(fn: Callable) -> Callable:
    """将返回 dict 或 str 的工具函数包装为返回 ToolResponse。"""
    import functools

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            result = await fn(*args, **kwargs)
            text = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            return ToolResponse(content=[TextBlock(type="text", text=text)])
        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            text = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            return ToolResponse(content=[TextBlock(type="text", text=text)])
        return sync_wrapper


# ---------------------------------------------------------------------------
# 工具注册辅助
# ---------------------------------------------------------------------------

def _build_toolkit() -> Toolkit:
    """构建并返回注册了所有工具的 Toolkit。"""
    tk = Toolkit()
    groups = {
        "explore":  "数据探索工具 —— 了解数据类型和结构",
        "ocr":      "OCR 工具 —— 图片文字识别和文本清洗",
        "extract":  "信息抽取工具 —— 从文本/图片中抽取医学实体",
        "io":       "IO 工具 —— 文件读写和目录管理",
    }
    for gname, gdesc in groups.items():
        tk.create_tool_group(group_name=gname, description=gdesc, active=True)

    # 探索工具
    for fn, desc in [
        (detect_input_type,  "检测输入路径的数据类型（image/text/csv/excel/jsonl/directory）"),
        (read_csv_sample,    "读取 CSV 文件的列名和样本数据，附带自动列分析"),
        (get_csv_full_info,  "获取 CSV 文件的完整可读分析报告（字符串）"),
        (read_excel_sample,  "读取 Excel 文件（.xlsx/.xls）的列名和样本数据，附带自动列分析"),
        (read_jsonl_sample,  "读取 JSONL 文件的顶层字段名和前 N 条记录摘要"),
        (scan_directory,     "扫描目录：返回患者分组、文件类型分布和摘要"),
        (list_directory,     "列出目录下的文件和子目录（非递归）"),
        (read_text_sample,   "读取文本文件前 N 个字符"),
    ]:
        tk.register_tool_function(_wrap_tool(fn), group_name="explore", func_description=desc)

    # OCR 工具
    for fn, desc in [
        (ocr_image,       "对图片文件进行 OCR 识别，返回提取的原始文本"),
        (preprocess_text, "清洗医学文本（去噪、规范化符号）"),
        (ocr_and_clean,   "OCR + 文本清洗一步完成"),
    ]:
        tk.register_tool_function(_wrap_tool(fn), group_name="ocr", func_description=desc)

    # 抽取工具
    for fn, desc in [
        (extract_from_text,   "使用 LLM 从医学文本中抽取结构化实体（async）"),
        (standardize_entities, "对已抽取实体列表进行批量术语标准化（async）"),
    ]:
        tk.register_tool_function(_wrap_tool(fn), group_name="extract", func_description=desc)

    # IO 工具
    for fn, desc in [
        (read_csv_all,      "读取 CSV 文件的全部数据（返回行数据列表）"),
        (save_rows_to_csv,  "将行数据列表保存为 CSV 文件"),
        (save_result_json,  "将处理结果字典保存为 JSON 文件"),
        (load_excel_table,  "读取 Excel 文件并按患者 ID 索引返回数据字典"),
        (get_patient_row,   "从已加载的 Excel 表格中获取指定患者的行数据"),
        (build_output_dir,  "在指定目录下创建带时间戳的子目录"),
    ]:
        tk.register_tool_function(_wrap_tool(fn), group_name="io", func_description=desc)

    return tk


# ---------------------------------------------------------------------------
# ReactMedicalAgent 主类
# ---------------------------------------------------------------------------

class ReactMedicalAgent(AgentBase):
    """
    基于 ReAct 框架的统一医学数据处理 Agent。

    工作方式：
    1. 用 ReActAgent 进行多轮 Thought/Action 探索数据
    2. 从最终回答中解析 ProcessingPlan
    3. Executor 按计划调用工具完成处理（CSV标准化/图片OCR+抽取/目录批处理）
    4. 将结果保存到输出目录
    """

    def __init__(
        self,
        name: str = "ReactMedicalAgent",
        model: Optional[OpenAIChatModel] = None,
        use_llm: bool = True,
        verbose: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            name: Agent 名称
            model: LLM 模型（None 则自动初始化）
            use_llm: 是否使用 LLM（False 时退化为纯规则处理）
            verbose: 是否打印详细日志
            output_dir: 输出根目录（None 则在输入文件同目录下创建 results_v2）
        """
        super().__init__()
        self.name = name
        self.use_llm = use_llm
        self.verbose = verbose
        self.output_dir = output_dir
        self.stats = {"total": 0, "success": 0, "failed": 0, "by_type": {}}

        # 初始化模型
        self.model: Optional[OpenAIChatModel] = model or (
            self._init_model() if use_llm else None
        )

        # 构建 Toolkit（所有工具共享）
        self._toolkit = _build_toolkit()

        # 构建 ReAct 探索 Agent
        self._explorer: Optional[ReActAgent] = None
        if self.model:
            self._explorer = ReActAgent(
                name=f"{name}_Explorer",
                sys_prompt=SYSTEM_PROMPT,
                model=self.model,
                formatter=OpenAIChatFormatter(),
                toolkit=self._toolkit,
            )

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_model(self) -> Optional[OpenAIChatModel]:
        cfg = get_api_config()
        if not cfg["api_key"]:
            if self.verbose:
                print(f"[{self.name}] 未找到 API Key，将使用规则模式")
            return None
        try:
            return OpenAIChatModel(
                config_name=f"{self.name}-model",
                model_name=cfg["model_name"],
                api_key=cfg["api_key"],
                client_kwargs={"base_url": cfg["api_base"], "timeout": cfg["timeout"]},
                generate_kwargs={"temperature": 0.0},
            )
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] 模型初始化失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def reply(self, msg: "Msg | dict | None" = None) -> Msg:
        """
        处理输入消息。

        Args:
            msg: content 为输入文件/目录路径

        Returns:
            Msg，metadata 包含完整处理结果
        """
        if msg is None:
            return Msg(self.name, "未提供输入", role="assistant")

        input_path: str = (
            msg.content if isinstance(msg, Msg)
            else msg.get("content", "") if isinstance(msg, dict)
            else str(msg)
        )
        input_path = input_path.strip()

        if self.verbose:
            print(f"\n[{self.name}] ─── 开始处理: {input_path} ───")

        # Phase 1：ReAct 探索 → 处理计划
        plan = await self._explore_and_plan(input_path)

        if self.verbose:
            print(f"[{self.name}] 处理计划:")
            print(f"  类型: {plan.data_type}")
            print(f"  理由: {plan.reasoning}")

        # Phase 2：按计划执行
        try:
            result = await self._execute(plan, input_path)
        except Exception as e:
            result = {"success": False, "error": str(e),
                      "data_type": plan.data_type, "plan": plan.to_dict()}
            self.stats["failed"] += 1
            if self.verbose:
                import traceback
                print(f"[{self.name}] 执行失败: {e}")
                traceback.print_exc()
        else:
            if result.get("success"):
                self.stats["success"] += 1
            else:
                self.stats["failed"] += 1

        self.stats["total"] += 1
        self.stats["by_type"][plan.data_type] = (
            self.stats["by_type"].get(plan.data_type, 0) + 1
        )

        summary = self._summarize(result)
        return Msg(self.name, content=summary, role="assistant", metadata=result)

    # ------------------------------------------------------------------
    # Phase 1：探索 → ProcessingPlan
    # ------------------------------------------------------------------

    async def _explore_and_plan(self, input_path: str) -> ProcessingPlan:
        """让 ReAct Agent 探索数据，解析其输出为 ProcessingPlan。"""
        if not self._explorer:
            return self._rule_based_plan(input_path)

        user_msg = Msg(
            name="User",
            content=f"请探索以下医学数据，充分了解数据情况后输出处理计划：\n\n输入路径：{input_path}",
            role="user",
        )
        try:
            resp = await self._explorer(user_msg)
            raw = self._parse_plan_json(resp.content or "", input_path)
            return ProcessingPlan(raw, input_path)
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] ReAct 探索异常，退化为规则计划: {e}")
            return self._rule_based_plan(input_path)

    def _parse_plan_json(self, content: Any, input_path: str) -> Dict[str, Any]:
        """从 ReAct 最终回答中提取 JSON 处理计划。"""
        # content 可能是 list（Msg.content 为 block 列表）或 str
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            content = "\n".join(parts)
        elif not isinstance(content, str):
            content = str(content)
        for marker in ["```json", "```"]:
            if marker in content:
                try:
                    raw = content.split(marker)[1].split("```")[0]
                    parsed = json.loads(raw.strip())
                    parsed.setdefault("input_path", input_path)
                    return parsed
                except (IndexError, json.JSONDecodeError):
                    pass
        try:
            s, e = content.find("{"), content.rfind("}") + 1
            if s != -1 and e > s:
                parsed = json.loads(content[s:e])
                parsed.setdefault("input_path", input_path)
                return parsed
        except json.JSONDecodeError:
            pass
        if self.verbose:
            print(f"[{self.name}] 无法解析 JSON 计划，退化为规则模式")
        return self._rule_based_plan(input_path).to_dict()

    def _rule_based_plan(self, input_path: str) -> ProcessingPlan:
        """无 LLM 时，纯规则生成处理计划。"""
        info = detect_input_type(input_path)
        data_type = info["data_type"]
        raw: Dict[str, Any] = {
            "data_type": data_type, "input_path": input_path,
            "csv_plan": {"extraction_columns": []},
            "jsonl_plan": {"text_fields": []},
            "directory_plan": {"has_patient_groups": False, "has_excel_table": False},
            "output_format": "both", "reasoning": "规则自动检测",
        }

        if data_type == "csv":
            sample = read_csv_sample(input_path)
            if sample["success"]:
                analysis = sample["column_analysis"]
                raw["csv_plan"]["extraction_columns"] = analysis.get("extraction_candidates", [])
            raw["reasoning"] = "CSV 文件，规则自动分析列"

        elif data_type == "excel":
            sample = read_excel_sample(input_path)
            if sample["success"]:
                analysis = sample["column_analysis"]
                raw["csv_plan"]["extraction_columns"] = analysis.get("extraction_candidates", [])
            raw["reasoning"] = "Excel 文件，规则自动分析列"

        elif data_type == "jsonl":
            sample = read_jsonl_sample(input_path, sample_rows=2)
            if sample["success"]:
                # 按关键词自动识别长文本字段
                text_kw = {"text", "note", "report", "description", "findings",
                           "impression", "narrative", "discharge", "就诊文本", "notes"}
                text_fields = [k for k in sample["top_keys"]
                               if any(kw in k.lower() for kw in text_kw)]
                raw["jsonl_plan"]["text_fields"] = text_fields
            raw["reasoning"] = "JSONL 文件，规则自动识别文本字段"

        elif data_type == "directory":
            folder_info = scan_directory(input_path)
            raw["directory_plan"]["has_patient_groups"] = bool(folder_info.get("patients"))
            raw["directory_plan"]["has_excel_table"] = folder_info.get("has_table", False)
            raw["reasoning"] = "目录输入，按患者分组处理"

        return ProcessingPlan(raw, input_path)

    # ------------------------------------------------------------------
    # Phase 2：Executor
    # ------------------------------------------------------------------

    async def _execute(self, plan: ProcessingPlan, input_path: str) -> Dict[str, Any]:
        """根据处理计划调用对应的执行流程。"""
        # 如果实际路径是目录，无论 plan 类型如何都走目录流程
        if os.path.isdir(input_path):
            return await self._run_directory(input_path, plan)
        dt = plan.data_type
        if dt == "csv":
            return await self._run_csv(input_path, plan)
        elif dt == "excel":
            return await self._run_excel(input_path, plan)
        elif dt == "jsonl":
            return await self._run_jsonl(input_path, plan)
        elif dt == "image":
            return await self._run_image(input_path)
        elif dt == "text":
            return await self._run_text(input_path)
        elif dt == "directory":
            return await self._run_directory(input_path, plan)
        else:
            # 再次自动检测
            info = detect_input_type(input_path)
            dt2 = info["data_type"]
            if dt2 == "csv":
                return await self._run_csv(input_path, plan)
            elif dt2 == "excel":
                return await self._run_excel(input_path, plan)
            elif dt2 == "jsonl":
                return await self._run_jsonl(input_path, plan)
            elif dt2 == "image":
                return await self._run_image(input_path)
            elif dt2 == "text":
                return await self._run_text(input_path)
            return {"success": False, "error": f"不支持的数据类型: {dt}", "data_type": dt}

    # ── JSONL 处理流程 ─────────────────────────────────────────────────

    async def _run_jsonl(self, file_path: str, plan: ProcessingPlan) -> Dict[str, Any]:
        """
        JSONL 处理流程：
          1. 逐行读取记录
          2. 对 text_fields 中的字段做信息抽取
          3. 保存结果（JSON + CSV）
        """
        if self.verbose:
            print(f"[{self.name}] [JSONL] 开始处理: {file_path}")

        records: List[Dict] = []
        skipped = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1

        if not records:
            return {"success": False, "error": "JSONL 文件无有效记录", "data_type": "jsonl"}

        stats = {
            "total_rows": len(records),
            "skipped_rows": skipped,
            "extracted_texts": 0,
            "extracted_entities": 0,
        }

        # 对每条记录的 text_fields 做信息抽取
        for rec in records:
            for field in plan.text_fields:
                # 支持点号路径（如 "就诊文本.notes"）
                parts = field.split(".", 1)
                if len(parts) == 2:
                    top_val = rec.get(parts[0])
                    field_val = top_val.get(parts[1]) if isinstance(top_val, dict) else None
                    field_key = field.replace(".", "_")  # 用于写回 rec 的键名
                else:
                    field_val = rec.get(field)
                    field_key = field

                if field_val is None:
                    continue
                # 支持字段值为 dict、list 或 str
                if isinstance(field_val, list):
                    texts = []
                    for item in field_val:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(str(item["text"]))
                        elif isinstance(item, str) and len(item) > 50:
                            texts.append(item)
                    text = "\n".join(texts)
                elif isinstance(field_val, dict):
                    # 尝试拼接所有 notes 的 text
                    texts = []
                    for sub_key, sub_val in field_val.items():
                        if isinstance(sub_val, list):
                            for item in sub_val:
                                if isinstance(item, dict) and "text" in item:
                                    texts.append(str(item["text"]))
                        elif isinstance(sub_val, str) and len(sub_val) > 50:
                            texts.append(sub_val)
                    text = "\n".join(texts)
                elif isinstance(field_val, str):
                    text = field_val
                else:
                    continue

                if len(text.strip()) < 50:
                    continue

                ext_result = await extract_from_text(text)
                if not ext_result["success"]:
                    continue

                stats["extracted_texts"] += 1
                entities = ext_result["entities"]
                stats["extracted_entities"] += len(entities)

                rec[f"{field_key}_entities_json"] = json.dumps(entities, ensure_ascii=False)
                rec[f"{field_key}_entity_count"] = len(entities)
                rec[f"{field_key}_impression"] = ext_result.get("impression", "")
                by_cat: Dict[str, List[str]] = {}
                for ent in entities:
                    cat = ent.get("category", "Other")
                    name = ent.get("name", "")
                    val = ent.get("value", "")
                    entry = f"{name}:{val}" if val else name
                    by_cat.setdefault(cat, []).append(entry)
                for cat, items in by_cat.items():
                    rec[f"{field_key}_{cat}"] = "; ".join(items)

        # 保存结果
        out_dir = self._resolve_output_dir(file_path)
        stem = os.path.splitext(os.path.basename(file_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(out_dir, f"{stem}_processed_{ts}.json")
        csv_path  = os.path.join(out_dir, f"{stem}_processed_{ts}.csv")

        result_json = {
            "success": True, "data_type": "jsonl",
            "file_path": file_path,
            "processed_at": datetime.now().isoformat(),
            "plan": plan.to_dict(),
            "statistics": stats,
            "output_json": json_path,
            "output_csv": csv_path,
            "data": records,
        }
        save_result_json(result_json, json_path)

        # 扁平化保存 CSV（只保留非 dict/list 的字段 + 新增抽取字段）
        flat_rows = []
        for rec in records:
            flat = {}
            for k, v in rec.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
                else:
                    flat[k] = v
            flat_rows.append(flat)
        save_rows_to_csv(flat_rows, csv_path)

        if self.verbose:
            print(f"[{self.name}] [JSONL] 完成: {stats}")
            print(f"[{self.name}] [JSONL] 输出: {json_path}")

        return result_json

    async def _run_csv(self, file_path: str, plan: ProcessingPlan) -> Dict[str, Any]:
        """
        CSV 处理流程：
          1. 读取全部数据
          2. 对长文本列做信息抽取
          3. 保存结果
        """
        if self.verbose:
            print(f"[{self.name}] [CSV] 开始处理: {file_path}")

        # 1. 读取数据
        csv_data = read_csv_all(file_path)
        if not csv_data["success"]:
            return {"success": False, "error": csv_data["error"], "data_type": "csv"}

        rows: List[Dict] = csv_data["rows"]
        stats = {
            "total_rows": len(rows),
            "extracted_texts": 0,
            "extracted_entities": 0,
        }

        # 2. 长文本列：信息抽取
        for col in plan.extraction_columns:
            if self.verbose:
                print(f"[{self.name}] [CSV] 抽取列: {col}")
            for row in rows:
                text = str(row.get(col, "")).strip()
                if len(text) < 50:
                    continue
                ext_result = await extract_from_text(text)
                if not ext_result["success"]:
                    continue
                stats["extracted_texts"] += 1
                entities = ext_result["entities"]
                stats["extracted_entities"] += len(entities)

                # 写回行
                row[f"{col}_entities_json"] = json.dumps(entities, ensure_ascii=False)
                row[f"{col}_entity_count"] = len(entities)
                row[f"{col}_impression"] = ext_result.get("impression", "")
                # 按类别汇总
                by_cat: Dict[str, List[str]] = {}
                for ent in entities:
                    cat = ent.get("category", "Other")
                    name = ent.get("name", "")
                    val = ent.get("value", "")
                    entry = f"{name}:{val}" if val else name
                    by_cat.setdefault(cat, []).append(entry)
                for cat, items in by_cat.items():
                    row[f"{col}_{cat}"] = "; ".join(items)

        # 3. 保存结果
        out_dir = self._resolve_output_dir(file_path)
        stem = os.path.splitext(os.path.basename(file_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{stem}_processed_{ts}.csv")
        json_path = os.path.join(out_dir, f"{stem}_processed_{ts}.json")

        save_rows_to_csv(rows, csv_path)
        result_json = {
            "success": True, "data_type": "csv",
            "file_path": file_path,
            "processed_at": datetime.now().isoformat(),
            "plan": plan.to_dict(),
            "statistics": stats,
            "output_csv": csv_path,
            "output_json": json_path,
            "data": rows,
        }
        save_result_json(result_json, json_path)

        if self.verbose:
            print(f"[{self.name}] [CSV] 完成: {stats}")
            print(f"[{self.name}] [CSV] 输出: {csv_path}")

        return result_json

    # ── Excel 处理流程 ─────────────────────────────────────────────────

    async def _run_excel(self, file_path: str, plan: ProcessingPlan) -> Dict[str, Any]:
        """
        Excel 处理流程：
          1. 读取全部数据
          2. 对长文本列做信息抽取
          3. 保存结果（CSV + JSON）
        """
        if self.verbose:
            print(f"[{self.name}] [Excel] 开始处理: {file_path}")

        # 1. 读取数据
        excel_data = read_excel_all(file_path)
        if not excel_data["success"]:
            return {"success": False, "error": excel_data["error"], "data_type": "excel"}

        rows: List[Dict] = excel_data["rows"]
        stats = {
            "total_rows": len(rows),
            "extracted_texts": 0,
            "extracted_entities": 0,
        }

        # 2. 长文本列：信息抽取
        for col in plan.extraction_columns:
            if self.verbose:
                print(f"[{self.name}] [Excel] 抽取列: {col}")
            for row in rows:
                text = str(row.get(col, "")).strip()
                if len(text) < 50:
                    continue
                ext_result = await extract_from_text(text)
                if not ext_result["success"]:
                    continue
                stats["extracted_texts"] += 1
                entities = ext_result["entities"]
                stats["extracted_entities"] += len(entities)
                row[f"{col}_entities_json"] = json.dumps(entities, ensure_ascii=False)
                row[f"{col}_entity_count"] = len(entities)
                row[f"{col}_impression"] = ext_result.get("impression", "")
                by_cat: Dict[str, List[str]] = {}
                for ent in entities:
                    cat = ent.get("category", "Other")
                    name = ent.get("name", "")
                    val = ent.get("value", "")
                    entry = f"{name}:{val}" if val else name
                    by_cat.setdefault(cat, []).append(entry)
                for cat, items in by_cat.items():
                    row[f"{col}_{cat}"] = "; ".join(items)

        # 3. 保存结果
        out_dir = self._resolve_output_dir(file_path)
        stem = os.path.splitext(os.path.basename(file_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{stem}_processed_{ts}.csv")
        json_path = os.path.join(out_dir, f"{stem}_processed_{ts}.json")

        save_rows_to_csv(rows, csv_path)
        result_json = {
            "success": True, "data_type": "excel",
            "file_path": file_path,
            "sheet_name": excel_data.get("sheet_name", ""),
            "processed_at": datetime.now().isoformat(),
            "plan": plan.to_dict(),
            "statistics": stats,
            "output_csv": csv_path,
            "output_json": json_path,
            "data": rows,
        }
        save_result_json(result_json, json_path)

        if self.verbose:
            print(f"[{self.name}] [Excel] 完成: {stats}")
            print(f"[{self.name}] [Excel] 输出: {csv_path}")

        return result_json

    # ── 图片处理流程 ──────────────────────────────────────────────────

    async def _run_image(self, file_path: str) -> Dict[str, Any]:
        """
        图片处理流程：
          OCR → 文本清洗 → 信息抽取 → 保存
        """
        if self.verbose:
            print(f"[{self.name}] [Image] OCR + 抽取: {os.path.basename(file_path)}")

        result: Dict[str, Any] = {
            "success": False, "data_type": "image", "file_path": file_path,
            "ocr_text": "", "cleaned_text": "",
            "extraction_result": {},
        }

        # OCR
        ocr = ocr_and_clean(file_path)
        result["ocr_text"] = ocr.get("ocr_text", "")
        result["cleaned_text"] = ocr.get("cleaned_text", "")
        if not ocr.get("success") or not result["cleaned_text"]:
            result["error"] = ocr.get("error", "OCR 失败")
            return result

        # 信息抽取
        ext = await extract_from_text(result["cleaned_text"])
        result["extraction_result"] = ext
        result["success"] = ext.get("success", False)
        if not result["success"]:
            result["error"] = ext.get("error", "抽取失败")

        return result

    # ── 文本处理流程 ──────────────────────────────────────────────────

    async def _run_text(self, input_path: str) -> Dict[str, Any]:
        """
        文本处理流程：
          读取文本 → 信息抽取 → 保存
        """
        if self.verbose:
            print(f"[{self.name}] [Text] 抽取: {os.path.basename(input_path)}")

        # 读取文本内容
        if os.path.exists(input_path):
            sample = read_text_sample(input_path, max_chars=8000)
            text = sample.get("content", "")
        else:
            text = input_path  # 直接是文本内容

        result: Dict[str, Any] = {
            "success": False, "data_type": "text",
            "file_path": input_path if os.path.exists(input_path) else None,
            "extraction_result": {},
        }

        if not text.strip():
            result["error"] = "文本内容为空"
            return result

        ext = await extract_from_text(text)
        result["extraction_result"] = ext
        result["success"] = ext.get("success", False)
        if not result["success"]:
            result["error"] = ext.get("error", "抽取失败")

        return result

    # ── 目录处理流程 ──────────────────────────────────────────────────

    async def _run_directory(self, dir_path: str, plan: ProcessingPlan) -> Dict[str, Any]:
        """
        目录处理流程：
          扫描目录 → 加载 Excel 表格 → 按患者分组处理文件 → 保存汇总结果
        """
        if self.verbose:
            print(f"[{self.name}] [Dir] 扫描目录: {dir_path}")

        folder_info = scan_directory(dir_path)
        if not folder_info.get("success"):
            return {"success": False, "error": folder_info.get("error"), "data_type": "directory"}

        patients = folder_info.get("patients", {})
        flat_files = folder_info.get("flat_files", [])

        # 无患者子目录，但根目录有数据文件 → 直接处理文件
        if not patients and flat_files:
            if self.verbose:
                print(f"[{self.name}] [Dir] 无患者子目录，处理根目录文件: {[f['filename'] for f in flat_files]}")
            results = []
            for finfo in flat_files:
                fpath = finfo["path"]
                ftype = finfo["file_type"]
                try:
                    if ftype == "jsonl":
                        res = await self._run_jsonl(fpath, plan)
                    elif ftype == "csv":
                        res = await self._run_csv(fpath, plan)
                    elif ftype == "excel":
                        res = await self._run_excel(fpath, plan)
                    elif ftype == "text":
                        res = await self._run_text(fpath)
                    elif ftype == "image":
                        res = await self._run_image(fpath)
                    else:
                        res = {"success": False, "error": f"不支持的类型: {ftype}"}
                    results.append(res)
                except Exception as e:
                    results.append({"success": False, "error": str(e), "file": fpath})
            # 返回最后一个结果（通常只有一个文件）
            return results[0] if len(results) == 1 else {
                "success": all(r.get("success") for r in results),
                "data_type": "directory",
                "files": results,
            }

        if not patients:
            return {"success": False, "error": "目录中没有找到可处理的患者文件", "data_type": "directory"}

        # 加载 Excel/CSV 表格
        # reorganized_output 格式：每个患者各有独立 table_path，分别加载
        # 全局 table_path 仅用于兼容旧格式（根目录单一表格）
        global_table_path = folder_info.get("table_path")
        # 判断是否为"每患者独立表格"模式：所有患者都有各自的 table_path
        per_patient_tables = all(
            pdata.get("table_path") for pdata in patients.values()
        )
        global_table_data: Dict[str, Any] = {}
        if global_table_path and not per_patient_tables:
            tbl = load_excel_table(global_table_path)
            if tbl.get("success"):
                global_table_data = tbl.get("data", {})
                if self.verbose:
                    print(f"[{self.name}] [Dir] 全局表格加载: {len(global_table_data)} 条记录")

        out_dir = self._resolve_output_dir(dir_path)
        all_results: Dict[str, Any] = {
            "success": True, "data_type": "directory",
            "root_path": dir_path,
            "processed_at": datetime.now().isoformat(),
            "plan": plan.to_dict(),
            "statistics": {
                "total_patients": len(patients),
                "total_files": folder_info["statistics"]["total_files"],
                "processed_files": 0, "failed_files": 0,
            },
            "table_record_count": len(global_table_data) if not per_patient_tables else len(patients),
            "patients": {},
            "output_dir": out_dir,
        }

        for pid, pdata in patients.items():
            if self.verbose:
                print(f"\n[{self.name}] [Dir] 患者 {pid} ({pdata['file_count']} 个文件)")

            # 按患者独立表格 > 全局表格顺序取表格数据
            if per_patient_tables and pdata.get("table_path"):
                tbl = load_excel_table(pdata["table_path"])
                patient_table = next(iter(tbl["data"].values()), {}) if tbl.get("success") and tbl.get("data") else {}
                if self.verbose and patient_table:
                    print(f"[{self.name}] [Dir] 患者 {pid} 表格加载: {len(patient_table)} 列")
            else:
                patient_table = get_patient_row(global_table_data, pid) if global_table_data else {}

            patient_result = await self._process_patient(pid, pdata, patient_table)

            all_results["patients"][pid] = patient_result
            all_results["statistics"]["processed_files"] += patient_result.get("processed_count", 0)
            all_results["statistics"]["failed_files"] += patient_result.get("failed_count", 0)

        # 保存汇总 JSON
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(out_dir, f"summary_{ts}.json")
        save_result_json(all_results, summary_path)

        # 导出 CSV：实体明细表
        entities_rows = []
        for pid, pdata in all_results["patients"].items():
            for ent in pdata.get("merged_entities", []):
                row = {"patient_id": pid}
                row.update(ent)
                entities_rows.append(row)
        entities_csv = os.path.join(out_dir, f"entities_{ts}.csv")
        if entities_rows:
            save_rows_to_csv(entities_rows, entities_csv)

        # 导出 CSV：患者统计表
        patient_rows = []
        for pid, pdata in all_results["patients"].items():
            structured = pdata.get("structured_data", {})
            patient_rows.append({
                "patient_id": pid,
                "file_count": pdata.get("file_count", 0),
                "processed_count": pdata.get("processed_count", 0),
                "failed_count": pdata.get("failed_count", 0),
                "entity_count": len(pdata.get("merged_entities", [])),
                "categories": "; ".join(pdata.get("categories", [])),
                "labevents_count": len(structured.get("labevents", [])),
                "diagnoses_count": len(structured.get("diagnoses_icd", [])),
                "prescriptions_count": len(structured.get("prescriptions", [])),
                "structured_tables": "; ".join(structured.keys()),
            })
        patients_csv = os.path.join(out_dir, f"patients_{ts}.csv")
        save_rows_to_csv(patient_rows, patients_csv)

        all_results["output_entities_csv"] = entities_csv
        all_results["output_patients_csv"] = patients_csv

        # 导出 CSV：合并宽表（表格列 + 紧凑抽取列 + 结构化数据摘要，每患者一行）
        has_any_table = any(
            p.get("table_data") for p in all_results["patients"].values()
        )
        merged_csv = None
        if has_any_table:
            extra_rows: Dict[str, Dict] = {}
            extra_cols: list = []
            seen_extra: set = set()

            def _add_col(col: str) -> None:
                if col not in seen_extra:
                    seen_extra.add(col)
                    extra_cols.append(col)

            for pid, pdata in all_results["patients"].items():
                row: Dict = {}

                # ── 1. 紧凑抽取列：每个 category 一列，分号分隔 ──────────
                by_cat: Dict[str, list] = {}
                for e in pdata.get("merged_entities", []):
                    by_cat.setdefault(e.get("category", "Other"), []).append(e)

                for cat, ents in by_cat.items():
                    parts = []
                    for e in ents:
                        n = str(e.get("name", "") or "")
                        v = str(e.get("value", "") or "")
                        u = str(e.get("unit", "") or "")
                        val_str = f"{v} {u}".strip() if u else v
                        parts.append(f"{n}: {val_str}" if val_str else n)
                    col = f"{cat}_extracted"
                    row[col] = "; ".join(parts)
                    _add_col(col)

                # ── 2. 笔记 impression（兼容 OCR 图片 + notes CSV 两个来源）──
                impressions = []
                for finfo in pdata.get("files", []):
                    ext = finfo.get("extraction_result") or {}
                    if isinstance(ext, dict) and ext.get("impression"):
                        impressions.append(ext["impression"])
                for imp_entry in pdata.get("impressions", []):
                    if imp_entry.get("impression"):
                        impressions.append(imp_entry["impression"])
                if impressions:
                    row["notes_impression"] = " | ".join(impressions)
                    _add_col("notes_impression")

                # ── 3. 结构化数据摘要（通用启发式，不绑定任何 schema）────
                for tname, trows in pdata.get("structured_data", {}).items():
                    if not trows:
                        continue
                    count_col = f"{tname}_count"
                    row[count_col] = len(trows)
                    _add_col(count_col)

                    # 识别"关键列"：列名含 code/name/drug/type/icd/label/flag 等关键词
                    cols_in_table = list(trows[0].keys()) if trows else []
                    _KEY_KWS = {"code", "name", "drug", "type", "icd", "label",
                                "category", "flag", "class", "route", "diag"}
                    key_cols = [c for c in cols_in_table
                                if any(kw in c.lower() for kw in _KEY_KWS)][:3]
                    if key_cols:
                        seen_vals: set = set()
                        vals: list = []
                        for r in trows:
                            for kc in key_cols:
                                v = str(r.get(kc, "") or "").strip()
                                if v and v not in seen_vals:
                                    seen_vals.add(v)
                                    vals.append(v)
                        summary_col = f"{tname}_summary"
                        row[summary_col] = ", ".join(vals[:30])
                        _add_col(summary_col)

                extra_rows[pid] = row

            # 取第一个有 table_data 的患者确定表格列顺序
            first_table_row = next(
                (p["table_data"] for p in all_results["patients"].values() if p.get("table_data")),
                {}
            )
            out_cols = list(first_table_row.keys()) + extra_cols
            merged_rows = []
            for pid in sorted(extra_rows.keys()):
                merged_row = dict(all_results["patients"][pid].get("table_data", {}))
                merged_row.update(extra_rows[pid])
                merged_rows.append(merged_row)

            merged_csv = os.path.join(out_dir, f"merged_patients_{ts}.csv")
            save_rows_to_csv(merged_rows, merged_csv, columns=out_cols)
            all_results["output_merged_csv"] = merged_csv


        if self.verbose:
            s = all_results["statistics"]
            print(f"\n[{self.name}] [Dir] 完成: 患者={s['total_patients']}, "
                  f"成功={s['processed_files']}, 失败={s['failed_files']}")
            print(f"[{self.name}] [Dir] 汇总: {summary_path}")
            print(f"[{self.name}] [Dir] 实体CSV: {entities_csv}")
            print(f"[{self.name}] [Dir] 患者CSV: {patients_csv}")
            if merged_csv:
                print(f"[{self.name}] [Dir] 合并宽表: {merged_csv}")

        return all_results

    # notes/ 下 CSV 需要做文本抽取的列名
    _NOTES_TEXT_COLS = ("text", "report_text", "notes", "findings", "impression", "description")

    async def _process_patient(
        self,
        patient_id: str,
        pdata: Dict[str, Any],
        patient_table: Dict[str, Any],
    ) -> Dict[str, Any]:
        """处理单个患者的所有文件。"""
        patient_result: Dict[str, Any] = {
            "patient_id": patient_id,
            "categories": pdata.get("categories", []),
            "file_count": pdata.get("file_count", 0),
            "processed_count": 0,
            "failed_count": 0,
            "table_data": patient_table,
            "structured_data": {},   # table/structured/ 下各 CSV 的原始行
            "files": [],
            "merged_entities": [],
        }

        # ── 1. 处理 OCR 图片 ──────────────────────────────────────────
        for finfo in pdata.get("files", []):
            fpath = finfo["path"]
            ftype = finfo.get("file_type", "unknown")
            category = finfo.get("category", "unknown")

            try:
                if ftype == "image":
                    res = await self._run_image(fpath)
                elif ftype == "text":
                    res = await self._run_text(fpath)
                else:
                    res = {"success": False, "error": f"不支持的类型: {ftype}"}

                file_entry: Dict[str, Any] = {
                    "path": fpath,
                    "filename": os.path.basename(fpath),
                    "category": category,
                    "file_type": ftype,
                    "success": res.get("success", False),
                }

                if res.get("success"):
                    patient_result["processed_count"] += 1
                    if res.get("ocr_text"):
                        file_entry["ocr_text"] = res["ocr_text"]
                    if res.get("cleaned_text"):
                        file_entry["cleaned_text"] = res["cleaned_text"]

                    ext = res.get("extraction_result", {})
                    if isinstance(ext, dict) and ext.get("entities"):
                        file_entry["entities"] = ext["entities"]
                        for ent in ext["entities"]:
                            ec = {**ent, "source_file": os.path.basename(fpath), "source_category": category}
                            patient_result["merged_entities"].append(ec)
                else:
                    patient_result["failed_count"] += 1
                    file_entry["error"] = res.get("error", "处理失败")

                patient_result["files"].append(file_entry)

            except Exception as e:
                patient_result["failed_count"] += 1
                patient_result["files"].append({
                    "path": fpath, "filename": os.path.basename(fpath),
                    "category": category, "success": False, "error": str(e),
                })
                if self.verbose:
                    print(f"    ❌ {os.path.basename(fpath)}: {e}")

        # ── 2. 处理 table/ 目录下的所有数据 ──────────────────────────
        table_path = pdata.get("table_path")
        if table_path and os.path.exists(table_path):
            table_dir = os.path.dirname(table_path)
            await self._process_table_dir(table_dir, patient_id, patient_result)

        # ── 3. 对 merged_entities 去重 ────────────────────────────────
        # 只去除同一来源行内的重复实体（同一 source_file + row_id + category + name + value）
        # 跨行、跨文件的相同实体保留，因为可能来自不同时间的真实记录
        seen_keys: set = set()
        deduped: List[Dict] = []
        for ent in patient_result["merged_entities"]:
            key = (
                ent.get("source_file", ""),
                ent.get("row_id", ""),
                ent.get("category", ""),
                str(ent.get("name", "")).strip().lower(),
                str(ent.get("value", "")).strip().lower(),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(ent)
        patient_result["merged_entities"] = deduped

        return patient_result

    async def _process_table_dir(
        self,
        table_dir: str,
        patient_id: str,
        patient_result: Dict[str, Any],
    ) -> None:
        """
        扫描 table/ 目录，分三类处理：
          - 根目录 CSV（如 cxr_sampled_with_reports.csv）：对文本列做实体抽取
          - notes/ 子目录 CSV：对 text 列做实体抽取
          - structured/ 子目录 CSV：直接读取原始行，存入 structured_data
        """
        for entry in sorted(os.listdir(table_dir)):
            full = os.path.join(table_dir, entry)

            # 根目录 CSV
            if os.path.isfile(full) and entry.lower().endswith(".csv"):
                await self._extract_from_csv(full, patient_id, patient_result, source_category="table")

            # notes/ 子目录
            elif os.path.isdir(full) and entry.lower() == "notes":
                for fname in sorted(os.listdir(full)):
                    if fname.lower().endswith(".csv"):
                        fpath = os.path.join(full, fname)
                        await self._extract_from_csv(fpath, patient_id, patient_result,
                                                     source_category=f"notes/{fname}")

            # structured/ 子目录
            elif os.path.isdir(full) and entry.lower() == "structured":
                for fname in sorted(os.listdir(full)):
                    if fname.lower().endswith(".csv"):
                        fpath = os.path.join(full, fname)
                        self._load_structured_csv(fpath, fname, patient_result)

    async def _extract_from_csv(
        self,
        csv_path: str,
        patient_id: str,
        patient_result: Dict[str, Any],
        source_category: str,
    ) -> None:
        """读取 CSV 每行，对识别到的文本列做实体抽取，结果合并到 merged_entities。"""
        csv_data = read_csv_all(csv_path)
        if not csv_data.get("success"):
            return

        rows: List[Dict] = csv_data.get("rows", [])
        cols_lower = {c.lower(): c for c in (csv_data.get("columns") or [])}
        text_cols = [cols_lower[k] for k in self._NOTES_TEXT_COLS if k in cols_lower]
        if not text_cols:
            return

        fname = os.path.basename(csv_path)
        for row in rows:
            row_id = str(row.get("note_id", row.get("dicom_id", row.get("study_id", "")))).strip()
            for col in text_cols:
                text = str(row.get(col, "")).strip()
                if len(text) < 50:
                    continue
                if self.verbose:
                    print(f"[{self.name}] [Dir] 患者 {patient_id} 文本抽取: {source_category}/{col} (id={row_id})")
                try:
                    ext = await extract_from_text(text)
                except Exception:
                    continue
                if not ext.get("success") or not ext.get("entities"):
                    continue
                for ent in ext["entities"]:
                    patient_result["merged_entities"].append({
                        **ent,
                        "source_file": fname,
                        "source_category": source_category,
                        "row_id": row_id,
                    })
                # 收集 impression，供 merged CSV 使用
                if ext.get("impression"):
                    patient_result.setdefault("impressions", []).append({
                        "source": f"{source_category}/{col}",
                        "row_id": row_id,
                        "impression": ext["impression"],
                    })

    def _load_structured_csv(
        self,
        csv_path: str,
        fname: str,
        patient_result: Dict[str, Any],
    ) -> None:
        """读取 structured/ 下的 CSV，原始行存入 patient_result['structured_data']。"""
        csv_data = read_csv_all(csv_path)
        if csv_data.get("success") and csv_data.get("rows"):
            key = os.path.splitext(fname)[0]  # 去掉 .csv 后缀作为 key
            patient_result["structured_data"][key] = csv_data["rows"]
            if self.verbose:
                print(f"[{self.name}] [Dir] 结构化数据加载: {fname} ({len(csv_data['rows'])} 行)")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _resolve_output_dir(self, input_path: str) -> str:
        """确定输出目录：用户指定 > 输入文件同级的 results_v2 子目录。"""
        if self.output_dir:
            base = self.output_dir
        elif os.path.isdir(input_path):
            base = os.path.join(input_path, "results_v2")
        else:
            base = os.path.join(os.path.dirname(input_path), "results_v2")
        return build_output_dir(base)

    def _summarize(self, result: Dict[str, Any]) -> str:
        lines = [
            "=== 处理结果 ===",
            f"类型: {result.get('data_type', '?')}",
            f"状态: {'✅ 成功' if result.get('success') else '❌ 失败'}",
        ]
        if result.get("error"):
            lines.append(f"错误: {result['error']}")
        if result.get("plan", {}).get("reasoning"):
            lines.append(f"策略: {result['plan']['reasoning']}")
        stats = result.get("statistics", {})
        if stats:
            lines.append("统计:")
            for k, v in stats.items():
                if not isinstance(v, (dict, list)):
                    lines.append(f"  {k}: {v}")
        # 目录处理：汇总实体数
        patients = result.get("patients", {})
        if patients:
            total_entities = sum(len(p.get("merged_entities", [])) for p in patients.values())
            lines.append(f"  total_entities: {total_entities}")
        if result.get("output_csv"):
            lines.append(f"输出 CSV: {result['output_csv']}")
        if result.get("output_dir"):
            lines.append(f"输出目录: {result['output_dir']}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

async def process_medical_data(
    input_path: str,
    use_llm: bool = True,
    verbose: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 ReactMedicalAgent 处理医学数据的便捷函数。

    Args:
        input_path: 输入文件路径或目录路径
        use_llm: 是否使用 LLM（False 则纯规则模式）
        verbose: 是否打印详细日志
        output_dir: 输出根目录

    Returns:
        处理结果字典
    """
    agent = ReactMedicalAgent(use_llm=use_llm, verbose=verbose, output_dir=output_dir)
    msg = Msg(name="User", content=input_path, role="user")
    resp = await agent.reply(msg)
    return resp.metadata if resp.metadata else {"content": resp.content}
