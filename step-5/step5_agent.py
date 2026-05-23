from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

STEP5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP5_DIR.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, STEP5_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runtime import create_openai_model_and_formatter  # noqa: E402
from step5_tools import register_step5_tools  # noqa: E402


STEP5_SYSTEM_PROMPT = """你是 Step5ReActAgent，负责对 Step4 输出的医疗宽表做数据清洗和质量修复。

你只能调用已注册的 Step5 细粒度工具完成工作。不要自己编造路径、列数或校验结果。

必须按顺序调用：
1. resolve_step5_input_tool
2. profile_step5_table_tool
3. classify_step5_column_risks_tool
4. run_step5_data_cleaning_tool
5. normalize_step5_outputs_tool
6. validate_step5_output_tool

风险策略固定：
- low-risk: 只报告，不清洗。
- medium-risk: 使用 LLM 生成列级清洗脚本并执行；执行后必须保持行数、列数、列顺序不变。
- high-risk: 使用 LLM 生成列级清洗脚本并执行；执行后必须保持行数、列数、列顺序不变。

任一工具返回 NEEDS_REPAIR 或 FAILED 时，停止后续工具调用，并用 JSON 总结失败。
最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def create_step5_agent() -> ReActAgent:
    toolkit = Toolkit()
    register_step5_tools(toolkit)
    model, formatter = create_openai_model_and_formatter("agent_5", "gpt-4.1-mini")
    agent = ReActAgent(
        name="Step5ReActAgent",
        sys_prompt=STEP5_SYSTEM_PROMPT,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=12,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


def build_step5_task_prompt(
    task_type: Literal["step5_only", "resume_from_step4"],
    input_path: str,
    output_root: str,
    workers: int = 4,
    llm_workers: int = 2,
    enable_llm: bool = True,
    memory_context: str | None = None,
) -> str:
    bounded_memory_context = str(memory_context or "").strip()[:6000]
    tool_sequence = [
        {"tool": "resolve_step5_input_tool", "args": {"input_path": input_path}},
        {
            "tool": "profile_step5_table_tool",
            "args": {
                "input_csv_path": "<use artifacts.input_csv_path from resolve_step5_input_tool>",
                "output_root": output_root,
                "workers": workers,
            },
        },
        {
            "tool": "classify_step5_column_risks_tool",
            "args": {
                "input_csv_path": "<use artifacts.input_csv_path from resolve_step5_input_tool>",
                "profile_report_path": "<use artifacts.profile_report_path from profile_step5_table_tool>",
                "output_root": output_root,
                "workers": workers,
            },
        },
        {
            "tool": "run_step5_data_cleaning_tool",
            "args": {
                "input_csv_path": "<use artifacts.input_csv_path from resolve_step5_input_tool>",
                "column_risk_report_path": "<use artifacts.column_risk_report_path from classify_step5_column_risks_tool>",
                "output_root": output_root,
                "workers": workers,
                "llm_workers": llm_workers,
                "enable_llm": enable_llm,
            },
        },
        {
            "tool": "normalize_step5_outputs_tool",
            "args": {
                "cleaned_csv_path": "<use artifacts.cleaned_csv_path from run_step5_data_cleaning_tool>",
                "data_quality_report_path": "<use artifacts.data_quality_report_path from run_step5_data_cleaning_tool>",
                "column_risk_report_path": "<use artifacts.column_risk_report_path from run_step5_data_cleaning_tool>",
                "output_root": output_root,
                "selection_report_path": "<use artifacts.selection_report_path from resolve_step5_input_tool if present>",
            },
        },
        {
            "tool": "validate_step5_output_tool",
            "args": {
                "input_csv_path": "<use artifacts.input_csv_path from resolve_step5_input_tool>",
                "cleaned_csv_path": "<use artifacts.cleaned_csv_path from run_step5_data_cleaning_tool>",
                "data_quality_report_path": "<use artifacts.data_quality_report_path from run_step5_data_cleaning_tool>",
                "column_risk_report_path": "<use artifacts.column_risk_report_path from run_step5_data_cleaning_tool>",
                "next_input_csv": "<use artifacts.next_input_csv from normalize_step5_outputs_tool>",
                "next_data_quality_report": "<use artifacts.next_data_quality_report from normalize_step5_outputs_tool>",
                "next_column_risk_report": "<use artifacts.next_column_risk_report from normalize_step5_outputs_tool>",
            },
        },
    ]
    payload: dict[str, Any] = {
        "task_type": task_type,
        "input_path": input_path,
        "output_root": output_root,
        "workers": workers,
        "llm_workers": llm_workers,
        "enable_llm": enable_llm,
        "memory_context": bounded_memory_context,
        "memory_context_usage": "仅作为历史清洗经验参考；风险分类、清洗结果和校验必须以本轮工具返回为准。",
        "risk_policy": {
            "low": "report_only",
            "medium": "builtin_deterministic_cleaning",
            "high": "llm_generated_script_cleaning",
        },
        "required_tool_sequence": tool_sequence,
        "final_output": "WorkerResult JSON only",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
