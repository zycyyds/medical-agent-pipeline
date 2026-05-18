from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

STEP4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP4_DIR.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, STEP4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runtime import create_openai_model_and_formatter  # noqa: E402
from step4_tools import register_step4_tools  # noqa: E402


STEP4_SYSTEM_PROMPT = """你是 Step4ReActAgent，负责对 Step2_3 输出的医疗宽表做任务驱动列筛选。

你只能调用已注册的 Step4 细粒度工具完成工作。不要自己编造路径、列数或校验结果。

必须按顺序调用：
1. resolve_step4_input_tool
2. resolve_step4_task_text_tool
3. run_step4_column_selection_tool
4. normalize_step4_outputs_tool
5. validate_step4_output_tool

任一工具返回 NEEDS_REPAIR 或 FAILED 时，停止后续工具调用，并用 JSON 总结失败。
最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def create_step4_agent() -> ReActAgent:
    toolkit = Toolkit()
    register_step4_tools(toolkit)
    model, formatter = create_openai_model_and_formatter("agent_4", "MiniMax-M2.7")
    agent = ReActAgent(
        name="Step4ReActAgent",
        sys_prompt=STEP4_SYSTEM_PROMPT,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=10,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


def build_step4_task_prompt(
    task_type: Literal["step4_only", "resume_from_step2_3"],
    input_path: str,
    output_root: str,
    task_text: str | None = None,
    memory_context: str | None = None,
) -> str:
    bounded_memory_context = str(memory_context or "").strip()[:6000]
    tool_sequence = [
        {
            "tool": "resolve_step4_input_tool",
            "args": {"input_path": input_path},
        },
        {
            "tool": "resolve_step4_task_text_tool",
            "args": {"task_text": task_text},
        },
        {
            "tool": "run_step4_column_selection_tool",
            "args": {
                "input_csv_path": "<use artifacts.input_csv_path from resolve_step4_input_tool>",
                "task_text": "<use artifacts.task_text from resolve_step4_task_text_tool>",
                "output_root": output_root,
                "memory_context": bounded_memory_context,
            },
        },
        {
            "tool": "normalize_step4_outputs_tool",
            "args": {
                "filtered_csv_path": "<use artifacts.filtered_csv_path from run_step4_column_selection_tool>",
                "selection_report_path": "<use artifacts.selection_report_path from run_step4_column_selection_tool>",
                "output_root": output_root,
            },
        },
        {
            "tool": "validate_step4_output_tool",
            "args": {
                "filtered_csv_path": "<use artifacts.filtered_csv_path from run_step4_column_selection_tool>",
                "selection_report_path": "<use artifacts.selection_report_path from run_step4_column_selection_tool>",
                "next_input_csv": "<use artifacts.next_input_csv from normalize_step4_outputs_tool>",
                "next_selection_report": "<use artifacts.next_selection_report from normalize_step4_outputs_tool>",
            },
        },
    ]
    payload: dict[str, Any] = {
        "task_type": task_type,
        "input_path": input_path,
        "output_root": output_root,
        "task_text": task_text,
        "memory_context": bounded_memory_context,
        "required_tool_sequence": tool_sequence,
        "final_output": "WorkerResult JSON only",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
