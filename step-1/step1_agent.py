from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

STEP1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP1_DIR.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, STEP1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runtime import create_openai_model_and_formatter  # noqa: E402
from step1_tools import register_step1_tools  # noqa: E402


STEP1_SYSTEM_PROMPT = """你是 Step1ReActAgent，负责把原始医疗数据目录整理成 Step1 records 和重组输出。

你只能调用已注册的 Step1 细粒度工具完成工作。不要自己编造文件数量、路径或校验结果。

从头运行 step1_only 时，必须按顺序调用：
1. scan_source_files_tool
2. build_records_parallel_tool
3. validate_records_tool
4. write_generated_reorganizer_tool
5. run_reorganize_from_records_tool
6. validate_step1_output_tool

从 records.json 续跑 resume_from_records 时，必须按顺序调用：
1. validate_records_tool
2. write_generated_reorganizer_tool
3. run_reorganize_from_records_tool
4. validate_step1_output_tool

任一工具返回 NEEDS_REPAIR 或 FAILED 时，停止后续工具调用，并用 JSON 总结失败。
最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def create_step1_agent() -> ReActAgent:
    toolkit = Toolkit()
    register_step1_tools(toolkit)
    model, formatter = create_openai_model_and_formatter("agent_1", "MiniMax-M2.7")
    agent = ReActAgent(
        name="Step1ReActAgent",
        sys_prompt=STEP1_SYSTEM_PROMPT,
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


def build_step1_task_prompt(
    task_type: Literal["step1_only", "resume_from_records"],
    records_path: str,
    output_root: str,
    generated_script_path: str,
    input_path: str | None = None,
    memory_context: str | None = None,
) -> str:
    bounded_memory_context = str(memory_context or "").strip()[:6000]
    if task_type == "step1_only":
        tool_sequence = [
            {
                "tool": "scan_source_files_tool",
                "args": {"input_path": input_path},
            },
            {
                "tool": "build_records_parallel_tool",
                "args": {"input_path": input_path, "records_path": records_path},
            },
            {
                "tool": "validate_records_tool",
                "args": {"records_path": records_path},
            },
            {
                "tool": "write_generated_reorganizer_tool",
                "args": {
                    "script_path": generated_script_path,
                    "records_path": records_path,
                    "output_root": output_root,
                    "input_root": input_path,
                },
            },
            {
                "tool": "run_reorganize_from_records_tool",
                "args": {"records_path": records_path, "output_root": output_root, "input_root": input_path},
            },
            {
                "tool": "validate_step1_output_tool",
                "args": {"output_root": output_root},
            },
        ]
    else:
        tool_sequence = [
            {
                "tool": "validate_records_tool",
                "args": {"records_path": records_path},
            },
            {
                "tool": "write_generated_reorganizer_tool",
                "args": {
                    "script_path": generated_script_path,
                    "records_path": records_path,
                    "output_root": output_root,
                    "input_root": input_path,
                },
            },
            {
                "tool": "run_reorganize_from_records_tool",
                "args": {"records_path": records_path, "output_root": output_root, "input_root": input_path},
            },
            {
                "tool": "validate_step1_output_tool",
                "args": {"output_root": output_root},
            },
        ]

    payload: dict[str, Any] = {
        "task_type": task_type,
        "input_path": input_path,
        "records_path": records_path,
        "output_root": output_root,
        "generated_script_path": generated_script_path,
        "memory_context": bounded_memory_context,
        "memory_context_usage": "仅作为历史经验参考；路径、数量和校验结果必须以本轮工具返回为准。",
        "required_tool_sequence": tool_sequence,
        "final_output": "WorkerResult JSON only",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
