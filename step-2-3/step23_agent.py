from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

STEP23_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP23_DIR.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, STEP23_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runtime import create_openai_model_and_formatter  # noqa: E402
from step23_tools import register_step23_tools  # noqa: E402


STEP23_SYSTEM_PROMPT = """你是 Step2_3ReActAgent，负责把 Step1 输出目录结构化为可供 Step4 使用的宽表。

你只能调用已注册的 Step2-3 细粒度工具完成工作。不要自己编造路径、文件数量或校验结果。

auto 模式必须先选择 pipeline：
1. resolve_step23_input_tool
2. scan_step23_input_tool
3. select_step23_pipeline_tool

如果 resolved_mode 是 directory，继续按顺序调用：
4. run_step23_directory_pipeline_tool
5. normalize_step23_outputs_tool
6. validate_step23_output_tool

如果 resolved_mode 是 ocr_fill，继续按顺序调用：
4. run_step23_ocr_tool
5. run_step23_extraction_tool
6. run_step23_fill_table_tool
7. normalize_step23_outputs_tool
8. validate_step23_output_tool

Step2-3 不再支持 single 单文件模式。输入必须是目录。

任一工具返回 NEEDS_REPAIR 或 FAILED 时，停止后续工具调用，并用 JSON 总结失败。
最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def create_step23_agent() -> ReActAgent:
    toolkit = Toolkit()
    register_step23_tools(toolkit)
    model, formatter = create_openai_model_and_formatter("agent_2_3", "MiniMax-M2.7")
    agent = ReActAgent(
        name="Step2_3ReActAgent",
        sys_prompt=STEP23_SYSTEM_PROMPT,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=14,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


def build_step23_task_prompt(
    task_type: Literal["step2_3_only", "full_pipeline_step2_3"],
    input_path: str,
    output_root: str,
    mode: Literal["auto", "directory", "ocr_fill"] = "auto",
    cache_path: str | None = None,
    ocr_workers: int = 8,
    concurrency: int = 8,
    use_llm: bool = True,
    user_hint: str = "",
    memory_context: str | None = None,
) -> str:
    bounded_memory_context = str(memory_context or "").strip()[:6000]
    tool_sequence: list[dict[str, Any]] = [
        {"tool": "resolve_step23_input_tool", "args": {"input_path": input_path, "mode": mode}},
        {"tool": "scan_step23_input_tool", "args": {"input_path": "<use artifacts.input_path from resolve_step23_input_tool>"}},
        {
            "tool": "select_step23_pipeline_tool",
            "args": {
                "input_path": "<use artifacts.input_path from resolve_step23_input_tool>",
                "requested_mode": mode,
                "user_hint": user_hint,
                "scan_artifacts": "<use artifacts from scan_step23_input_tool>",
            },
        },
        {
            "branch": "if resolved_mode == directory",
            "tools": [
                {
                    "tool": "run_step23_directory_pipeline_tool",
                    "args": {
                        "input_path": "<use artifacts.input_path from resolve_step23_input_tool>",
                        "output_root": output_root,
                        "concurrency": concurrency,
                        "use_llm": use_llm,
                    },
                },
                {
                    "tool": "normalize_step23_outputs_tool",
                    "args": {
                        "output_root": output_root,
                        "fallback_csv": "<use artifacts.admission_wide_csv from run_step23_directory_pipeline_tool>",
                        "summary": "<merge selection artifacts and directory pipeline artifacts>",
                    },
                },
            ],
        },
        {
            "branch": "if resolved_mode == ocr_fill",
            "tools": [
                {
                    "tool": "run_step23_ocr_tool",
                    "args": {
                        "input_path": "<use artifacts.input_path from resolve_step23_input_tool>",
                        "output_root": output_root,
                        "ocr_workers": ocr_workers,
                        "cache_path": cache_path,
                    },
                },
                {
                    "tool": "run_step23_extraction_tool",
                    "args": {
                        "ocr_cache_path": "<use artifacts.ocr_cache_path from run_step23_ocr_tool>",
                        "results_dir": "<use artifacts.results_dir from run_step23_ocr_tool>",
                        "concurrency": concurrency,
                    },
                },
                {
                    "tool": "run_step23_fill_table_tool",
                    "args": {
                        "input_path": "<use artifacts.input_path from resolve_step23_input_tool>",
                        "results_dir": "<use artifacts.results_dir from run_step23_ocr_tool>",
                        "use_llm": use_llm,
                    },
                },
                {
                    "tool": "normalize_step23_outputs_tool",
                    "args": {
                        "output_root": output_root,
                        "filled_merged_csv": "<use artifacts.filled_merged_csv from run_step23_fill_table_tool>",
                        "fallback_csv": "<use artifacts.patients_csv from run_step23_extraction_tool>",
                        "summary": "<merge selection artifacts and OCR/extraction/fill artifacts>",
                    },
                },
            ],
        },
        {
            "tool": "validate_step23_output_tool",
            "args": {
                "next_input_csv": "<use artifacts.next_input_csv from normalize_step23_outputs_tool>",
                "next_summary_json": "<use artifacts.next_summary_json from normalize_step23_outputs_tool>",
            },
        },
    ]

    payload = {
        "task_type": task_type,
        "mode": mode,
        "input_path": input_path,
        "output_root": output_root,
        "cache_path": cache_path,
        "ocr_workers": ocr_workers,
        "concurrency": concurrency,
        "use_llm": use_llm,
        "user_hint": user_hint,
        "memory_context": bounded_memory_context,
        "memory_context_usage": "仅作为历史经验参考；pipeline 选择、路径、数量和校验结果必须以本轮工具返回为准。",
        "required_tool_sequence": tool_sequence,
        "final_output": "WorkerResult JSON only",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
