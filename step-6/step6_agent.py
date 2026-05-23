from __future__ import annotations

import json
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.token import CharTokenCounter
from agentscope.tool import Toolkit

import step6_runtime as runtime
from step6_tools import register_step6_tools


STEP6_SYSTEM_PROMPT = """你是 Step6ReActAgent，负责对 Step5 输出的医疗宽表执行患者级一致性验证。

你只能调用已注册的 Step6 细粒度工具完成工作。不要自己编造路径、患者数或一致性评分。

必须按顺序调用：
1. load_data_overview
2. analyze_all_patients_consistency
3. save_step6_report

报告末尾必须包含并原样保留第二步工具返回的 PASSED_PATIENTS 标记。
如果第二步返回 PASSED_PATIENTS: <stored_in_step6_tool_cache>，不要展开 ID 列表，保存工具会读取缓存。

任一工具失败时，停止后续工具调用，并用 JSON 总结失败。
最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def create_step6_agent():
    toolkit = Toolkit()
    register_step6_tools(toolkit)
    agent = ReActAgent(
        name="ConsistencyAgent",
        sys_prompt=STEP6_SYSTEM_PROMPT,
        model=runtime.create_model(),
        formatter=OpenAIChatFormatter(
            token_counter=CharTokenCounter(),
            max_tokens=runtime._FORMATTER_MAX_CHARS,
        ),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=8,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


def create_consistency_agent():
    return create_step6_agent()


def build_step6_task_prompt(
    input_csv: str,
    output_root: str,
    raw_csv: str | None = None,
    memory_context: str | None = None,
    reuse_existing: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "task_type": "step6_consistency_validation",
        "input_csv": input_csv,
        "raw_csv": raw_csv or input_csv,
        "output_root": output_root,
        "reuse_existing": reuse_existing,
        "memory_context": str(memory_context or "").strip()[:6000],
        "required_tool_sequence": [
            {"tool": "load_data_overview", "args": {}},
            {"tool": "analyze_all_patients_consistency", "args": {}},
            {"tool": "save_step6_report", "args": {"report_content": "<include PASSED_PATIENTS marker from analyze tool>"}},
        ],
        "final_output": "WorkerResult JSON only",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
