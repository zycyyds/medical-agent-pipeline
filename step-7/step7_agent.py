from __future__ import annotations

import json
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.token import CharTokenCounter
from agentscope.tool import Toolkit

import step7_runtime as runtime
from step7_tools import register_step7_tools


STEP7_SYSTEM_PROMPT = """你是 Step7ReActAgent，负责把通过一致性验证的医疗数据转化为机器学习训练集。

你只能调用已注册的 Step7 细粒度工具完成工作。不要自己编造路径、患者数、标签列或模型配置。

关键约束：
- 必须严格围绕本轮 task_text / ml_task_goal 生成数据集。
- 禁止生成与任务目标无关的探索性任务。
- 必须使用 Step5 filtered.csv、Step4 selection_report.json、Step6 passed_patients_json 三个输入。
- 缺少任一输入或任务目标无法构造标签时，返回 NEEDS_REPAIR。

最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def create_step7_agent():
    toolkit = Toolkit()
    register_step7_tools(toolkit)
    agent = ReActAgent(
        name="DataPrepAgent",
        sys_prompt=STEP7_SYSTEM_PROMPT,
        model=runtime.create_model(),
        formatter=OpenAIChatFormatter(
            token_counter=CharTokenCounter(),
            max_tokens=runtime._FORMATTER_MAX_CHARS,
        ),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=60,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


def create_dataprep_agent():
    return create_step7_agent()


def build_step7_task_prompt(
    input_csv: str,
    selection_report: str,
    passed_patients_json: str,
    output_root: str,
    task_text: str,
    memory_context: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "task_type": "step7_ml_dataset_generation",
        "input_csv": input_csv,
        "selection_report": selection_report,
        "passed_patients_json": passed_patients_json,
        "output_root": output_root,
        "task_text": task_text,
        "memory_context": str(memory_context or "").strip()[:6000],
        "required_policy": "strict_task_match",
        "required_inputs": ["filtered_csv", "selection_report", "passed_patients_json"],
        "final_output": "WorkerResult JSON only",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
