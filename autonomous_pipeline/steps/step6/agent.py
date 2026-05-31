from __future__ import annotations

from autonomous_pipeline.agent_factory import create_react_agent

from . import tools
from .prompt import SYSTEM_PROMPT, build_task_prompt

STEP_TITLE = "Step6 ML 数据集生成"


def create_agent():
    return create_react_agent("step6", SYSTEM_PROMPT, tools.TOOL_FUNCTIONS, max_iters=14)


__all__ = ["STEP_TITLE", "create_agent", "build_task_prompt"]
