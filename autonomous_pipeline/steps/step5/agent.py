from __future__ import annotations

from autonomous_pipeline.agent_factory import create_react_agent

from . import tools
from .prompt import SYSTEM_PROMPT, build_task_prompt

STEP_TITLE = "Step5 一致性验证"


def create_agent():
    return create_react_agent("step5", SYSTEM_PROMPT, tools.TOOL_FUNCTIONS, max_iters=14)


__all__ = ["STEP_TITLE", "create_agent", "build_task_prompt"]
