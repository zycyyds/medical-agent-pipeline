from __future__ import annotations

from autonomous_pipeline.agent_factory import create_react_agent

from . import tools
from .prompt import SYSTEM_PROMPT, build_task_prompt

STEP_TITLE = "TaskAnalysis 任务分析"


def create_agent():
    return create_react_agent("task_analysis", SYSTEM_PROMPT, tools.TOOL_FUNCTIONS, max_iters=8)


__all__ = ["STEP_TITLE", "create_agent", "build_task_prompt"]
