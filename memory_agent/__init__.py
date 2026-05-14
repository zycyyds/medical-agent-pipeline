"""
Memory Agent - 记忆反思智能体 (v4.5.0)

基于 ACE (Agentic Context Engineering) 框架，
主链已切换到结构化 trace / reflection / curator delta。
"""

__version__ = "4.5.0"

from .memory_tool import (
    curator_apply_reflection,
    curator_grow_and_refine,
    curator_process_execution,
    get_playbook_context_data,
    playbook_add_bullet,
    playbook_get_context,
    playbook_get_statistics,
    playbook_list_bullets,
    reflector_analyze,
    update_strategy_count,
)

__all__ = [
    "curator_apply_reflection",
    "curator_grow_and_refine",
    "curator_process_execution",
    "get_playbook_context_data",
    "playbook_add_bullet",
    "playbook_get_context",
    "playbook_get_statistics",
    "playbook_list_bullets",
    "reflector_analyze",
    "update_strategy_count",
]
