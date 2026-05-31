from __future__ import annotations

import functools
import json
import sys
import time
from typing import Any, Callable, Iterable

from .config import get_agent_config


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) if isinstance(payload, (dict, list)) else str(payload)


def _compact(value: Any, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _artifact_brief(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else {}
    if not isinstance(artifacts, dict):
        return {}
    preferred = (
        "file_count",
        "records_count",
        "visual_count",
        "table_count",
        "split_tables",
        "patch_count",
        "written_files",
        "output_file_count",
        "modality_counts",
        "tasks_path",
        "base_records_path",
        "visual_patches_path",
        "table_split_patches_path",
        "records_path",
        "tool_trace_path",
        "device",
    )
    return {key: artifacts[key] for key in preferred if key in artifacts}


def _tool_reason(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    reason = kwargs.get("reason")
    if reason:
        return str(reason).strip()
    return ""


def wrap_tool_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a plain tool function so AgentScope receives a ToolResponse."""
    from agentscope.message import TextBlock
    from agentscope.tool import ToolResponse

    if hasattr(func, "__wrapped_for_agentscope__"):
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> ToolResponse:
        started = time.time()
        reason = _tool_reason(args, kwargs)
        if reason:
            print(f"[Agent 决策] {func.__name__}: {reason}", file=sys.stderr, flush=True)
        print(f"[Agent 编排] 调用工具 {func.__name__} args={_compact(kwargs or args)}", file=sys.stderr, flush=True)
        try:
            payload = func(*args, **kwargs)
        except Exception as exc:  # Tools should not crash the ReAct loop.
            payload = {
                "status": "FAILED",
                "summary": f"工具 {func.__name__} 执行异常。",
                "artifacts": {},
                "issues": [str(exc)],
                "next_recommendation": "根据错误修正参数或选择其他工具。",
            }
        duration_ms = int((time.time() - started) * 1000)
        status = payload.get("status") if isinstance(payload, dict) else "UNKNOWN"
        summary = payload.get("summary", "") if isinstance(payload, dict) else ""
        print(
            f"[Agent 编排] 完成工具 {func.__name__} status={status} duration_ms={duration_ms} summary={_compact(summary, 160)} artifacts={_compact(_artifact_brief(payload), 360)}",
            file=sys.stderr,
            flush=True,
        )
        return ToolResponse(content=[TextBlock(type="text", text=_json_text(payload))])

    wrapper.__wrapped_for_agentscope__ = True  # type: ignore[attr-defined]
    return wrapper


def create_toolkit(tool_functions: Iterable[Callable[..., Any]]):
    from agentscope.tool import Toolkit

    toolkit = Toolkit()
    for func in tool_functions:
        description = (func.__doc__ or "").strip().split("\n\n", 1)[0] or func.__name__
        toolkit.register_tool_function(wrap_tool_function(func), func_description=description)
    return toolkit


def create_model_and_formatter(step_name: str):
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.model import OpenAIChatModel

    cfg = get_agent_config(step_name)
    model = OpenAIChatModel(
        model_name=cfg.model,
        api_key=cfg.api_key or None,
        stream=False,
        client_kwargs={"base_url": cfg.base_url, "timeout": cfg.timeout, "max_retries": 2},
        generate_kwargs={"temperature": cfg.temperature, "seed": cfg.seed},
    )
    return model, OpenAIChatFormatter()


def create_react_agent(step_name: str, system_prompt: str, tool_functions: Iterable[Callable[..., Any]], max_iters: int = 16):
    from agentscope.agent import ReActAgent
    from agentscope.memory import InMemoryMemory

    model, formatter = create_model_and_formatter(step_name)
    agent = ReActAgent(
        name=f"{step_name.title()}AutonomousAgent",
        sys_prompt=system_prompt,
        model=model,
        formatter=formatter,
        toolkit=create_toolkit(tool_functions),
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=max_iters,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent
