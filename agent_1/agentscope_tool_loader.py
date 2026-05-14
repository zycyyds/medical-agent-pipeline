import importlib
import inspect
import json
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

DEFAULT_TOOL_RESPONSE_TEXT_LIMIT = 60_000


@dataclass(frozen=True)
class ToolFunctionSpec:
    name: str
    preset_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolSpec:
    module: str
    functions: list[ToolFunctionSpec]


def _build_tool_response_content(value: Any) -> list[TextBlock]:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    text = _truncate_tool_text(text)
    return [TextBlock(type="text", text=text)]


def _tool_response_text_limit() -> int:
    raw_limit = os.environ.get("AGENT1_TOOL_RESPONSE_TEXT_LIMIT", str(DEFAULT_TOOL_RESPONSE_TEXT_LIMIT))
    try:
        limit = int(raw_limit)
    except ValueError:
        return DEFAULT_TOOL_RESPONSE_TEXT_LIMIT
    return max(1_000, limit)


def _truncate_tool_text(text: str) -> str:
    limit = _tool_response_text_limit()
    if len(text) <= limit:
        return text

    notice = (
        f"\n\n[tool output truncated from {len(text)} characters; "
        f"set AGENT1_TOOL_RESPONSE_TEXT_LIMIT to change this cap.]"
    )
    keep = max(0, limit - len(notice))
    return f"{text[:keep]}{notice}"


def _truncate_tool_response(response: ToolResponse) -> ToolResponse:
    changed = False
    content = []
    raw_content = response.content
    blocks = [TextBlock(type="text", text=raw_content)] if isinstance(raw_content, str) else raw_content

    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            original_text = str(block.get("text", ""))
            text = _truncate_tool_text(original_text)
            if text != original_text:
                changed = True
                content.append(TextBlock(type="text", text=text))
            else:
                content.append(block)
        else:
            content.append(block)

    if not changed:
        return response

    metadata = dict(response.metadata or {})
    metadata["tool_output_truncated"] = True
    return ToolResponse(
        content=content,
        metadata=metadata,
        stream=response.stream,
        is_last=response.is_last,
        is_interrupted=response.is_interrupted,
        id=response.id,
    )


def wrap_tool_function(fn):
    if inspect.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            result = await fn(*args, **kwargs)
            if isinstance(result, ToolResponse):
                return _truncate_tool_response(result)
            return ToolResponse(content=_build_tool_response_content(result))

        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, ToolResponse):
            return _truncate_tool_response(result)
        return ToolResponse(content=_build_tool_response_content(result))

    return sync_wrapper


def load_toolkit_from_config(config_path: str) -> Toolkit:
    abs_path = os.path.abspath(config_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    specs: list[ToolSpec] = []
    for item in raw.get("tools", []):
        module = item["module"]
        fn_specs: list[ToolFunctionSpec] = []
        for fn in item.get("functions", []):
            if isinstance(fn, str):
                fn_specs.append(ToolFunctionSpec(name=fn, preset_kwargs=None))
            else:
                fn_specs.append(
                    ToolFunctionSpec(
                        name=str(fn.get("name")),
                        preset_kwargs=dict(fn.get("preset_kwargs") or {}),
                    )
                )
        specs.append(ToolSpec(module=module, functions=fn_specs))

    toolkit = Toolkit()
    for spec in specs:
        mod = importlib.import_module(spec.module)
        for fn_spec in spec.functions:
            fn = wrap_tool_function(getattr(mod, fn_spec.name))
            preset = fn_spec.preset_kwargs
            if preset:
                toolkit.register_tool_function(fn, preset_kwargs=preset)
            else:
                toolkit.register_tool_function(fn)
    return toolkit
