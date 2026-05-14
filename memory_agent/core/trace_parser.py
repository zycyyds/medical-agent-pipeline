from __future__ import annotations

import ast
import json
import re
from typing import Any

from .models import ExecutionTrace


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _extract_assistant_outputs_from_messages(messages: list[Any]) -> list[str]:
    outputs: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "assistant":
            continue
        for block in _coerce_list(message.get("content_blocks")):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "text":
                text = str(block.get("text") or "").strip()
            elif block_type == "raw":
                text = str(block.get("value") or "").strip()
            else:
                text = ""
            if text:
                outputs.append(text)
    return outputs


def normalize_trace_payload(trace_payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(trace_payload, str):
        try:
            raw = json.loads(trace_payload)
        except Exception:
            legacy = parse_execution_trace_to_model(trace_payload)
            return {
                "input_text": legacy.input_text,
                "duration_seconds": legacy.duration,
                "orchestrator_summary": legacy.output_text,
                "retrieved_bullet_ids": legacy.used_bullet_ids,
                "steps": [],
                "raw_trace_text": legacy.raw_text,
                "success": legacy.success,
                "failure_signals": legacy.failure_signals,
            }
    elif isinstance(trace_payload, dict):
        raw = dict(trace_payload)
    else:
        raw = {}

    steps = []
    for step in _coerce_list(raw.get("steps")):
        if not isinstance(step, dict):
            continue
        raw_messages: list[Any] = []
        full_raw_messages: list[Any] = []
        full_tool_events = _coerce_list(step.get("full_tool_events"))
        raw_tool_events = _coerce_list(step.get("tool_events"))
        tool_events_preview = _coerce_list(step.get("tool_events_preview"))
        assistant_outputs = [str(item) for item in _coerce_list(step.get("assistant_outputs")) if str(item).strip()]
        steps.append(
            {
                "step_name": str(step.get("step_name") or ""),
                "timestamp": str(step.get("timestamp") or ""),
                "input_data": str(step.get("input_data") or ""),
                "context_preview": str(step.get("context_preview") or ""),
                "assistant_outputs": assistant_outputs,
                "tool_events": full_tool_events or raw_tool_events,
                "tool_events_preview": tool_events_preview or (raw_tool_events if full_tool_events else []),
                "full_tool_events": full_tool_events,
                "final_output": str(step.get("final_output") or ""),
                "raw_messages": [],
                "full_raw_messages": [],
                "truncated_flags": [str(item) for item in _coerce_list(step.get("truncated_flags")) if str(item).strip()],
            }
        )

    return {
        "input_text": str(raw.get("input_text") or ""),
        "duration_seconds": float(raw.get("duration_seconds") or 0.0),
        "orchestrator_summary": str(raw.get("orchestrator_summary") or ""),
        "retrieved_bullet_ids": [
            str(item) for item in _coerce_list(raw.get("retrieved_bullet_ids")) if str(item).strip()
        ],
        "steps": steps,
        "raw_trace_text": str(raw.get("raw_trace_text") or ""),
        "full_raw_trace_text": str(raw.get("full_raw_trace_text") or raw.get("raw_trace_text") or ""),
        "success": bool(raw.get("success", True)),
        "failure_signals": [str(item) for item in _coerce_list(raw.get("failure_signals")) if str(item).strip()],
    }


def _extract_section_between_markers(
    trace_text: str,
    start_marker: str,
    end_markers: tuple[str, ...],
) -> str:
    text = str(trace_text or "")
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""

    section_start = start_idx + len(start_marker)
    section_end = len(text)
    for marker in end_markers:
        idx = text.find(marker, section_start)
        if idx != -1 and idx < section_end:
            section_end = idx
    return text[section_start:section_end].strip()


def _extract_user_visible_text_from_blocks(content: str) -> str:
    stripped = str(content or "").strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return stripped

    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        return stripped

    if not isinstance(parsed, list):
        return stripped

    text_parts: list[str] = []
    fallback_parts: list[str] = []
    for block in parsed:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    text_parts.append(text)
            elif block_type == "tool_result":
                output = str(block.get("output", "")).strip()
                if output:
                    fallback_parts.append(output)
        else:
            fallback_parts.append(str(block).strip())

    if text_parts:
        return "\n".join(text_parts).strip()
    return "\n".join(part for part in fallback_parts if part).strip()


def _parse_possible_list(raw_text: str) -> list[str]:
    text = str(raw_text or "").strip()
    if not text:
        return []

    match = re.search(r"(\[.*\])", text)
    if match:
        text = match.group(1)

    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except Exception:
            continue

    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def parse_execution_trace_to_model(trace_text: str) -> ExecutionTrace:
    raw = str(trace_text or "")
    lines = raw.splitlines()

    input_text = next(
        (line.split("用户原始输入:", 1)[1].strip() for line in lines if "用户原始输入:" in line),
        "",
    )
    output_text = _extract_section_between_markers(
        raw,
        "Orchestrator 最终总结:",
        (
            "\n============================================================",
            "\n============================================================\n",
            "\nStep ",
            "\n本次运行尝试使用的记忆条目 ID 有：",
            "\n本次运行尝试使用的记忆条目ID有：",
            "\n策略 IDs:",
            "\nused_bullet_ids:",
        ),
    )
    if not output_text:
        output_text = next(
            (
                line.split("Orchestrator 最终总结:", 1)[1].strip()
                for line in lines
                if "Orchestrator 最终总结:" in line
            ),
            raw[-1000:],
        )
    output_text = _extract_user_visible_text_from_blocks(output_text)

    duration = 0.0
    duration_match = re.search(r"耗时:\s*([0-9.]+)s", raw)
    if duration_match:
        try:
            duration = float(duration_match.group(1))
        except Exception:
            duration = 0.0

    tool_sequence: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Step ") and ":" in stripped:
            tool_name = stripped.split(":", 1)[1].strip()
            if tool_name and tool_name not in tool_sequence:
                tool_sequence.append(tool_name)

    used_bullet_ids: list[str] = []
    id_prefixes = (
        "本次运行尝试使用的记忆条目 ID 有：",
        "本次运行尝试使用的记忆条目ID有：",
        "策略 IDs:",
        "used_bullet_ids:",
    )
    for line in lines:
        for prefix in id_prefixes:
            if prefix in line:
                used_bullet_ids = _parse_possible_list(line.split(prefix, 1)[1].strip())
                if used_bullet_ids:
                    break
        if used_bullet_ids:
            break

    failure_signals = []
    for keyword in ("Traceback", "ERROR", "失败", "崩溃", "AssertionError", "warning"):
        if keyword in raw:
            failure_signals.append(keyword)

    success = not any(keyword in raw for keyword in ("Traceback", "ERROR", "失败", "崩溃"))
    feedback_quality = (
        "strong"
        if any(key in raw for key in ("Traceback", "AssertionError", "单元测试", "验证", "error"))
        else "weak"
    )

    return ExecutionTrace(
        raw_text=raw,
        input_text=input_text,
        output_text=output_text,
        tool_sequence=tool_sequence,
        used_bullet_ids=used_bullet_ids,
        success=success,
        duration=duration,
        trace_length=len(raw),
        failure_signals=failure_signals,
        feedback_quality=feedback_quality,
    )
