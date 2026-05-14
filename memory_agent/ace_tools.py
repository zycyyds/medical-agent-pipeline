"""
ACE 工具兼容层

保留旧的 `memory_agent.ace_tools` 导入路径，
但内部主链已经切到结构化 `memory_tool` 接口。
"""

import json

from agentscope.tool import ToolResponse

from . import memory_tool


def _get_playbook():
    return memory_tool._get_playbook()


def playbook_get_context(max_bullets: int = 15, query_text: str = "") -> ToolResponse:
    return memory_tool.playbook_get_context(max_bullets=max_bullets, query_text=query_text)


def playbook_list_bullets() -> ToolResponse:
    return memory_tool.playbook_list_bullets()


def playbook_add_bullet(content: str, bullet_type: str = "general") -> ToolResponse:
    return memory_tool.playbook_add_bullet(content=content, bullet_type=bullet_type)


def playbook_update_counts(bullet_id: str, is_helpful: bool) -> ToolResponse:
    result = memory_tool.update_strategy_count([bullet_id], is_helpful=is_helpful)
    data = json.loads(result.content)
    return ToolResponse(
        content=json.dumps(
            {
                "operation": "update_counts",
                "success": bool(data.get("success_count", 0)),
                "bullet_id": bullet_id,
                "is_helpful": is_helpful,
                "updated_ids": data.get("updated_ids", []),
                "missing_ids": data.get("missing_ids", []),
            },
            ensure_ascii=False,
        )
    )


def playbook_get_statistics() -> ToolResponse:
    return memory_tool.playbook_get_statistics()


def parse_execution_trace(trace_text: str) -> ToolResponse:
    parsed = memory_tool.normalize_trace_payload(trace_text)
    return ToolResponse(
        content=json.dumps(
            {
                "operation": "parse_execution_trace",
                "input_text": parsed.get("input_text", ""),
                "output_text": parsed.get("orchestrator_summary", ""),
                "tool_sequence": [
                    str(event.get("tool_name") or "")
                    for step in parsed.get("steps", [])
                    for event in step.get("tool_events", [])
                    if str(event.get("tool_name") or "")
                ],
                "success": bool(parsed.get("success", True)),
                "used_bullet_ids": parsed.get("retrieved_bullet_ids", []),
                "trace_json": parsed,
            },
            ensure_ascii=False,
        )
    )


async def reflector_analyze(
    input_text: str,
    output_text: str,
    tool_sequence: list[str],
    success: bool,
    used_bullet_ids: list[str] | None = None,
) -> ToolResponse:
    trace_payload = {
        "input_text": input_text,
        "duration_seconds": 0.0,
        "orchestrator_summary": output_text,
        "retrieved_bullet_ids": used_bullet_ids or [],
        "steps": [
            {
                "step_name": "legacy_execution",
                "timestamp": "",
                "input_data": input_text,
                "context_preview": "",
                "tool_events": [
                    {
                        "tool_name": tool_name,
                        "tool_input": {},
                        "tool_output": "",
                        "status": "succeeded" if success else "failed",
                        "error": "" if success else "legacy_execution_failed",
                    }
                    for tool_name in (tool_sequence or [])
                ],
                "final_output": output_text,
                "raw_messages": [],
                "truncated_flags": [],
            }
        ],
        "raw_trace_text": "",
        "success": success,
    }
    return await memory_tool.reflector_analyze(
        trace_json=json.dumps(trace_payload, ensure_ascii=False),
        max_refinement_rounds=3,
    )


async def curator_process_execution(
    input_text: str,
    output_text: str,
    tool_sequence: list[str],
    success: bool,
    used_bullet_ids: list[str] | None = None,
) -> ToolResponse:
    trace_payload = {
        "input_text": input_text,
        "duration_seconds": 0.0,
        "orchestrator_summary": output_text,
        "retrieved_bullet_ids": used_bullet_ids or [],
        "steps": [
            {
                "step_name": "legacy_execution",
                "timestamp": "",
                "input_data": input_text,
                "context_preview": "",
                "tool_events": [
                    {
                        "tool_name": tool_name,
                        "tool_input": {},
                        "tool_output": "",
                        "status": "succeeded" if success else "failed",
                        "error": "" if success else "legacy_execution_failed",
                    }
                    for tool_name in (tool_sequence or [])
                ],
                "final_output": output_text,
                "raw_messages": [],
                "truncated_flags": [],
            }
        ],
        "raw_trace_text": "",
        "success": success,
    }
    return await memory_tool.curator_process_execution(json.dumps(trace_payload, ensure_ascii=False))


async def curator_grow_and_refine() -> ToolResponse:
    return memory_tool.curator_grow_and_refine()
