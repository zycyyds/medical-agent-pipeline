import asyncio
import json
from typing import Any

from agentscope.model import OllamaChatModel
from agentscope.tool import ToolResponse

from memory_agent.core import (
    ACECurator,
    ACEPlaybookManager,
    ACEReflector,
    Bullet,
    ReflectionResult,
    normalize_trace_payload,
)
from memory_agent.config import config


MEMORY_BANK_PATH = config.get_memory_bank_path()
LLM_MODEL = config.LLM_MODEL
LLM_TEMPERATURE = config.LLM_TEMPERATURE
LLM_SEED = config.LLM_SEED
LLM_ENABLE_THINKING = config.LLM_ENABLE_THINKING

_PLAYBOOK_INSTANCE: ACEPlaybookManager | None = None
_CURRENT_TRACE_PAYLOAD: dict[str, Any] | None = None
_CURRENT_REFLECTION_PAYLOAD: dict[str, Any] | None = None


def make_chat_model_factory(_agent_key: str = "memory_agent"):
    return lambda: OllamaChatModel(
        model_name=LLM_MODEL,
        enable_thinking=LLM_ENABLE_THINKING,
        options={
            "temperature": LLM_TEMPERATURE,
            "seed": LLM_SEED,
            "num_predict": 4096,
        },
    )


def _get_playbook() -> ACEPlaybookManager:
    global _PLAYBOOK_INSTANCE
    if _PLAYBOOK_INSTANCE is None:
        _PLAYBOOK_INSTANCE = ACEPlaybookManager(MEMORY_BANK_PATH)
    return _PLAYBOOK_INSTANCE


def set_current_trace_payload(trace_payload: str | dict[str, Any] | None) -> None:
    global _CURRENT_TRACE_PAYLOAD
    if trace_payload is None:
        _CURRENT_TRACE_PAYLOAD = None
        return
    _CURRENT_TRACE_PAYLOAD = normalize_trace_payload(trace_payload)


def get_current_trace_payload() -> dict[str, Any]:
    return dict(_CURRENT_TRACE_PAYLOAD or {})


def _create_reflector() -> ACEReflector:
    return ACEReflector(
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        enable_thinking=LLM_ENABLE_THINKING,
        model_factory=make_chat_model_factory("memory_agent"),
    )


def _create_curator() -> ACECurator:
    return ACECurator(
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        enable_thinking=LLM_ENABLE_THINKING,
        model_factory=make_chat_model_factory("memory_agent"),
    )


def _load_trace_payload(trace_json: str | dict[str, Any]) -> dict[str, Any]:
    if (trace_json == "" or trace_json is None) and _CURRENT_TRACE_PAYLOAD is not None:
        return dict(_CURRENT_TRACE_PAYLOAD)
    return normalize_trace_payload(trace_json)


def _reflection_from_json(reflection_json: str | dict[str, Any], success: bool = True) -> ReflectionResult:
    if isinstance(reflection_json, str):
        try:
            data = json.loads(reflection_json)
        except Exception:
            data = {}
    else:
        data = dict(reflection_json or {})

    bullet_tags = []
    for item in data.get("bullet_tags", []):
        if not isinstance(item, dict):
            continue
        bullet_id = str(item.get("id") or "").strip()
        tag = str(item.get("tag") or "neutral").strip().lower()
        if bullet_id and tag:
            from memory_agent.core.models import BulletTag

            bullet_tags.append(BulletTag(id=bullet_id, tag=tag))

    if not bullet_tags:
        from memory_agent.core.models import BulletTag

        for bullet_id in data.get("helpful_bullet_ids", []):
            if str(bullet_id).strip():
                bullet_tags.append(BulletTag(id=str(bullet_id).strip(), tag="helpful"))
        for bullet_id in data.get("harmful_bullet_ids", []):
            if str(bullet_id).strip():
                bullet_tags.append(BulletTag(id=str(bullet_id).strip(), tag="harmful"))

    return ReflectionResult(
        reasoning=str(data.get("analysis") or data.get("reasoning") or "").strip(),
        error_identification=str(data.get("error_identification") or "").strip(),
        root_cause_analysis=str(data.get("root_cause_analysis") or "").strip(),
        correct_approach=str(data.get("correct_approach") or "").strip(),
        key_insight=str(data.get("key_insight") or "").strip(),
        key_insight_section=str(data.get("key_insight_section") or "").strip(),
        bullet_tags=bullet_tags,
        is_success=bool(data.get("is_success", success)),
    )


_REFLECTION_REQUIRED_STRING_FIELDS = (
    "analysis",
    "error_identification",
    "root_cause_analysis",
    "correct_approach",
    "key_insight",
    "key_insight_section",
)
_REFLECTION_REQUIRED_LIST_FIELDS = (
    "helpful_bullet_ids",
    "harmful_bullet_ids",
    "neutral_bullet_ids",
    "bullet_tags",
)


def _standardize_reflection_payload(
    reflection_payload: str | dict[str, Any] | None,
    success: bool = True,
) -> dict[str, Any]:
    if reflection_payload is None:
        return {}
    standardized = _reflection_from_json(reflection_payload, success=success).to_dict()
    operation = ""
    if isinstance(reflection_payload, str):
        try:
            raw_payload = json.loads(reflection_payload)
        except Exception:
            raw_payload = {}
    else:
        raw_payload = dict(reflection_payload or {})
    operation = str(raw_payload.get("operation") or "").strip()
    if operation:
        standardized["operation"] = operation
    return standardized


def set_current_reflection_payload(reflection_payload: str | dict[str, Any] | None) -> None:
    global _CURRENT_REFLECTION_PAYLOAD
    if reflection_payload is None:
        _CURRENT_REFLECTION_PAYLOAD = None
        return
    normalized = _standardize_reflection_payload(reflection_payload)
    _CURRENT_REFLECTION_PAYLOAD = normalized or None


def get_current_reflection_payload() -> dict[str, Any]:
    return dict(_CURRENT_REFLECTION_PAYLOAD or {})


def _load_reflection_payload(reflection_json: str | dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if isinstance(reflection_json, str):
        if reflection_json.strip():
            return _standardize_reflection_payload(reflection_json), "explicit"
    elif reflection_json:
        return _standardize_reflection_payload(reflection_json), "explicit"

    if _CURRENT_REFLECTION_PAYLOAD is not None:
        return dict(_CURRENT_REFLECTION_PAYLOAD), "current_reflection"
    return {}, "missing"


def _inspect_reflection_payload_quality(reflection_payload: dict[str, Any]) -> tuple[str, list[str]]:
    missing_fields: list[str] = []
    for field in _REFLECTION_REQUIRED_STRING_FIELDS:
        if not str(reflection_payload.get(field) or "").strip():
            missing_fields.append(field)
    for field in _REFLECTION_REQUIRED_LIST_FIELDS:
        if not isinstance(reflection_payload.get(field), list):
            missing_fields.append(field)
    return ("complete" if not missing_fields else "degraded"), missing_fields


def get_playbook_context_data(
    max_bullets: int = 15,
    query_text: str = "",
) -> tuple[str, list[str]]:
    manager = _get_playbook()
    context_str, used_ids, _ = manager.format_playbook(query_text=query_text, max_bullets=max_bullets)
    return context_str, used_ids


def playbook_get_context(
    max_bullets: int = 15,
    query_text: str = "",
) -> ToolResponse:
    manager = _get_playbook()
    context_str, used_ids, retrieval = manager.format_playbook(query_text=query_text, max_bullets=max_bullets)
    return ToolResponse(
        content=json.dumps(
            {
                "operation": "get_context",
                "context": context_str,
                "used_bullet_ids": used_ids,
                "retrieval": retrieval,
            },
            ensure_ascii=False,
        )
    )


def playbook_list_bullets() -> ToolResponse:
    manager = _get_playbook()
    return ToolResponse(
        content=json.dumps(
            {
                "operation": "list_bullets",
                "bullets": [item.to_dict() for item in manager.get_active_bullets()],
            },
            ensure_ascii=False,
        )
    )


def playbook_get_statistics() -> ToolResponse:
    manager = _get_playbook()
    return ToolResponse(
        content=json.dumps(
            {
                "operation": "get_statistics",
                "stats": manager.get_statistics(),
            },
            ensure_ascii=False,
        )
    )


def update_strategy_count(bullet_ids: list[str], is_helpful: bool) -> ToolResponse:
    manager = _get_playbook()
    if isinstance(bullet_ids, str):
        try:
            bullet_ids = json.loads(bullet_ids.replace("'", '"'))
        except Exception:
            bullet_ids = [bullet_ids]
    result = manager.apply_feedback(
        bullet_ids=[str(item) for item in bullet_ids],
        tag="helpful" if is_helpful else "harmful",
    )
    return ToolResponse(
        content=json.dumps(
            {
                "operation": "batch_update_counts",
                "success_count": len(result["updated_ids"]),
                "missing_count": len(result["missing_ids"]),
                "updated_ids": result["updated_ids"],
                "missing_ids": result["missing_ids"],
                "is_helpful": is_helpful,
            },
            ensure_ascii=False,
        )
    )


def playbook_add_bullet(
    content: str,
    bullet_type: str = "general",
    section: str | None = None,
) -> ToolResponse:
    manager = _get_playbook()
    result = manager.add_bullet(
        content=content,
        section=section,
        bullet_type=bullet_type,
        metadata={"source": "manual_add"},
    )
    return ToolResponse(
        content=json.dumps(
            {
                "operation": "add_bullet",
                **result,
            },
            ensure_ascii=False,
        )
    )


async def reflector_analyze(
    trace_json: str = "",
    max_refinement_rounds: int = 3,
) -> ToolResponse:
    manager = _get_playbook()
    trace_data = _load_trace_payload(trace_json)
    if not trace_data.get("steps") and not trace_data.get("input_text") and not trace_data.get("raw_trace_text"):
        return ToolResponse(
            content=json.dumps(
                {
                    "operation": "analyze",
                    "analysis": "",
                    "error_identification": "未提供可用的结构化 trace。",
                    "root_cause_analysis": "MemorySupervisor 在调用 Reflector 前没有设置当前 trace，也没有显式传入 trace_json。",
                    "correct_approach": "请先预加载当前 trace，或在调用 reflector_analyze 时显式传入 trace_json。",
                    "key_insight": "调用 Reflector 前必须确保当前结构化 trace 已预加载或显式传入，避免基于猜测生成反思。",
                    "key_insight_section": "tool_usage",
                    "is_success": False,
                    "helpful_bullet_ids": [],
                    "harmful_bullet_ids": [],
                    "neutral_bullet_ids": [],
                    "bullet_tags": [],
                },
                ensure_ascii=False,
            )
        )
    used_bullets = manager.get_bullets_by_ids(trace_data.get("retrieved_bullet_ids", []))
    reflector = _create_reflector()
    result = await reflector.analyze(
        trace_json=trace_data,
        used_bullets=used_bullets,
        max_refinement_rounds=max_refinement_rounds,
    )
    payload = result.to_dict()
    if reflector.last_debug_artifact:
        payload.update(reflector.last_debug_artifact)
    set_current_reflection_payload(payload)
    return ToolResponse(content=json.dumps(payload, ensure_ascii=False))


async def curator_apply_reflection(
    reflection_json: str = "",
    trace_json: str = "",
) -> ToolResponse:
    manager = _get_playbook()
    reflection_payload, reflection_source = _load_reflection_payload(reflection_json)
    if not reflection_payload:
        return ToolResponse(
            content=json.dumps(
                {
                    "operation": "curator_apply_delta",
                    "reasoning": "未提供可用的 reflection，无法安全生成增量更新。",
                    "applied_operations": [],
                    "new_bullet_ids": [],
                    "new_bullets": 0,
                    "skipped_duplicates": 0,
                    "error": "missing_reflection_context",
                    "reflection_source": reflection_source,
                },
                ensure_ascii=False,
            )
        )
    reflection_quality, missing_fields = _inspect_reflection_payload_quality(reflection_payload)
    reflection = _reflection_from_json(reflection_payload)
    if reflection_quality == "degraded" and not reflection.key_insight:
        return ToolResponse(
            content=json.dumps(
                {
                    "operation": "curator_apply_delta",
                    "reasoning": "reflection 输入缺少关键字段，已拒绝继续生成增量更新。",
                    "applied_operations": [],
                    "new_bullet_ids": [],
                    "new_bullets": 0,
                    "skipped_duplicates": 0,
                    "error": "degraded_reflection_input",
                    "reflection_source": reflection_source,
                    "reflection_quality": reflection_quality,
                    "missing_fields": missing_fields,
                },
                ensure_ascii=False,
            )
        )
    trace_data = _load_trace_payload(trace_json)
    if not trace_data.get("steps") and not trace_data.get("input_text") and not trace_data.get("raw_trace_text"):
        return ToolResponse(
            content=json.dumps(
                {
                    "operation": "curator_apply_delta",
                    "reasoning": "未提供可用的结构化 trace，无法安全生成增量更新。",
                    "applied_operations": [],
                    "new_bullet_ids": [],
                    "new_bullets": 0,
                    "skipped_duplicates": 0,
                    "error": "missing_trace_context",
                },
                ensure_ascii=False,
            )
        )
    curator = _create_curator()
    delta = await curator.propose_delta(
        manager=manager,
        reflection=reflection,
        trace_json=trace_data,
    )
    result = curator.apply_delta(manager=manager, delta=delta, source="curator", trace_data=trace_data)
    result["delta"] = delta.to_dict()
    result["reflection_source"] = reflection_source
    result["reflection_quality"] = reflection_quality
    if missing_fields:
        result["reflection_warnings"] = [f"reflection 缺少核心字段: {', '.join(missing_fields)}"]
    return ToolResponse(content=json.dumps(result, ensure_ascii=False))


def curator_grow_and_refine() -> ToolResponse:
    manager = _get_playbook()
    result = manager.grow_and_refine(record_execution=True)
    return ToolResponse(content=json.dumps(result, ensure_ascii=False))


async def curator_process_execution(trace_json: str) -> ToolResponse:
    trace_data = _load_trace_payload(trace_json)

    reflection_res = await reflector_analyze(
        trace_json=json.dumps(trace_data, ensure_ascii=False),
        max_refinement_rounds=3,
    )
    reflection_data = json.loads(reflection_res.content)

    helpful_ids = reflection_data.get("helpful_bullet_ids", [])
    harmful_ids = reflection_data.get("harmful_bullet_ids", [])

    if helpful_ids:
        update_strategy_count(helpful_ids, is_helpful=True)
    if harmful_ids:
        update_strategy_count(harmful_ids, is_helpful=False)

    curator_res = await curator_apply_reflection(
        trace_json=json.dumps(trace_data, ensure_ascii=False),
    )
    curator_data = json.loads(curator_res.content)

    refine_data = _get_playbook().grow_and_refine(record_execution=True)

    return ToolResponse(
        content=json.dumps(
            {
                "operation": "curator_complete_process",
                "trace": trace_data,
                "reflection": reflection_data,
                "curator": curator_data,
                "updates": {
                    "helpful": len(helpful_ids),
                    "harmful": len(harmful_ids),
                    "new_bullets": curator_data.get("new_bullets", 0),
                    "skipped_duplicates": curator_data.get("skipped_duplicates", 0),
                    "removed_bullets": refine_data.get("removed_count", 0),
                    "merged_duplicates": refine_data.get("merged_count", 0),
                },
                "new_bullet_ids": curator_data.get("new_bullet_ids", []),
            },
            ensure_ascii=False,
        )
    )
