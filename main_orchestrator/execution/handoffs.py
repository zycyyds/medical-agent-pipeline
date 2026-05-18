from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from agentscope.tool import ToolResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
STEP1_DIR = PROJECT_ROOT / "step-1"
STEP4_DIR = PROJECT_ROOT / "step-4"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR, EXECUTION_DIR, STEP1_DIR, STEP4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runtime import (  # noqa: E402
    collect_tool_results,
    has_model_credentials,
    json_tool_response,
    make_user_msg,
    parse_agent_text_json,
    parse_tool_response,
)
from contracts import WorkerResult, WorkerStatus  # noqa: E402
from step1_agent import build_step1_task_prompt, create_step1_agent  # noqa: E402
from step4_agent import build_step4_task_prompt, create_step4_agent  # noqa: E402
from step4_runtime import validate_step4_output as runtime_validate_step4_output  # noqa: E402
from validators import validate_records, validate_step1_output  # noqa: E402


def worker_result_from_dict(payload: dict[str, Any]) -> WorkerResult:
    return WorkerResult(
        status=WorkerStatus(str(payload.get("status") or WorkerStatus.FAILED.value)),
        summary=str(payload.get("summary") or ""),
        artifacts=dict(payload.get("artifacts") or {}),
        issues=list(payload.get("issues") or []),
        next_recommendation=str(payload.get("next_recommendation") or ""),
    )


def worker_result_tool_response(result: WorkerResult) -> ToolResponse:
    return json_tool_response(result.to_dict())


def _artifact_dict(
    input_path: str | None,
    records_path: str,
    output_root: str,
    generated_script_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = {
        "input_path": input_path or "",
        "records_path": records_path,
        "generated_script_path": generated_script_path,
        "step1_output_root": output_root,
    }
    artifacts.update(extra or {})
    return artifacts


def _finalize_step1_result(
    *,
    task_type: str,
    input_path: str | None,
    records_path: str,
    output_root: str,
    generated_script_path: str,
    tool_results: dict[str, list[dict[str, Any]]] | None = None,
    agent_summary: str = "",
) -> WorkerResult:
    record_validation = validate_records(records_path)
    if not record_validation.passed:
        return WorkerResult.failure(
            summary="Step1 records 校验失败。",
            issues=record_validation.issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_artifact_dict(input_path, records_path, output_root, generated_script_path, record_validation.details),
        )

    tool_results = tool_results or {}
    run_payload = (tool_results.get("run_reorganize_from_records_tool") or [None])[-1] or {}
    run_artifacts = dict(run_payload.get("artifacts") or {})
    if run_payload and run_payload.get("status") != WorkerStatus.SUCCESS.value:
        return WorkerResult.failure(
            summary="Step1 重组脚本执行失败。",
            issues=list(run_payload.get("issues") or run_artifacts.get("errors") or []),
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_artifact_dict(input_path, records_path, output_root, generated_script_path, run_artifacts),
            next_recommendation="进入 Step1 repair worker。",
        )

    expected_min = int(run_artifacts.get("processed_records") or run_artifacts.get("records_count") or 1)
    output_validation = validate_step1_output(output_root, expected_min_files=max(1, expected_min))
    if not output_validation.passed:
        return WorkerResult.failure(
            summary="Step1 输出验收失败。",
            issues=output_validation.issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_artifact_dict(
                input_path,
                records_path,
                output_root,
                generated_script_path,
                {**record_validation.details, **run_artifacts, **output_validation.details},
            ),
        )

    output_details = dict(output_validation.details)
    output_details["output_modality_counts"] = output_details.pop("modality_counts", {})
    artifacts = _artifact_dict(
        input_path,
        records_path,
        output_root,
        generated_script_path,
        {**record_validation.details, **run_artifacts, **output_details},
    )
    written_files = run_artifacts.get("written_files") or output_details.get("output_file_count") or 0
    return WorkerResult.success(
        summary=agent_summary
        or (
            f"Step1 handoff 完成：{task_type} 已由 Step1ReActAgent 执行，"
            f"输出文件 {written_files} 个。"
        ),
        artifacts=artifacts,
    )


async def handoff_to_step1_agent(
    task_type: str,
    records_path: str,
    output_root: str,
    generated_script_path: str,
    input_path: str | None = None,
) -> ToolResponse:
    """Create and invoke the Step1ReActAgent worker, then return WorkerResult JSON."""
    if task_type not in {"step1_only", "resume_from_records"}:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step1 handoff 收到不支持的 task_type。",
                issues=[f"Unsupported task_type: {task_type}"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_artifact_dict(input_path, records_path, output_root, generated_script_path),
            )
        )
    if task_type == "step1_only" and not input_path:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step1 handoff 缺少 input_path。",
                issues=["step1_only 必须提供原始数据目录 input_path。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_artifact_dict(input_path, records_path, output_root, generated_script_path),
            )
        )
    if not has_model_credentials("agent_1"):
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step1ReActAgent 未配置模型密钥，无法执行 handoff。",
                issues=["请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.agent_1.api_key。"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_artifact_dict(input_path, records_path, output_root, generated_script_path),
            )
        )

    agent = create_step1_agent()
    task_prompt = build_step1_task_prompt(
        task_type=task_type,
        input_path=input_path,
        records_path=records_path,
        output_root=output_root,
        generated_script_path=generated_script_path,
    )
    try:
        res = await agent(make_user_msg("OrchestratorAgent", task_prompt))
        tool_results = await collect_tool_results(agent)
        parsed_final = parse_agent_text_json(res.get_text_content() or "")
        if parsed_final and parsed_final.get("status") not in {None, WorkerStatus.SUCCESS.value}:
            return worker_result_tool_response(worker_result_from_dict(parsed_final))
        result = _finalize_step1_result(
            task_type=task_type,
            input_path=input_path,
            records_path=records_path,
            output_root=output_root,
            generated_script_path=generated_script_path,
            tool_results=tool_results,
        )
        return worker_result_tool_response(result)
    except Exception as exc:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step1ReActAgent handoff 执行失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_artifact_dict(input_path, records_path, output_root, generated_script_path),
            )
        )


async def run_step1_handoff(
    task_type: Literal["step1_only", "resume_from_records"],
    records_path: str,
    output_root: str,
    generated_script_path: str,
    input_path: str | None = None,
) -> WorkerResult:
    response = await handoff_to_step1_agent(
        task_type=task_type,
        input_path=input_path,
        records_path=records_path,
        output_root=output_root,
        generated_script_path=generated_script_path,
    )
    return worker_result_from_dict(parse_tool_response(response))


def _step4_artifact_dict(
    input_path: str,
    output_root: str,
    task_text: str | None = None,
    memory_context: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = {
        "input_path": input_path,
        "output_root": output_root,
        "task_text": task_text or "",
        "memory_context_used": bool(str(memory_context or "").strip()),
    }
    artifacts.update(extra or {})
    return artifacts


def _finalize_step4_result(
    *,
    task_type: str,
    input_path: str,
    output_root: str,
    task_text: str | None,
    memory_context: str | None = None,
    tool_results: dict[str, list[dict[str, Any]]] | None = None,
    agent_summary: str = "",
) -> WorkerResult:
    tool_results = tool_results or {}
    run_payload = (tool_results.get("run_step4_column_selection_tool") or [None])[-1] or {}
    normalize_payload = (tool_results.get("normalize_step4_outputs_tool") or [None])[-1] or {}
    task_payload = (tool_results.get("resolve_step4_task_text_tool") or [None])[-1] or {}
    resolved_task_text = (
        ((task_payload.get("artifacts") or {}).get("task_text"))
        if isinstance(task_payload.get("artifacts"), dict)
        else None
    ) or task_text

    for tool_name in (
        "resolve_step4_input_tool",
        "resolve_step4_task_text_tool",
        "run_step4_column_selection_tool",
        "normalize_step4_outputs_tool",
        "validate_step4_output_tool",
    ):
        payload = (tool_results.get(tool_name) or [None])[-1] or {}
        if payload and payload.get("status") != WorkerStatus.SUCCESS.value:
            return WorkerResult.failure(
                summary=str(payload.get("summary") or f"{tool_name} 执行失败。"),
                issues=list(payload.get("issues") or []),
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step4_artifact_dict(
                    input_path,
                    output_root,
                    resolved_task_text,
                    memory_context,
                    dict(payload.get("artifacts") or {}),
                ),
                next_recommendation="进入 Step4 repair worker。",
            )

    run_artifacts = dict(run_payload.get("artifacts") or {})
    normalize_artifacts = dict(normalize_payload.get("artifacts") or {})
    filtered_csv = normalize_artifacts.get("filtered_csv_path") or run_artifacts.get("filtered_csv_path")
    selection_report = normalize_artifacts.get("selection_report_path") or run_artifacts.get("selection_report_path")
    next_input_csv = normalize_artifacts.get("next_input_csv")
    next_selection_report = normalize_artifacts.get("next_selection_report")
    if not filtered_csv or not selection_report:
        return WorkerResult.failure(
            summary="Step4 handoff 未返回必要产物。",
            issues=["缺少 filtered_csv_path 或 selection_report_path。"],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step4_artifact_dict(
                input_path,
                output_root,
                resolved_task_text,
                memory_context,
                {**run_artifacts, **normalize_artifacts},
            ),
        )

    validation = runtime_validate_step4_output(
        filtered_csv_path=filtered_csv,
        selection_report_path=selection_report,
        next_input_csv=next_input_csv,
        next_selection_report=next_selection_report,
    )
    artifacts = _step4_artifact_dict(
        input_path,
        output_root,
        resolved_task_text,
        memory_context,
        {**run_artifacts, **normalize_artifacts, **validation.details},
    )
    if not validation.passed:
        return WorkerResult.failure(
            summary="Step4 输出验收失败。",
            issues=validation.issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=artifacts,
        )

    return WorkerResult.success(
        summary=agent_summary or f"Step4 handoff 完成：{task_type} 已由 Step4ReActAgent 执行。",
        artifacts=artifacts,
    )


async def handoff_to_step4_agent(
    task_type: str,
    input_path: str,
    output_root: str,
    task_text: str | None = None,
    memory_context: str | None = None,
) -> ToolResponse:
    """Create and invoke the Step4ReActAgent worker, then return WorkerResult JSON."""
    if task_type not in {"step4_only", "resume_from_step2_3"}:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step4 handoff 收到不支持的 task_type。",
                issues=[f"Unsupported task_type: {task_type}"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step4_artifact_dict(input_path, output_root, task_text, memory_context),
            )
        )
    if not input_path:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step4 handoff 缺少 input_path。",
                issues=["Step4 必须提供输入表格 input_path。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step4_artifact_dict(input_path, output_root, task_text, memory_context),
            )
        )
    if not has_model_credentials("agent_4"):
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step4ReActAgent 未配置模型密钥，无法执行 handoff。",
                issues=["请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.agent_4.api_key。"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step4_artifact_dict(input_path, output_root, task_text, memory_context),
            )
        )

    agent = create_step4_agent()
    task_prompt = build_step4_task_prompt(
        task_type="resume_from_step2_3" if task_type == "resume_from_step2_3" else "step4_only",
        input_path=input_path,
        output_root=output_root,
        task_text=task_text,
        memory_context=memory_context,
    )
    try:
        res = await agent(make_user_msg("OrchestratorAgent", task_prompt))
        tool_results = await collect_tool_results(agent)
        parsed_final = parse_agent_text_json(res.get_text_content() or "")
        if parsed_final and parsed_final.get("status") not in {None, WorkerStatus.SUCCESS.value}:
            return worker_result_tool_response(worker_result_from_dict(parsed_final))
        result = _finalize_step4_result(
            task_type=task_type,
            input_path=input_path,
            output_root=output_root,
            task_text=task_text,
            memory_context=memory_context,
            tool_results=tool_results,
        )
        return worker_result_tool_response(result)
    except Exception as exc:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step4ReActAgent handoff 执行失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step4_artifact_dict(input_path, output_root, task_text, memory_context),
            )
        )


async def run_step4_handoff(
    task_type: Literal["step4_only", "resume_from_step2_3"],
    input_path: str,
    output_root: str,
    task_text: str | None = None,
    memory_context: str | None = None,
) -> WorkerResult:
    response = await handoff_to_step4_agent(
        task_type=task_type,
        input_path=input_path,
        output_root=output_root,
        task_text=task_text,
        memory_context=memory_context,
    )
    return worker_result_from_dict(parse_tool_response(response))
