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
STEP23_DIR = PROJECT_ROOT / "step-2-3"
STEP4_DIR = PROJECT_ROOT / "step-4"
STEP5_DIR = PROJECT_ROOT / "step-5"
STEP6_DIR = PROJECT_ROOT / "step-6"
STEP7_DIR = PROJECT_ROOT / "step-7"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR, EXECUTION_DIR, STEP1_DIR, STEP23_DIR, STEP4_DIR, STEP5_DIR, STEP6_DIR, STEP7_DIR):
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
from step23_agent import build_step23_task_prompt, create_step23_agent  # noqa: E402
from step23_runtime import validate_step23_output as runtime_validate_step23_output  # noqa: E402
from step4_agent import build_step4_task_prompt, create_step4_agent  # noqa: E402
from step4_runtime import validate_step4_output as runtime_validate_step4_output  # noqa: E402
from step5_agent import build_step5_task_prompt, create_step5_agent  # noqa: E402
from step5_runtime import validate_step5_output as runtime_validate_step5_output  # noqa: E402
from run_step6_ml_pipeline import get_default_output_step6_dir, run_step6_only  # noqa: E402
from run_step7_ml_pipeline import get_default_output_step7_dir, run_step7_only  # noqa: E402
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
    memory_context: str | None = None,
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
        memory_context=memory_context,
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
    memory_context: str | None = None,
) -> WorkerResult:
    response = await handoff_to_step1_agent(
        task_type=task_type,
        input_path=input_path,
        records_path=records_path,
        output_root=output_root,
        generated_script_path=generated_script_path,
        memory_context=memory_context,
    )
    return worker_result_from_dict(parse_tool_response(response))


def _step23_artifact_dict(
    input_path: str,
    output_root: str,
    mode: str = "auto",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = {
        "input_path": input_path,
        "output_root": output_root,
        "mode": mode,
    }
    artifacts.update(extra or {})
    return artifacts


def _finalize_step23_result(
    *,
    task_type: str,
    input_path: str,
    output_root: str,
    mode: str = "auto",
    tool_results: dict[str, list[dict[str, Any]]] | None = None,
    agent_summary: str = "",
) -> WorkerResult:
    tool_results = tool_results or {}
    common_tools = (
        "resolve_step23_input_tool",
        "scan_step23_input_tool",
        "select_step23_pipeline_tool",
    )
    collected: dict[str, Any] = {}

    def _collect_required(tool_name: str) -> WorkerResult | None:
        payload = (tool_results.get(tool_name) or [None])[-1] or {}
        if not payload:
            return WorkerResult.failure(
                summary=f"Step2-3 handoff 未执行必要工具: {tool_name}",
                issues=[f"missing tool result: {tool_name}"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step23_artifact_dict(input_path, output_root, mode, collected),
            )
        artifacts = dict(payload.get("artifacts") or {})
        collected.update(artifacts)
        if payload.get("status") != WorkerStatus.SUCCESS.value:
            return WorkerResult.failure(
                summary=str(payload.get("summary") or f"{tool_name} 执行失败。"),
                issues=list(payload.get("issues") or []),
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step23_artifact_dict(input_path, output_root, mode, collected),
                next_recommendation="进入 Step2-3 repair worker。",
            )
        return None

    for tool_name in common_tools:
        failure = _collect_required(tool_name)
        if failure is not None:
            return failure

    resolved_mode = str(collected.get("resolved_mode") or mode or "auto")
    if resolved_mode == "directory":
        required_tools = (
            "run_step23_directory_pipeline_tool",
            "normalize_step23_outputs_tool",
            "validate_step23_output_tool",
        )
    elif resolved_mode == "ocr_fill":
        required_tools = (
            "run_step23_ocr_tool",
            "run_step23_extraction_tool",
            "run_step23_fill_table_tool",
            "normalize_step23_outputs_tool",
            "validate_step23_output_tool",
        )
    else:
        return WorkerResult.failure(
            summary="Step2-3 handoff 未能确定 pipeline 模式。",
            issues=[f"resolved_mode 不支持或为空: {resolved_mode}"],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step23_artifact_dict(input_path, output_root, mode, collected),
        )

    for tool_name in required_tools:
        failure = _collect_required(tool_name)
        if failure is not None:
            return failure

    next_input_csv = collected.get("next_input_csv")
    next_summary_json = collected.get("next_summary_json")
    if not next_input_csv:
        return WorkerResult.failure(
            summary="Step2-3 handoff 未返回 next_input/input.csv。",
            issues=["缺少 next_input_csv。"],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step23_artifact_dict(input_path, output_root, mode, collected),
        )

    validation = runtime_validate_step23_output(
        next_input_csv=next_input_csv,
        next_summary_json=next_summary_json,
    )
    artifacts = _step23_artifact_dict(
        input_path,
        output_root,
        mode,
        {**collected, **dict(validation.get("details") or {})},
    )
    if not validation.get("passed"):
        return WorkerResult.failure(
            summary="Step2-3 输出验收失败。",
            issues=list(validation.get("issues") or []),
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=artifacts,
        )

    return WorkerResult.success(
        summary=agent_summary or f"Step2-3 handoff 完成：{task_type} 已由 Step2_3ReActAgent 执行。",
        artifacts=artifacts,
    )


async def handoff_to_step23_agent(
    task_type: str,
    input_path: str,
    output_root: str,
    mode: Literal["auto", "directory", "ocr_fill"] = "auto",
    cache_path: str | None = None,
    ocr_workers: int = 8,
    concurrency: int = 8,
    use_llm: bool = True,
    user_hint: str = "",
    memory_context: str | None = None,
) -> ToolResponse:
    """Create and invoke the Step2_3ReActAgent worker, then return WorkerResult JSON."""
    if task_type not in {"step2_3_only", "full_pipeline_step2_3"}:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step2-3 handoff 收到不支持的 task_type。",
                issues=[f"Unsupported task_type: {task_type}"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step23_artifact_dict(input_path, output_root, mode),
            )
        )
    if not input_path:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step2-3 handoff 缺少 input_path。",
                issues=["Step2-3 必须提供输入路径 input_path。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step23_artifact_dict(input_path, output_root, mode),
            )
        )
    if not has_model_credentials("agent_2_3"):
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step2_3ReActAgent 未配置模型密钥，无法执行 handoff。",
                issues=["请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.agent_2_3.api_key。"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step23_artifact_dict(input_path, output_root, mode),
            )
        )

    agent = create_step23_agent()
    task_prompt = build_step23_task_prompt(
        task_type=task_type,
        input_path=input_path,
        output_root=output_root,
        mode=mode,
        cache_path=cache_path,
        ocr_workers=ocr_workers,
        concurrency=concurrency,
        use_llm=use_llm,
        user_hint=user_hint,
        memory_context=memory_context,
    )
    try:
        res = await agent(make_user_msg("OrchestratorAgent", task_prompt))
        tool_results = await collect_tool_results(agent)
        parsed_final = parse_agent_text_json(res.get_text_content() or "")
        if parsed_final and parsed_final.get("status") not in {None, WorkerStatus.SUCCESS.value}:
            return worker_result_tool_response(worker_result_from_dict(parsed_final))
        result = _finalize_step23_result(
            task_type=task_type,
            input_path=input_path,
            output_root=output_root,
            mode=mode,
            tool_results=tool_results,
        )
        return worker_result_tool_response(result)
    except Exception as exc:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step2_3ReActAgent handoff 执行失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step23_artifact_dict(input_path, output_root, mode),
            )
        )


async def run_step23_handoff(
    task_type: Literal["step2_3_only", "full_pipeline_step2_3"],
    input_path: str,
    output_root: str,
    mode: Literal["auto", "directory", "ocr_fill"] = "auto",
    cache_path: str | None = None,
    ocr_workers: int = 8,
    concurrency: int = 8,
    use_llm: bool = True,
    user_hint: str = "",
    memory_context: str | None = None,
) -> WorkerResult:
    response = await handoff_to_step23_agent(
        task_type=task_type,
        input_path=input_path,
        output_root=output_root,
        mode=mode,
        cache_path=cache_path,
        ocr_workers=ocr_workers,
        concurrency=concurrency,
        use_llm=use_llm,
        user_hint=user_hint,
        memory_context=memory_context,
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


def _step5_artifact_dict(
    input_path: str,
    output_root: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = {
        "input_path": input_path,
        "output_root": output_root,
    }
    artifacts.update(extra or {})
    return artifacts


def _finalize_step5_result(
    *,
    task_type: str,
    input_path: str,
    output_root: str,
    tool_results: dict[str, list[dict[str, Any]]] | None = None,
    agent_summary: str = "",
) -> WorkerResult:
    tool_results = tool_results or {}
    collected: dict[str, Any] = {}

    def _collect_required(tool_name: str) -> WorkerResult | None:
        payload = (tool_results.get(tool_name) or [None])[-1] or {}
        if not payload:
            return WorkerResult.failure(
                summary=f"Step5 handoff 未执行必要工具: {tool_name}",
                issues=[f"missing tool result: {tool_name}"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step5_artifact_dict(input_path, output_root, collected),
            )
        artifacts = dict(payload.get("artifacts") or {})
        collected.update(artifacts)
        if payload.get("status") != WorkerStatus.SUCCESS.value:
            return WorkerResult.failure(
                summary=str(payload.get("summary") or f"{tool_name} 执行失败。"),
                issues=list(payload.get("issues") or []),
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step5_artifact_dict(input_path, output_root, collected),
                next_recommendation="进入 Step5 repair worker。",
            )
        return None

    for tool_name in (
        "resolve_step5_input_tool",
        "profile_step5_table_tool",
        "classify_step5_column_risks_tool",
        "run_step5_data_cleaning_tool",
        "normalize_step5_outputs_tool",
        "validate_step5_output_tool",
    ):
        failure = _collect_required(tool_name)
        if failure is not None:
            return failure

    cleaned_csv = collected.get("cleaned_csv_path")
    next_input_csv = collected.get("next_input_csv")
    if not cleaned_csv or not next_input_csv:
        return WorkerResult.failure(
            summary="Step5 handoff 未返回必要 cleaned/next_input 产物。",
            issues=["缺少 cleaned_csv_path 或 next_input_csv。"],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step5_artifact_dict(input_path, output_root, collected),
        )

    validation = runtime_validate_step5_output(
        input_csv_path=collected.get("input_csv_path") or input_path,
        cleaned_csv_path=cleaned_csv,
        data_quality_report_path=collected.get("data_quality_report_path"),
        column_risk_report_path=collected.get("column_risk_report_path"),
        next_input_csv=next_input_csv,
        next_data_quality_report=collected.get("next_data_quality_report"),
        next_column_risk_report=collected.get("next_column_risk_report"),
    )
    artifacts = _step5_artifact_dict(input_path, output_root, {**collected, **validation.details})
    if not validation.passed:
        return WorkerResult.failure(
            summary="Step5 输出验收失败。",
            issues=validation.issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=artifacts,
        )

    return WorkerResult.success(
        summary=agent_summary or f"Step5 handoff 完成：{task_type} 已由 Step5ReActAgent 执行。",
        artifacts=artifacts,
    )


async def handoff_to_step5_agent(
    task_type: str,
    input_path: str,
    output_root: str,
    workers: int = 4,
    llm_workers: int = 2,
    enable_llm: bool = True,
    memory_context: str | None = None,
) -> ToolResponse:
    """Create and invoke the Step5ReActAgent worker, then return WorkerResult JSON."""
    if task_type not in {"step5_only", "resume_from_step4"}:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step5 handoff 收到不支持的 task_type。",
                issues=[f"Unsupported task_type: {task_type}"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step5_artifact_dict(input_path, output_root),
            )
        )
    if not input_path:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step5 handoff 缺少 input_path。",
                issues=["Step5 必须提供输入表格 input_path。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step5_artifact_dict(input_path, output_root),
            )
        )
    if not has_model_credentials("agent_5"):
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step5ReActAgent 未配置模型密钥，无法执行 handoff。",
                issues=["请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.agent_5.api_key。"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step5_artifact_dict(input_path, output_root),
            )
        )

    agent = create_step5_agent()
    task_prompt = build_step5_task_prompt(
        task_type="resume_from_step4" if task_type == "resume_from_step4" else "step5_only",
        input_path=input_path,
        output_root=output_root,
        workers=workers,
        llm_workers=llm_workers,
        enable_llm=enable_llm,
        memory_context=memory_context,
    )
    try:
        res = await agent(make_user_msg("OrchestratorAgent", task_prompt))
        tool_results = await collect_tool_results(agent)
        parsed_final = parse_agent_text_json(res.get_text_content() or "")
        if parsed_final and parsed_final.get("status") not in {None, WorkerStatus.SUCCESS.value}:
            return worker_result_tool_response(worker_result_from_dict(parsed_final))
        result = _finalize_step5_result(
            task_type=task_type,
            input_path=input_path,
            output_root=output_root,
            tool_results=tool_results,
        )
        return worker_result_tool_response(result)
    except Exception as exc:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step5ReActAgent handoff 执行失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step5_artifact_dict(input_path, output_root),
            )
        )


async def run_step5_handoff(
    task_type: Literal["step5_only", "resume_from_step4"],
    input_path: str,
    output_root: str,
    workers: int = 4,
    llm_workers: int = 2,
    enable_llm: bool = True,
    memory_context: str | None = None,
) -> WorkerResult:
    response = await handoff_to_step5_agent(
        task_type=task_type,
        input_path=input_path,
        output_root=output_root,
        workers=workers,
        llm_workers=llm_workers,
        enable_llm=enable_llm,
        memory_context=memory_context,
    )
    return worker_result_from_dict(parse_tool_response(response))


def _passed_count_from_json(path_text: str) -> int:
    if not path_text:
        return 0
    try:
        with Path(path_text).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return int(payload.get("passed_count") or len(payload.get("passed_patient_ids") or []))
    except Exception:
        return 0


def _step6_artifact_dict(
    input_path: str,
    output_root: str,
    result: dict[str, Any] | None = None,
    memory_context: str | None = None,
) -> dict[str, Any]:
    result = result or {}
    passed_ids = result.get("passed_patient_ids") if isinstance(result.get("passed_patient_ids"), list) else []
    passed_json = str(result.get("passed_patients_json") or "")
    passed_count = len(passed_ids) if passed_ids else _passed_count_from_json(passed_json)
    return {
        "input_path": str(input_path or result.get("input_csv") or ""),
        "input_csv": str(result.get("input_csv") or input_path or ""),
        "raw_csv": str(result.get("raw_csv") or input_path or ""),
        "output_step6": str(result.get("output_step6") or output_root),
        "step6_report_txt": str(result.get("step6_report_txt") or ""),
        "step6_report_json": str(result.get("step6_report_json") or ""),
        "passed_patients_json": passed_json,
        "passed_patient_count": passed_count,
        "passed_patient_ids_sample": [str(item) for item in passed_ids[:20]],
        "memory_context_used": bool(str(memory_context or "").strip()),
    }


def _validate_step6_artifacts(artifacts: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in ("step6_report_txt", "step6_report_json", "passed_patients_json"):
        path_text = str(artifacts.get(key) or "")
        if not path_text:
            issues.append(f"缺少 {key}。")
        elif not Path(path_text).is_file():
            issues.append(f"{key} 不存在: {path_text}")
    if int(artifacts.get("passed_patient_count") or 0) < 0:
        issues.append("passed_patient_count 非法。")
    return issues


async def handoff_to_step6_agent(
    task_type: str,
    input_path: str,
    output_root: str = "",
    raw_csv: str | None = None,
    memory_context: str | None = None,
    reuse_existing: bool = False,
) -> ToolResponse:
    """Invoke the Step6 standalone ConsistencyAgent and return WorkerResult JSON."""
    output_root = output_root or str(get_default_output_step6_dir())
    if task_type not in {"step6_only", "resume_from_step5"}:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step6 handoff 收到不支持的 task_type。",
                issues=[f"Unsupported task_type: {task_type}"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step6_artifact_dict(input_path, output_root, memory_context=memory_context),
            )
        )
    if not input_path:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step6 handoff 缺少 input_path。",
                issues=["Step6 必须提供 Step5 输出 CSV input_path。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step6_artifact_dict(input_path, output_root, memory_context=memory_context),
            )
        )

    try:
        result = await run_step6_only(
            {
                "input_csv": input_path,
                "raw_csv": raw_csv or input_path,
                "output_step6_dir": output_root,
                "memory_context": memory_context or "",
                "reuse_existing": reuse_existing,
            }
        )
        artifacts = _step6_artifact_dict(input_path, output_root, result=result, memory_context=memory_context)
        if not result.get("success"):
            return worker_result_tool_response(
                WorkerResult.failure(
                    summary="Step6 一致性验证失败。",
                    issues=[str(result.get("error") or "Step6 returned success=false.")],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=artifacts,
                )
            )
        validation_issues = _validate_step6_artifacts(artifacts)
        if validation_issues:
            return worker_result_tool_response(
                WorkerResult.failure(
                    summary="Step6 handoff 后输出验收失败。",
                    issues=validation_issues,
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=artifacts,
                )
            )
        return worker_result_tool_response(WorkerResult.success(summary="Step6 一致性验证完成。", artifacts=artifacts))
    except Exception as exc:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step6ReActAgent handoff 执行失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step6_artifact_dict(input_path, output_root, memory_context=memory_context),
            )
        )


async def run_step6_handoff(
    task_type: Literal["step6_only", "resume_from_step5"],
    input_path: str,
    output_root: str,
    raw_csv: str | None = None,
    memory_context: str | None = None,
    reuse_existing: bool = False,
) -> WorkerResult:
    response = await handoff_to_step6_agent(
        task_type=task_type,
        input_path=input_path,
        output_root=output_root,
        raw_csv=raw_csv,
        memory_context=memory_context,
        reuse_existing=reuse_existing,
    )
    return worker_result_from_dict(parse_tool_response(response))


def _step7_artifact_dict(
    input_path: str,
    output_root: str,
    selection_report: str | None = None,
    passed_patients_json: str | None = None,
    task_text: str | None = None,
    result: dict[str, Any] | None = None,
    memory_context: str | None = None,
) -> dict[str, Any]:
    result = result or {}
    dataset_csvs = [str(item) for item in result.get("step7_dataset_csvs") or []]
    dataset_jsonls = [str(item) for item in result.get("step7_dataset_jsonls") or []]
    dataset_jsons = [str(item) for item in result.get("step7_dataset_jsons") or []]
    model_configs = [str(item) for item in result.get("model_config_jsons") or []]
    return {
        "input_path": str(input_path or result.get("input_csv") or ""),
        "input_csv": str(result.get("input_csv") or input_path or ""),
        "selection_report": str(result.get("selection_report") or selection_report or ""),
        "passed_patients_json": str(result.get("passed_patients_json") or passed_patients_json or ""),
        "passed_patient_count": int(result.get("passed_patient_count") or 0),
        "output_step7": str(result.get("output_step7") or output_root),
        "task_text": str(result.get("task_text") or task_text or ""),
        "step7_dataset_csvs": dataset_csvs,
        "step7_dataset_jsonls": dataset_jsonls,
        "step7_dataset_jsons": dataset_jsons,
        "model_config_jsons": model_configs,
        "step7_dataset_count": len(dataset_csvs),
        "memory_context_used": bool(str(memory_context or "").strip()),
    }


def _validate_step7_artifacts(artifacts: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in ("step7_dataset_csvs", "step7_dataset_jsonls", "step7_dataset_jsons", "model_config_jsons"):
        values = artifacts.get(key)
        if not isinstance(values, list) or not values:
            issues.append(f"缺少 {key}。")
            continue
        missing = [str(path) for path in values if not Path(str(path)).is_file()]
        if missing:
            issues.append(f"{key} 存在缺失文件: {missing[:3]}")
    for key in ("input_csv", "selection_report", "passed_patients_json"):
        path_text = str(artifacts.get(key) or "")
        if not path_text:
            issues.append(f"缺少 {key}。")
        elif not Path(path_text).is_file():
            issues.append(f"{key} 不存在: {path_text}")
    return issues


async def handoff_to_step7_agent(
    task_type: str,
    input_path: str,
    selection_report: str,
    passed_patients_json: str,
    output_root: str = "",
    task_text: str | None = None,
    memory_context: str | None = None,
) -> ToolResponse:
    """Invoke the Step7 standalone DataPrepAgent and return WorkerResult JSON."""
    output_root = output_root or str(get_default_output_step7_dir())
    if task_type not in {"step7_only", "resume_from_step6"}:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step7 handoff 收到不支持的 task_type。",
                issues=[f"Unsupported task_type: {task_type}"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=_step7_artifact_dict(input_path, output_root, selection_report, passed_patients_json, task_text=task_text, memory_context=memory_context),
            )
        )
    if not input_path or not selection_report or not passed_patients_json or not str(task_text or "").strip():
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step7 handoff 缺少必要输入。",
                issues=["Step7 必须提供 input_path、selection_report、passed_patients_json 和 task_text。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step7_artifact_dict(input_path, output_root, selection_report, passed_patients_json, task_text=task_text, memory_context=memory_context),
            )
        )

    try:
        result = await run_step7_only(
            {
                "input_csv": input_path,
                "selection_report": selection_report,
                "passed_patients_json": passed_patients_json,
                "output_step7_dir": output_root,
                "task_text": task_text or "",
                "memory_context": memory_context or "",
            }
        )
        artifacts = _step7_artifact_dict(
            input_path,
            output_root,
            selection_report=selection_report,
            passed_patients_json=passed_patients_json,
            task_text=task_text,
            result=result,
            memory_context=memory_context,
        )
        if not result.get("success"):
            return worker_result_tool_response(
                WorkerResult.failure(
                    summary="Step7 ML 训练数据生成失败。",
                    issues=[str(result.get("error") or "Step7 returned success=false.")],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=artifacts,
                )
            )
        validation_issues = _validate_step7_artifacts(artifacts)
        if validation_issues:
            return worker_result_tool_response(
                WorkerResult.failure(
                    summary="Step7 handoff 后输出验收失败。",
                    issues=validation_issues,
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=artifacts,
                )
            )
        return worker_result_tool_response(WorkerResult.success(summary="Step7 ML 训练数据生成完成。", artifacts=artifacts))
    except Exception as exc:
        return worker_result_tool_response(
            WorkerResult.failure(
                summary="Step7ReActAgent handoff 执行失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=_step7_artifact_dict(input_path, output_root, selection_report, passed_patients_json, task_text=task_text, memory_context=memory_context),
            )
        )


async def run_step7_handoff(
    task_type: Literal["step7_only", "resume_from_step6"],
    input_path: str,
    selection_report: str,
    passed_patients_json: str,
    output_root: str,
    task_text: str | None = None,
    memory_context: str | None = None,
) -> WorkerResult:
    response = await handoff_to_step7_agent(
        task_type=task_type,
        input_path=input_path,
        selection_report=selection_report,
        passed_patients_json=passed_patients_json,
        output_root=output_root,
        task_text=task_text,
        memory_context=memory_context,
    )
    return worker_result_from_dict(parse_tool_response(response))
