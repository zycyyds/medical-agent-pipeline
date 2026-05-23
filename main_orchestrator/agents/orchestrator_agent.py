from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
MEMORY_DIR = ORCHESTRATOR_DIR / "memory"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR, EXECUTION_DIR, MEMORY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, ToolResponse

from agent_runtime import (
    collect_tool_results,
    create_openai_model_and_formatter,
    has_model_credentials,
    json_tool_response,
    make_user_msg,
    parse_agent_text_json,
    parse_tool_response,
)
from contracts import PipelineRunResult, PipelineState, RepairTicket, TaskSpec, TaskType, WorkerResult, WorkerStatus
from handoffs import (
    handoff_to_step1_agent,
    handoff_to_step23_agent,
    handoff_to_step4_agent,
    handoff_to_step5_agent,
    handoff_to_step6_agent,
    handoff_to_step7_agent,
    worker_result_from_dict,
)

try:
    from memory_supervisor import get_pipeline_context, get_step_memory_context, report_pipeline_result
except Exception:  # pragma: no cover - memory is optional in minimal test/runtime environments
    get_pipeline_context = None
    get_step_memory_context = None
    report_pipeline_result = None

STEP_ORDER = ["step1", "step2_3", "step4", "step5", "step6", "step7"]
MEMORY_TOOL_NAMES = [
    "memory_search_rules_tool",
    "memory_search_step_tool",
    "memory_skip_tool",
    "memory_record_trace_tool",
]
STEP_TO_STATE = {
    "step1": PipelineState.STEP1_RECORDING,
    "step2_3": PipelineState.STEP2_3_RUNNING,
    "step4": PipelineState.STEP4_RUNNING,
    "step5": PipelineState.STEP5_RUNNING,
    "step6": PipelineState.STEP6_RUNNING,
    "step7": PipelineState.STEP7_RUNNING,
}
STEP_LABELS = {
    "step1": "Step1 数据重组",
    "step2_3": "Step2-3 结构化",
    "step4": "Step4 任务裁剪",
    "step5": "Step5 数据清洗",
    "step6": "Step6 一致性验证",
    "step7": "Step7 ML 数据集生成",
}

STEP1_TASK_TYPES = {TaskType.STEP1_ONLY, TaskType.RESUME_FROM_RECORDS}
STEP23_TASK_TYPES = {TaskType.STEP2_3_ONLY}
STEP4_TASK_TYPES = {TaskType.STEP4_ONLY, TaskType.RESUME_FROM_STEP2_3, TaskType.RESUME_FROM_STEP4}
STEP5_TASK_TYPES = {TaskType.STEP5_ONLY, TaskType.RESUME_FROM_STEP4, TaskType.RESUME_FROM_STEP5}
STEP6_TASK_TYPES = {TaskType.STEP6_ONLY, TaskType.RESUME_FROM_STEP5, TaskType.RESUME_FROM_STEP6}
STEP7_TASK_TYPES = {TaskType.STEP7_ONLY, TaskType.RESUME_FROM_STEP6}


def _important_artifacts_for_step(step: str, artifacts: dict[str, Any]) -> list[tuple[str, Any]]:
    keys_by_step = {
        "step1": ("input_path", "records_path", "step1_output_root"),
        "step2_3": ("input_path", "step1_output_root", "next_input_csv"),
        "step4": ("input_csv", "input_path", "task_text", "next_input_csv", "next_selection_report"),
        "step5": ("filtered_csv", "input_csv", "cleaned_csv_path", "next_input_csv"),
        "step6": ("filtered_csv", "input_csv", "passed_patients_json", "passed_patient_count"),
        "step7": ("filtered_csv", "selection_report", "passed_patients_json", "task_text", "step7_dataset_count"),
    }
    keys = keys_by_step.get(step, ())
    selected: list[tuple[str, Any]] = []
    for key in keys:
        value = artifacts.get(key)
        if value not in (None, "", [], {}):
            selected.append((key, value))
    return selected[:6]


def _print_step_start(step: str, artifacts: dict[str, Any]) -> None:
    label = STEP_LABELS.get(step, step)
    print("\n" + "=" * 72)
    print(f"[Orchestrator] START {label}")
    for key, value in _important_artifacts_for_step(step, artifacts):
        print(f"[Orchestrator]   {key}: {value}")
    print("=" * 72)


def _print_handoff_decision(
    step: str,
    *,
    required_steps: list[str],
    completed_steps: list[str],
    expected_next_step: str,
    artifacts: dict[str, Any],
) -> None:
    label = STEP_LABELS.get(step, step)
    print("\n" + "=" * 72)
    print(f"[Orchestrator] DECISION handoff -> {label}")
    print(f"[Orchestrator]   required_steps: {required_steps}")
    print(f"[Orchestrator]   completed_steps: {completed_steps}")
    print(f"[Orchestrator]   expected_next_step: {expected_next_step or '(none)'}")
    print(f"[Orchestrator]   reason: 当前编排边界内，下一个未完成步骤是 {expected_next_step or '(none)'}。")
    for key, value in _important_artifacts_for_step(step, artifacts):
        print(f"[Orchestrator]   input.{key}: {value}")
    print("=" * 72)


def _print_step_end(step: str, result: WorkerResult) -> None:
    label = STEP_LABELS.get(step, step)
    print("\n" + "-" * 72)
    print(f"[Orchestrator] END {label} | status={result.status.value}")
    if result.summary:
        print(f"[Orchestrator]   summary: {result.summary}")
    for key, value in _important_artifacts_for_step(step, result.artifacts):
        print(f"[Orchestrator]   {key}: {value}")
    if result.issues:
        print(f"[Orchestrator]   issues: {len(result.issues)}")
    print("-" * 72)


def _print_planner_overview(task_spec: TaskSpec) -> None:
    required_steps = _required_steps(task_spec)
    print("\n" + "=" * 72)
    print("[Orchestrator] PLANNER OVERVIEW")
    print(f"[Orchestrator]   task_type: {task_spec.task_type.value}")
    print(f"[Orchestrator]   execution_scope: {task_spec.execution_scope or '(unset)'}")
    print(f"[Orchestrator]   start_step: {task_spec.start_step or '(unset)'}")
    print(f"[Orchestrator]   allowed_steps: {_default_allowed_steps(task_spec)}")
    print(f"[Orchestrator]   required_steps: {required_steps}")
    if task_spec.intent_summary:
        print(f"[Orchestrator]   route_reason: {task_spec.intent_summary}")
    task_text = task_spec.entry_artifacts.get("task_text") or task_spec.entry_artifacts.get("ml_task_goal")
    if task_text:
        print(f"[Orchestrator]   task_text: {task_text}")
    print("[Orchestrator]   decision_policy: 首次 handoff 前必须显式决策 Memory；工具层强制 required_steps 顺序。")
    print("=" * 72)


ORCHESTRATOR_SYSTEM_PROMPT = """你是 OrchestratorAgent，负责在已经路由好的 TaskSpec 边界内编排 Step handoff。

你会收到 TaskSpec、artifact_inventory、allowed_steps、required_steps 和可选 memory context。
你只能调用本轮注册给你的 handoff_step*_tool，禁止手写或改写路径参数。

## Memory 使用边界
- 你可以按需调用 memory_search_rules_tool 查询规则记忆。
- 你可以按需调用 memory_search_step_tool 查询某个 Step 的历史经验。
- 首次 handoff 前必须先显式做一次 Memory 决策：需要上下文就调用 memory_search_*；确认不需要就调用 memory_skip_tool，并说明原因。
- Memory 只提供参考上下文，不能修改 TaskSpec、allowed_steps、required_steps 或任何路径。
- 如果某 Step 输入不确定、之前失败、或你需要复用历史经验，优先查询对应 memory。
- 你可以在最终前调用 memory_record_trace_tool 写入结构化执行摘要；如果漏掉，外层 supervisor 会兜底记录。

## 执行边界
- execution_scope=only：必须且只能完成 allowed_steps 中唯一 Step。
- execution_scope=from_step_to_end：必须按 allowed_steps 顺序完成从起点到 Step7 的所有 Step。
- execution_scope=full_pipeline：必须按 step1 → step2_3 → step4 → step5 → step6 → step7 完成全流程。
- 不能调用 allowed_steps 之外的 Step；工具层会二次拒绝越界调用。

## 产物传递约束
- 每个 Step 工具会从锁定 TaskSpec 和上一步产物中读取路径；你不要猜路径。
- Step4 必须使用 TaskSpec 中的任务裁剪目标 task_text。
- Step7 必须同时具备 Step5 filtered.csv、Step4 selection_report.json、Step6 passed_patients_json 和本轮 task_text。
- 任一必要产物缺失或任一 Step 失败时，停止后续调用并输出失败 JSON。

## 最终输出
最终回答只输出 JSON，字段包括 execution_plan、called_steps、missing_steps、status、summary。
"""


def _orchestrator_system_prompt(enable_memory_agent: bool) -> str:
    if enable_memory_agent:
        return ORCHESTRATOR_SYSTEM_PROMPT
    return ORCHESTRATOR_SYSTEM_PROMPT.replace(
        """## Memory 使用边界
- 你可以按需调用 memory_search_rules_tool 查询规则记忆。
- 你可以按需调用 memory_search_step_tool 查询某个 Step 的历史经验。
- 首次 handoff 前必须先显式做一次 Memory 决策：需要上下文就调用 memory_search_*；确认不需要就调用 memory_skip_tool，并说明原因。
- Memory 只提供参考上下文，不能修改 TaskSpec、allowed_steps、required_steps 或任何路径。
- 如果某 Step 输入不确定、之前失败、或你需要复用历史经验，优先查询对应 memory。
- 你可以在最终前调用 memory_record_trace_tool 写入结构化执行摘要；如果漏掉，外层 supervisor 会兜底记录。
""",
        """## Memory 使用边界
- 本轮未注册 Memory 工具，直接按 required_steps 顺序调用 handoff_step*_tool。
- 禁止编造 memory 检索结果，也不要尝试调用未注册的 memory 工具。
""",
    )


def _step1_call_payload(task_spec: TaskSpec, memory_context: str = "") -> dict[str, Any]:
    artifacts = task_spec.entry_artifacts
    default_records = PROJECT_ROOT / "reorganized_output" / "_meta" / "records.json"
    default_output = PROJECT_ROOT / "program" / "output" / "step1_results"
    default_script = PROJECT_ROOT / "step-1" / "generated_reorganizer.py"
    if task_spec.task_type == TaskType.RESUME_FROM_RECORDS:
        return {
            "task_type": "resume_from_records",
            "records_path": str(artifacts.get("records_path") or ""),
            "output_root": str(artifacts.get("output_root") or default_output),
            "generated_script_path": str(artifacts.get("generated_script_path") or default_script),
            "input_path": str(artifacts.get("input_path") or "") or None,
            "memory_context": memory_context,
        }
    return {
        "task_type": "step1_only",
        "input_path": str(artifacts.get("input_path") or ""),
        "records_path": str(artifacts.get("records_path") or default_records),
        "output_root": str(artifacts.get("output_root") or default_output),
        "generated_script_path": str(artifacts.get("generated_script_path") or default_script),
        "memory_context": memory_context,
    }


def _step1_artifacts_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_path": payload.get("input_path") or "",
        "records_path": payload.get("records_path") or "",
        "generated_script_path": payload.get("generated_script_path") or "",
        "step1_output_root": payload.get("output_root") or "",
    }


def _validate_step1_task_spec(task_spec: TaskSpec) -> WorkerResult | None:
    if task_spec.task_type not in STEP1_TASK_TYPES:
        return WorkerResult.failure(
            summary="OrchestratorAgent 收到不支持的 Step1 task_type。",
            issues=[f"Unsupported task_type: {task_spec.task_type.value}"],
            status=WorkerStatus.NEEDS_ESCALATION,
            artifacts=dict(task_spec.entry_artifacts),
        )

    payload = _step1_call_payload(task_spec)
    issues: list[str] = []
    for key in ("task_type", "records_path", "output_root", "generated_script_path"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            issues.append(f"Step1 handoff 参数缺失或类型错误: {key}")

    if task_spec.task_type == TaskType.STEP1_ONLY:
        input_path = str(payload.get("input_path") or "")
        if not input_path:
            issues.append("step1_only 必须提供原始数据目录 input_path。")
        elif not Path(input_path).exists():
            issues.append(f"input_path 不存在: {input_path}")
        elif not Path(input_path).is_dir():
            issues.append(f"input_path 不是目录: {input_path}")

    if task_spec.task_type == TaskType.RESUME_FROM_RECORDS:
        records_path = str(payload.get("records_path") or "")
        if not records_path:
            issues.append("resume_from_records 必须提供 records_path。")
        elif not Path(records_path).exists():
            issues.append(f"records_path 不存在: {records_path}")
        elif not Path(records_path).is_file():
            issues.append(f"records_path 不是文件: {records_path}")

    if issues:
        return WorkerResult.failure(
            summary="Step1 handoff 参数校验失败。",
            issues=issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step1_artifacts_from_payload(payload),
        )
    return None


def _step23_call_payload(task_spec: TaskSpec, memory_context: str = "") -> dict[str, Any]:
    artifacts = task_spec.entry_artifacts
    default_output = PROJECT_ROOT / "program" / "output" / "step2_3_results"
    return {
        "task_type": "step2_3_only",
        "input_path": str(artifacts.get("input_path") or artifacts.get("step1_output_root") or ""),
        "output_root": str(artifacts.get("step23_output_root") or default_output),
        "mode": str(artifacts.get("step23_mode") or artifacts.get("mode") or "auto"),
        "cache_path": artifacts.get("cache_path"),
        "ocr_workers": int(artifacts.get("ocr_workers") or 8),
        "concurrency": int(artifacts.get("concurrency") or 8),
        "use_llm": bool(artifacts.get("use_llm", True)),
        "user_hint": str(artifacts.get("user_hint") or artifacts.get("raw_input") or ""),
        "memory_context": memory_context,
    }


def _step23_artifacts_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_path": payload.get("input_path") or "",
        "output_root": payload.get("output_root") or "",
        "mode": payload.get("mode") or "auto",
    }


def _validate_step23_task_spec(task_spec: TaskSpec) -> WorkerResult | None:
    if task_spec.task_type not in STEP23_TASK_TYPES:
        return WorkerResult.failure(
            summary="OrchestratorAgent 收到不支持的 Step2-3 task_type。",
            issues=[f"Unsupported task_type: {task_spec.task_type.value}"],
            status=WorkerStatus.NEEDS_ESCALATION,
            artifacts=dict(task_spec.entry_artifacts),
        )

    payload = _step23_call_payload(task_spec)
    issues: list[str] = []
    for key in ("task_type", "input_path", "output_root", "mode"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            issues.append(f"Step2-3 handoff 参数缺失或类型错误: {key}")

    mode = str(payload.get("mode") or "")
    input_path = str(payload.get("input_path") or "")
    if mode not in {"auto", "directory", "ocr_fill"}:
        issues.append(f"Step2-3 mode 不支持: {mode}")
    if input_path:
        path = Path(input_path)
        if not path.exists():
            issues.append(f"input_path 不存在: {input_path}")
        elif not path.is_dir():
            issues.append(f"Step2-3 只接受目录 input_path: {input_path}")

    if issues:
        return WorkerResult.failure(
            summary="Step2-3 handoff 参数校验失败。",
            issues=issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step23_artifacts_from_payload(payload),
        )
    return None


def _step4_call_payload(task_spec: TaskSpec, memory_context: str = "") -> dict[str, Any]:
    artifacts = task_spec.entry_artifacts
    default_output = PROJECT_ROOT / "program" / "output" / "step4_results"
    return {
        "task_type": "resume_from_step2_3",
        "input_path": str(artifacts.get("input_csv") or artifacts.get("input_path") or ""),
        "output_root": str(artifacts.get("step4_output_root") or default_output),
        "task_text": str(artifacts.get("task_text") or "") or None,
        "memory_context": memory_context,
    }


def _step4_artifacts_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_path": payload.get("input_path") or "",
        "output_root": payload.get("output_root") or "",
        "task_text": payload.get("task_text") or "",
    }


def _validate_step4_task_spec(task_spec: TaskSpec) -> WorkerResult | None:
    if task_spec.task_type not in STEP4_TASK_TYPES:
        return WorkerResult.failure(
            summary="OrchestratorAgent 收到不支持的 Step4 task_type。",
            issues=[f"Unsupported task_type: {task_spec.task_type.value}"],
            status=WorkerStatus.NEEDS_ESCALATION,
            artifacts=dict(task_spec.entry_artifacts),
        )

    payload = _step4_call_payload(task_spec)
    issues: list[str] = []
    for key in ("task_type", "input_path", "output_root"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            issues.append(f"Step4 handoff 参数缺失或类型错误: {key}")
    if not str(payload.get("task_text") or "").strip():
        issues.append("Step4 必须提供本轮交互指定的任务裁剪目标 task_text。")

    input_path = str(payload.get("input_path") or "")
    if input_path:
        path = Path(input_path)
        if not path.exists():
            issues.append(f"input_path 不存在: {input_path}")
        elif path.is_file() and path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            issues.append(f"input_path 不是 Step4 支持的表格文件: {input_path}")

    if issues:
        return WorkerResult.failure(
            summary="Step4 handoff 参数校验失败。",
            issues=issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step4_artifacts_from_payload(payload),
        )
    return None


def _step5_call_payload(task_spec: TaskSpec, memory_context: str = "") -> dict[str, Any]:
    artifacts = task_spec.entry_artifacts
    default_output = PROJECT_ROOT / "program" / "output" / "step5_results"
    return {
        "task_type": "resume_from_step4",
        "input_path": str(artifacts.get("filtered_csv") or artifacts.get("input_csv") or artifacts.get("input_path") or ""),
        "output_root": str(artifacts.get("step5_output_root") or default_output),
        "workers": int(artifacts.get("workers") or 4),
        "llm_workers": int(artifacts.get("llm_workers") or 2),
        "enable_llm": bool(artifacts.get("enable_llm", True)),
        "memory_context": memory_context,
    }


def _step5_artifacts_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_path": payload.get("input_path") or "",
        "output_root": payload.get("output_root") or "",
    }


def _validate_step5_task_spec(task_spec: TaskSpec) -> WorkerResult | None:
    if task_spec.task_type not in STEP5_TASK_TYPES:
        return WorkerResult.failure(
            summary="OrchestratorAgent 收到不支持的 Step5 task_type。",
            issues=[f"Unsupported task_type: {task_spec.task_type.value}"],
            status=WorkerStatus.NEEDS_ESCALATION,
            artifacts=dict(task_spec.entry_artifacts),
        )

    payload = _step5_call_payload(task_spec)
    issues: list[str] = []
    for key in ("task_type", "input_path", "output_root"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            issues.append(f"Step5 handoff 参数缺失或类型错误: {key}")

    input_path = str(payload.get("input_path") or "")
    if input_path:
        path = Path(input_path)
        if not path.exists():
            issues.append(f"input_path 不存在: {input_path}")
        elif path.is_file() and path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            issues.append(f"input_path 不是 Step5 支持的表格文件: {input_path}")

    if issues:
        return WorkerResult.failure(
            summary="Step5 handoff 参数校验失败。",
            issues=issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=_step5_artifacts_from_payload(payload),
        )
    return None


def _step6_call_payload(task_spec: TaskSpec, memory_context: str = "") -> dict[str, Any]:
    artifacts = task_spec.entry_artifacts
    default_output = PROJECT_ROOT / "program" / "output" / "step6_results"
    input_path = str(
        artifacts.get("filtered_csv")
        or artifacts.get("next_input_csv")
        or artifacts.get("input_csv")
        or artifacts.get("input_path")
        or ""
    )
    return {
        "task_type": "step6_only" if task_spec.task_type == TaskType.STEP6_ONLY else "resume_from_step5",
        "input_path": input_path,
        "raw_csv": str(artifacts.get("raw_csv") or input_path),
        "output_root": str(artifacts.get("step6_output_root") or artifacts.get("output_step6") or default_output),
        "memory_context": memory_context,
    }


def _step7_call_payload(task_spec: TaskSpec, memory_context: str = "") -> dict[str, Any]:
    artifacts = task_spec.entry_artifacts
    default_output = PROJECT_ROOT / "program" / "output" / "step7_results"
    return {
        "task_type": "step7_only" if task_spec.task_type == TaskType.STEP7_ONLY else "resume_from_step6",
        "input_path": str(
            artifacts.get("filtered_csv")
            or artifacts.get("next_input_csv")
            or artifacts.get("input_csv")
            or artifacts.get("input_path")
            or ""
        ),
        "selection_report": str(
            artifacts.get("selection_report")
            or artifacts.get("next_selection_report")
            or ""
        ),
        "passed_patients_json": str(artifacts.get("passed_patients_json") or ""),
        "output_root": str(artifacts.get("step7_output_root") or artifacts.get("output_step7") or default_output),
        "task_text": str(artifacts.get("task_text") or artifacts.get("ml_task_goal") or ""),
        "memory_context": memory_context,
    }


def _make_locked_step1_handoff_tool(task_spec: TaskSpec, memory_context: str = ""):
    async def handoff_step1_from_task_spec_tool(task_type: str = "") -> ToolResponse:
        """Handoff the locked TaskSpec to Step1ReActAgent. Do not pass paths manually."""
        if task_type and task_type != task_spec.task_type.value:
            return json_tool_response(
                WorkerResult.failure(
                    summary="OrchestratorAgent 请求的 task_type 与锁定 TaskSpec 不一致。",
                    issues=[f"requested={task_type}, locked={task_spec.task_type.value}"],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=dict(task_spec.entry_artifacts),
                ).to_dict()
            )
        validation_error = _validate_step1_task_spec(task_spec)
        if validation_error is not None:
            return json_tool_response(validation_error.to_dict())
        return await handoff_to_step1_agent(**_step1_call_payload(task_spec, memory_context=memory_context))

    handoff_step1_from_task_spec_tool.__name__ = "handoff_step1_from_task_spec_tool"
    return handoff_step1_from_task_spec_tool


def _make_locked_step23_handoff_tool(task_spec: TaskSpec, memory_context: str = ""):
    async def handoff_step2_3_from_task_spec_tool(task_type: str = "") -> ToolResponse:
        """Handoff the locked TaskSpec to Step2_3ReActAgent. Do not pass paths manually."""
        if task_type and task_type != task_spec.task_type.value:
            return json_tool_response(
                WorkerResult.failure(
                    summary="OrchestratorAgent 请求的 task_type 与锁定 TaskSpec 不一致。",
                    issues=[f"requested={task_type}, locked={task_spec.task_type.value}"],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=dict(task_spec.entry_artifacts),
                ).to_dict()
            )
        validation_error = _validate_step23_task_spec(task_spec)
        if validation_error is not None:
            return json_tool_response(validation_error.to_dict())
        return await handoff_to_step23_agent(**_step23_call_payload(task_spec, memory_context=memory_context))

    handoff_step2_3_from_task_spec_tool.__name__ = "handoff_step2_3_from_task_spec_tool"
    return handoff_step2_3_from_task_spec_tool


def _make_locked_step4_handoff_tool(task_spec: TaskSpec, memory_context: str = ""):
    async def handoff_step4_from_task_spec_tool(task_type: str = "") -> ToolResponse:
        """Handoff the locked TaskSpec to Step4ReActAgent. Do not pass paths manually."""
        if task_type and task_type != task_spec.task_type.value:
            return json_tool_response(
                WorkerResult.failure(
                    summary="OrchestratorAgent 请求的 task_type 与锁定 TaskSpec 不一致。",
                    issues=[f"requested={task_type}, locked={task_spec.task_type.value}"],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=dict(task_spec.entry_artifacts),
                ).to_dict()
            )
        validation_error = _validate_step4_task_spec(task_spec)
        if validation_error is not None:
            return json_tool_response(validation_error.to_dict())
        return await handoff_to_step4_agent(**_step4_call_payload(task_spec, memory_context=memory_context))

    handoff_step4_from_task_spec_tool.__name__ = "handoff_step4_from_task_spec_tool"
    return handoff_step4_from_task_spec_tool


def _make_locked_step5_handoff_tool(task_spec: TaskSpec, memory_context: str = ""):
    async def handoff_step5_from_task_spec_tool(task_type: str = "") -> ToolResponse:
        """Handoff the locked TaskSpec to Step5ReActAgent. Do not pass paths manually."""
        if task_type and task_type != task_spec.task_type.value:
            return json_tool_response(
                WorkerResult.failure(
                    summary="OrchestratorAgent 请求的 task_type 与锁定 TaskSpec 不一致。",
                    issues=[f"requested={task_type}, locked={task_spec.task_type.value}"],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=dict(task_spec.entry_artifacts),
                ).to_dict()
            )
        validation_error = _validate_step5_task_spec(task_spec)
        if validation_error is not None:
            return json_tool_response(validation_error.to_dict())
        return await handoff_to_step5_agent(**_step5_call_payload(task_spec, memory_context=memory_context))

    handoff_step5_from_task_spec_tool.__name__ = "handoff_step5_from_task_spec_tool"
    return handoff_step5_from_task_spec_tool


def _default_allowed_steps(task_spec: TaskSpec) -> list[str]:
    if task_spec.allowed_steps:
        return list(task_spec.allowed_steps)
    if task_spec.task_type == TaskType.FULL_PIPELINE:
        return list(STEP_ORDER)
    if task_spec.task_type == TaskType.STEP1_ONLY or task_spec.task_type == TaskType.RESUME_FROM_RECORDS:
        return ["step1"]
    if task_spec.task_type == TaskType.STEP2_3_ONLY:
        return ["step2_3"]
    if task_spec.task_type == TaskType.STEP4_ONLY:
        return ["step4"]
    if task_spec.task_type == TaskType.STEP5_ONLY:
        return ["step5"]
    if task_spec.task_type == TaskType.STEP6_ONLY:
        return ["step6"]
    if task_spec.task_type == TaskType.STEP7_ONLY:
        return ["step7"]
    if task_spec.task_type == TaskType.RESUME_FROM_STEP2_3:
        return ["step4", "step5", "step6", "step7"]
    if task_spec.task_type == TaskType.RESUME_FROM_STEP4:
        return ["step5", "step6", "step7"]
    if task_spec.task_type == TaskType.RESUME_FROM_STEP5:
        return ["step6", "step7"]
    if task_spec.task_type == TaskType.RESUME_FROM_STEP6:
        return ["step7"]
    return []


def _required_steps(task_spec: TaskSpec) -> list[str]:
    return _default_allowed_steps(task_spec)


class HandoffPlannerRuntime:
    """Mutable tool runtime used by OrchestratorAgent to compose handoff calls."""

    def __init__(self, task_spec: TaskSpec, context: str = "", enable_memory_agent: bool = True) -> None:
        self.task_spec = task_spec
        self.context = context
        self.enable_memory_agent = enable_memory_agent
        self.allowed_steps = _default_allowed_steps(task_spec)
        self.artifacts: dict[str, Any] = dict(task_spec.entry_artifacts)
        self.results: list[tuple[str, WorkerResult]] = []
        self.memory_used_ids: list[str] = []
        self.memory_events: list[dict[str, Any]] = []
        self.memory_decision_made = False
        self.memory_decision_action = ""
        self.memory_recorded = False

    def register_tools(self, toolkit: Toolkit) -> None:
        if self.enable_memory_agent:
            toolkit.register_tool_function(self.memory_search_rules_tool)
            toolkit.register_tool_function(self.memory_search_step_tool)
            toolkit.register_tool_function(self.memory_skip_tool)
            toolkit.register_tool_function(self.memory_record_trace_tool)
        tool_map = {
            "step1": self.handoff_step1_tool,
            "step2_3": self.handoff_step2_3_tool,
            "step4": self.handoff_step4_tool,
            "step5": self.handoff_step5_tool,
            "step6": self.handoff_step6_tool,
            "step7": self.handoff_step7_tool,
        }
        for step in self.allowed_steps:
            tool = tool_map.get(step)
            if tool is not None:
                toolkit.register_tool_function(tool)

    def _disallowed(self, step: str) -> ToolResponse:
        result = WorkerResult.failure(
            summary=f"本次编排不允许调用 {step}。",
            issues=[f"{step} not in allowed_steps"],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts={
                "requested_step": step,
                "allowed_steps": list(self.allowed_steps),
                **dict(self.artifacts),
            },
        )
        self.results.append((step, result))
        _print_step_end(step, result)
        return json_tool_response(result.to_dict())

    def _record(self, step: str, result: WorkerResult) -> ToolResponse:
        self.results.append((step, result))
        new_artifacts = dict(result.artifacts or {})
        new_artifacts.pop("output_root", None)
        self.artifacts.update(new_artifacts)
        if step == "step1":
            output_root = result.artifacts.get("step1_output_root") or result.artifacts.get("output_root")
            if output_root:
                self.artifacts["step1_output_root"] = output_root
                self.artifacts["input_path"] = output_root
        elif step == "step2_3":
            next_input = result.artifacts.get("next_input_csv")
            if next_input:
                self.artifacts["input_csv"] = next_input
        elif step == "step4":
            filtered = result.artifacts.get("next_input_csv") or result.artifacts.get("filtered_csv_path")
            selection = result.artifacts.get("next_selection_report") or result.artifacts.get("selection_report_path")
            if filtered:
                self.artifacts["filtered_csv"] = filtered
            if selection:
                self.artifacts["selection_report"] = selection
        elif step == "step5":
            filtered = result.artifacts.get("next_input_csv") or result.artifacts.get("cleaned_csv_path")
            if filtered:
                self.artifacts["filtered_csv"] = filtered
        elif step == "step6":
            passed_json = result.artifacts.get("passed_patients_json")
            if passed_json:
                self.artifacts["passed_patients_json"] = passed_json
            input_csv = result.artifacts.get("input_csv")
            if input_csv:
                self.artifacts["filtered_csv"] = input_csv
        _print_step_end(step, result)
        return json_tool_response(result.to_dict())

    def _memory_context(self) -> str:
        return self.context if self.enable_memory_agent else ""

    def _memory_query_text(self, query_text: str = "") -> str:
        return str(query_text or self.task_spec.intent_summary or self.artifacts.get("raw_input") or "").strip()

    def _append_memory_context(self, context: str) -> None:
        text = str(context or "").strip()
        if not text:
            return
        if self.context.strip():
            self.context = f"{self.context.strip()}\n\n{text}"
        else:
            self.context = text

    def _record_memory_event(
        self,
        action: str,
        *,
        status: WorkerStatus,
        summary: str,
        artifacts: dict[str, Any] | None = None,
        issues: list[str] | None = None,
    ) -> None:
        result = WorkerResult(
            status=status,
            summary=summary,
            artifacts={"memory_action": action, **dict(artifacts or {})},
            issues=list(issues or []),
        )
        self.memory_events.append({"state": PipelineState.MEMORY_REFLECTION.value, "result": result.to_dict()})

    def _mark_memory_decision(self, action: str) -> None:
        self.memory_decision_made = True
        self.memory_decision_action = action

    def _memory_disabled_response(self, action: str) -> ToolResponse:
        self._mark_memory_decision(action)
        payload = {
            "status": "DISABLED",
            "summary": "Memory tools are disabled for this run.",
            "memory_action": action,
            "used_memory_ids": [],
            "context": "",
        }
        self._record_memory_event(
            action,
            status=WorkerStatus.SUCCESS,
            summary="Memory tool disabled.",
            artifacts={"used_memory_ids": []},
        )
        return json_tool_response(payload)

    async def memory_search_rules_tool(self, query_text: str = "") -> ToolResponse:
        """Search rule memory on demand. This cannot modify TaskSpec or step order."""
        action = "search_rules"
        if not self.enable_memory_agent or get_pipeline_context is None:
            return self._memory_disabled_response(action)
        self._mark_memory_decision(action)
        query = self._memory_query_text(query_text)
        print("\n" + "=" * 72)
        print("[Orchestrator] DECISION memory_search -> rules")
        print(f"[Orchestrator]   query: {query or '(empty)'}")
        print("=" * 72)
        try:
            context, used_ids = await get_pipeline_context(query_text=query, task_spec=self._runtime_task_spec())
            self.memory_used_ids.extend(item for item in used_ids if item not in self.memory_used_ids)
            self._append_memory_context(context)
            payload = {
                "status": "SUCCESS",
                "summary": "Rule memory retrieved.",
                "memory_action": action,
                "used_memory_ids": used_ids,
                "context": context,
            }
            self._record_memory_event(
                action,
                status=WorkerStatus.SUCCESS,
                summary=f"Rule memory retrieved: memory_ids={len(used_ids)}",
                artifacts={"used_memory_ids": used_ids},
            )
            return json_tool_response(payload)
        except Exception as exc:
            payload = {
                "status": "FAILED",
                "summary": "Rule memory retrieval failed; continue without memory context.",
                "memory_action": action,
                "used_memory_ids": [],
                "context": "",
                "error": str(exc),
            }
            self._record_memory_event(
                action,
                status=WorkerStatus.SUCCESS,
                summary="Rule memory retrieval failed but did not block orchestration.",
                artifacts={"used_memory_ids": []},
                issues=[str(exc)],
            )
            return json_tool_response(payload)

    async def memory_search_step_tool(self, step_name: str, query_text: str = "") -> ToolResponse:
        """Search Step experience memory on demand. Only allowed for this run's allowed_steps."""
        action = "search_step"
        step = str(step_name or "").strip()
        if not self.enable_memory_agent or get_step_memory_context is None:
            return self._memory_disabled_response(action)
        if step not in self.allowed_steps:
            payload = {
                "status": "NEEDS_REPAIR",
                "summary": f"Memory search for {step} is outside allowed_steps.",
                "memory_action": action,
                "requested_step": step,
                "allowed_steps": list(self.allowed_steps),
                "used_memory_ids": [],
                "context": "",
            }
            self._record_memory_event(
                action,
                status=WorkerStatus.SUCCESS,
                summary=f"Rejected memory search outside allowed_steps: {step}",
                artifacts={"requested_step": step, "allowed_steps": list(self.allowed_steps), "used_memory_ids": []},
            )
            return json_tool_response(payload)
        self._mark_memory_decision(action)
        query = self._memory_query_text(query_text)
        print("\n" + "=" * 72)
        print(f"[Orchestrator] DECISION memory_search -> {step}")
        print(f"[Orchestrator]   query: {query or '(empty)'}")
        print("=" * 72)
        try:
            context, used_ids = await get_step_memory_context(
                step_name=step,
                query_text=query,
                task_spec=self._runtime_task_spec(),
            )
            self.memory_used_ids.extend(item for item in used_ids if item not in self.memory_used_ids)
            self._append_memory_context(context)
            payload = {
                "status": "SUCCESS",
                "summary": f"{step} memory retrieved.",
                "memory_action": action,
                "step_name": step,
                "used_memory_ids": used_ids,
                "context": context,
            }
            self._record_memory_event(
                action,
                status=WorkerStatus.SUCCESS,
                summary=f"{step} memory retrieved: memory_ids={len(used_ids)}",
                artifacts={"step_name": step, "used_memory_ids": used_ids},
            )
            return json_tool_response(payload)
        except Exception as exc:
            payload = {
                "status": "FAILED",
                "summary": f"{step} memory retrieval failed; continue without memory context.",
                "memory_action": action,
                "step_name": step,
                "used_memory_ids": [],
                "context": "",
                "error": str(exc),
            }
            self._record_memory_event(
                action,
                status=WorkerStatus.SUCCESS,
                summary=f"{step} memory retrieval failed but did not block orchestration.",
                artifacts={"step_name": step, "used_memory_ids": []},
                issues=[str(exc)],
            )
            return json_tool_response(payload)

    async def memory_skip_tool(self, reason: str = "") -> ToolResponse:
        """Explicitly skip memory retrieval for this run before the first handoff."""
        action = "skip"
        self._mark_memory_decision(action)
        reason_text = str(reason or "").strip() or "Orchestrator judged no memory context was needed."
        print("\n" + "=" * 72)
        print("[Orchestrator] DECISION memory_skip")
        print(f"[Orchestrator]   reason: {reason_text}")
        print("=" * 72)
        payload = {
            "status": "SUCCESS",
            "summary": "Memory retrieval explicitly skipped.",
            "memory_action": action,
            "reason": reason_text,
            "used_memory_ids": [],
            "context": "",
        }
        self._record_memory_event(
            action,
            status=WorkerStatus.SUCCESS,
            summary="Memory retrieval explicitly skipped by Orchestrator.",
            artifacts={"used_memory_ids": [], "reason": reason_text},
        )
        return json_tool_response(payload)

    async def memory_record_trace_tool(self, trace_summary_json: str = "") -> ToolResponse:
        """Record a compact orchestration trace. This cannot modify business artifacts."""
        action = "record_trace"
        if not self.enable_memory_agent or report_pipeline_result is None:
            return self._memory_disabled_response(action)
        try:
            trace_payload: dict[str, Any]
            if str(trace_summary_json or "").strip():
                try:
                    parsed = json.loads(trace_summary_json)
                    trace_payload = parsed if isinstance(parsed, dict) else {"raw_trace_text": trace_summary_json}
                except json.JSONDecodeError:
                    trace_payload = {"raw_trace_text": trace_summary_json}
            else:
                trace_payload = {
                    "orchestrator_summary": "Orchestrator requested memory trace recording.",
                    "task_spec": self.task_spec.to_dict(),
                    "steps": [
                        {"step": step, "result": result.to_dict()}
                        for step, result in self.results
                    ],
                    "success": all(result.status == WorkerStatus.SUCCESS for _, result in self.results),
                }
            summary = await report_pipeline_result(trace_payload, used_memory_ids=self.memory_used_ids)
            self.memory_recorded = True
            payload = {
                "status": "SUCCESS",
                "summary": summary,
                "memory_action": action,
                "used_memory_ids": list(self.memory_used_ids),
                "memory_recorded": True,
            }
            self._record_memory_event(
                action,
                status=WorkerStatus.SUCCESS,
                summary="Memory trace recorded by Orchestrator.",
                artifacts={"used_memory_ids": list(self.memory_used_ids), "memory_recorded": True},
            )
            return json_tool_response(payload)
        except Exception as exc:
            payload = {
                "status": "FAILED",
                "summary": "Memory trace recording failed; supervisor fallback may still record.",
                "memory_action": action,
                "used_memory_ids": list(self.memory_used_ids),
                "memory_recorded": False,
                "error": str(exc),
            }
            self._record_memory_event(
                action,
                status=WorkerStatus.SUCCESS,
                summary="Memory trace recording failed but did not block orchestration.",
                artifacts={"used_memory_ids": list(self.memory_used_ids), "memory_recorded": False},
                issues=[str(exc)],
            )
            return json_tool_response(payload)

    def _runtime_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_type=self.task_spec.task_type,
            entry_artifacts=dict(self.artifacts),
            resume_from_step=self.task_spec.resume_from_step,
            intent_summary=self.task_spec.intent_summary,
            confidence=self.task_spec.confidence,
            execution_scope=self.task_spec.execution_scope,
            start_step=self.task_spec.start_step,
            allowed_steps=list(self.allowed_steps),
        )

    def _completed_required_steps(self) -> list[str]:
        required = _required_steps(self.task_spec)
        completed: list[str] = []
        for previous_step, previous_result in self.results:
            if previous_step in required and previous_result.status == WorkerStatus.SUCCESS and previous_step not in completed:
                completed.append(previous_step)
        return completed

    def _expected_next_step(self) -> str:
        completed = set(self._completed_required_steps())
        for required_step in _required_steps(self.task_spec):
            if required_step not in completed:
                return required_step
        return ""

    def _print_decision_for_step(self, step: str) -> None:
        _print_handoff_decision(
            step,
            required_steps=_required_steps(self.task_spec),
            completed_steps=self._completed_required_steps(),
            expected_next_step=self._expected_next_step(),
            artifacts=self.artifacts,
        )

    def _require_step(self, step: str) -> ToolResponse | None:
        if step not in self.allowed_steps:
            return self._disallowed(step)
        if self.enable_memory_agent and not self.memory_decision_made:
            result = WorkerResult.failure(
                summary=f"调用 {step} 前必须先显式决定是否使用 Memory。",
                issues=[
                    "先调用 memory_search_rules_tool、memory_search_step_tool，或 memory_skip_tool。",
                    "Memory 决策只影响上下文，不允许修改 TaskSpec、路径或 Step 顺序。",
                ],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts={
                    "requested_step": step,
                    "available_memory_tools": [
                        "memory_search_rules_tool",
                        "memory_search_step_tool",
                        "memory_skip_tool",
                    ],
                    **dict(self.artifacts),
                },
            )
            self._record_memory_event(
                "required_before_handoff",
                status=WorkerStatus.NEEDS_REPAIR,
                summary=result.summary,
                artifacts={"requested_step": step},
                issues=list(result.issues),
            )
            return json_tool_response(result.to_dict())
        for previous_step, previous_result in self.results:
            if previous_result.status != WorkerStatus.SUCCESS:
                result = WorkerResult.failure(
                    summary=f"前置步骤 {previous_step} 未成功，禁止继续调用 {step}。",
                    issues=[
                        f"previous_step={previous_step}",
                        f"previous_status={previous_result.status.value}",
                        "任一 Step 返回 NEEDS_REPAIR/FAILED 后必须停止后续 handoff。",
                    ],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts={
                        "requested_step": step,
                        "failed_step": previous_step,
                        **dict(self.artifacts),
                    },
                )
                self.results.append((step, result))
                _print_step_end(step, result)
                return json_tool_response(result.to_dict())
        expected_step = self._expected_next_step()
        if expected_step and step != expected_step:
            result = WorkerResult.failure(
                summary=f"当前应先执行 {expected_step}，禁止提前调用 {step}。",
                issues=[
                    f"requested_step={step}",
                    f"expected_next_step={expected_step}",
                    f"required_steps={_required_steps(self.task_spec)}",
                    f"completed_steps={self._completed_required_steps()}",
                    "LLM 编排必须按 required_steps 顺序 handoff。",
                ],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts={
                    "requested_step": step,
                    "expected_next_step": expected_step,
                    "required_steps": _required_steps(self.task_spec),
                    "completed_steps": self._completed_required_steps(),
                    **dict(self.artifacts),
                },
            )
            self.results.append((step, result))
            _print_step_end(step, result)
            return json_tool_response(result.to_dict())
        if not expected_step:
            result = WorkerResult.failure(
                summary=f"required_steps 已完成，禁止额外调用 {step}。",
                issues=[
                    f"requested_step={step}",
                    f"required_steps={_required_steps(self.task_spec)}",
                    f"completed_steps={self._completed_required_steps()}",
                ],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts={
                    "requested_step": step,
                    "required_steps": _required_steps(self.task_spec),
                    "completed_steps": self._completed_required_steps(),
                    **dict(self.artifacts),
                },
            )
            self.results.append((step, result))
            _print_step_end(step, result)
            return json_tool_response(result.to_dict())
        return None

    async def handoff_step1_tool(self) -> ToolResponse:
        """Handoff to Step1ReActAgent only when step1 is in allowed_steps."""
        if (blocked := self._require_step("step1")) is not None:
            return blocked
        self._print_decision_for_step("step1")
        _print_step_start("step1", self.artifacts)
        response = await handoff_to_step1_agent(**_step1_call_payload(self._runtime_task_spec(), memory_context=self._memory_context()))
        result = worker_result_from_dict(parse_tool_response(response))
        return self._record("step1", result)

    async def handoff_step2_3_tool(self) -> ToolResponse:
        """Handoff to Step2_3ReActAgent only when step2_3 is in allowed_steps."""
        if (blocked := self._require_step("step2_3")) is not None:
            return blocked
        self._print_decision_for_step("step2_3")
        _print_step_start("step2_3", self.artifacts)
        response = await handoff_to_step23_agent(**_step23_call_payload(self._runtime_task_spec(), memory_context=self._memory_context()))
        result = worker_result_from_dict(parse_tool_response(response))
        return self._record("step2_3", result)

    async def handoff_step4_tool(self) -> ToolResponse:
        """Handoff to Step4ReActAgent only when step4 is in allowed_steps."""
        if (blocked := self._require_step("step4")) is not None:
            return blocked
        self._print_decision_for_step("step4")
        _print_step_start("step4", self.artifacts)
        task_text = str(self.artifacts.get("task_text") or "")
        if not task_text.strip():
            result = WorkerResult.failure(
                summary="Step4 缺少任务裁剪目标 task_text。",
                issues=["Step4 requires task_text from interactive user input."],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=dict(self.artifacts),
            )
            return self._record("step4", result)
        response = await handoff_to_step4_agent(**_step4_call_payload(self._runtime_task_spec(), memory_context=self._memory_context()))
        result = worker_result_from_dict(parse_tool_response(response))
        return self._record("step4", result)

    async def handoff_step5_tool(self) -> ToolResponse:
        """Handoff to Step5ReActAgent only when step5 is in allowed_steps."""
        if (blocked := self._require_step("step5")) is not None:
            return blocked
        self._print_decision_for_step("step5")
        _print_step_start("step5", self.artifacts)
        response = await handoff_to_step5_agent(**_step5_call_payload(self._runtime_task_spec(), memory_context=self._memory_context()))
        result = worker_result_from_dict(parse_tool_response(response))
        return self._record("step5", result)

    async def handoff_step6_tool(self) -> ToolResponse:
        """Handoff to Step6 ConsistencyAgent only when step6 is in allowed_steps."""
        if (blocked := self._require_step("step6")) is not None:
            return blocked
        self._print_decision_for_step("step6")
        _print_step_start("step6", self.artifacts)
        response = await handoff_to_step6_agent(**_step6_call_payload(self._runtime_task_spec(), memory_context=self._memory_context()))
        result = worker_result_from_dict(parse_tool_response(response))
        return self._record("step6", result)

    async def handoff_step7_tool(self) -> ToolResponse:
        """Handoff to Step7 DataPrepAgent only when step7 is in allowed_steps."""
        if (blocked := self._require_step("step7")) is not None:
            return blocked
        self._print_decision_for_step("step7")
        _print_step_start("step7", self.artifacts)
        filtered_csv = str(self.artifacts.get("filtered_csv") or self.artifacts.get("input_csv") or self.artifacts.get("next_input_csv") or "")
        selection_report = str(self.artifacts.get("selection_report") or self.artifacts.get("next_selection_report") or "")
        passed_patients_json = str(self.artifacts.get("passed_patients_json") or "")
        task_text = str(self.artifacts.get("task_text") or self.artifacts.get("ml_task_goal") or "")
        missing_inputs = []
        if not filtered_csv.strip():
            missing_inputs.append("filtered_csv")
        if not selection_report.strip():
            missing_inputs.append("selection_report")
        if not passed_patients_json.strip():
            missing_inputs.append("passed_patients_json")
        if not task_text.strip():
            missing_inputs.append("task_text")
        if missing_inputs:
            result = WorkerResult.failure(
                summary="Step7 缺少必要上游输入。",
                issues=[f"Step7 requires filtered_csv, selection_report, passed_patients_json and task_text; missing={missing_inputs}"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=dict(self.artifacts),
            )
            return self._record("step7", result)
        response = await handoff_to_step7_agent(**_step7_call_payload(self._runtime_task_spec(), memory_context=self._memory_context()))
        result = worker_result_from_dict(parse_tool_response(response))
        return self._record("step7", result)


def create_orchestrator_agent(
    task_spec: TaskSpec | None = None,
    context: str = "",
    planner_runtime: HandoffPlannerRuntime | None = None,
) -> ReActAgent:
    toolkit = Toolkit()
    planner_runtime = planner_runtime or (HandoffPlannerRuntime(task_spec, context=context) if task_spec is not None else None)
    if planner_runtime is not None:
        planner_runtime.register_tools(toolkit)
    else:
        toolkit.register_tool_function(handoff_to_step1_agent)
        toolkit.register_tool_function(handoff_to_step23_agent)
        toolkit.register_tool_function(handoff_to_step4_agent)
        toolkit.register_tool_function(handoff_to_step5_agent)
        toolkit.register_tool_function(handoff_to_step6_agent)
        toolkit.register_tool_function(handoff_to_step7_agent)
    model, formatter = create_openai_model_and_formatter("main_orchestrator", "gpt-4.1-mini")
    agent = ReActAgent(
        name="OrchestratorAgent",
        sys_prompt=_orchestrator_system_prompt(
            planner_runtime.enable_memory_agent if planner_runtime is not None else False
        ),
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=12,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


def _pipeline_from_step4_result(task_spec: TaskSpec, result: WorkerResult) -> PipelineRunResult:
    state = PipelineState.STEP4_RUNNING
    states = [PipelineState.INIT, state]
    worker_results = [{"state": state.value, "result": result.to_dict()}]
    if result.status == WorkerStatus.SUCCESS:
        states.append(PipelineState.DONE)
        return PipelineRunResult(
            status=WorkerStatus.SUCCESS,
            task_spec=task_spec,
            states_visited=states,
            worker_results=worker_results,
            summary="Step4 任务裁剪完成。",
        )

    states.extend([PipelineState.REPAIR_PENDING, PipelineState.FAILED])
    repair_ticket = RepairTicket(
        repair_type="missing_artifact_repair" if "selection_report" in " ".join(result.issues) else "generic_repair",
        failed_state=state,
        input_artifacts=dict(result.artifacts),
        validator_errors=list(result.issues),
        resume_target_state=state,
    )
    return PipelineRunResult(
        status=WorkerStatus.NEEDS_REPAIR if result.status != WorkerStatus.NEEDS_ESCALATION else WorkerStatus.NEEDS_ESCALATION,
        task_spec=task_spec,
        states_visited=states,
        worker_results=worker_results,
        repair_ticket=repair_ticket,
        summary=result.summary or "Step4 handoff 失败。",
    )


def _pipeline_from_step5_result(task_spec: TaskSpec, result: WorkerResult) -> PipelineRunResult:
    state = PipelineState.STEP5_RUNNING
    states = [PipelineState.INIT, state]
    worker_results = [{"state": state.value, "result": result.to_dict()}]
    if result.status == WorkerStatus.SUCCESS:
        states.append(PipelineState.DONE)
        return PipelineRunResult(
            status=WorkerStatus.SUCCESS,
            task_spec=task_spec,
            states_visited=states,
            worker_results=worker_results,
            summary="Step5 数据清洗完成。",
        )

    states.extend([PipelineState.REPAIR_PENDING, PipelineState.FAILED])
    repair_ticket = RepairTicket(
        repair_type="step5_data_quality_repair",
        failed_state=state,
        input_artifacts=dict(result.artifacts),
        validator_errors=list(result.issues),
        resume_target_state=state,
    )
    return PipelineRunResult(
        status=WorkerStatus.NEEDS_REPAIR if result.status != WorkerStatus.NEEDS_ESCALATION else WorkerStatus.NEEDS_ESCALATION,
        task_spec=task_spec,
        states_visited=states,
        worker_results=worker_results,
        repair_ticket=repair_ticket,
        summary=result.summary or "Step5 handoff 失败。",
    )


def _pipeline_from_step23_result(task_spec: TaskSpec, result: WorkerResult) -> PipelineRunResult:
    state = PipelineState.STEP2_3_RUNNING
    states = [PipelineState.INIT, state]
    worker_results = [{"state": state.value, "result": result.to_dict()}]
    if result.status == WorkerStatus.SUCCESS:
        states.append(PipelineState.DONE)
        return PipelineRunResult(
            status=WorkerStatus.SUCCESS,
            task_spec=task_spec,
            states_visited=states,
            worker_results=worker_results,
            summary="Step2-3 医疗结构化与回填完成。",
        )

    states.extend([PipelineState.REPAIR_PENDING, PipelineState.FAILED])
    repair_ticket = RepairTicket(
        repair_type="step2_3_repair",
        failed_state=state,
        input_artifacts=dict(result.artifacts),
        validator_errors=list(result.issues),
        resume_target_state=state,
    )
    return PipelineRunResult(
        status=WorkerStatus.NEEDS_REPAIR if result.status != WorkerStatus.NEEDS_ESCALATION else WorkerStatus.NEEDS_ESCALATION,
        task_spec=task_spec,
        states_visited=states,
        worker_results=worker_results,
        repair_ticket=repair_ticket,
        summary=result.summary or "Step2-3 handoff 失败。",
    )


def _pipeline_from_step1_result(task_spec: TaskSpec, result: WorkerResult) -> PipelineRunResult:
    state = PipelineState.STEP1_CODEGEN if task_spec.task_type == TaskType.RESUME_FROM_RECORDS else PipelineState.STEP1_RECORDING
    states = [PipelineState.INIT, state]
    worker_results = [{"state": state.value, "result": result.to_dict()}]
    if result.status == WorkerStatus.SUCCESS:
        states.append(PipelineState.DONE)
        summary = "已从 records.json 完成 Step1 续跑。" if task_spec.task_type == TaskType.RESUME_FROM_RECORDS else "Step1-only 任务完成。"
        return PipelineRunResult(
            status=WorkerStatus.SUCCESS,
            task_spec=task_spec,
            states_visited=states,
            worker_results=worker_results,
            summary=summary,
        )

    states.extend([PipelineState.REPAIR_PENDING, PipelineState.FAILED])
    repair_ticket = RepairTicket(
        repair_type="records_repair" if "records" in " ".join(result.issues).lower() else "generated_script_repair",
        failed_state=state,
        input_artifacts=dict(result.artifacts),
        validator_errors=list(result.issues),
        resume_target_state=state,
    )
    return PipelineRunResult(
        status=WorkerStatus.NEEDS_REPAIR if result.status != WorkerStatus.NEEDS_ESCALATION else WorkerStatus.NEEDS_ESCALATION,
        task_spec=task_spec,
        states_visited=states,
        worker_results=worker_results,
        repair_ticket=repair_ticket,
        summary=result.summary or "Step1 handoff 失败。",
    )


def _tool_name_to_step(tool_name: str) -> str:
    mapping = {
        "handoff_step1_tool": "step1",
        "handoff_step1_from_task_spec_tool": "step1",
        "handoff_step2_3_tool": "step2_3",
        "handoff_step2_3_from_task_spec_tool": "step2_3",
        "handoff_step4_tool": "step4",
        "handoff_step4_from_task_spec_tool": "step4",
        "handoff_step5_tool": "step5",
        "handoff_step5_from_task_spec_tool": "step5",
        "handoff_step6_tool": "step6",
        "handoff_step7_tool": "step7",
    }
    return mapping.get(tool_name, "")


def _planner_results_from_collected_tools(tool_results: dict[str, list[dict[str, Any]]]) -> list[tuple[str, WorkerResult]]:
    ordered: list[tuple[str, WorkerResult]] = []
    for step in STEP_ORDER:
        for tool_name, payloads in tool_results.items():
            if _tool_name_to_step(tool_name) != step:
                continue
            for payload in payloads:
                ordered.append((step, worker_result_from_dict(payload)))
    return ordered


def _pipeline_from_planner_worker_results(
    task_spec: TaskSpec,
    results: list[tuple[str, WorkerResult]],
) -> PipelineRunResult:
    states = [PipelineState.INIT]
    worker_results: list[dict[str, Any]] = []
    first_failure: tuple[str, WorkerResult] | None = None
    for step, result in results:
        state = STEP_TO_STATE.get(step, PipelineState.REPAIR_PENDING)
        states.append(state)
        worker_results.append({"state": state.value, "result": result.to_dict()})
        if result.status != WorkerStatus.SUCCESS and first_failure is None:
            first_failure = (step, result)

    if first_failure is None:
        required = _required_steps(task_spec)
        successful_steps = [step for step, result in results if step in required and result.status == WorkerStatus.SUCCESS]
        seen_steps: list[str] = []
        for step in successful_steps:
            if step not in seen_steps:
                seen_steps.append(step)
        missing_steps = [step for step in required if step not in seen_steps]
        if missing_steps or seen_steps != required:
            issue_parts: list[str] = []
            if missing_steps:
                issue_parts.append(f"missing_steps={missing_steps}")
            if seen_steps != required:
                issue_parts.append(f"called_steps={seen_steps}, required_steps={required}")
            guard_result = WorkerResult.failure(
                summary="OrchestratorAgent 编排未完成 required_steps。",
                issues=issue_parts,
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts={
                    "required_steps": required,
                    "called_steps": seen_steps,
                    "missing_steps": missing_steps,
                },
            )
            states.extend([PipelineState.REPAIR_PENDING, PipelineState.FAILED])
            worker_results.append({"state": PipelineState.REPAIR_PENDING.value, "result": guard_result.to_dict()})
            return PipelineRunResult(
                status=WorkerStatus.NEEDS_REPAIR,
                task_spec=task_spec,
                states_visited=states,
                worker_results=worker_results,
                repair_ticket=RepairTicket(
                    repair_type="llm_orchestration_missing_steps",
                    failed_state=PipelineState.REPAIR_PENDING,
                    input_artifacts=guard_result.artifacts,
                    validator_errors=guard_result.issues,
                    resume_target_state=STEP_TO_STATE.get(missing_steps[0], PipelineState.REPAIR_PENDING) if missing_steps else PipelineState.REPAIR_PENDING,
                ),
                summary=guard_result.summary,
            )
        states.extend([PipelineState.MEMORY_REFLECTION, PipelineState.DONE])
        return PipelineRunResult(
            status=WorkerStatus.SUCCESS,
            task_spec=task_spec,
            states_visited=states,
            worker_results=worker_results,
            summary="OrchestratorAgent LLM 编排完成。",
        )

    failed_step, failed_result = first_failure
    failed_state = STEP_TO_STATE.get(failed_step, PipelineState.REPAIR_PENDING)
    states.extend([PipelineState.REPAIR_PENDING, PipelineState.FAILED])
    repair_ticket = RepairTicket(
        repair_type="llm_orchestration_repair",
        failed_state=failed_state,
        input_artifacts=dict(failed_result.artifacts),
        validator_errors=list(failed_result.issues),
        resume_target_state=failed_state,
    )
    return PipelineRunResult(
        status=WorkerStatus.NEEDS_ESCALATION if failed_result.status == WorkerStatus.NEEDS_ESCALATION else WorkerStatus.NEEDS_REPAIR,
        task_spec=task_spec,
        states_visited=states,
        worker_results=worker_results,
        repair_ticket=repair_ticket,
        summary=failed_result.summary or "OrchestratorAgent LLM 编排失败。",
    )


def _attach_memory_events(result: PipelineRunResult, planner_runtime: HandoffPlannerRuntime) -> PipelineRunResult:
    if not planner_runtime.memory_events:
        return result
    if PipelineState.MEMORY_REFLECTION not in result.states_visited:
        if result.states_visited and result.states_visited[-1] == PipelineState.DONE:
            result.states_visited.insert(len(result.states_visited) - 1, PipelineState.MEMORY_REFLECTION)
        else:
            result.states_visited.append(PipelineState.MEMORY_REFLECTION)
    result.worker_results.extend(planner_runtime.memory_events)
    return result


def _validate_planner_task_spec(task_spec: TaskSpec) -> WorkerResult | None:
    allowed = _default_allowed_steps(task_spec)
    artifacts = task_spec.entry_artifacts
    issues: list[str] = []
    if not allowed:
        issues.append("allowed_steps 为空，无法编排 Step 工具。")
    if task_spec.task_type == TaskType.RESUME_FROM_RECORDS:
        if not str(artifacts.get("records_path") or "").strip():
            issues.append("resume_from_records 必须提供 records_path。")
    elif "step1" in allowed and (task_spec.start_step or "step1") == "step1":
        if not str(artifacts.get("input_path") or "").strip():
            issues.append("step1 必须提供原始数据目录 input_path。")
    if "step4" in allowed and (task_spec.execution_scope in {"only", "full_pipeline"} or task_spec.start_step == "step4"):
        if not str(artifacts.get("task_text") or "").strip():
            issues.append("Step4 必须提供本轮交互指定的任务裁剪目标 task_text。")
    if "step7" in allowed and not str(artifacts.get("task_text") or artifacts.get("ml_task_goal") or "").strip():
        issues.append("Step7 必须提供本轮交互指定的 ML 任务目标 task_text。")
    if issues:
        return WorkerResult.failure(
            summary="OrchestratorAgent 编排前参数校验失败。",
            issues=issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=dict(artifacts),
        )
    return None


async def run_with_orchestrator_agent(
    task_spec: TaskSpec,
    context: str = "",
    enable_memory_agent: bool = True,
) -> PipelineRunResult:
    _ = enable_memory_agent
    validation_error = _validate_planner_task_spec(task_spec)
    if validation_error is not None:
        return _pipeline_from_planner_worker_results(task_spec, [("orchestrator", validation_error)])
    if not has_model_credentials("main_orchestrator"):
        result = WorkerResult.failure(
            summary="OrchestratorAgent 未配置模型密钥，无法执行 LLM 编排。",
            issues=["请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.main_orchestrator.api_key。"],
            status=WorkerStatus.NEEDS_ESCALATION,
            artifacts=dict(task_spec.entry_artifacts),
        )
        return _pipeline_from_planner_worker_results(task_spec, [("orchestrator", result)])

    _print_planner_overview(task_spec)
    planner_runtime = HandoffPlannerRuntime(task_spec, context=context, enable_memory_agent=enable_memory_agent)
    try:
        agent = create_orchestrator_agent(task_spec, context=context, planner_runtime=planner_runtime)
    except TypeError:
        agent = create_orchestrator_agent(task_spec, context=context)
    payload = {
        "task_spec": task_spec.to_dict(),
        "artifact_inventory": dict(task_spec.entry_artifacts),
        "context": context,
        "allowed_steps": _default_allowed_steps(task_spec),
        "required_steps": _required_steps(task_spec),
        "memory_tools": MEMORY_TOOL_NAMES if enable_memory_agent else [],
        "required_action": "按 required_steps 顺序调用对应 handoff_step*_tool；可按需调用 memory 工具获取参考；不要少跑、跳步、越界或手写路径参数。",
        "required_output": "JSON: {execution_plan, called_steps, missing_steps, status, summary}",
    }
    try:
        res = await agent(make_user_msg("TaskRouterAgent", json.dumps(payload, ensure_ascii=False, indent=2)))
        if planner_runtime.results:
            return _attach_memory_events(
                _pipeline_from_planner_worker_results(task_spec, list(planner_runtime.results)),
                planner_runtime,
            )
        tool_results = await collect_tool_results(agent)
        planner_results = _planner_results_from_collected_tools(tool_results)
        if planner_results:
            return _attach_memory_events(
                _pipeline_from_planner_worker_results(task_spec, planner_results),
                planner_runtime,
            )
        parsed_final = parse_agent_text_json(res.get_text_content() or "")
        if parsed_final and parsed_final.get("worker_results"):
            state_to_step = {state: step for step, state in STEP_TO_STATE.items()}
            parsed_results: list[tuple[str, WorkerResult]] = []
            for item in parsed_final.get("worker_results", []):
                if not isinstance(item, dict):
                    continue
                state = PipelineState._value2member_map_.get(str(item.get("state") or ""))
                step = state_to_step.get(state)
                result_payload = item.get("result")
                if step and isinstance(result_payload, dict):
                    parsed_results.append((step, worker_result_from_dict(result_payload)))
            if parsed_results:
                return _attach_memory_events(
                    _pipeline_from_planner_worker_results(task_spec, parsed_results),
                    planner_runtime,
                )
        result = WorkerResult.failure(
            summary="OrchestratorAgent 未调用任何 Step 工具。",
            issues=["未检测到 handoff_step*_tool 的工具返回。"],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=dict(task_spec.entry_artifacts),
        )
        return _attach_memory_events(
            _pipeline_from_planner_worker_results(task_spec, [("orchestrator", result)]),
            planner_runtime,
        )
    except Exception as exc:
        if planner_runtime.results:
            return _attach_memory_events(
                _pipeline_from_planner_worker_results(task_spec, list(planner_runtime.results)),
                planner_runtime,
            )
        result = WorkerResult.failure(
            summary="OrchestratorAgent LLM 编排执行失败。",
            issues=[str(exc)],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=dict(task_spec.entry_artifacts),
        )
        return _attach_memory_events(
            _pipeline_from_planner_worker_results(task_spec, [("orchestrator", result)]),
            planner_runtime,
        )
