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
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR, EXECUTION_DIR):
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
)
from contracts import PipelineRunResult, PipelineState, RepairTicket, TaskSpec, TaskType, WorkerResult, WorkerStatus
from handoffs import handoff_to_step1_agent, handoff_to_step4_agent, worker_result_from_dict
from state_machine import StateMachineOrchestrator

STEP1_TASK_TYPES = {TaskType.STEP1_ONLY, TaskType.RESUME_FROM_RECORDS}
STEP4_TASK_TYPES = {TaskType.RESUME_FROM_STEP2_3}


ORCHESTRATOR_SYSTEM_PROMPT = """你是 OrchestratorAgent，只负责根据已经路由好的 TaskSpec 把任务 handoff 给对应子 Agent。

当前已接入的子 Agent：
- step1_only 必须调用 handoff_step1_from_task_spec_tool
- resume_from_records 必须调用 handoff_step1_from_task_spec_tool
- resume_from_step2_3 必须调用 handoff_step4_from_task_spec_tool

你不要直接调用 Step1/Step4 细粒度工具。你也不要重新填写 input_path、records_path、output_root、generated_script_path。
这些参数已经由 TaskSpec 锁定，handoff 工具会从锁定的 TaskSpec 中读取参数。
最终回答只输出 handoff 工具返回的 JSON，不要添加解释。
"""


def _step1_call_payload(task_spec: TaskSpec) -> dict[str, Any]:
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
        }
    return {
        "task_type": "step1_only",
        "input_path": str(artifacts.get("input_path") or ""),
        "records_path": str(artifacts.get("records_path") or default_records),
        "output_root": str(artifacts.get("output_root") or default_output),
        "generated_script_path": str(artifacts.get("generated_script_path") or default_script),
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


def _step4_call_payload(task_spec: TaskSpec, memory_context: str = "") -> dict[str, Any]:
    artifacts = task_spec.entry_artifacts
    default_output = PROJECT_ROOT / "program" / "output" / "step4_results"
    return {
        "task_type": "resume_from_step2_3",
        "input_path": str(artifacts.get("input_csv") or artifacts.get("input_path") or ""),
        "output_root": str(artifacts.get("output_root") or default_output),
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


def _make_locked_step1_handoff_tool(task_spec: TaskSpec):
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
        return await handoff_to_step1_agent(**_step1_call_payload(task_spec))

    handoff_step1_from_task_spec_tool.__name__ = "handoff_step1_from_task_spec_tool"
    return handoff_step1_from_task_spec_tool


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


def create_orchestrator_agent(task_spec: TaskSpec | None = None, context: str = "") -> ReActAgent:
    toolkit = Toolkit()
    if task_spec is not None and task_spec.task_type in STEP1_TASK_TYPES:
        toolkit.register_tool_function(_make_locked_step1_handoff_tool(task_spec))
    elif task_spec is not None and task_spec.task_type in STEP4_TASK_TYPES:
        toolkit.register_tool_function(_make_locked_step4_handoff_tool(task_spec, memory_context=context))
    else:
        toolkit.register_tool_function(handoff_to_step1_agent)
        toolkit.register_tool_function(handoff_to_step4_agent)
    model, formatter = create_openai_model_and_formatter("main_orchestrator", "gpt-4.1-mini")
    agent = ReActAgent(
        name="OrchestratorAgent",
        sys_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=6,
        print_hint_msg=True,
    )
    agent._disable_console_output = False
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


async def run_with_orchestrator_agent(
    task_spec: TaskSpec,
    context: str = "",
    enable_memory_agent: bool = True,
) -> PipelineRunResult:
    if task_spec.task_type not in STEP1_TASK_TYPES and task_spec.task_type not in STEP4_TASK_TYPES:
        return await StateMachineOrchestrator(
            task_spec=task_spec,
            context=context,
            enable_memory_agent=enable_memory_agent,
        ).run()

    if task_spec.task_type in STEP4_TASK_TYPES:
        validation_error = _validate_step4_task_spec(task_spec)
        if validation_error is not None:
            return _pipeline_from_step4_result(task_spec, validation_error)
        if not has_model_credentials("main_orchestrator"):
            return _pipeline_from_step4_result(
                task_spec,
                WorkerResult.failure(
                    summary="OrchestratorAgent 未配置模型密钥，无法执行 Agent 编排。",
                    issues=["请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.main_orchestrator.api_key。"],
                    status=WorkerStatus.NEEDS_ESCALATION,
                    artifacts=dict(task_spec.entry_artifacts),
                ),
            )

        agent = create_orchestrator_agent(task_spec, context=context)
        payload = {
            "task_spec": task_spec.to_dict(),
            "context": context,
            "required_action": "调用 handoff_step4_from_task_spec_tool，禁止手写或改写路径参数。",
            "required_output": "WorkerResult JSON",
        }
        try:
            res = await agent(make_user_msg("TaskRouterAgent", json.dumps(payload, ensure_ascii=False, indent=2)))
            tool_results = await collect_tool_results(agent)
            tool_payload = (tool_results.get("handoff_step4_from_task_spec_tool") or [None])[-1]
            if tool_payload:
                return _pipeline_from_step4_result(task_spec, worker_result_from_dict(tool_payload))
            parsed_final = parse_agent_text_json(res.get_text_content() or "")
            if parsed_final:
                return _pipeline_from_step4_result(task_spec, worker_result_from_dict(parsed_final))
            return _pipeline_from_step4_result(
                task_spec,
                WorkerResult.failure(
                    summary="OrchestratorAgent 未调用 Step4 handoff 工具。",
                    issues=["未检测到 handoff_step4_from_task_spec_tool 的工具返回。"],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=dict(task_spec.entry_artifacts),
                ),
            )
        except Exception as exc:
            return _pipeline_from_step4_result(
                task_spec,
                WorkerResult.failure(
                    summary="OrchestratorAgent 执行失败。",
                    issues=[str(exc)],
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=dict(task_spec.entry_artifacts),
                ),
            )

    validation_error = _validate_step1_task_spec(task_spec)
    if validation_error is not None:
        return _pipeline_from_step1_result(task_spec, validation_error)
    if not has_model_credentials("main_orchestrator"):
        return _pipeline_from_step1_result(
            task_spec,
            WorkerResult.failure(
                summary="OrchestratorAgent 未配置模型密钥，无法执行 Agent 编排。",
                issues=["请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.main_orchestrator.api_key。"],
                status=WorkerStatus.NEEDS_ESCALATION,
                artifacts=dict(task_spec.entry_artifacts),
            ),
        )

    agent = create_orchestrator_agent(task_spec, context=context)
    payload = {
        "task_spec": task_spec.to_dict(),
        "context": context,
        "required_action": "调用 handoff_step1_from_task_spec_tool，禁止手写或改写路径参数。",
        "required_output": "WorkerResult JSON",
    }
    try:
        res = await agent(make_user_msg("TaskRouterAgent", json.dumps(payload, ensure_ascii=False, indent=2)))
        tool_results = await collect_tool_results(agent)
        tool_payload = (tool_results.get("handoff_step1_from_task_spec_tool") or [None])[-1]
        if tool_payload:
            return _pipeline_from_step1_result(task_spec, worker_result_from_dict(tool_payload))
        parsed_final = parse_agent_text_json(res.get_text_content() or "")
        if parsed_final:
            return _pipeline_from_step1_result(task_spec, worker_result_from_dict(parsed_final))
        return _pipeline_from_step1_result(
            task_spec,
            WorkerResult.failure(
                summary="OrchestratorAgent 未调用 Step1 handoff 工具。",
                issues=["未检测到 handoff_step1_from_task_spec_tool 的工具返回。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=dict(task_spec.entry_artifacts),
            ),
        )
    except Exception as exc:
        return _pipeline_from_step1_result(
            task_spec,
            WorkerResult.failure(
                summary="OrchestratorAgent 执行失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=dict(task_spec.entry_artifacts),
            ),
        )
