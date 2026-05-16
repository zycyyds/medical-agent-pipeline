from __future__ import annotations

from pathlib import Path
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from agent_runtime import (
    create_openai_model_and_formatter,
)
from contracts import PipelineRunResult, PipelineState, RepairTicket, TaskSpec, TaskType, WorkerResult, WorkerStatus
from handoffs import handoff_to_step1_agent, run_step1_handoff
from state_machine import StateMachineOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ORCHESTRATOR_SYSTEM_PROMPT = """你是 OrchestratorAgent，只负责根据 TaskSpec 把任务 handoff 给对应子 Agent。

第一阶段只有 Step1ReActAgent 已接入：
- step1_only 必须调用 handoff_to_step1_agent
- resume_from_records 必须调用 handoff_to_step1_agent

你不要直接调用 Step1 细粒度工具。handoff_to_step1_agent 是唯一允许的 Step1 子 Agent 入口。
最终回答只输出 handoff 工具返回的 JSON，不要添加解释。
"""


def create_orchestrator_agent() -> ReActAgent:
    toolkit = Toolkit()
    toolkit.register_tool_function(handoff_to_step1_agent)
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
    if task_spec.task_type not in {TaskType.STEP1_ONLY, TaskType.RESUME_FROM_RECORDS}:
        return await StateMachineOrchestrator(
            task_spec=task_spec,
            context=context,
            enable_memory_agent=enable_memory_agent,
        ).run()

    direct_result = await run_step1_handoff(**_step1_call_payload(task_spec))
    return _pipeline_from_step1_result(task_spec, direct_result)
