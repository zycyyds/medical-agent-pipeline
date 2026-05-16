from __future__ import annotations

import sys
from pathlib import Path
from typing import Awaitable, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, CORE_DIR, EXECUTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contracts import (
    PipelineRunResult,
    PipelineState,
    RepairTicket,
    TaskSpec,
    TaskType,
    WorkerResult,
    WorkerStatus,
)
from supervisors import ArtifactRepairSupervisor, LegacyStepSupervisor, Step1Supervisor


class StateMachineOrchestrator:
    def __init__(
        self,
        task_spec: TaskSpec,
        context: str = "",
        enable_memory_agent: bool = True,
        step1_supervisor: Step1Supervisor | None = None,
        legacy_supervisor: LegacyStepSupervisor | None = None,
    ) -> None:
        self.task_spec = task_spec
        self.context = context
        self.enable_memory_agent = enable_memory_agent
        self.step1 = step1_supervisor or Step1Supervisor()
        self.legacy = legacy_supervisor or LegacyStepSupervisor(
            context=context,
            enable_memory_agent=enable_memory_agent,
        )
        self.states_visited: list[PipelineState] = [PipelineState.INIT]
        self.worker_results: list[dict[str, object]] = []
        self.repair_ticket: RepairTicket | None = None

    async def run(self) -> PipelineRunResult:
        task_type = self.task_spec.task_type
        if task_type == TaskType.MEMORY_ONLY:
            return self._finish(WorkerStatus.SUCCESS, "memory_only 任务已路由；主链路未执行。")
        if task_type == TaskType.REPAIR_TASK:
            return await self._run_repair()
        if task_type == TaskType.RESUME_FROM_RECORDS:
            return await self._run_resume_from_records()
        if task_type == TaskType.RESUME_FROM_STEP2_3:
            return await self._run_step4_to_end()
        if task_type == TaskType.RESUME_FROM_STEP4:
            return await self._run_step5_to_end()
        if task_type == TaskType.RESUME_FROM_STEP5:
            return await self._run_step6_to_end()
        if task_type == TaskType.STEP1_ONLY:
            return await self._run_step1_only()
        return await self._run_full_pipeline()

    async def _run_step1_only(self) -> PipelineRunResult:
        result = await self._run_step(PipelineState.STEP1_RECORDING, self.step1.run_from_input, self._input_path())
        if result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP1_RECORDING, result, PipelineState.STEP1_RECORDING)
        self.states_visited.append(PipelineState.DONE)
        return self._finish(WorkerStatus.SUCCESS, "Step1-only 任务完成。")

    async def _run_resume_from_records(self) -> PipelineRunResult:
        records_path = str(self.task_spec.entry_artifacts.get("records_path") or "")
        result = await self._run_step(PipelineState.STEP1_CODEGEN, self.step1.run_from_records, records_path)
        if result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP1_CODEGEN, result, PipelineState.STEP1_CODEGEN)
        self.states_visited.append(PipelineState.DONE)
        return self._finish(WorkerStatus.SUCCESS, "已从 records.json 完成 Step1 续跑。")

    async def _run_full_pipeline(self) -> PipelineRunResult:
        step1_result = await self._run_step(PipelineState.STEP1_RECORDING, self.step1.run_from_input, self._input_path())
        if step1_result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP1_RECORDING, step1_result, PipelineState.STEP1_RECORDING)
        return await self._run_step2_3_to_end(seed_input=self._input_path())

    async def _run_step2_3_to_end(self, seed_input: str | None = None) -> PipelineRunResult:
        input_data = seed_input or str(self.task_spec.entry_artifacts.get("input_csv") or self._input_path())
        result = await self._run_step(PipelineState.STEP2_3_RUNNING, self.legacy.run_step2_3, input_data)
        if result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP2_3_RUNNING, result, PipelineState.STEP2_3_RUNNING)
        return await self._run_step4_to_end(seed_input=input_data)

    async def _run_step4_to_end(self, seed_input: str | None = None) -> PipelineRunResult:
        input_data = seed_input or str(self.task_spec.entry_artifacts.get("input_csv") or self._input_path())
        result = await self._run_step(PipelineState.STEP4_RUNNING, self.legacy.run_step4, input_data)
        if result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP4_RUNNING, result, PipelineState.STEP4_RUNNING)
        return await self._run_step5_to_end(seed_input=input_data)

    async def _run_step5_to_end(self, seed_input: str | None = None) -> PipelineRunResult:
        input_data = seed_input or str(self.task_spec.entry_artifacts.get("filtered_csv") or self._input_path())
        result = await self._run_step(PipelineState.STEP5_RUNNING, self.legacy.run_step5, input_data)
        if result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP5_RUNNING, result, PipelineState.STEP5_RUNNING)
        return await self._run_step6_to_end(seed_input=input_data)

    async def _run_step6_to_end(self, seed_input: str | None = None) -> PipelineRunResult:
        input_data = seed_input or str(self.task_spec.entry_artifacts.get("filtered_csv") or self._input_path())
        step6_result = await self._run_step(PipelineState.STEP6_RUNNING, self.legacy.run_step6, input_data)
        if step6_result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP6_RUNNING, step6_result, PipelineState.STEP6_RUNNING)
        step7_result = await self._run_step(PipelineState.STEP7_RUNNING, self.legacy.run_step7, input_data)
        if step7_result.status != WorkerStatus.SUCCESS:
            return self._needs_repair(PipelineState.STEP7_RUNNING, step7_result, PipelineState.STEP7_RUNNING)
        self.states_visited.append(PipelineState.MEMORY_REFLECTION)
        self.states_visited.append(PipelineState.DONE)
        return self._finish(WorkerStatus.SUCCESS, "流水线执行完成。")

    async def _run_repair(self) -> PipelineRunResult:
        self.states_visited.append(PipelineState.REPAIR_PENDING)
        result = ArtifactRepairSupervisor().run(
            artifacts=dict(self.task_spec.entry_artifacts),
            failed_state=self.task_spec.resume_from_step,
        )
        self._record_result(PipelineState.REPAIR_PENDING, result)
        if result.status == WorkerStatus.SUCCESS:
            self.states_visited.append(PipelineState.DONE)
            return self._finish(WorkerStatus.SUCCESS, "repair 检查完成。")
        return self._needs_repair(PipelineState.REPAIR_PENDING, result, PipelineState.REPAIR_PENDING)

    async def _run_step(
        self,
        state: PipelineState,
        call: Callable[[str], Awaitable[WorkerResult]],
        input_data: str,
    ) -> WorkerResult:
        self.states_visited.append(state)
        result = await call(input_data)
        self._record_result(state, result)
        return result

    def _record_result(self, state: PipelineState, result: WorkerResult) -> None:
        self.worker_results.append({"state": state.value, "result": result.to_dict()})

    def _needs_repair(
        self,
        failed_state: PipelineState,
        result: WorkerResult,
        resume_target_state: PipelineState,
    ) -> PipelineRunResult:
        self.states_visited.append(PipelineState.REPAIR_PENDING)
        self.repair_ticket = RepairTicket(
            repair_type=self._guess_repair_type(failed_state, result),
            failed_state=failed_state,
            input_artifacts=dict(result.artifacts),
            validator_errors=list(result.issues),
            resume_target_state=resume_target_state,
        )
        self.states_visited.append(PipelineState.FAILED)
        return self._finish(WorkerStatus.NEEDS_REPAIR, result.summary or "任务需要 repair 后继续。")

    def _finish(self, status: WorkerStatus, summary: str) -> PipelineRunResult:
        return PipelineRunResult(
            status=status,
            task_spec=self.task_spec,
            states_visited=list(self.states_visited),
            worker_results=list(self.worker_results),
            repair_ticket=self.repair_ticket,
            summary=summary,
        )

    def _input_path(self) -> str:
        artifacts = self.task_spec.entry_artifacts
        for key in ("input_path", "target", "filtered_csv", "input_csv", "records_path"):
            value = artifacts.get(key)
            if value:
                return str(value)
        return str(Path.cwd())

    @staticmethod
    def _guess_repair_type(state: PipelineState, result: WorkerResult) -> str:
        text = " ".join(result.issues + [result.summary]).lower()
        if state in {PipelineState.STEP1_RECORDING, PipelineState.STEP1_CODEGEN, PipelineState.STEP1_VALIDATING}:
            if "records" in text:
                return "records_repair"
            return "generated_script_repair"
        if "selection_report" in text or "filtered.csv" in text:
            return "missing_artifact_repair"
        return "generic_repair"
