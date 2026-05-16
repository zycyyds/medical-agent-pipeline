from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    FULL_PIPELINE = "full_pipeline"
    STEP1_ONLY = "step1_only"
    RESUME_FROM_RECORDS = "resume_from_records"
    RESUME_FROM_STEP2_3 = "resume_from_step2_3"
    RESUME_FROM_STEP4 = "resume_from_step4"
    RESUME_FROM_STEP5 = "resume_from_step5"
    REPAIR_TASK = "repair_task"
    MEMORY_ONLY = "memory_only"


class PipelineState(str, Enum):
    INIT = "INIT"
    STEP1_RECORDING = "STEP1_RECORDING"
    STEP1_CODEGEN = "STEP1_CODEGEN"
    STEP1_VALIDATING = "STEP1_VALIDATING"
    STEP2_3_RUNNING = "STEP2_3_RUNNING"
    STEP4_RUNNING = "STEP4_RUNNING"
    STEP5_RUNNING = "STEP5_RUNNING"
    STEP6_RUNNING = "STEP6_RUNNING"
    STEP7_RUNNING = "STEP7_RUNNING"
    REPAIR_PENDING = "REPAIR_PENDING"
    MEMORY_REFLECTION = "MEMORY_REFLECTION"
    DONE = "DONE"
    FAILED = "FAILED"


class WorkerStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NEEDS_REPAIR = "NEEDS_REPAIR"
    NEEDS_ESCALATION = "NEEDS_ESCALATION"


@dataclass
class TaskSpec:
    task_type: TaskType
    entry_artifacts: dict[str, Any] = field(default_factory=dict)
    resume_from_step: str = ""
    intent_summary: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "entry_artifacts": dict(self.entry_artifacts),
            "resume_from_step": self.resume_from_step,
            "intent_summary": self.intent_summary,
            "confidence": self.confidence,
        }


@dataclass
class WorkerResult:
    status: WorkerStatus
    summary: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    next_recommendation: str = ""

    @classmethod
    def success(
        cls,
        summary: str = "",
        artifacts: dict[str, Any] | None = None,
        next_recommendation: str = "",
    ) -> "WorkerResult":
        return cls(
            status=WorkerStatus.SUCCESS,
            summary=summary,
            artifacts=dict(artifacts or {}),
            next_recommendation=next_recommendation,
        )

    @classmethod
    def failure(
        cls,
        summary: str,
        issues: list[str] | None = None,
        status: WorkerStatus = WorkerStatus.FAILED,
        artifacts: dict[str, Any] | None = None,
        next_recommendation: str = "",
    ) -> "WorkerResult":
        return cls(
            status=status,
            summary=summary,
            artifacts=dict(artifacts or {}),
            issues=list(issues or []),
            next_recommendation=next_recommendation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "artifacts": dict(self.artifacts),
            "issues": list(self.issues),
            "next_recommendation": self.next_recommendation,
        }


@dataclass
class ValidatorResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_worker_result(self, summary: str, artifacts: dict[str, Any] | None = None) -> WorkerResult:
        if self.passed:
            return WorkerResult.success(summary=summary, artifacts=artifacts or self.details)
        return WorkerResult.failure(
            summary=summary,
            issues=self.issues,
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts=artifacts or self.details,
            next_recommendation="进入 repair_task 处理校验失败产物。",
        )


@dataclass
class RepairTicket:
    repair_type: str
    failed_state: PipelineState
    input_artifacts: dict[str, Any] = field(default_factory=dict)
    validator_errors: list[str] = field(default_factory=list)
    resume_target_state: PipelineState = PipelineState.INIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_type": self.repair_type,
            "failed_state": self.failed_state.value,
            "input_artifacts": dict(self.input_artifacts),
            "validator_errors": list(self.validator_errors),
            "resume_target_state": self.resume_target_state.value,
        }


@dataclass
class PipelineRunResult:
    status: WorkerStatus
    task_spec: TaskSpec
    states_visited: list[PipelineState] = field(default_factory=list)
    worker_results: list[dict[str, Any]] = field(default_factory=list)
    repair_ticket: RepairTicket | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "task_spec": self.task_spec.to_dict(),
            "states_visited": [state.value for state in self.states_visited],
            "worker_results": list(self.worker_results),
            "repair_ticket": self.repair_ticket.to_dict() if self.repair_ticket else None,
            "summary": self.summary,
        }
