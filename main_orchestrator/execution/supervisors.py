from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
STEP1_DIR = PROJECT_ROOT / "step-1"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, CORE_DIR, EXECUTION_DIR, STEP1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contracts import WorkerResult, WorkerStatus  # noqa: E402
from handoffs import run_step1_handoff  # noqa: E402
from validators import validate_records, validate_required_files, validate_step1_output  # noqa: E402


def _default_records_path() -> Path:
    return PROJECT_ROOT / "reorganized_output" / "_meta" / "records.json"


def _default_step1_output_root() -> Path:
    return PROJECT_ROOT / "program" / "output" / "step1_results"


def _default_generated_script_path() -> Path:
    return PROJECT_ROOT / "step-1" / "generated_reorganizer.py"


class Step1Supervisor:
    def __init__(
        self,
        records_path: str | Path | None = None,
        output_root: str | Path | None = None,
        generated_script_path: str | Path | None = None,
    ) -> None:
        self.records_path = Path(records_path) if records_path else _default_records_path()
        self.output_root = Path(output_root) if output_root else _default_step1_output_root()
        self.generated_script_path = Path(generated_script_path) if generated_script_path else _default_generated_script_path()

    async def run_from_input(self, input_path: str | Path) -> WorkerResult:
        try:
            result = await run_step1_handoff(
                task_type="step1_only",
                input_path=str(input_path),
                records_path=str(self.records_path),
                output_root=str(self.output_root),
                generated_script_path=str(self.generated_script_path),
            )
            return self._post_handoff_validate(result, input_path=input_path)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step1 并行处理失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path),
            )

    async def run_from_records(self, records_path: str | Path, input_root: str | Path | None = None) -> WorkerResult:
        self.records_path = Path(records_path)
        try:
            result = await run_step1_handoff(
                task_type="resume_from_records",
                input_path=str(input_root) if input_root is not None else None,
                records_path=str(self.records_path),
                output_root=str(self.output_root),
                generated_script_path=str(self.generated_script_path),
            )
            return self._post_handoff_validate(result, input_path=input_root)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step1 从 records 续跑失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(),
            )

    def _post_handoff_validate(self, result: WorkerResult, input_path: str | Path | None = None) -> WorkerResult:
        if result.status != WorkerStatus.SUCCESS:
            return result

        record_validation = validate_records(self.records_path)
        if not record_validation.passed:
            return WorkerResult.failure(
                summary="Step1 handoff 后 records 校验失败。",
                issues=record_validation.issues,
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, extra={**result.artifacts, **record_validation.details}),
            )

        expected_min = int(
            result.artifacts.get("processed_records")
            or result.artifacts.get("records_count")
            or result.artifacts.get("output_file_count")
            or 1
        )
        output_validation = validate_step1_output(self.output_root, expected_min_files=max(1, expected_min))
        if not output_validation.passed:
            return WorkerResult.failure(
                summary="Step1 handoff 后输出验收失败。",
                issues=output_validation.issues,
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(
                    input_path=input_path,
                    extra={**result.artifacts, **record_validation.details, **output_validation.details},
                ),
            )
        return result

    def _artifact_dict(self, input_path: str | Path | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        artifacts = {
            "input_path": str(input_path or ""),
            "records_path": str(self.records_path),
            "generated_script_path": str(self.generated_script_path),
            "step1_output_root": str(self.output_root),
        }
        artifacts.update(extra or {})
        return artifacts


class LegacyStepSupervisor:
    def __init__(self, context: str = "", enable_memory_agent: bool = True) -> None:
        self.context = context
        self.enable_memory_agent = enable_memory_agent

    async def run_step2_3(self, input_data: str) -> WorkerResult:
        return self._not_connected("Step2_3", input_data)

    async def run_step4(self, input_data: str) -> WorkerResult:
        return self._not_connected("Step4", input_data)

    async def run_step5(self, input_data: str) -> WorkerResult:
        return self._not_connected("Step5", input_data)

    async def run_step6(self, input_data: str) -> WorkerResult:
        return self._not_connected("Step6", input_data)

    async def run_step7(self, input_data: str) -> WorkerResult:
        return self._not_connected("Step7", input_data)

    @staticmethod
    def _not_connected(label: str, input_data: str) -> WorkerResult:
        return WorkerResult.failure(
            summary=f"{label} 尚未接入新主链路。",
            issues=[f"{label} 的旧包装层已删除，需要用新的 supervisor/tool 重新接入。"],
            status=WorkerStatus.NEEDS_ESCALATION,
            artifacts={"input_data": input_data},
            next_recommendation=f"实现 {label}Supervisor 后从该状态续跑。",
        )


class ArtifactRepairSupervisor:
    def run(self, artifacts: dict[str, Any], failed_state: str = "") -> WorkerResult:
        missing = {
            key: value
            for key, value in artifacts.items()
            if key.startswith("missing") or (isinstance(value, str) and value.endswith(("selection_report.json", "filtered.csv")))
        }
        validation = validate_required_files({key: value for key, value in missing.items() if isinstance(value, str)})
        if validation.passed:
            return WorkerResult.success(summary="repair 检查未发现缺失文件。", artifacts=validation.details)
        return WorkerResult.failure(
            summary=f"{failed_state or 'repair'} 需要人工或上游步骤补齐产物。",
            issues=validation.issues,
            status=WorkerStatus.NEEDS_ESCALATION,
            artifacts=validation.details,
        )
