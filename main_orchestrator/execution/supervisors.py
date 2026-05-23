from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
STEP1_DIR = PROJECT_ROOT / "step-1"
STEP23_DIR = PROJECT_ROOT / "step-2-3"
STEP4_DIR = PROJECT_ROOT / "step-4"
STEP5_DIR = PROJECT_ROOT / "step-5"
STEP6_DIR = PROJECT_ROOT / "step-6"
STEP7_DIR = PROJECT_ROOT / "step-7"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, CORE_DIR, EXECUTION_DIR, STEP1_DIR, STEP23_DIR, STEP4_DIR, STEP5_DIR, STEP6_DIR, STEP7_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contracts import WorkerResult, WorkerStatus  # noqa: E402
from handoffs import run_step1_handoff, run_step23_handoff, run_step4_handoff, run_step5_handoff, run_step6_handoff, run_step7_handoff  # noqa: E402
from step23_runtime import get_default_output_root as get_default_step23_output_root  # noqa: E402
from step23_runtime import validate_step23_output as runtime_validate_step23_output  # noqa: E402
from step4_runtime import get_default_output_root, validate_step4_output as runtime_validate_step4_output  # noqa: E402
from step5_runtime import get_default_output_root as get_default_step5_output_root  # noqa: E402
from step5_runtime import validate_step5_output as runtime_validate_step5_output  # noqa: E402
from run_step6_ml_pipeline import get_default_output_step6_dir  # noqa: E402
from run_step7_ml_pipeline import get_default_output_step7_dir  # noqa: E402
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

    async def run_from_input(self, input_path: str | Path, memory_context: str | None = None) -> WorkerResult:
        try:
            kwargs = {
                "task_type": "step1_only",
                "input_path": str(input_path),
                "records_path": str(self.records_path),
                "output_root": str(self.output_root),
                "generated_script_path": str(self.generated_script_path),
            }
            if memory_context is not None:
                kwargs["memory_context"] = memory_context
            result = await run_step1_handoff(**kwargs)
            return self._post_handoff_validate(result, input_path=input_path)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step1 并行处理失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path),
            )

    async def run_from_records(
        self,
        records_path: str | Path,
        input_root: str | Path | None = None,
        memory_context: str | None = None,
    ) -> WorkerResult:
        self.records_path = Path(records_path)
        try:
            kwargs = {
                "task_type": "resume_from_records",
                "input_path": str(input_root) if input_root is not None else None,
                "records_path": str(self.records_path),
                "output_root": str(self.output_root),
                "generated_script_path": str(self.generated_script_path),
            }
            if memory_context is not None:
                kwargs["memory_context"] = memory_context
            result = await run_step1_handoff(**kwargs)
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


class Step4Supervisor:
    def __init__(self, output_root: str | Path | None = None, memory_context: str = "") -> None:
        self.output_root = Path(output_root) if output_root else Path(get_default_output_root())
        self.memory_context = memory_context

    async def run_from_input(
        self,
        input_path: str | Path,
        task_text: str | None = None,
        memory_context: str | None = None,
    ) -> WorkerResult:
        try:
            result = await run_step4_handoff(
                task_type="resume_from_step2_3",
                input_path=str(input_path),
                output_root=str(self.output_root),
                task_text=task_text,
                memory_context=memory_context if memory_context is not None else self.memory_context,
            )
            return self._post_handoff_validate(result, input_path=input_path, task_text=task_text)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step4 列筛选失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, task_text=task_text),
            )

    def _post_handoff_validate(
        self,
        result: WorkerResult,
        input_path: str | Path | None = None,
        task_text: str | None = None,
    ) -> WorkerResult:
        if result.status != WorkerStatus.SUCCESS:
            return result

        filtered_csv = result.artifacts.get("filtered_csv_path") or result.artifacts.get("next_input_csv")
        selection_report = result.artifacts.get("selection_report_path") or result.artifacts.get("next_selection_report")
        if not filtered_csv or not selection_report:
            return WorkerResult.failure(
                summary="Step4 handoff 后缺少关键产物。",
                issues=["缺少 filtered_csv_path 或 selection_report_path。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, task_text=task_text, extra=result.artifacts),
            )

        validation = runtime_validate_step4_output(
            filtered_csv_path=filtered_csv,
            selection_report_path=selection_report,
            next_input_csv=result.artifacts.get("next_input_csv"),
            next_selection_report=result.artifacts.get("next_selection_report"),
        )
        if not validation.passed:
            return WorkerResult.failure(
                summary="Step4 handoff 后输出验收失败。",
                issues=validation.issues,
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(
                    input_path=input_path,
                    task_text=task_text,
                    extra={**result.artifacts, **validation.details},
                ),
            )
        return result

    def _artifact_dict(
        self,
        input_path: str | Path | None = None,
        task_text: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts = {
            "input_path": str(input_path or ""),
            "output_root": str(self.output_root),
            "task_text": task_text or "",
        }
        artifacts.update(extra or {})
        return artifacts


class Step2_3Supervisor:
    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = Path(output_root) if output_root else Path(get_default_step23_output_root())

    async def run_from_input(
        self,
        input_path: str | Path,
        mode: str = "auto",
        user_hint: str = "",
        memory_context: str | None = None,
    ) -> WorkerResult:
        try:
            kwargs = {
                "task_type": "step2_3_only",
                "input_path": str(input_path),
                "output_root": str(self.output_root),
                "mode": mode if mode in {"auto", "directory", "ocr_fill"} else "auto",
                "user_hint": user_hint,
            }
            if memory_context is not None:
                kwargs["memory_context"] = memory_context
            result = await run_step23_handoff(**kwargs)
            return self._post_handoff_validate(result, input_path=input_path, mode=mode)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step2-3 医疗结构化与回填失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, mode=mode),
            )

    def _post_handoff_validate(
        self,
        result: WorkerResult,
        input_path: str | Path | None = None,
        mode: str = "auto",
    ) -> WorkerResult:
        if result.status != WorkerStatus.SUCCESS:
            return result
        next_input_csv = result.artifacts.get("next_input_csv")
        if not next_input_csv:
            return WorkerResult.failure(
                summary="Step2-3 handoff 后缺少 next_input/input.csv。",
                issues=["缺少 next_input_csv。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, mode=mode, extra=result.artifacts),
            )
        validation = runtime_validate_step23_output(
            next_input_csv=str(next_input_csv),
            next_summary_json=result.artifacts.get("next_summary_json"),
        )
        if not validation.get("passed"):
            return WorkerResult.failure(
                summary="Step2-3 handoff 后输出验收失败。",
                issues=list(validation.get("issues") or []),
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(
                    input_path=input_path,
                    mode=mode,
                    extra={**result.artifacts, **dict(validation.get("details") or {})},
                ),
            )
        return result

    def _artifact_dict(
        self,
        input_path: str | Path | None = None,
        mode: str = "auto",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts = {
            "input_path": str(input_path or ""),
            "output_root": str(self.output_root),
            "mode": mode,
        }
        artifacts.update(extra or {})
        return artifacts


class Step5Supervisor:
    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = Path(output_root) if output_root else Path(get_default_step5_output_root())

    async def run_from_input(self, input_path: str | Path, memory_context: str | None = None) -> WorkerResult:
        try:
            kwargs = {
                "task_type": "resume_from_step4",
                "input_path": str(input_path),
                "output_root": str(self.output_root),
            }
            if memory_context is not None:
                kwargs["memory_context"] = memory_context
            result = await run_step5_handoff(**kwargs)
            return self._post_handoff_validate(result, input_path=input_path)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step5 数据清洗失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path),
            )

    def _post_handoff_validate(self, result: WorkerResult, input_path: str | Path | None = None) -> WorkerResult:
        if result.status != WorkerStatus.SUCCESS:
            return result
        cleaned_csv = result.artifacts.get("cleaned_csv_path") or result.artifacts.get("next_input_csv")
        if not cleaned_csv:
            return WorkerResult.failure(
                summary="Step5 handoff 后缺少 cleaned CSV。",
                issues=["缺少 cleaned_csv_path 或 next_input_csv。"],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, extra=result.artifacts),
            )
        validation = runtime_validate_step5_output(
            input_csv_path=str(input_path or result.artifacts.get("input_csv_path") or ""),
            cleaned_csv_path=str(cleaned_csv),
            data_quality_report_path=result.artifacts.get("data_quality_report_path"),
            column_risk_report_path=result.artifacts.get("column_risk_report_path"),
            next_input_csv=result.artifacts.get("next_input_csv"),
            next_data_quality_report=result.artifacts.get("next_data_quality_report"),
            next_column_risk_report=result.artifacts.get("next_column_risk_report"),
        )
        if not validation.passed:
            return WorkerResult.failure(
                summary="Step5 handoff 后输出验收失败。",
                issues=validation.issues,
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, extra={**result.artifacts, **validation.details}),
            )
        return result

    def _artifact_dict(self, input_path: str | Path | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        artifacts = {
            "input_path": str(input_path or ""),
            "output_root": str(self.output_root),
        }
        artifacts.update(extra or {})
        return artifacts


class Step6Supervisor:
    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = Path(output_root) if output_root else Path(get_default_output_step6_dir())

    async def run_from_input(self, input_path: str | Path, memory_context: str | None = None) -> WorkerResult:
        try:
            result = await run_step6_handoff(
                task_type="resume_from_step5",
                input_path=str(input_path),
                raw_csv=str(input_path),
                output_root=str(self.output_root),
                memory_context=memory_context or "",
            )
            if result.status != WorkerStatus.SUCCESS:
                return result

            artifacts = self._artifact_dict(input_path=input_path, result=result.artifacts, memory_context=memory_context)
            validation_issues = self._validate_artifacts(artifacts)
            if validation_issues:
                return WorkerResult.failure(
                    summary="Step6 handoff 后输出验收失败。",
                    issues=validation_issues,
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=artifacts,
                )
            return WorkerResult.success(summary=result.summary or "Step6 一致性验证完成。", artifacts=artifacts)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step6 一致性验证失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_path, memory_context=memory_context),
            )

    def _artifact_dict(
        self,
        input_path: str | Path | None = None,
        result: dict[str, Any] | None = None,
        memory_context: str | None = None,
    ) -> dict[str, Any]:
        result = result or {}
        passed_ids = result.get("passed_patient_ids") if isinstance(result.get("passed_patient_ids"), list) else []
        passed_sample = result.get("passed_patient_ids_sample") if isinstance(result.get("passed_patient_ids_sample"), list) else []
        passed_json = str(result.get("passed_patients_json") or "")
        passed_count = len(passed_ids) if passed_ids else self._passed_count_from_json(passed_json)
        return {
            "input_path": str(input_path or result.get("input_csv") or ""),
            "input_csv": str(result.get("input_csv") or input_path or ""),
            "raw_csv": str(result.get("raw_csv") or input_path or ""),
            "output_step6": str(result.get("output_step6") or self.output_root),
            "step6_report_txt": str(result.get("step6_report_txt") or ""),
            "step6_report_json": str(result.get("step6_report_json") or ""),
            "passed_patients_json": passed_json,
            "passed_patient_count": passed_count,
            "passed_patient_ids_sample": [str(item) for item in (passed_ids[:20] if passed_ids else passed_sample[:20])],
            "memory_context_used": bool(str(memory_context or "").strip()),
        }

    @staticmethod
    def _passed_count_from_json(path_text: str) -> int:
        if not path_text:
            return 0
        try:
            with Path(path_text).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return int(payload.get("passed_count") or len(payload.get("passed_patient_ids") or []))
        except Exception:
            return 0

    @staticmethod
    def _validate_artifacts(artifacts: dict[str, Any]) -> list[str]:
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


class Step7Supervisor:
    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = Path(output_root) if output_root else Path(get_default_output_step7_dir())

    async def run_from_input(
        self,
        input_path: str | Path,
        selection_report: str | Path | None = None,
        passed_patients_json: str | Path | None = None,
        task_text: str | None = None,
        memory_context: str | None = None,
    ) -> WorkerResult:
        try:
            result = await run_step7_handoff(
                task_type="resume_from_step6",
                input_path=str(input_path),
                selection_report=str(selection_report or ""),
                passed_patients_json=str(passed_patients_json or ""),
                output_root=str(self.output_root),
                task_text=task_text or "",
                memory_context=memory_context or "",
            )
            if result.status != WorkerStatus.SUCCESS:
                return result
            artifacts = self._artifact_dict(
                input_path=input_path,
                selection_report=selection_report,
                passed_patients_json=passed_patients_json,
                task_text=task_text,
                result=result.artifacts,
                memory_context=memory_context,
            )
            validation_issues = self._validate_artifacts(artifacts)
            if validation_issues:
                return WorkerResult.failure(
                    summary="Step7 handoff 后输出验收失败。",
                    issues=validation_issues,
                    status=WorkerStatus.NEEDS_REPAIR,
                    artifacts=artifacts,
                )
            return WorkerResult.success(summary=result.summary or "Step7 ML 训练数据生成完成。", artifacts=artifacts)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step7 ML 训练数据生成失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(
                    input_path=input_path,
                    selection_report=selection_report,
                    passed_patients_json=passed_patients_json,
                    task_text=task_text,
                    memory_context=memory_context,
                ),
            )

    def _artifact_dict(
        self,
        input_path: str | Path | None = None,
        selection_report: str | Path | None = None,
        passed_patients_json: str | Path | None = None,
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
            "output_step7": str(result.get("output_step7") or self.output_root),
            "task_text": str(result.get("task_text") or task_text or ""),
            "step7_dataset_csvs": dataset_csvs,
            "step7_dataset_jsonls": dataset_jsonls,
            "step7_dataset_jsons": dataset_jsons,
            "model_config_jsons": model_configs,
            "step7_dataset_count": len(dataset_csvs),
            "memory_context_used": bool(str(memory_context or "").strip()),
        }

    @staticmethod
    def _validate_artifacts(artifacts: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        required_lists = ("step7_dataset_csvs", "step7_dataset_jsonls", "step7_dataset_jsons", "model_config_jsons")
        for key in required_lists:
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
