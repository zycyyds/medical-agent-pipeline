from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from agentscope.tool import ToolResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
STEP1_DIR = PROJECT_ROOT / "step-1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))

from contracts import WorkerResult, WorkerStatus  # noqa: E402
from step1_runtime_core import scan_source_files  # noqa: E402
from step1_tools_core import (  # noqa: E402
    build_records_parallel_tool,
    run_reorganize_from_records_tool,
    validate_records_tool,
    validate_step1_output_tool,
    write_generated_reorganizer_tool,
)
from validators import validate_records, validate_required_files, validate_step1_output  # noqa: E402


def _default_records_path() -> Path:
    return PROJECT_ROOT / "reorganized_output" / "_meta" / "records.json"


def _default_step1_output_root() -> Path:
    return PROJECT_ROOT / "program" / "output" / "step1_results"


def _default_generated_script_path() -> Path:
    return PROJECT_ROOT / "step-1" / "generated_reorganizer.py"


def _parse_json_tool_response(response: ToolResponse) -> dict[str, Any]:
    payload = json.loads(str(response.content))
    if not isinstance(payload, dict):
        raise ValueError("工具返回不是对象 JSON")
    return payload


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
            return await asyncio.to_thread(self._run_from_input_sync, input_path)
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
            return await asyncio.to_thread(self._codegen_execute_validate_sync, input_root)
        except Exception as exc:
            return WorkerResult.failure(
                summary="Step1 从 records 续跑失败。",
                issues=[str(exc)],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(),
            )

    def _run_from_input_sync(self, input_path: str | Path) -> WorkerResult:
        tasks = scan_source_files(input_path)
        build_payload = _parse_json_tool_response(
            build_records_parallel_tool(
                input_path=str(input_path),
                records_path=str(self.records_path),
            )
        )
        build_result = dict(build_payload.get("artifacts") or {})
        record_validation = validate_records(self.records_path, expected_count=len(tasks))
        if not record_validation.passed:
            return record_validation.to_worker_result(
                summary="Step1 records 校验失败。",
                artifacts=self._artifact_dict(input_path=input_path, extra=build_result),
            )
        return self._codegen_execute_validate_sync(input_path, preflight_artifacts=build_result)

    def _codegen_execute_validate_sync(
        self,
        input_root: str | Path | None = None,
        preflight_artifacts: dict[str, Any] | None = None,
    ) -> WorkerResult:
        record_payload = _parse_json_tool_response(
            validate_records_tool(str(self.records_path))
        )
        if record_payload.get("status") != "SUCCESS":
            return WorkerResult.failure(
                summary="Step1 records 校验失败。",
                issues=list(record_payload.get("issues") or []),
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_root),
            )
        record_artifacts = dict(preflight_artifacts or {})
        record_artifacts.setdefault(
            "records_modality_counts",
            (record_payload.get("artifacts") or {}).get("modality_counts", {}),
        )
        record_artifacts.setdefault("record_status_counts", (record_payload.get("artifacts") or {}).get("status_counts", {}))
        record_artifacts.setdefault(
            "processable_records",
            (record_payload.get("artifacts") or {}).get("processable_records"),
        )
        record_artifacts.setdefault(
            "unsupported_records",
            (record_payload.get("artifacts") or {}).get("unsupported_records"),
        )

        script_payload = _parse_json_tool_response(
            write_generated_reorganizer_tool(
                script_path=str(self.generated_script_path),
                records_path=str(self.records_path),
                output_root=str(self.output_root),
                input_root=str(input_root) if input_root is not None else None,
            )
        )
        generated_script = str((script_payload.get("artifacts") or {}).get("generated_script_path") or self.generated_script_path)
        run_payload = _parse_json_tool_response(
            run_reorganize_from_records_tool(
                records_path=str(self.records_path),
                output_root=str(self.output_root),
                input_root=str(input_root) if input_root is not None else None,
            )
        )
        run_result = dict(run_payload.get("artifacts") or {})
        if run_payload.get("status") != "SUCCESS":
            return WorkerResult.failure(
                summary="Step1 重组脚本执行失败。",
                issues=list(run_payload.get("issues") or run_result.get("errors") or []),
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts=self._artifact_dict(input_path=input_root, extra=run_result),
                next_recommendation="进入 GeneratedScriptRepairWorker。",
            )

        output_payload = _parse_json_tool_response(
            validate_step1_output_tool(
                output_root=str(self.output_root),
                expected_min_files=max(
                    1,
                    int(run_result.get("processed_records") or run_result.get("records_count") or 1),
                ),
            )
        )
        artifacts = self._artifact_dict(input_path=input_root, extra={**record_artifacts, **run_result})
        artifacts["generated_script_path"] = generated_script
        if output_payload.get("status") != "SUCCESS":
            return WorkerResult.failure(
                summary="Step1 输出验收失败。",
                issues=list(output_payload.get("issues") or []),
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts={**artifacts, **dict(output_payload.get("artifacts") or {})},
            )
        output_details = dict(output_payload.get("artifacts") or {})
        output_details["output_modality_counts"] = output_details.pop("modality_counts", {})

        return WorkerResult.success(
            summary=(
                "Step1 完成：records 已生成/校验，重组脚本已生成并执行，"
                f"输出文件 {run_result.get('written_files', 0)} 个。"
            ),
            artifacts={**artifacts, **output_details},
        )

    def _artifact_dict(self, input_path: str | Path | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        artifacts = {
            "input_path": str(input_path or ""),
            "records_path": str(self.records_path),
            "generated_script_path": str(self.generated_script_path),
            "step1_output_root": str(self.output_root),
        }
        artifacts.update(extra or {})
        return artifacts


async def _wrap_tool_response(
    label: str,
    call: Callable[..., Awaitable[ToolResponse]],
    input_data: str,
    context: str = "",
    enable_memory_agent: bool = True,
) -> WorkerResult:
    try:
        response = await call(input_data, context=context, enable_memory_agent=enable_memory_agent)
        content = str(response.content)
        failed = any(token in content for token in ("失败", "不存在", "错误", "Traceback"))
        if failed:
            return WorkerResult.failure(
                summary=f"{label} 返回失败。",
                issues=[content],
                status=WorkerStatus.NEEDS_REPAIR,
                artifacts={"raw_output": content},
            )
        return WorkerResult.success(summary=content, artifacts={"raw_output": content})
    except TypeError:
        response = await call(input_data, context=context)
        return WorkerResult.success(summary=str(response.content), artifacts={"raw_output": str(response.content)})
    except Exception as exc:
        return WorkerResult.failure(
            summary=f"{label} 执行异常。",
            issues=[str(exc)],
            status=WorkerStatus.NEEDS_REPAIR,
        )


class LegacyStepSupervisor:
    def __init__(self, context: str = "", enable_memory_agent: bool = True) -> None:
        self.context = context
        self.enable_memory_agent = enable_memory_agent

    async def run_step2_3(self, input_data: str) -> WorkerResult:
        from step_wrappers import run_step2_3_medical_data_cleaner

        return await _wrap_tool_response(
            "Step2_3",
            run_step2_3_medical_data_cleaner,
            input_data,
            self.context,
            self.enable_memory_agent,
        )

    async def run_step4(self, input_data: str) -> WorkerResult:
        from step_wrappers import run_step4_cut

        return await _wrap_tool_response("Step4", run_step4_cut, input_data, self.context, self.enable_memory_agent)

    async def run_step5(self, input_data: str) -> WorkerResult:
        from step_wrappers import run_step5_clean

        return await _wrap_tool_response("Step5", run_step5_clean, input_data, self.context, self.enable_memory_agent)

    async def run_step6(self, input_data: str) -> WorkerResult:
        from step_wrappers import run_step6_consistency_verification

        return await _wrap_tool_response("Step6", run_step6_consistency_verification, input_data, self.context, self.enable_memory_agent)

    async def run_step7(self, input_data: str) -> WorkerResult:
        from step_wrappers import run_step7_phenotype_knowledge_confirmation

        return await _wrap_tool_response("Step7", run_step7_phenotype_knowledge_confirmation, input_data, self.context, self.enable_memory_agent)


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
