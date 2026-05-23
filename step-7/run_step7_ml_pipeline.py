from __future__ import annotations

import asyncio
from typing import Any

import step7_runtime as _runtime
from step7_runtime import *  # noqa: F401,F403


_ORIGINAL_FORMAT_AND_SAVE_ML_DATASET = _runtime.format_and_save_ml_dataset


def __getattr__(name: str):
    return getattr(_runtime, name)


def create_dataprep_agent():
    from step7_agent import create_step7_agent

    return create_step7_agent()


def format_and_save_ml_dataset(*args, **kwargs):
    return _ORIGINAL_FORMAT_AND_SAVE_ML_DATASET(*args, **kwargs)


async def run_step7_only(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    _runtime.create_dataprep_agent = create_dataprep_agent
    if "_local_icd10_lookup" in globals():
        _runtime._local_icd10_lookup = globals()["_local_icd10_lookup"]
    result = await _runtime.run_step7_only(runtime_overrides)
    return result


async def run_step7() -> None:
    result = await run_step7_only()
    if not result.get("success"):
        raise RuntimeError(str(result.get("error") or "Step7 执行失败"))


def main() -> None:
    parser = _runtime._build_parser()
    args = parser.parse_args()
    overrides = {
        "input_csv": args.input_csv,
        "selection_report": args.selection_report,
        "passed_patients_json": args.passed_patients_json,
        "output_step7_dir": args.output_step7_dir,
        "patient_id_col": args.patient_id_col,
        "data_source": args.data_source,
        "task_text": args.task_text,
    }
    result = asyncio.run(run_step7_only(overrides))
    if not result.get("success"):
        raise SystemExit(result.get("error") or "Step7 failed")
    print(f"[Step7] success, datasets={len(result.get('step7_dataset_csvs') or [])}")


if __name__ == "__main__":
    main()
