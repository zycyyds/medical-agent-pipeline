from __future__ import annotations

import asyncio
from typing import Any

import step6_runtime as _runtime
from step6_runtime import *  # noqa: F401,F403


_ORIGINAL_SAVE_STEP6_REPORT = _runtime.save_step6_report
_LAST_ANALYSIS_PASSED_IDS = _runtime._LAST_ANALYSIS_PASSED_IDS
_LAST_SAVED_ARTIFACTS = _runtime._LAST_SAVED_ARTIFACTS


def __getattr__(name: str):
    return getattr(_runtime, name)


def create_consistency_agent():
    from step6_agent import create_step6_agent

    return create_step6_agent()


def save_step6_report(report_content: str):
    cached_passed_ids = globals().get("_LAST_ANALYSIS_PASSED_IDS")
    if cached_passed_ids is not None:
        _runtime._LAST_ANALYSIS_PASSED_IDS = cached_passed_ids
    response = _ORIGINAL_SAVE_STEP6_REPORT(report_content)
    globals()["_LAST_ANALYSIS_PASSED_IDS"] = _runtime._LAST_ANALYSIS_PASSED_IDS
    globals()["_LAST_SAVED_ARTIFACTS"] = _runtime._LAST_SAVED_ARTIFACTS
    return response


async def run_step6_only(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    _runtime.create_consistency_agent = create_consistency_agent
    _runtime.save_step6_report = save_step6_report
    result = await _runtime.run_step6_only(runtime_overrides)
    globals()["_LAST_ANALYSIS_PASSED_IDS"] = _runtime._LAST_ANALYSIS_PASSED_IDS
    globals()["_LAST_SAVED_ARTIFACTS"] = _runtime._LAST_SAVED_ARTIFACTS
    return result


async def run_step6() -> None:
    result = await run_step6_only()
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Step6 failed")
    print("\n" + "=" * 65)
    print("  Step 6 全部完成")
    print(f"  报告目录: {result.get('output_step6')}/")
    print("=" * 65)


def main() -> None:
    parser = _runtime._build_parser()
    args = parser.parse_args()
    overrides = {
        "input_csv": args.input,
        "raw_csv": args.raw_csv,
        "output_step6_dir": args.output,
        "patient_id_col": args.patient_id_col,
        "data_source": args.data_source,
        "reuse_existing": args.reuse_existing,
    }
    result = asyncio.run(run_step6_only({key: value for key, value in overrides.items() if value not in ("", None)}))
    if not result.get("success"):
        raise SystemExit(result.get("error") or "Step6 failed")
    print(f"[Step6] success, passed={len(result.get('passed_patient_ids') or [])}")


if __name__ == "__main__":
    main()
