from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP5_DIR = PROJECT_ROOT / "step-5"
for path in (PROJECT_ROOT, STEP5_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step5_runtime import get_default_output_root, run_step5_pipeline  # noqa: E402


async def run_data_quality_repair(
    input_dir: str,
    workspace_dir: str | None = None,
    workers: int = 4,
    llm_workers: int = 2,
    enable_llm: bool = True,
) -> dict[str, Any]:
    """Run Agent5 Step5 data quality repair."""

    return run_step5_pipeline(
        input_path=input_dir,
        output_root=workspace_dir or get_default_output_root(),
        workers=workers,
        llm_workers=llm_workers,
        enable_llm=enable_llm,
        emit=lambda message: print(message, file=sys.stderr, flush=True),
    )


def format_run_summary(result: dict[str, Any]) -> str:
    status = result.get("status", "UNKNOWN")
    issues = result.get("issues") or []
    lines = [
        f"Agent5 Step5 数据清洗完成: {status}",
        f"- cleaned_csv: {result.get('cleaned_csv_path', '')}",
        f"- next_input: {result.get('next_input_csv', '')}",
        f"- changed_cells: {result.get('changed_cell_count', 0)}",
        f"- risk_counts: {result.get('risk_counts', {})}",
    ]
    if issues:
        lines.append("- issues:")
        lines.extend(f"  - {item}" for item in issues)
    return "\n".join(lines)
