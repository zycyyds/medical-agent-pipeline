from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

AGENT1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT1_DIR.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
STEP1_DIR = PROJECT_ROOT / "step-1"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR, EXECUTION_DIR, STEP1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from supervisors import Step1Supervisor  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step1 standalone with the same tools used by main_orchestrator.")
    parser.add_argument("input_path", help="原始输入目录，或 records.json 路径")
    parser.add_argument(
        "--task-type",
        choices=["step1_only", "resume_from_records"],
        default="step1_only",
        help="step1_only 从原始目录扫描；resume_from_records 从 records.json 续跑",
    )
    parser.add_argument("--records-path", default=str(PROJECT_ROOT / "reorganized_output" / "_meta" / "records.json"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "program" / "output" / "step1_results"))
    parser.add_argument("--script-path", default=str(PROJECT_ROOT / "step-1" / "generated_reorganizer.py"))
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    return parser.parse_args(argv)


def _worker_result_to_dict(result) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "summary": result.summary,
        "artifacts": result.artifacts,
        "issues": result.issues,
        "next_recommendation": result.next_recommendation,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = str(Path(args.input_path).expanduser().resolve())
    supervisor = Step1Supervisor(
        records_path=Path(args.records_path).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        generated_script_path=Path(args.script_path).expanduser().resolve(),
    )
    if args.task_type == "resume_from_records":
        result = await supervisor.run_from_records(input_path)
    else:
        result = await supervisor.run_from_input(input_path)
    return {
        "status": result.status.value,
        "task_type": args.task_type,
        "result": _worker_result_to_dict(result),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.json:
        os.environ["STEP1_PROGRESS_ENABLED"] = "false"
    payload = asyncio.run(run(args))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["result"]["summary"])
        if payload["result"]["issues"]:
            print(json.dumps(payload["result"]["issues"], ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
