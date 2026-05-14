"""Agent 4 standalone entrypoint: task-oriented column clipping.

The implementation stays in ``agent_5.medical_column_selector`` for now to keep
the code move small. This file defines the Agent 4-facing defaults and CLI.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_4.data_quality_repair import *  # noqa: F401,F403 - keep old imports compatible.
from agent_5.medical_column_selector.main import (
    TaskDrivenColumnSelector,
    get_default_task_text,
    resolve_input_csv,
)


def get_agent_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_project_root() -> str:
    return str(PROJECT_ROOT)


def get_default_input_dir() -> str:
    return os.path.join(get_agent_root(), "data_input")


def get_default_output_dir() -> str:
    return os.path.join(get_project_root(), "program", "output", "step4_results")


def run_task_oriented_clipping(
    input_path: str | None = None,
    output_dir: str | None = None,
    task_text: str | None = None,
) -> dict:
    input_csv = resolve_input_csv(input_path or get_default_input_dir())
    target_output_dir = output_dir or get_default_output_dir()
    selector = TaskDrivenColumnSelector()
    return selector.run(
        input_csv_path=input_csv,
        task_text=task_text or get_default_task_text(),
        output_dir=target_output_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent_4 task-oriented column clipping.")
    parser.add_argument(
        "--input",
        "-i",
        default=get_default_input_dir(),
        help=f"输入 CSV/XLSX 文件或目录（默认: {get_default_input_dir()}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=get_default_output_dir(),
        help=f"输出目录（默认: {get_default_output_dir()}）",
    )
    parser.add_argument(
        "--task",
        "-t",
        default=get_default_task_text(),
        help="任务文本；也可通过 AGENT4_TASK_TEXT 环境变量设置。",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    result = run_task_oriented_clipping(
        input_path=args.input,
        output_dir=args.output,
        task_text=args.task,
    )
    print("Agent4 task clipping completed.")
    print(f"Filtered CSV: {result['filtered_csv_path']}")
    print(f"Selection Report: {result['selection_report_path']}")
    return result


if __name__ == "__main__":
    main()
