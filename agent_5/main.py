"""Agent 5 standalone entrypoint: data quality repair."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_4.data_quality_repair import *  # noqa: F401,F403 - expose cleaning helpers through agent_5.
from agent_4 import data_quality_repair as _repair


def get_agent_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_project_root() -> str:
    return str(PROJECT_ROOT)


def get_default_input_dir() -> str:
    return os.path.join(get_agent_root(), "data_input")


def get_default_workspace_dir() -> str:
    return os.path.join(get_project_root(), "program", "output", "step5_results")


def get_default_validation_config_path() -> str:
    return str(PROJECT_ROOT / "agent_4" / "validation_config.json")


async def run_data_quality_repair_standalone(
    input_dir: str | None = None,
    workspace_dir: str | None = None,
    validation_config_path: str | None = None,
) -> dict:
    return await _repair.run_data_quality_repair(
        input_dir=input_dir or get_default_input_dir(),
        workspace_dir=workspace_dir or get_default_workspace_dir(),
        validation_config_path=validation_config_path or get_default_validation_config_path(),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent_5 data quality repair.")
    parser.add_argument(
        "--input",
        "-i",
        default=get_default_input_dir(),
        help=f"输入 CSV/JSON 文件或目录（默认: {get_default_input_dir()}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=get_default_workspace_dir(),
        help=f"输出目录（默认: {get_default_workspace_dir()}）",
    )
    parser.add_argument(
        "--validation-config",
        default=get_default_validation_config_path(),
        help=f"验证配置路径（默认: {get_default_validation_config_path()}）",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    result = asyncio.run(
        run_data_quality_repair_standalone(
            input_dir=args.input,
            workspace_dir=args.output,
            validation_config_path=args.validation_config,
        )
    )
    print(_repair.format_run_summary(result))
    return result


if __name__ == "__main__":
    main()
