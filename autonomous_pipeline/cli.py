from __future__ import annotations

import argparse
import asyncio
import importlib
import re
import sys
from pathlib import Path
from typing import Any

from agentscope.message import Msg

from .config import DEFAULT_OUTPUT_ROOT

STEPS = ("step1", "step2", "step3", "step4", "step5", "step6")


def _load_agent_module(step: str):
    if step not in STEPS:
        raise SystemExit(f"Unsupported step '{step}'. Expected one of: {', '.join(STEPS)}")
    return importlib.import_module(f"autonomous_pipeline.steps.{step}.agent")


def _prompt_for(step: str, input_path: str, output_root: str, task_text: str) -> str:
    mod = _load_agent_module(step)
    return mod.build_task_prompt(input_path=input_path, output_root=output_root, task_text=task_text)


def _is_connection_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    if name in {"APIConnectionError", "RemoteProtocolError", "ConnectError", "ReadTimeout", "APITimeoutError"}:
        return True
    return "connection error" in text or "server disconnected" in text or "remote protocol" in text


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _clean_model_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def print_response(response: Any) -> None:
    print(_clean_model_text(_message_text(response)))


async def _run_step(step: str, input_path: str, output_root: str, task_text: str, retries: int = 2) -> Any:
    mod = _load_agent_module(step)
    agent = mod.create_agent()
    prompt = mod.build_task_prompt(input_path=input_path, output_root=output_root, task_text=task_text)
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return await agent(Msg(name="user", role="user", content=prompt))
        except Exception as exc:
            last_error = exc
            if not _is_connection_error(exc) or attempt >= retries:
                raise
            wait_seconds = min(2 ** attempt, 8)
            print(f"[LLM Retry] 模型连接异常，{wait_seconds}s 后重试 ({attempt + 1}/{retries})：{type(exc).__name__}: {exc}", file=sys.stderr)
            await asyncio.sleep(wait_seconds)
    raise RuntimeError(f"Step {step} failed before tool execution: {last_error}")


def list_steps() -> None:
    for step in STEPS:
        mod = _load_agent_module(step)
        print(f"{step}: {mod.STEP_TITLE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous medical data pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-steps", help="List autonomous pipeline steps")

    run_step = sub.add_parser("run-step", help="Run one autonomous step")
    run_step.add_argument("--step", required=True, choices=STEPS)
    run_step.add_argument("--input", required=True)
    run_step.add_argument("--task", default="")
    run_step.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run_step.add_argument("--dry-run", action="store_true", help="Print the prompt without calling the model")
    run_step.add_argument("--retries", type=int, default=2, help="Retry transient model connection errors before failing")

    run_all = sub.add_parser("run-all", help="Run Step1-Step6 in stage order; each step internally chooses tools")
    run_all.add_argument("--input", required=True)
    run_all.add_argument("--task", default="")
    run_all.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run_all.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd == "list-steps":
        list_steps()
        return

    output_root = str(Path(args.output_root).resolve())
    if args.cmd == "run-step":
        if args.dry_run:
            print(_prompt_for(args.step, args.input, output_root, args.task))
            return
        try:
            response = asyncio.run(_run_step(args.step, args.input, output_root, args.task, retries=args.retries))
        except Exception as exc:
            print(f"[run-step failed] {type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print_response(response)
        return

    current_input = args.input
    for step in STEPS:
        if args.dry_run:
            print(f"\n===== {step} =====")
            print(_prompt_for(step, current_input, output_root, args.task))
            continue
        response = asyncio.run(_run_step(step, current_input, output_root, args.task))
        print_response(response)
        # First version keeps stage order but does not infer every next artifact from agent text.
        # Operators can run individual steps with explicit inputs for controlled experiments.


if __name__ == "__main__":
    main()
