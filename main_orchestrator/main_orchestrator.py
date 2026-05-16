from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
MEMORY_DIR = ORCHESTRATOR_DIR / "memory"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR, EXECUTION_DIR, MEMORY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.model import OpenAIChatModel
except Exception:  # pragma: no cover - compatibility for minimal test envs
    OpenAIChatFormatter = None
    OpenAIChatModel = None

from configs.loader import get_agent_config
from contracts import PipelineRunResult, TaskType
from orchestrator_agent import run_with_orchestrator_agent
from router import route_task
from router_agent import route_task_with_agent

try:
    from memory_supervisor import get_pipeline_context, report_pipeline_result
except Exception:  # pragma: no cover
    get_pipeline_context = None
    report_pipeline_result = None


AGENT_CFG = get_agent_config("main_orchestrator")
EXIT_COMMANDS = {"exit", "quit", "q", "退出", "结束"}
HELP_COMMANDS = {"help", "?", "帮助"}


def resolve_model_name(agent_key: str, default: str) -> str:
    env_model = os.environ.get("MODEL_NAME")
    if env_model:
        return env_model
    cfg = get_agent_config(agent_key)
    return str(cfg.get("model_name") or cfg.get("model") or default)


def _build_orchestrator(context_str: str, enable_memory_agent: bool = True):
    """Compatibility shim for old tests; new runtime uses StateMachineOrchestrator."""
    _ = context_str, enable_memory_agent
    formatter = OpenAIChatFormatter() if OpenAIChatFormatter else None
    model = None
    if OpenAIChatModel:
        model = OpenAIChatModel(
            model_name=resolve_model_name("main_orchestrator", "gpt-4.1-mini"),
            api_key=os.environ.get("OPENAI_API_KEY", AGENT_CFG.get("api_key", "")),
            client_kwargs={
                "base_url": os.environ.get(
                    "OPENAI_API_BASE",
                    AGENT_CFG.get("base_url") or AGENT_CFG.get("api_base") or "https://api.openai.com/v1",
                )
            },
        )
    return SimpleNamespace(name="StateMachineOrchestrator", model=model, formatter=formatter)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the task-aware multi-agent pipeline orchestrator.")
    parser.add_argument("input", nargs="?", default="", help="Input path or task text. Omit it to start interactive mode.")
    parser.add_argument(
        "--task-type",
        choices=[item.value for item in TaskType],
        default="",
        help="Override Router task type.",
    )
    parser.add_argument(
        "--disable-memory-agent",
        action="store_true",
        help="Disable Playbook context injection and post-run reflection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final PipelineRunResult as JSON only.",
    )
    return parser.parse_args(argv)


async def run_pipeline(
    user_input: str,
    task_type: str = "",
    enable_memory_agent: bool = True,
) -> PipelineRunResult:
    if task_type:
        task_spec = route_task(user_input, project_root=PROJECT_ROOT)
    else:
        task_spec = await route_task_with_agent(user_input, project_root=PROJECT_ROOT)
    if task_type:
        task_spec.task_type = TaskType(task_type)
        task_spec.intent_summary = f"用户显式指定 task_type={task_type}"

    context_str = ""
    used_bullet_ids: list[str] = []
    if enable_memory_agent and get_pipeline_context is not None:
        try:
            context_str, used_bullet_ids = await get_pipeline_context(query_text=task_spec.intent_summary or user_input)
        except Exception as exc:
            context_str = f"ACE Playbook 读取失败，主流程继续执行: {exc}"

    result = await run_with_orchestrator_agent(
        task_spec=task_spec,
        context=context_str,
        enable_memory_agent=enable_memory_agent,
    )

    if enable_memory_agent and report_pipeline_result is not None and task_spec.task_type != TaskType.MEMORY_ONLY:
        try:
            await report_pipeline_result(
                {
                    "input_text": user_input,
                    "duration_seconds": 0.0,
                    "orchestrator_summary": result.summary,
                    "retrieved_bullet_ids": used_bullet_ids,
                    "steps": result.worker_results,
                    "raw_trace_text": json.dumps(result.to_dict(), ensure_ascii=False),
                    "success": result.status.value == "SUCCESS",
                    "failure_signals": result.repair_ticket.validator_errors if result.repair_ticket else [],
                },
                used_bullet_ids=used_bullet_ids,
            )
        except Exception as exc:
            result.worker_results.append(
                {
                    "state": "MEMORY_REFLECTION",
                    "result": {
                        "status": "FAILED",
                        "summary": f"Memory reflection 失败但不阻塞主流程: {exc}",
                        "artifacts": {},
                        "issues": [str(exc)],
                        "next_recommendation": "",
                    },
                }
            )
    return result


async def main_async(args: argparse.Namespace) -> PipelineRunResult:
    if args.json:
        os.environ["STEP1_PROGRESS_ENABLED"] = "false"
    user_input = args.input or str(PROJECT_ROOT / "mimic-10")
    return await run_pipeline(
        user_input=user_input,
        task_type=args.task_type,
        enable_memory_agent=not args.disable_memory_agent,
    )


def _print_result(result: PipelineRunResult, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    print("\n[Orchestrator v2] 执行完成")
    print(f"状态: {result.status.value}")
    print(f"任务: {result.task_spec.task_type.value}")
    print(f"状态流: {' -> '.join(state.value for state in result.states_visited)}")
    print(f"摘要: {result.summary}")
    if result.repair_ticket:
        print("\n[Repair Ticket]")
        print(json.dumps(result.repair_ticket.to_dict(), ensure_ascii=False, indent=2))
    if result.worker_results:
        print("\n[Worker Results]")
        for item in result.worker_results:
            state = item.get("state")
            worker_result = item.get("result") or {}
            print(f"- {state}: {worker_result.get('status')} | {worker_result.get('summary')}")


def _should_enter_interactive(args: argparse.Namespace) -> bool:
    return not str(args.input or "").strip()


def _print_interactive_help() -> None:
    print(
        "\n可输入自然语言任务或路径，例如：\n"
        "  处理 mimic-10，只跑 step1\n"
        "  从 reorganized_output/_meta/records.json 继续跑\n"
        "  修复 program/output/step1_results\n"
        "\n快捷命令：help / ? / 帮助 查看说明，exit / quit / q / 退出 结束。\n"
    )


def _resolve_interactive_input(user_input: str, last_result: PipelineRunResult | None) -> str:
    text = user_input.strip()
    if text in {"继续", "继续跑", "继续运行", "continue"} and last_result is not None:
        for item in reversed(last_result.worker_results):
            result = item.get("result") if isinstance(item, dict) else {}
            artifacts = result.get("artifacts") if isinstance(result, dict) else {}
            if isinstance(artifacts, dict) and artifacts.get("records_path"):
                return f"从 {artifacts['records_path']} 继续跑"
    return text


async def interactive_loop(args: argparse.Namespace) -> None:
    if args.json:
        os.environ["STEP1_PROGRESS_ENABLED"] = "false"

    print("[MultiAgent] 交互模式已启动。直接输入任务，输入 help 查看示例，输入 exit 退出。")
    if args.task_type:
        print(f"[MultiAgent] 当前所有输入都会被 --task-type={args.task_type} 覆盖。")

    last_result: PipelineRunResult | None = None
    while True:
        try:
            user_input = input("\nMultiAgent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[MultiAgent] 已退出。")
            return

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS or user_input in EXIT_COMMANDS:
            print("[MultiAgent] 已退出。")
            return
        if user_input.lower() in HELP_COMMANDS or user_input in HELP_COMMANDS:
            _print_interactive_help()
            continue

        effective_input = _resolve_interactive_input(user_input, last_result)
        result = await run_pipeline(
            user_input=effective_input,
            task_type=args.task_type,
            enable_memory_agent=not args.disable_memory_agent,
        )
        _print_result(result, as_json=args.json)
        last_result = result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if _should_enter_interactive(args):
        asyncio.run(interactive_loop(args))
        return
    result = asyncio.run(main_async(args))
    _print_result(result, as_json=args.json)


if __name__ == "__main__":
    main()
