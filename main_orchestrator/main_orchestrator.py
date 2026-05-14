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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

try:
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.model import OpenAIChatModel
except Exception:  # pragma: no cover - compatibility for minimal test envs
    OpenAIChatFormatter = None
    OpenAIChatModel = None

from configs.loader import get_agent_config
from contracts import PipelineRunResult, TaskType
from router import route_task
from state_machine import StateMachineOrchestrator

try:
    from memory_supervisor import get_pipeline_context, report_pipeline_result
except Exception:  # pragma: no cover
    get_pipeline_context = None
    report_pipeline_result = None


AGENT_CFG = get_agent_config("main_orchestrator")


def resolve_model_name(agent_key: str, default: str) -> str:
    env_model = os.environ.get("MODEL_NAME")
    if env_model:
        return env_model
    cfg = get_agent_config(agent_key)
    return str(cfg.get("model_name") or cfg.get("model") or default)


def _build_orchestrator(context_str: str, enable_memory_agent: bool = True, verbose_agent_output: bool = False):
    """Compatibility shim for old tests; new runtime uses StateMachineOrchestrator."""
    _ = context_str, enable_memory_agent, verbose_agent_output
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
    parser.add_argument("input", nargs="?", default="", help="Input path or middle artifact path.")
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
        "--verbose-agent-output",
        action="store_true",
        help="Reserved for compatibility; state-machine runtime prints structured summaries.",
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
    task_spec = route_task(user_input, project_root=PROJECT_ROOT)
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

    orchestrator = StateMachineOrchestrator(
        task_spec=task_spec,
        context=context_str,
        enable_memory_agent=enable_memory_agent,
    )
    result = await orchestrator.run()

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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = asyncio.run(main_async(args))
    _print_result(result, as_json=args.json)


if __name__ == "__main__":
    main()
