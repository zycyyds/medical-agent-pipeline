from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
CORE_DIR = ORCHESTRATOR_DIR / "core"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
MEMORY_DIR = ORCHESTRATOR_DIR / "memory"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR, EXECUTION_DIR, MEMORY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.loader import get_agent_config
from contracts import PipelineRunResult, TaskType
from orchestrator_agent import run_with_orchestrator_agent
from router import route_task
from router_agent import route_task_with_agent

try:
    from memory_supervisor import report_pipeline_result
except Exception:  # pragma: no cover
    report_pipeline_result = None


AGENT_CFG = get_agent_config("main_orchestrator")
EXIT_COMMANDS = {"exit", "quit", "q", "退出", "结束"}
HELP_COMMANDS = {"help", "?", "帮助"}
TASK_TEXT_REQUIRED_TYPES = {
    TaskType.FULL_PIPELINE,
    TaskType.STEP4_ONLY,
    TaskType.STEP7_ONLY,
    TaskType.RESUME_FROM_STEP2_3,
    TaskType.RESUME_FROM_STEP4,
    TaskType.RESUME_FROM_STEP5,
    TaskType.RESUME_FROM_STEP6,
}
STEP_ORDER = ["step1", "step2_3", "step4", "step5", "step6", "step7"]


def resolve_model_name(agent_key: str, default: str) -> str:
    env_model = os.environ.get("MODEL_NAME")
    if env_model:
        return env_model
    cfg = get_agent_config(agent_key)
    return str(cfg.get("model_name") or cfg.get("model") or default)


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
        help="Disable Orchestrator-controlled ReMe memory tools and post-run memory recording.",
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
        _apply_task_type_override(task_spec, TaskType(task_type))
        task_spec.intent_summary = f"用户显式指定 task_type={task_type}"

    result = await run_with_orchestrator_agent(
        task_spec=task_spec,
        context="",
        enable_memory_agent=enable_memory_agent,
    )

    used_memory_ids = _memory_ids_from_result(result)
    if (
        enable_memory_agent
        and report_pipeline_result is not None
        and task_spec.task_type != TaskType.MEMORY_ONLY
        and not _memory_recorded_by_orchestrator(result)
    ):
        try:
            await report_pipeline_result(
                {
                    "input_text": user_input,
                    "duration_seconds": 0.0,
                    "orchestrator_summary": result.summary,
                    "retrieved_memory_ids": used_memory_ids,
                    "steps": result.worker_results,
                    "raw_trace_text": json.dumps(result.to_dict(), ensure_ascii=False),
                    "success": result.status.value == "SUCCESS",
                    "failure_signals": result.repair_ticket.validator_errors if result.repair_ticket else [],
                },
                used_memory_ids=used_memory_ids,
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


def _memory_ids_from_result(result: PipelineRunResult) -> list[str]:
    ids: list[str] = []
    for item in result.worker_results:
        payload = item.get("result") if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            continue
        artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
        for memory_id in artifacts.get("used_memory_ids") or []:
            if memory_id not in ids:
                ids.append(memory_id)
    return ids


def _memory_recorded_by_orchestrator(result: PipelineRunResult) -> bool:
    for item in result.worker_results:
        payload = item.get("result") if isinstance(item, dict) else None
        if not isinstance(payload, dict):
            continue
        artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
        if artifacts.get("memory_recorded") is True:
            return True
    return False


def _memory_step_name_for_task(task_type: TaskType) -> str:
    if task_type in {TaskType.STEP1_ONLY, TaskType.RESUME_FROM_RECORDS}:
        return "step1"
    if task_type == TaskType.STEP2_3_ONLY:
        return "step2_3"
    if task_type in {TaskType.STEP4_ONLY, TaskType.RESUME_FROM_STEP2_3}:
        return "step4"
    if task_type in {TaskType.STEP5_ONLY, TaskType.RESUME_FROM_STEP4}:
        return "step5"
    if task_type in {TaskType.STEP6_ONLY, TaskType.RESUME_FROM_STEP5}:
        return "step6"
    if task_type in {TaskType.STEP7_ONLY, TaskType.RESUME_FROM_STEP6}:
        return "step7"
    return task_type.value


def _steps_from(start_step: str) -> list[str]:
    if start_step not in STEP_ORDER:
        return []
    return STEP_ORDER[STEP_ORDER.index(start_step):]


def _apply_task_type_override(spec, task_type: TaskType):
    spec.task_type = task_type
    if task_type == TaskType.FULL_PIPELINE:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "full_pipeline", "step1", list(STEP_ORDER)
    elif task_type == TaskType.STEP1_ONLY:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "only", "step1", ["step1"]
    elif task_type == TaskType.STEP2_3_ONLY:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "only", "step2_3", ["step2_3"]
    elif task_type == TaskType.STEP4_ONLY:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "only", "step4", ["step4"]
    elif task_type == TaskType.STEP5_ONLY:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "only", "step5", ["step5"]
    elif task_type == TaskType.STEP6_ONLY:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "only", "step6", ["step6"]
    elif task_type == TaskType.STEP7_ONLY:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "only", "step7", ["step7"]
    elif task_type == TaskType.RESUME_FROM_STEP2_3:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "from_step_to_end", "step4", _steps_from("step4")
    elif task_type == TaskType.RESUME_FROM_STEP4:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "from_step_to_end", "step5", _steps_from("step5")
    elif task_type == TaskType.RESUME_FROM_STEP5:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "from_step_to_end", "step6", _steps_from("step6")
    elif task_type == TaskType.RESUME_FROM_STEP6:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "from_step_to_end", "step7", ["step7"]
    elif task_type == TaskType.RESUME_FROM_RECORDS:
        spec.execution_scope, spec.start_step, spec.allowed_steps = "only", "step1", ["step1"]
    return spec


async def main_async(args: argparse.Namespace) -> PipelineRunResult:
    if args.json:
        os.environ["STEP1_PROGRESS_ENABLED"] = "false"
        os.environ["MEMORY_PROGRESS_ENABLED"] = "false"
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


def _preview_task_spec_for_task_text(user_input: str, task_type: str = ""):
    spec = route_task(user_input, project_root=PROJECT_ROOT)
    if task_type:
        _apply_task_type_override(spec, TaskType(task_type))
    return spec


def _needs_interactive_task_text(user_input: str, task_type: str = "") -> bool:
    spec = _preview_task_spec_for_task_text(user_input, task_type=task_type)
    if spec.task_type not in TASK_TEXT_REQUIRED_TYPES and not {"step4", "step7"}.intersection(set(spec.allowed_steps)):
        return False
    return not str(spec.entry_artifacts.get("task_text") or "").strip()


def _append_interactive_task_text(user_input: str, task_text: str) -> str:
    return f"{user_input.strip()}\n本次任务裁剪的目标是{task_text.strip()}"


async def interactive_loop(args: argparse.Namespace) -> None:
    if args.json:
        os.environ["STEP1_PROGRESS_ENABLED"] = "false"
        os.environ["MEMORY_PROGRESS_ENABLED"] = "false"

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
        if _needs_interactive_task_text(effective_input, task_type=args.task_type):
            try:
                task_text = input("[MultiAgent] 请输入本次任务裁剪的目标: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[MultiAgent] 已退出。")
                return
            if not task_text:
                print("[MultiAgent] 缺少任务裁剪目标，本次任务未启动。")
                continue
            effective_input = _append_interactive_task_text(effective_input, task_text)
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
