from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from autonomous_pipeline.agent_factory import create_react_agent
from autonomous_pipeline.config import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT

from . import tools
from .prompt import SYSTEM_PROMPT, build_task_prompt

STEP_TITLE = "Planner 主编排"


def create_agent():
    return create_react_agent("planner", SYSTEM_PROMPT, tools.TOOL_FUNCTIONS, max_iters=24)


def _extract_existing_paths(text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"(?:~|/)[^\s，,；;]+", text):
        raw = match.group(0).strip().strip("\"'")
        if raw not in paths:
            paths.append(raw)
    return paths


def _extract_after_label(text: str, labels: tuple[str, ...]) -> str:
    joined = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{joined})\s*[:：]\s*([^，,；;\n]+)", text)
    return match.group(1).strip() if match else ""


def _infer_default_gold_dir(task_text: str) -> str:
    lower = task_text.lower()
    if "眩晕" in task_text or "前庭" in task_text or "vestibular" in lower:
        return str(PROJECT_ROOT / "gold" / "vestibular")
    return str(PROJECT_ROOT / "gold" / "liver")


def parse_natural_request(text: str) -> dict[str, str | int]:
    """Parse a natural-language terminal request for the planner."""
    paths = _extract_existing_paths(text)
    explicit_gold = _extract_after_label(text, ("gold", "gold-dir", "金标准", "验证目录"))
    task_text = _extract_after_label(text, ("任务", "目标", "task"))
    if not task_text:
        # Drop obvious path fragments and command words, leaving a short task hint if present.
        cleaned = text
        for path in paths:
            cleaned = cleaned.replace(path, " ")
        cleaned = re.sub(r"\b(process|run|处理|运行|数据集|目录)\b", " ", cleaned, flags=re.IGNORECASE)
        task_text = re.sub(r"\s+", " ", cleaned).strip(" ，,；;")
    max_rounds = 1
    round_match = re.search(r"(?:max[-_ ]?rounds?|轮数|最多)\s*[:：]?\s*(\d+)", text, flags=re.IGNORECASE)
    if not round_match:
        round_match = re.search(r"(\d+)\s*轮", text)
    if round_match:
        max_rounds = max(1, int(round_match.group(1)))
    output_root = _extract_after_label(text, ("输出", "output", "output-root")) or str(DEFAULT_OUTPUT_ROOT)
    input_path = paths[0] if paths else ""
    gold_dir = explicit_gold or (paths[1] if len(paths) > 1 else "")
    return {
        "input_path": input_path,
        "task_text": task_text,
        "gold_dir": gold_dir,
        "output_root": output_root,
        "max_rounds": max_rounds,
    }


def _prompt_missing(value: str, prompt: str, default: str = "") -> str:
    if value:
        return value
    suffix = f" [{default}]" if default else ""
    entered = input(f"{prompt}{suffix}: ").strip()
    return entered or default


def _gold_has_visible_jsonl(path: str) -> bool:
    visible = Path(path).expanduser() / "visible"
    return visible.is_dir() and any(visible.glob("*.jsonl"))


def _resolve_interactive_request(text: str, require_gold_visible: bool = True) -> dict[str, str | int]:
    parsed = parse_natural_request(text)
    input_path = _prompt_missing(str(parsed["input_path"]), "原始数据路径")
    task_text = _prompt_missing(str(parsed["task_text"]), "任务目标")
    default_gold = _infer_default_gold_dir(task_text)
    gold_dir = str(parsed["gold_dir"]) or default_gold
    if require_gold_visible and not _gold_has_visible_jsonl(gold_dir):
        print(f"[Planner] 默认金标准目录没有 visible/*.jsonl: {gold_dir}", file=sys.stderr)
        gold_dir = _prompt_missing("", "金标准目录", default_gold)
    return {
        "input_path": input_path,
        "task_text": task_text,
        "gold_dir": gold_dir,
        "output_root": str(parsed["output_root"] or DEFAULT_OUTPUT_ROOT),
        "max_rounds": int(parsed["max_rounds"] or 1),
    }


def _dry_run_payload(request: dict[str, str | int]) -> dict[str, str | int]:
    from .prompt import build_task_prompt as build_planner_prompt
    from autonomous_pipeline.task_analysis.prompt import build_task_prompt as build_analysis_prompt

    output_root = Path(str(request["output_root"])).expanduser()
    run_root = output_root / "planner_runs" / "interactive_dry_run"
    round_root = run_root / "rounds" / "round_01"
    task_spec = run_root / "task_spec.json"
    print("===== task_analysis =====")
    print(build_analysis_prompt(str(request["input_path"]), str(run_root), str(request["task_text"]), str(request["gold_dir"])))
    print("\n===== planner round 01 =====")
    print(build_planner_prompt(str(request["input_path"]), str(round_root), str(request["task_text"]), str(task_spec)))
    return request


def run_once(text: str, dry_run: bool = False) -> dict:
    from .loop import run_planner_loop_sync

    request = _resolve_interactive_request(text, require_gold_visible=not dry_run)
    if dry_run:
        _dry_run_payload(request)
        return {"status": "DRY_RUN", **request}
    result = run_planner_loop_sync(
        input_path=str(request["input_path"]),
        task_text=str(request["task_text"]),
        gold_dir=str(request["gold_dir"]),
        output_root=str(request["output_root"]),
        max_rounds=int(request["max_rounds"]),
    )
    return result


def interactive_main(dry_run: bool = False) -> None:
    print("PlannerAgent 已启动。直接输入自然语言请求，例如：处理 /path/to/mimic-mini，任务：肝病诊断。输入 exit 退出。")
    while True:
        try:
            text = input("Planner> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            return
        try:
            result = run_once(text, dry_run=dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[Planner failed] {type(exc).__name__}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive PlannerAgent for the autonomous medical pipeline")
    parser.add_argument("--once", default="", help="Run one natural-language request and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling models")
    args = parser.parse_args()
    if args.once:
        result = run_once(args.once, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    interactive_main(dry_run=args.dry_run)


__all__ = ["STEP_TITLE", "create_agent", "build_task_prompt"]


if __name__ == "__main__":
    main()
