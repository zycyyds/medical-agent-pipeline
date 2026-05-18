from __future__ import annotations

import re
from pathlib import Path

from contracts import TaskSpec, TaskType


def _resolve_path(raw_path: str | Path, project_root: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _has_adjacent_selection_report(path: Path) -> bool:
    return (path.parent / "selection_report.json").is_file()


def _path_contains(path: Path, *parts: str) -> bool:
    normalized = [item.lower() for item in path.parts]
    return all(part.lower() in normalized for part in parts)


def _mentions_step1_only(lower: str) -> bool:
    compact = re.sub(r"[\s_\-—–]+", "", lower)
    mentions_step1 = (
        "step1" in compact
        or "第1步" in compact
        or "第一步" in compact
        or "步骤1" in compact
    )
    if not mentions_step1:
        return False
    return any(token in compact for token in ("只跑", "只运行", "只执行", "仅跑", "仅运行", "仅执行", "only"))


def _mentions_step4(lower: str) -> bool:
    compact = re.sub(r"[\s_\-—–]+", "", lower)
    return (
        "step4" in compact
        or "agent4" in compact
        or "第4步" in compact
        or "第四步" in compact
        or "步骤4" in compact
    )


def _trim_to_existing_path(token: str, project_root: Path) -> str | None:
    candidate_text = token.strip().strip("\"'“”‘’")
    if not candidate_text:
        return None

    if "/" in candidate_text:
        slash_index = candidate_text.find("/")
        if slash_index > 0:
            candidate_text = candidate_text[slash_index:]

    path = _resolve_path(candidate_text, project_root)
    if path.exists():
        return candidate_text

    for end in range(len(candidate_text) - 1, 0, -1):
        probe = candidate_text[:end].rstrip("/\\")
        if not probe:
            continue
        if _resolve_path(probe, project_root).exists():
            return probe
    return None


def _extract_path_like_text(raw: str, project_root: Path) -> str:
    cleaned = raw.strip().strip("\"'")
    if not cleaned:
        return cleaned

    direct = _resolve_path(cleaned, project_root)
    if direct.exists():
        return cleaned
    if cleaned.endswith((".json", ".csv", ".xlsx", ".xls")) and not re.search(r"[\s，。；;：:]", cleaned):
        return cleaned

    absolute_candidates = re.findall(r"(/[^\s，。；;：:]+)", cleaned)
    for token in absolute_candidates:
        existing = _trim_to_existing_path(token, project_root)
        if existing:
            return existing

    tokens = [token.strip().strip("\"'") for token in re.split(r"[\s，。；;：:]+", cleaned) if token.strip()]
    for token in reversed(tokens):
        if not token:
            continue
        existing = _trim_to_existing_path(token, project_root)
        if existing:
            return existing
        if "/" in token or "\\" in token or token.endswith((".json", ".csv", ".xlsx", ".xls", ".tsv")):
            return token
        candidate = _resolve_path(token, project_root)
        if candidate.exists():
            return token
    return cleaned


def route_task(user_input: str, project_root: str | Path | None = None) -> TaskSpec:
    root = Path(project_root or Path.cwd()).resolve()
    raw = str(user_input or "").strip()
    lower = raw.lower()

    if not raw:
        return TaskSpec(
            task_type=TaskType.FULL_PIPELINE,
            intent_summary="未提供输入，默认等待原始数据目录。",
            confidence=0.0,
        )

    if "memory" in lower or "playbook" in lower:
        return TaskSpec(
            task_type=TaskType.MEMORY_ONLY,
            entry_artifacts={"raw_input": raw},
            intent_summary="用户请求 memory/playbook 相关任务。",
        )

    if any(token in lower for token in ("修复", "repair", "失败", "报错")):
        path = _resolve_path(_extract_path_like_text(raw, root), root)
        return TaskSpec(
            task_type=TaskType.REPAIR_TASK,
            entry_artifacts={"target": str(path)},
            resume_from_step="repair",
            intent_summary="用户显式请求修复已有产物。",
        )

    path_text = _extract_path_like_text(raw, root)
    path = _resolve_path(path_text, root)

    if path.name == "records.json" or _path_contains(path, "reorganized_output", "_meta"):
        return TaskSpec(
            task_type=TaskType.RESUME_FROM_RECORDS,
            entry_artifacts={"records_path": str(path)},
            resume_from_step="step1_codegen",
            intent_summary="检测到 Step1 records.json，直接从脚本生成与验收续跑。",
        )

    if path.name == "input.csv" and _path_contains(path, "step2_3_results", "next_input"):
        return TaskSpec(
            task_type=TaskType.RESUME_FROM_STEP2_3,
            entry_artifacts={"input_csv": str(path)},
            resume_from_step="step4",
            intent_summary="检测到 Step2_3 next_input/input.csv，从 Step4 开始续跑。",
        )

    if path.name == "filtered.csv" and _path_contains(path, "step4_results", "next_input"):
        if _has_adjacent_selection_report(path):
            return TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP4,
                entry_artifacts={
                    "filtered_csv": str(path),
                    "selection_report": str(path.parent / "selection_report.json"),
                },
                resume_from_step="step5",
                intent_summary="检测到 Step4 filtered.csv 和 selection_report.json，从 Step5 开始续跑。",
            )
        return TaskSpec(
            task_type=TaskType.REPAIR_TASK,
            entry_artifacts={"filtered_csv": str(path), "missing": str(path.parent / "selection_report.json")},
            resume_from_step="step5",
            intent_summary="Step4 filtered.csv 缺少 selection_report.json，进入修复模式。",
            confidence=0.8,
        )

    if path.name == "filtered.csv" and _path_contains(path, "step5_results", "next_input"):
        if _has_adjacent_selection_report(path):
            return TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP5,
                entry_artifacts={
                    "filtered_csv": str(path),
                    "selection_report": str(path.parent / "selection_report.json"),
                },
                resume_from_step="step6",
                intent_summary="检测到 Step5 filtered.csv 和 selection_report.json，从 Step6/7 开始续跑。",
            )
        return TaskSpec(
            task_type=TaskType.REPAIR_TASK,
            entry_artifacts={"filtered_csv": str(path), "missing": str(path.parent / "selection_report.json")},
            resume_from_step="step6",
            intent_summary="Step5 filtered.csv 缺少 selection_report.json，进入修复模式。",
            confidence=0.8,
        )

    if _mentions_step1_only(lower):
        return TaskSpec(
            task_type=TaskType.STEP1_ONLY,
            entry_artifacts={"input_path": str(path)},
            resume_from_step="step1",
            intent_summary="用户请求只运行 Step1。",
        )

    if _mentions_step4(lower):
        return TaskSpec(
            task_type=TaskType.RESUME_FROM_STEP2_3,
            entry_artifacts={"input_csv": str(path)},
            resume_from_step="step4",
            intent_summary="用户请求运行 Step4 任务裁剪。",
        )

    return TaskSpec(
        task_type=TaskType.FULL_PIPELINE,
        entry_artifacts={"input_path": str(path)},
        resume_from_step="step1",
        intent_summary="默认按原始输入目录启动完整流水线。",
    )
