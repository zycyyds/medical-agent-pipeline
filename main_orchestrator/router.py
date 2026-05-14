from __future__ import annotations

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
        path = _resolve_path(raw.split()[-1], root)
        return TaskSpec(
            task_type=TaskType.REPAIR_TASK,
            entry_artifacts={"target": str(path)},
            resume_from_step="repair",
            intent_summary="用户显式请求修复已有产物。",
        )

    path = _resolve_path(raw, root)

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

    if "step1" in lower or "只跑step1" in lower:
        return TaskSpec(
            task_type=TaskType.STEP1_ONLY,
            entry_artifacts={"input_path": str(path)},
            resume_from_step="step1",
            intent_summary="用户请求只运行 Step1。",
        )

    return TaskSpec(
        task_type=TaskType.FULL_PIPELINE,
        entry_artifacts={"input_path": str(path)},
        resume_from_step="step1",
        intent_summary="默认按原始输入目录启动完整流水线。",
    )
