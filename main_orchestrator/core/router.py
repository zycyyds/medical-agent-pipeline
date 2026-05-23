from __future__ import annotations

import re
from pathlib import Path

from contracts import TaskSpec, TaskType

STEP_ORDER = ["step1", "step2_3", "step4", "step5", "step6", "step7"]
ONLY_TASK_TYPES = {
    "step1": TaskType.STEP1_ONLY,
    "step2_3": TaskType.STEP2_3_ONLY,
    "step4": TaskType.STEP4_ONLY,
    "step5": TaskType.STEP5_ONLY,
    "step6": TaskType.STEP6_ONLY,
    "step7": TaskType.STEP7_ONLY,
}
FROM_STEP_TASK_TYPES = {
    "step1": TaskType.FULL_PIPELINE,
    "step2_3": TaskType.RESUME_FROM_STEP2_3,
    "step4": TaskType.RESUME_FROM_STEP4,
    "step5": TaskType.RESUME_FROM_STEP5,
    "step6": TaskType.RESUME_FROM_STEP6,
    "step7": TaskType.STEP7_ONLY,
}


def _steps_from(start_step: str) -> list[str]:
    if start_step not in STEP_ORDER:
        return []
    return STEP_ORDER[STEP_ORDER.index(start_step):]


def _execution_scope_markers(raw_lower: str) -> tuple[bool, bool]:
    compact = re.sub(r"[\s_\-—–]+", "", raw_lower)
    is_only = any(token in compact for token in ("只跑", "只运行", "只执行", "仅跑", "仅运行", "仅执行", "only"))
    to_end = any(
        token in compact
        for token in (
            "到最后",
            "跑到最后",
            "运行到最后",
            "执行到最后",
            "继续到最后",
            "一直到最后",
            "toend",
        )
    )
    return is_only, to_end


def _set_execution_boundary(
    spec: TaskSpec,
    scope: str,
    start_step: str,
    allowed_steps: list[str],
) -> TaskSpec:
    spec.execution_scope = scope
    spec.start_step = start_step
    spec.allowed_steps = list(allowed_steps)
    return spec


def _default_boundary(spec: TaskSpec) -> tuple[str, str, list[str]]:
    task_type = spec.task_type
    if task_type == TaskType.FULL_PIPELINE:
        return "full_pipeline", "step1", list(STEP_ORDER)
    if task_type == TaskType.STEP1_ONLY:
        return "only", "step1", ["step1"]
    if task_type == TaskType.STEP2_3_ONLY:
        return "only", "step2_3", ["step2_3"]
    if task_type == TaskType.STEP4_ONLY:
        return "only", "step4", ["step4"]
    if task_type == TaskType.STEP5_ONLY:
        return "only", "step5", ["step5"]
    if task_type == TaskType.STEP6_ONLY:
        return "only", "step6", ["step6"]
    if task_type == TaskType.STEP7_ONLY:
        return "only", "step7", ["step7"]
    if task_type == TaskType.RESUME_FROM_STEP2_3:
        return "from_step_to_end", "step4", _steps_from("step4")
    if task_type == TaskType.RESUME_FROM_STEP4:
        return "from_step_to_end", "step5", _steps_from("step5")
    if task_type == TaskType.RESUME_FROM_STEP5:
        return "from_step_to_end", "step6", _steps_from("step6")
    if task_type == TaskType.RESUME_FROM_STEP6:
        return "from_step_to_end", "step7", ["step7"]
    if task_type == TaskType.RESUME_FROM_RECORDS:
        return "only", "step1", ["step1"]
    return "", "", []


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


def _mentions_step5(lower: str) -> bool:
    compact = re.sub(r"[\s_\-—–]+", "", lower)
    return (
        "step5" in compact
        or "agent5" in compact
        or "第5步" in compact
        or "第五步" in compact
        or "步骤5" in compact
        or "数据清洗" in compact
        or "质量修复" in compact
        or "dataquality" in compact
    )


def _mentions_step6(lower: str) -> bool:
    compact = re.sub(r"[\s_\-—–]+", "", lower)
    return (
        "step6" in compact
        or "agent6" in compact
        or "第6步" in compact
        or "第六步" in compact
        or "步骤6" in compact
        or "一致性验证" in compact
        or "一致性检查" in compact
    )


def _mentions_step7(lower: str) -> bool:
    compact = re.sub(r"[\s_\-—–]+", "", lower)
    return (
        "step7" in compact
        or "agent7" in compact
        or "第7步" in compact
        or "第七步" in compact
        or "步骤7" in compact
        or "ml训练" in compact
        or "机器学习" in compact
        or "训练数据集" in compact
    )


def _mentions_step2_3(lower: str) -> bool:
    compact = re.sub(r"[\s_\-—–]+", "", lower)
    return (
        "step2" in compact
        or "step3" in compact
        or "step23" in compact
        or "step2-3" in lower
        or "agent2-3" in lower
        or "agent23" in compact
        or "第2步" in compact
        or "第二步" in compact
        or "第3步" in compact
        or "第三步" in compact
        or "步骤2" in compact
        or "步骤3" in compact
    )


def _extract_task_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    patterns = (
        r"(?:本次)?任务裁剪(?:的)?目标(?:是|为|:|：)\s*(.+)",
        r"(?:本次)?任务裁剪(?:的)?任务(?:是|为|:|：)\s*(.+)",
        r"(?:本次)?任务裁剪以\s*(.+?)\s*进行裁剪",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        task_text = str(match.group(1) or "").strip().strip("\"'“”‘’")
        task_text = re.split(r"[\n\r。；;]", task_text, maxsplit=1)[0].strip()
        task_text = re.split(r"[，,]\s*(?:处理|输入|文件|路径|数据集|然后|并且)", task_text, maxsplit=1)[0].strip()
        return task_text.strip(" ，,。；;：:")
    return ""


def _finalize_spec(
    spec: TaskSpec,
    task_text: str,
    scope: str | None = None,
    start_step: str | None = None,
    allowed_steps: list[str] | None = None,
) -> TaskSpec:
    if scope is None or start_step is None or allowed_steps is None:
        scope, start_step, allowed_steps = _default_boundary(spec)
    _set_execution_boundary(spec, scope, start_step, allowed_steps)
    task = str(task_text or "").strip()
    if not task:
        return spec
    spec.entry_artifacts["task_text"] = task
    if "任务裁剪目标" not in spec.intent_summary:
        spec.intent_summary = f"{spec.intent_summary} 任务裁剪目标：{task}".strip()
    return spec


def _attach_task_text(spec: TaskSpec, task_text: str) -> TaskSpec:
    return _finalize_spec(spec, task_text)


def _infer_step23_mode(raw: str) -> str:
    compact = re.sub(r"[\s_\-—–]+", "", raw.lower())
    directory_tokens = (
        "mimic",
        "目录处理",
        "不需要ocr",
        "不要ocr",
        "跳过ocr",
        "结构化",
        "notes",
        "structured",
        "reporttext",
    )
    ocr_tokens = (
        "rawdata",
        "ocr回填",
        "图片ocr",
        "填回表格",
        "ocr提取",
        "回填",
    )
    if any(token in compact for token in directory_tokens):
        return "directory"
    if any(token in compact for token in ocr_tokens):
        return "ocr_fill"
    return "auto"


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


def _extract_existing_paths(raw: str, project_root: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    def add(candidate: str | None) -> None:
        if not candidate:
            return
        path = _resolve_path(candidate, project_root)
        if not path.exists():
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        paths.append(path)

    for token in re.findall(r"(/[^\s，。；;]+)", raw):
        add(_trim_to_existing_path(token, project_root))

    for token in [item.strip().strip("\"'“”‘’") for item in re.split(r"[\s，。；;]+", raw) if item.strip()]:
        if "/" in token or "\\" in token or token.endswith((".json", ".csv", ".xlsx", ".xls", ".tsv")):
            add(_trim_to_existing_path(token, project_root) or token)
    return paths


def _step7_artifacts_from_paths(paths: list[Path], raw: str) -> dict[str, str]:
    artifacts: dict[str, str] = {"raw_input": raw}
    for path in paths:
        if path.name == "selection_report.json":
            artifacts["selection_report"] = str(path)
        elif path.name.startswith("passed_patients") and path.suffix == ".json":
            artifacts["passed_patients_json"] = str(path)
        elif path.name == "filtered.csv" or path.suffix == ".csv":
            artifacts["filtered_csv"] = str(path)
    return artifacts


def route_task(user_input: str, project_root: str | Path | None = None) -> TaskSpec:
    root = Path(project_root or Path.cwd()).resolve()
    raw = str(user_input or "").strip()
    lower = raw.lower()
    task_text = _extract_task_text(raw)
    is_only, to_end = _execution_scope_markers(lower)

    if not raw:
        return _attach_task_text(TaskSpec(
            task_type=TaskType.FULL_PIPELINE,
            intent_summary="未提供输入，默认等待原始数据目录。",
            confidence=0.0,
        ), task_text)

    if "memory" in lower or "playbook" in lower:
        return _attach_task_text(TaskSpec(
            task_type=TaskType.MEMORY_ONLY,
            entry_artifacts={"raw_input": raw},
            intent_summary="用户请求 memory/playbook 相关任务。",
        ), task_text)

    if any(token in lower for token in ("修复", "repair", "失败", "报错")) and not _mentions_step5(lower):
        path = _resolve_path(_extract_path_like_text(raw, root), root)
        return _attach_task_text(TaskSpec(
            task_type=TaskType.REPAIR_TASK,
            entry_artifacts={"target": str(path)},
            resume_from_step="repair",
            intent_summary="用户显式请求修复已有产物。",
        ), task_text)

    path_text = _extract_path_like_text(raw, root)
    path = _resolve_path(path_text, root)
    existing_paths = _extract_existing_paths(raw, root)

    if _mentions_step7(lower) and len(existing_paths) > 1:
        artifacts = _step7_artifacts_from_paths(existing_paths, raw)
        if any(key in artifacts for key in ("filtered_csv", "selection_report", "passed_patients_json")):
            return _finalize_spec(TaskSpec(
                task_type=TaskType.STEP7_ONLY if is_only else TaskType.RESUME_FROM_STEP6,
                entry_artifacts=artifacts,
                resume_from_step="step7",
                intent_summary="用户请求运行 Step7 ML 训练数据生成。",
            ), task_text, "only" if is_only else "from_step_to_end", "step7", ["step7"])

    if path.name == "records.json" or _path_contains(path, "reorganized_output", "_meta"):
        return _attach_task_text(TaskSpec(
            task_type=TaskType.RESUME_FROM_RECORDS,
            entry_artifacts={"records_path": str(path)},
            resume_from_step="step1_codegen",
            intent_summary="检测到 Step1 records.json，直接从脚本生成与验收续跑。",
        ), task_text)

    if path.name == "input.csv" and _path_contains(path, "step2_3_results", "next_input"):
        return _attach_task_text(TaskSpec(
            task_type=TaskType.RESUME_FROM_STEP2_3,
            entry_artifacts={"input_csv": str(path)},
            resume_from_step="step4",
            intent_summary="检测到 Step2_3 next_input/input.csv，从 Step4 开始续跑。",
        ), task_text)

    if path.name == "filtered.csv" and _path_contains(path, "step4_results", "next_input"):
        if _mentions_step5(lower) and is_only:
            return _finalize_spec(TaskSpec(
                task_type=TaskType.STEP5_ONLY,
                entry_artifacts={
                    "filtered_csv": str(path),
                    "selection_report": str(path.parent / "selection_report.json") if _has_adjacent_selection_report(path) else "",
                },
                resume_from_step="step5",
                intent_summary="用户请求只运行 Step5 数据清洗。",
            ), task_text, "only", "step5", ["step5"])
        if _mentions_step5(lower) and to_end:
            return _finalize_spec(TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP5,
                entry_artifacts={
                    "filtered_csv": str(path),
                    "selection_report": str(path.parent / "selection_report.json") if _has_adjacent_selection_report(path) else "",
                },
                resume_from_step="step5",
                intent_summary="用户请求从 Step5 继续运行到 Step7。",
            ), task_text, "from_step_to_end", "step5", _steps_from("step5"))
        return _attach_task_text(TaskSpec(
            task_type=TaskType.RESUME_FROM_STEP4,
            entry_artifacts={
                "filtered_csv": str(path),
                "selection_report": str(path.parent / "selection_report.json") if _has_adjacent_selection_report(path) else "",
            },
            resume_from_step="step5",
            intent_summary="检测到 Step4 filtered.csv，从 Step5 数据清洗开始续跑。",
        ), task_text)

    if path.name == "filtered.csv" and _path_contains(path, "step5_results", "next_input"):
        if _has_adjacent_selection_report(path):
            if _mentions_step6(lower) and not _mentions_step7(lower):
                if to_end:
                    return _finalize_spec(TaskSpec(
                        task_type=TaskType.RESUME_FROM_STEP6,
                        entry_artifacts={
                            "filtered_csv": str(path),
                            "selection_report": str(path.parent / "selection_report.json"),
                        },
                        resume_from_step="step6",
                        intent_summary="用户请求从 Step6 继续运行到 Step7。",
                    ), task_text, "from_step_to_end", "step6", _steps_from("step6"))
                return _finalize_spec(TaskSpec(
                    task_type=TaskType.STEP6_ONLY,
                    entry_artifacts={
                        "filtered_csv": str(path),
                        "selection_report": str(path.parent / "selection_report.json"),
                    },
                    resume_from_step="step6",
                    intent_summary="用户请求只运行 Step6 一致性验证。",
                ), task_text, "only", "step6", ["step6"])
            if _mentions_step7(lower):
                return _finalize_spec(TaskSpec(
                    task_type=TaskType.STEP7_ONLY if is_only else TaskType.RESUME_FROM_STEP6,
                    entry_artifacts={
                        "filtered_csv": str(path),
                        "selection_report": str(path.parent / "selection_report.json"),
                    },
                    resume_from_step="step7",
                    intent_summary="用户请求运行 Step7 ML 训练数据生成。",
                ), task_text, "only" if is_only else "from_step_to_end", "step7", ["step7"])
            return _attach_task_text(TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP5,
                entry_artifacts={
                    "filtered_csv": str(path),
                    "selection_report": str(path.parent / "selection_report.json"),
                },
                resume_from_step="step6",
                intent_summary="检测到 Step5 filtered.csv 和 selection_report.json，从 Step6/7 开始续跑。",
            ), task_text)
        return _attach_task_text(TaskSpec(
            task_type=TaskType.REPAIR_TASK,
            entry_artifacts={"filtered_csv": str(path), "missing": str(path.parent / "selection_report.json")},
            resume_from_step="step6",
            intent_summary="Step5 filtered.csv 缺少 selection_report.json，进入修复模式。",
            confidence=0.8,
        ), task_text)

    if path.name.startswith("passed_patients_") and path.suffix == ".json":
        return _finalize_spec(TaskSpec(
            task_type=TaskType.STEP7_ONLY if is_only or _mentions_step7(lower) else TaskType.RESUME_FROM_STEP6,
            entry_artifacts={"passed_patients_json": str(path), "raw_input": raw},
            resume_from_step="step7",
            intent_summary="检测到 Step6 passed_patients.json，从 Step7 开始续跑。",
        ), task_text, "only" if is_only or _mentions_step7(lower) else "from_step_to_end", "step7", ["step7"])

    if _mentions_step7(lower):
        artifacts: dict[str, str] = {"raw_input": raw}
        if path.name.endswith("selection_report.json"):
            artifacts["selection_report"] = str(path)
        elif path.name.startswith("passed_patients_") and path.suffix == ".json":
            artifacts["passed_patients_json"] = str(path)
        elif path.suffix == ".csv":
            artifacts["filtered_csv"] = str(path)
        return _finalize_spec(TaskSpec(
            task_type=TaskType.STEP7_ONLY if is_only else TaskType.RESUME_FROM_STEP6,
            entry_artifacts=artifacts,
            resume_from_step="step7",
            intent_summary="用户请求运行 Step7 ML 训练数据生成。",
        ), task_text, "only" if is_only else "from_step_to_end", "step7", ["step7"])

    if _mentions_step6(lower):
        if to_end:
            return _finalize_spec(TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP6,
                entry_artifacts={"filtered_csv": str(path), "raw_input": raw},
                resume_from_step="step6",
                intent_summary="用户请求从 Step6 继续运行到 Step7。",
            ), task_text, "from_step_to_end", "step6", _steps_from("step6"))
        return _finalize_spec(TaskSpec(
            task_type=TaskType.STEP6_ONLY,
            entry_artifacts={"filtered_csv": str(path), "raw_input": raw},
            resume_from_step="step6",
            intent_summary="用户请求运行 Step6 一致性验证。",
        ), task_text, "only", "step6", ["step6"])

    if _mentions_step4(lower):
        if to_end:
            return _finalize_spec(TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP4,
                entry_artifacts={"input_csv": str(path)},
                resume_from_step="step4",
                intent_summary="用户请求从 Step4 继续运行到 Step7。",
            ), task_text, "from_step_to_end", "step4", _steps_from("step4"))
        return _finalize_spec(TaskSpec(
            task_type=TaskType.STEP4_ONLY,
            entry_artifacts={"input_csv": str(path)},
            resume_from_step="step4",
            intent_summary="用户请求运行 Step4 任务裁剪。",
        ), task_text, "only", "step4", ["step4"])

    if _mentions_step5(lower):
        if to_end:
            return _finalize_spec(TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP5,
                entry_artifacts={"filtered_csv": str(path), "raw_input": raw},
                resume_from_step="step5",
                intent_summary="用户请求从 Step5 继续运行到 Step7。",
            ), task_text, "from_step_to_end", "step5", _steps_from("step5"))
        return _finalize_spec(TaskSpec(
            task_type=TaskType.STEP5_ONLY,
            entry_artifacts={"filtered_csv": str(path), "raw_input": raw},
            resume_from_step="step5",
            intent_summary="用户请求运行 Step5 数据清洗。",
        ), task_text, "only", "step5", ["step5"])

    if _mentions_step2_3(lower):
        if to_end:
            return _finalize_spec(TaskSpec(
                task_type=TaskType.RESUME_FROM_STEP2_3,
                entry_artifacts={"input_path": str(path), "step23_mode": _infer_step23_mode(raw), "raw_input": raw},
                resume_from_step="step2_3",
                intent_summary="用户请求从 Step2-3 继续运行到 Step7。",
            ), task_text, "from_step_to_end", "step2_3", _steps_from("step2_3"))
        return _finalize_spec(TaskSpec(
            task_type=TaskType.STEP2_3_ONLY,
            entry_artifacts={"input_path": str(path), "step23_mode": _infer_step23_mode(raw), "raw_input": raw},
            resume_from_step="step2_3",
            intent_summary="用户请求只运行 Step2-3 医疗结构化与回填。",
        ), task_text, "only", "step2_3", ["step2_3"])

    if _mentions_step1_only(lower):
        return _attach_task_text(TaskSpec(
            task_type=TaskType.STEP1_ONLY,
            entry_artifacts={"input_path": str(path)},
            resume_from_step="step1",
            intent_summary="用户请求只运行 Step1。",
        ), task_text)

    return _attach_task_text(TaskSpec(
        task_type=TaskType.FULL_PIPELINE,
        entry_artifacts={"input_path": str(path), "step23_mode": _infer_step23_mode(raw), "raw_input": raw},
        resume_from_step="step1",
        intent_summary="默认按原始输入目录启动完整流水线。",
    ), task_text)
