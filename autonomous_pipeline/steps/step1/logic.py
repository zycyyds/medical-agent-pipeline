from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from autonomous_pipeline.table_io import ensure_dir, read_rows, write_json, write_rows
from .doclayout import classify_visual_records

TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}
VISUAL_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".xml"}
SUPPORTED_SUFFIXES = TABULAR_SUFFIXES | VISUAL_SUFFIXES | TEXT_SUFFIXES
EXCLUDED_FILE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
EXCLUDED_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache", "program", "tmp"}
PATIENT_DIR_PATTERN = re.compile(r"p?(\d{5,})", re.IGNORECASE)
PATIENT_TEXT_PATTERN = re.compile(r"patient[-_ ]?(\d{3,})", re.IGNORECASE)


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def get_step1_max_workers() -> int:
    fallback = _read_positive_int_env("STEP1_WORKERS", 4)
    return _read_positive_int_env("STEP1_MAX_WORKERS", fallback)


def get_step1_reorganize_max_workers() -> int:
    return _read_positive_int_env("STEP1_REORGANIZE_MAX_WORKERS", get_step1_max_workers())


def _worker_count(total: int, configured: int | None = None) -> int:
    if total <= 0:
        return 0
    configured = get_step1_max_workers() if configured is None else configured
    return max(1, min(configured, total))


def _progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        return
    percent = current / total * 100
    print(f"[Step1][Progress] {label}: {current}/{total} ({percent:.1f}%)", file=sys.stderr, flush=True)


def _step_root(output_root: str | Path) -> Path:
    return Path(output_root) / "step1_results"


def _meta_dir(output_root: str | Path) -> Path:
    return ensure_dir(_step_root(output_root) / "_meta")


def append_tool_trace(output_root: str | Path, tool: str, arguments: dict[str, Any], payload: dict[str, Any], started_at: float) -> str:
    path = _meta_dir(output_root) / "tool_trace.json"
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, list):
                existing = parsed
        except Exception:
            existing = []
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else {}
    artifact_summary = {}
    if isinstance(artifacts, dict):
        for key, value in artifacts.items():
            if key == "tasks":
                artifact_summary[key] = f"{len(value)} tasks" if isinstance(value, list) else "tasks"
            elif key in {"records", "patches"}:
                artifact_summary[key] = f"{len(value)} items" if isinstance(value, list) else "items"
            else:
                artifact_summary[key] = value
    existing.append(
        {
            "tool": tool,
            "arguments": arguments,
            "status": payload.get("status") if isinstance(payload, dict) else "UNKNOWN",
            "artifacts": artifact_summary,
            "issues": payload.get("issues", []) if isinstance(payload, dict) else [],
            "duration_ms": int((time.time() - started_at) * 1000),
        }
    )
    write_json(path, existing)
    return str(path)


def _is_under_program_output(path: Path) -> bool:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "program" and index + 1 < len(parts) and parts[index + 1] == "output":
            return True
    return False


def _is_recordable_input_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in EXCLUDED_FILE_NAMES or path.name.startswith("."):
        return False
    if path.suffix.lower() in {".tmp", ".temp", ".bak", ".swp"}:
        return False
    if any(part.startswith(".") for part in path.parts):
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    return not _is_under_program_output(path)


def _is_supported_input_file(path: Path) -> bool:
    return _is_recordable_input_file(path) and path.suffix.lower() in SUPPORTED_SUFFIXES


def safe_patient_id(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    path_match = PATIENT_DIR_PATTERN.fullmatch(text)
    if path_match:
        return path_match.group(1)
    patient_match = PATIENT_TEXT_PATTERN.search(text)
    if patient_match:
        return patient_match.group(1)
    numeric_match = re.search(r"p?(\d{5,})", text, re.IGNORECASE)
    if numeric_match:
        return numeric_match.group(1)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return cleaned or fallback


def extract_patient_id(source_path: str, content_text: str = "") -> str | None:
    path = Path(source_path or "")
    for part in path.parts:
        match = PATIENT_DIR_PATTERN.fullmatch(str(part))
        if match:
            return match.group(1)
    for text in (source_path, path.name, content_text):
        patient_match = PATIENT_TEXT_PATTERN.search(text or "")
        if patient_match:
            return patient_match.group(1)
        numeric_match = re.search(r"p?(\d{5,})", text or "", re.IGNORECASE)
        if numeric_match:
            return numeric_match.group(1)
    return None


def _file_task(path: Path, base: Path) -> dict[str, Any]:
    relative_path = str(path.relative_to(base)) if path != base else path.name
    return {
        "source_path": str(path),
        "source_name": path.name,
        "relative_path": relative_path,
        "suffix": path.suffix.lower(),
        "patient_id": extract_patient_id(str(path), ""),
        "processable": _is_supported_input_file(path),
    }


def scan_source_files(input_path: str) -> dict[str, Any]:
    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        files = [root] if _is_recordable_input_file(root) else []
        base = root.parent
    elif root.is_dir():
        files = [item for item in sorted(root.rglob("*")) if _is_recordable_input_file(item)]
        base = root
    else:
        raise FileNotFoundError(input_path)
    tasks = [_file_task(path, base) for path in files]
    by_suffix: dict[str, int] = {}
    for task in tasks:
        by_suffix[task["suffix"] or "<none>"] = by_suffix.get(task["suffix"] or "<none>", 0) + 1
    return {"input_path": str(root), "file_count": len(tasks), "by_suffix": by_suffix, "tasks": tasks, "sample_tasks": tasks[:20]}


def _base_modality(task: dict[str, Any]) -> str:
    suffix = str(task.get("suffix") or "").lower()
    if suffix in TABULAR_SUFFIXES:
        return "table"
    if suffix in VISUAL_SUFFIXES:
        return "ocr"
    if suffix in TEXT_SUFFIXES:
        return "text"
    return "unsupported"


def write_source_tasks(tasks: list[dict[str, Any]], output_root: str) -> str:
    path = _meta_dir(output_root) / "source_tasks.json"
    write_json(path, tasks)
    return str(path)


def load_source_tasks(tasks_path: str | Path) -> list[dict[str, Any]]:
    parsed = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("source_tasks JSON 顶层必须是 list")
    return [dict(item) for item in parsed]


def _base_record_from_task(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(task.get("source_path")))
    return {
        "source_path": str(path),
        "source_name": path.name,
        "patient_id": task.get("patient_id"),
        "relative_path": task.get("relative_path") or path.name,
        "file_info": {
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "processable": bool(task.get("processable")),
        },
        "observations": {
            "modality": _base_modality(task),
            "should_split": False,
            "id_column": None,
            "status": "ready" if task.get("processable") else "unsupported",
        },
        "worker_evidence": {"modality": "suffix"},
    }


def build_base_records(tasks: list[dict[str, Any]], output_root: str) -> dict[str, Any]:
    ordered_tasks = sorted(tasks, key=lambda item: str(item.get("source_path") or ""))
    workers = _worker_count(len(ordered_tasks))
    print(f"[Step1][Runtime] SourceRecordWorker 开始生成 records: {len(ordered_tasks)} 条", file=sys.stderr, flush=True)
    print(f"[Step1][Runtime] SourceRecordWorker 并行执行: workers={workers}", file=sys.stderr, flush=True)
    records: list[dict[str, Any]] = []
    if ordered_tasks:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_base_record_from_task, task): task for task in ordered_tasks}
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                _progress("SourceRecordWorker records", done, len(ordered_tasks))
    records = sorted(records, key=lambda item: str(item.get("source_path") or ""))
    path = _meta_dir(output_root) / "base_records.json"
    write_json(path, records)
    return {"base_records_path": str(path), "records_count": len(records), "workers": workers}


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("records JSON 顶层必须是 list")
    return parsed


def infer_visual_modality(base_records_path: str, output_root: str, model_path: str | None = None) -> dict[str, Any]:
    records = _load_records(base_records_path)
    result = classify_visual_records(records, model_path=model_path)
    path = _meta_dir(output_root) / "visual_modality_patches.json"
    write_json(path, result["patches"])
    counts: dict[str, int] = {}
    for patch in result["patches"]:
        counts[patch["modality"]] = counts.get(patch["modality"], 0) + 1
    return {
        "visual_patches_path": str(path),
        "visual_count": result["visual_count"],
        "patch_count": len(result["patches"]),
        "modality_counts": counts,
        "doclayout_used": result["doclayout_used"],
        "model_path": result["model_path"],
        "device": result.get("device"),
    }


def _detect_id_column(columns: list[str]) -> str | None:
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", ""): str(c) for c in columns}
    for item in ("subjectid", "patientid", "hadmid", "stayid", "id", "病例号", "病人id", "患者id", "患者编号", "病历号"):
        if item in normalized:
            return normalized[item]
    return None


def _table_split_patch(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source_path")
    try:
        columns, _ = read_rows(source, limit=5)
        id_column = _detect_id_column(columns)
        return {
            "source_path": source,
            "should_split": id_column is not None,
            "id_column": id_column,
            "evidence": "detected id column" if id_column else "no id column",
        }
    except Exception as exc:
        return {"source_path": source, "should_split": False, "id_column": None, "evidence": "fallback on error", "error": str(exc)}


def infer_table_split_strategy(base_records_path: str, output_root: str) -> dict[str, Any]:
    records = _load_records(base_records_path)
    table_records = [record for record in records if str(record.get("file_info", {}).get("suffix") or "").lower() in TABULAR_SUFFIXES]
    workers = _worker_count(len(table_records))
    print(f"[Step1][Runtime] TableSplitWorkerPool 开始判断表格拆分: tables={len(table_records)}", file=sys.stderr, flush=True)
    print(f"[Step1][Runtime] TableSplitWorkerPool 并行执行: workers={workers}", file=sys.stderr, flush=True)
    patches: list[dict[str, Any]] = []
    if table_records:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_table_split_patch, record): record for record in table_records}
            for done, future in enumerate(as_completed(futures), 1):
                patches.append(future.result())
                _progress("TableSplitWorkerPool", done, len(table_records))
    patches = sorted(patches, key=lambda item: str(item.get("source_path") or ""))
    path = _meta_dir(output_root) / "table_split_patches.json"
    write_json(path, patches)
    return {"table_split_patches_path": str(path), "table_count": len(patches), "split_tables": sum(1 for p in patches if p.get("should_split")), "workers": workers}


def reduce_records(base_records_path: str, visual_patches_path: str, table_split_patches_path: str, output_root: str) -> dict[str, Any]:
    records = _load_records(base_records_path)
    by_path = {record["source_path"]: record for record in records}
    visual_patches = json.loads(Path(visual_patches_path).read_text(encoding="utf-8")) if visual_patches_path else []
    table_patches = json.loads(Path(table_split_patches_path).read_text(encoding="utf-8")) if table_split_patches_path else []
    for patch in visual_patches:
        record = by_path.get(patch.get("source_path"))
        if not record:
            continue
        record["observations"]["modality"] = patch.get("modality") or record["observations"].get("modality")
        record["observations"]["layout_analysis"] = patch.get("layout_analysis", {})
        record.setdefault("worker_evidence", {})["modality"] = patch.get("evidence") or "doclayout-yolo"
    for patch in table_patches:
        record = by_path.get(patch.get("source_path"))
        if not record:
            continue
        record["observations"]["modality"] = "table"
        record["observations"]["should_split"] = bool(patch.get("should_split"))
        record["observations"]["id_column"] = patch.get("id_column")
        record.setdefault("worker_evidence", {})["table_split"] = patch.get("evidence")
    ordered = sorted(records, key=lambda item: item["source_path"])
    path = _meta_dir(output_root) / "records.json"
    write_json(path, ordered)
    modality_counts: dict[str, int] = {}
    for record in ordered:
        mod = record.get("observations", {}).get("modality")
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
    return {"records_path": str(path), "records_count": len(ordered), "modality_counts": modality_counts}


def validate_records(records_path: str, expected_count: int | None = None) -> dict[str, Any]:
    issues: list[str] = []
    try:
        records = _load_records(records_path)
    except Exception as exc:
        return {"passed": False, "records_count": 0, "issues": [str(exc)]}
    if expected_count is not None and len(records) != expected_count:
        issues.append(f"records 数量不匹配: records={len(records)}, expected={expected_count}")
    required = {"source_path", "source_name", "patient_id", "relative_path", "file_info", "observations", "worker_evidence"}
    for index, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            issues.append(f"record[{index}] 缺少字段: {', '.join(missing)}")
        if not record.get("source_path"):
            issues.append(f"record[{index}] 缺少 source_path")
        observations = record.get("observations") or {}
        if not observations.get("modality"):
            issues.append(f"record[{index}] 缺少 observations.modality")
    modality_counts: dict[str, int] = {}
    for record in records:
        mod = record.get("observations", {}).get("modality")
        if mod:
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
    return {"passed": not issues, "records_path": records_path, "records_count": len(records), "modality_counts": modality_counts, "issues": sorted(set(issues))}


def _relative_parent(source: Path, input_root: Path, record: dict[str, Any]) -> Path:
    try:
        return source.parent.relative_to(input_root)
    except ValueError:
        return Path(str(record.get("relative_path") or source.name)).parent


def _write_group_table(rows: list[dict[str, Any]], fieldnames: list[str], target: Path) -> None:
    write_rows(target, rows, fieldnames)


def _copy_or_write_record_file(record: dict[str, Any], input_root: Path, output_root: Path) -> int:
    source = Path(str(record.get("source_path"))).resolve()
    observations = record.get("observations") or {}
    modality = str(observations.get("modality") or "ocr")
    patient_id = str(record.get("patient_id") or "") or safe_patient_id(source.stem)
    relative_parent = _relative_parent(source, input_root, record)
    if modality == "table" and observations.get("should_split") and observations.get("id_column"):
        fieldnames, rows = read_rows(source)
        id_column = str(observations["id_column"])
        if id_column not in fieldnames:
            raise ValueError(f"表格缺少 id_column={id_column}: {source}")
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(safe_patient_id(row.get(id_column), patient_id), []).append(row)
        for group_patient_id, group_rows in groups.items():
            target = output_root / group_patient_id / modality / relative_parent / source.name
            _write_group_table(group_rows, fieldnames, target)
        return len(groups)
    target = output_root / patient_id / modality / relative_parent / source.name
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return 1


def _infer_input_root(records: list[dict[str, Any]]) -> Path:
    source_paths = [Path(str(record.get("source_path"))).resolve() for record in records if record.get("source_path")]
    if not source_paths:
        return Path.cwd()
    common = Path(str(Path(*Path(source_paths[0]).parts[:1])))
    import os

    return Path(os.path.commonpath([str(path.parent) for path in source_paths]))


def reorganize_from_records(records_path: str, output_root: str, input_root: str | None = None) -> dict[str, Any]:
    records = _load_records(records_path)
    root = Path(input_root).expanduser().resolve() if input_root else _infer_input_root(records)
    step_root = _step_root(output_root)
    meta_payload = None
    trace_payload = None
    if Path(records_path).exists():
        meta_payload = Path(records_path).read_text(encoding="utf-8")
    existing_trace = step_root / "_meta" / "tool_trace.json"
    if existing_trace.exists():
        trace_payload = existing_trace.read_text(encoding="utf-8")
    if step_root.exists():
        shutil.rmtree(step_root)
    _meta_dir(output_root)
    if meta_payload is not None:
        (_meta_dir(output_root) / "records.json").write_text(meta_payload, encoding="utf-8")
    if trace_payload is not None:
        (_meta_dir(output_root) / "tool_trace.json").write_text(trace_payload, encoding="utf-8")
    processable_records: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        observations = record.get("observations") or {}
        if observations.get("status") == "unsupported" or observations.get("modality") == "unsupported":
            skipped += 1
        else:
            processable_records.append(record)
    workers = _worker_count(len(processable_records), get_step1_reorganize_max_workers())
    print(f"[Step1][Runtime] ReorganizeWorker 开始重组输出: records={len(records)}", file=sys.stderr, flush=True)
    print(f"[Step1][Runtime] ReorganizeWorker 并行执行: workers={workers}, processable_records={len(processable_records)}", file=sys.stderr, flush=True)
    written_files = 0
    errors: list[str] = []
    if processable_records:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_copy_or_write_record_file, record, root, step_root): record for record in processable_records}
            for done, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                try:
                    written_files += future.result()
                except Exception as exc:
                    errors.append(f"{record.get('source_path')}: {exc}")
                _progress("ReorganizeWorker records", done, len(processable_records))
    return {"status": "SUCCESS" if not errors else "FAILED", "output_root": str(step_root), "input_root": str(root), "records_count": len(records), "processed_records": len(processable_records), "skipped_records": skipped, "written_files": written_files, "errors": errors, "workers": workers}


def validate_output(output_root: str, expected_min_files: int = 1) -> dict[str, Any]:
    root = _step_root(output_root)
    issues: list[str] = []
    modality_counts: dict[str, int] = {}
    sample_paths: list[str] = []
    if not root.is_dir():
        return {"passed": False, "issues": [f"Step1 输出目录不存在: {root}"], "output_root": str(root), "output_file_count": 0}
    files = [item for item in sorted(root.rglob("*")) if item.is_file() and "_meta" not in item.parts]
    if len(files) < expected_min_files:
        issues.append(f"Step1 输出文件数不足: output_files={len(files)}, expected_min={expected_min_files}")
    for index, file_path in enumerate(files):
        rel = file_path.relative_to(root)
        if index < 20:
            sample_paths.append(str(rel))
        if len(rel.parts) < 3:
            issues.append(f"输出路径层级不足: {rel}")
        else:
            modality = rel.parts[1]
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
    records_path = root / "_meta" / "records.json"
    if not records_path.exists():
        issues.append("缺少 _meta/records.json")
    trace_path = root / "_meta" / "tool_trace.json"
    return {"passed": not issues, "issues": sorted(set(issues)), "output_root": str(root), "records_path": str(records_path), "tool_trace_path": str(trace_path), "output_file_count": len(files), "modality_counts": modality_counts, "sample_paths": sample_paths}


def build_records_compat(input_path: str, output_root: str) -> dict[str, Any]:
    scan = scan_source_files(input_path)
    base = build_base_records(scan["tasks"], output_root)
    visual = infer_visual_modality(base["base_records_path"], output_root)
    table = infer_table_split_strategy(base["base_records_path"], output_root)
    reduced = reduce_records(base["base_records_path"], visual["visual_patches_path"], table["table_split_patches_path"], output_root)
    return {"records_path": reduced["records_path"], "record_count": reduced["records_count"], "scan": scan, "modality_counts": reduced["modality_counts"]}
