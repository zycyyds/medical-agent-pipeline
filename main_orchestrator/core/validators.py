from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts import ValidatorResult


def load_records(records_path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(records_path)
    if not path.is_file():
        return [], [f"records.json 不存在: {path}"]
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"records.json 不是合法 JSON: {exc}"]
    if not isinstance(parsed, list):
        return [], ["records.json 顶层必须是 list"]
    invalid = [idx for idx, item in enumerate(parsed) if not isinstance(item, dict)]
    if invalid:
        return [], [f"records.json 中存在非对象记录: index={invalid[:5]}"]
    return parsed, []


def validate_records(
    records_path: str | Path,
    expected_count: int | None = None,
) -> ValidatorResult:
    records, issues = load_records(records_path)
    seen_source_paths: set[str] = set()
    modality_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    split_tables = 0

    if expected_count is not None and not issues and len(records) != expected_count:
        issues.append(f"records 数量不匹配: records={len(records)}, expected={expected_count}")

    for idx, record in enumerate(records):
        source_path = str(record.get("source_path") or "").strip()
        source_name = str(record.get("source_name") or "").strip()
        observations = record.get("observations")
        if not source_path:
            issues.append(f"record[{idx}] 缺少 source_path")
        if not source_name:
            issues.append(f"record[{idx}] 缺少 source_name")
        if source_path in seen_source_paths:
            issues.append(f"重复 source_path: {source_path}")
        seen_source_paths.add(source_path)
        if observations is None or not isinstance(observations, dict):
            issues.append(f"record[{idx}] 缺少 observations 对象")
            continue
        modality = str(observations.get("modality") or "").strip()
        if not modality:
            issues.append(f"record[{idx}] 缺少 observations.modality")
        else:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        status = str(observations.get("status") or "ready").strip()
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if observations.get("should_split") and not observations.get("id_column"):
            issues.append(f"record[{idx}] should_split=true 但缺少 id_column")
        if observations.get("should_split"):
            split_tables += 1

    return ValidatorResult(
        passed=not issues,
        issues=sorted(set(issues)),
        details={
            "records_path": str(Path(records_path)),
            "records_count": len(records),
            "expected_count": expected_count,
            "modality_counts": modality_counts,
            "status_counts": status_counts,
            "processable_records": len(records) - status_counts.get("unsupported", 0),
            "unsupported_records": status_counts.get("unsupported", 0),
            "split_tables": split_tables,
        },
    )


def validate_required_files(required: dict[str, str | Path]) -> ValidatorResult:
    issues = []
    details = {}
    for name, value in required.items():
        path = Path(value)
        details[name] = str(path)
        if not path.is_file():
            issues.append(f"缺少必需文件 {name}: {path}")
    return ValidatorResult(passed=not issues, issues=issues, details=details)


def validate_step1_output(output_root: str | Path, expected_min_files: int = 1) -> ValidatorResult:
    root = Path(output_root)
    issues: list[str] = []
    modality_counts: dict[str, int] = {}
    sample_modality_counts: dict[str, int] = {}
    sample_paths: list[str] = []

    if not root.is_dir():
        return ValidatorResult(
            passed=False,
            issues=[f"Step1 输出目录不存在: {root}"],
            details={"output_root": str(root), "output_file_count": 0},
        )

    files = [item for item in sorted(root.rglob("*")) if item.is_file() and "_meta" not in item.parts]
    if len(files) < expected_min_files:
        issues.append(f"Step1 输出文件数不足: output_files={len(files)}, expected_min={expected_min_files}")

    for index, file_path in enumerate(files):
        rel = file_path.relative_to(root)
        if index < 20:
            sample_paths.append(str(rel))
        if len(rel.parts) < 3:
            issues.append(f"输出路径层级不足: {rel}")
        elif len(rel.parts) >= 2:
            modality = rel.parts[1]
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            if index < 20:
                sample_modality_counts[modality] = sample_modality_counts.get(modality, 0) + 1

    return ValidatorResult(
        passed=not issues,
        issues=sorted(set(issues)),
        details={
            "output_root": str(root),
            "output_file_count": len(files),
            "expected_min_files": expected_min_files,
            "modality_counts": modality_counts,
            "sample_modality_counts": sample_modality_counts,
            "sample_paths": sample_paths,
        },
    )
