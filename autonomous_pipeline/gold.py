from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .table_io import ensure_dir, write_json


def _read_jsonl_file(path: Path, prefix: str) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    issues: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                issues.append(f"{path.name}:{line_no} JSONL 解析失败: {exc}")
                continue
            if not isinstance(payload, dict):
                issues.append(f"{path.name}:{line_no} 不是 JSON object")
                continue
            case_id = payload.get("case_id") or payload.get("patient_id") or f"{prefix}_{len(cases) + 1:04d}"
            cases.append({"case_id": str(case_id), "source_file": str(path), "line_no": line_no, "content": payload})
    return cases, issues


def _jsonl_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.glob("*.jsonl") if p.is_file())


def load_gold_standard(gold_dir: str | Path, include_hidden: bool = False) -> dict[str, Any]:
    """Load visible and optionally hidden gold-standard JSONL cases."""
    root = Path(gold_dir).expanduser().resolve()
    visible_root = root / "visible" if root.is_dir() else root
    hidden_root = root / "hidden" if root.is_dir() else root / "__missing_hidden__"

    visible_cases: list[dict[str, Any]] = []
    hidden_cases: list[dict[str, Any]] = []
    issues: list[str] = []
    visible_files = _jsonl_files(visible_root)
    if not visible_files and root.is_file() and root.suffix.lower() == ".jsonl":
        visible_files = [root]

    for path in visible_files:
        rows, row_issues = _read_jsonl_file(path, "visible")
        visible_cases.extend(rows)
        issues.extend(row_issues)

    hidden_files: list[Path] = []
    if include_hidden:
        hidden_files = _jsonl_files(hidden_root)
        for path in hidden_files:
            rows, row_issues = _read_jsonl_file(path, "hidden")
            hidden_cases.extend(rows)
            issues.extend(row_issues)

    mapping_path = root / "mapping.csv" if root.is_dir() else root.with_name("mapping.csv")
    mappings: list[dict[str, str]] = []
    if mapping_path.is_file():
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                mappings.append({str(k): "" if v is None else str(v) for k, v in row.items()})

    return {
        "gold_dir": str(root),
        "visible_files": [str(p) for p in visible_files],
        "hidden_files": [str(p) for p in hidden_files],
        "visible_cases": visible_cases,
        "hidden_cases": hidden_cases,
        "visible_count": len(visible_cases),
        "hidden_count": len(hidden_cases),
        "mapping_path": str(mapping_path) if mapping_path.is_file() else "",
        "mapping_count": len(mappings),
        "mappings": mappings,
        "issues": issues,
    }


def validate_gold_standard(gold_dir: str | Path, output_root: str | Path | None = None, include_hidden: bool = False) -> dict[str, Any]:
    payload = load_gold_standard(gold_dir, include_hidden=include_hidden)
    issues = list(payload["issues"])
    if payload["visible_count"] <= 0:
        issues.append("gold_standard/visible 中没有可用 JSONL 病例，Planner 不能启动。")
    payload["valid"] = not issues
    payload["issues"] = issues
    if output_root:
        ensure_dir(output_root)
        payload["summary_path"] = write_json(Path(output_root) / "gold_summary.json", {k: v for k, v in payload.items() if k not in {"visible_cases", "hidden_cases"}})
    return payload


def summarize_case_shape(cases: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
    top_keys: dict[str, int] = {}
    nested_keys: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for case in cases:
        content = case.get("content") if isinstance(case, dict) else {}
        if not isinstance(content, dict):
            continue
        for key, value in content.items():
            top_keys[str(key)] = top_keys.get(str(key), 0) + 1
            if isinstance(value, dict):
                for nested in value:
                    nested_keys[f"{key}.{nested}"] = nested_keys.get(f"{key}.{nested}", 0) + 1
        if len(examples) < limit:
            examples.append({"case_id": case.get("case_id"), "top_keys": list(content.keys())[:20]})
    return {
        "case_count": len(cases),
        "top_keys": sorted(top_keys, key=top_keys.get, reverse=True),
        "nested_keys": sorted(nested_keys, key=nested_keys.get, reverse=True)[:50],
        "examples": examples,
    }
