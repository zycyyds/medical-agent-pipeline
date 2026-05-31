from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from autonomous_pipeline.gold import summarize_case_shape, validate_gold_standard
from autonomous_pipeline.table_io import TABLE_SUFFIXES, TEXT_SUFFIXES, ensure_dir, file_modality, read_rows, write_json


def _compact(value: Any, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def append_tool_trace(output_root: str | Path, tool: str, arguments: dict[str, Any], payload: dict[str, Any], started_at: float) -> str:
    root = ensure_dir(Path(output_root) / "_meta")
    path = root / "tool_trace.json"
    try:
        trace = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        trace = []
    trace.append(
        {
            "tool": tool,
            "arguments": {k: _compact(v, 240) for k, v in arguments.items() if k != "reason"},
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "artifacts": payload.get("artifacts", {}),
            "issues": payload.get("issues", []),
            "duration_ms": int((time.time() - started_at) * 1000),
            "finished_at": datetime.now().isoformat(),
        }
    )
    write_json(path, trace)
    return str(path)


def sample_raw_dataset(input_path: str | Path, output_root: str | Path, max_files: int = 40) -> dict[str, Any]:
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"原始输入路径不存在: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    suffix_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for path in files:
        suffix = path.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        modality = file_modality(path)
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
        if len(samples) >= max_files:
            continue
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        item: dict[str, Any] = {"path": rel, "suffix": suffix, "modality": modality, "size": path.stat().st_size}
        if suffix in TABLE_SUFFIXES:
            try:
                headers, rows = read_rows(path, limit=2)
                item["columns"] = headers[:40]
                item["sample_rows"] = rows[:2]
            except Exception as exc:
                item["read_error"] = str(exc)
        elif suffix in TEXT_SUFFIXES:
            try:
                item["snippet"] = _compact(path.read_text(encoding="utf-8", errors="ignore"), 500)
            except Exception as exc:
                item["read_error"] = str(exc)
        samples.append(item)
    output = ensure_dir(output_root)
    summary = {
        "input_path": str(root),
        "file_count": len(files),
        "suffix_counts": suffix_counts,
        "modality_counts": modality_counts,
        "samples": samples,
    }
    summary["raw_sample_path"] = write_json(output / "raw_sample.json", summary)
    return summary


def build_task_spec(input_path: str | Path, gold_dir: str | Path, output_root: str | Path, task_text: str = "") -> dict[str, Any]:
    output = ensure_dir(output_root)
    gold = validate_gold_standard(gold_dir, output_root=output, include_hidden=False)
    if not gold["valid"]:
        raise ValueError("; ".join(gold["issues"]))
    raw_sample = sample_raw_dataset(input_path, output, max_files=40)
    shape = summarize_case_shape(gold["visible_cases"])
    focus_terms = _extract_focus_terms(task_text, shape)
    spec = {
        "task_goal": task_text or "未指定医疗数据结构化任务",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gold_standard": {
            "gold_dir": gold["gold_dir"],
            "visible_count": gold["visible_count"],
            "hidden_available": False,
            "mapping_available": bool(gold.get("mapping_path")),
            "shape": shape,
        },
        "raw_dataset": {
            "input_path": raw_sample["input_path"],
            "file_count": raw_sample["file_count"],
            "suffix_counts": raw_sample["suffix_counts"],
            "modality_counts": raw_sample["modality_counts"],
        },
        "content_expectations": shape["top_keys"],
        "medical_focus": focus_terms,
        "source_priority": _infer_source_priority(raw_sample),
        "extraction_requirements": [
            "保留明确出现的诊断、检查、用药、手术/操作和关键文本证据",
            "优先覆盖与用户任务和可见人工病例内容相关的信息",
            "不要为了匹配格式而编造未在原始数据中出现的医学事实",
        ],
        "filtering_requirements": [
            "任务相关诊断、ICD、实验室指标、用药和文本证据列优先保留",
            "高缺失或明显无关列可以裁剪，但患者 ID 和标签来源列不得丢失",
        ],
        "cleaning_requirements": [
            "清洗只能做格式标准化和噪声修复，不把缺失值填成业务值",
            "患者级病例输出以内容语义为准，结构允许与人工 JSONL 不完全一致",
        ],
        "evaluation_rubric": [
            "患者病例内容覆盖",
            "诊断和并发症准确性",
            "关键检查/用药/文本证据保留",
            "无关信息比例",
            "ML 数据集标签合理性、泄漏风险和类别可训练性",
        ],
    }
    task_spec_path = write_json(output / "task_spec.json", spec)
    return {"task_spec_path": task_spec_path, "task_spec": spec, "gold_summary_path": gold.get("summary_path", ""), "raw_sample_path": raw_sample["raw_sample_path"]}


def _extract_focus_terms(task_text: str, shape: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    blob = f"{task_text} {' '.join(shape.get('top_keys') or [])} {' '.join(shape.get('nested_keys') or [])}".lower()
    candidates = {
        "肝": ["肝病", "肝硬化", "肝衰竭", "腹水", "肝性脑病", "ALT", "AST", "胆红素", "INR", "白蛋白", "ICD"],
        "liver": ["liver disease", "cirrhosis", "hepatic failure", "ascites", "ALT", "AST", "bilirubin", "INR", "albumin", "ICD"],
        "眩晕": ["前庭", "眩晕", "眼震", "定位诊断", "半规管", "BPPV", "冷热试验", "vHIT"],
        "vestibular": ["vestibular", "vertigo", "nystagmus", "localization", "BPPV", "vHIT"],
    }
    for marker, values in candidates.items():
        if marker in blob:
            terms.extend(values)
    for key in shape.get("top_keys", [])[:12]:
        if key not in terms:
            terms.append(str(key))
    return terms[:30]


def _infer_source_priority(raw_sample: dict[str, Any]) -> list[str]:
    priority: list[str] = []
    for item in raw_sample.get("samples", []):
        columns = " ".join(str(c).lower() for c in item.get("columns", []))
        path = str(item.get("path", "")).lower()
        if "diagnoses_icd" in path or "icd" in columns:
            priority.append("diagnoses_icd")
        if "note" in path or "text" in columns or "report" in columns:
            priority.append("clinical_notes")
        if "labevents" in path or "lab" in columns or "itemid" in columns:
            priority.append("labevents")
        if "prescriptions" in path or "drug" in columns:
            priority.append("prescriptions")
    result: list[str] = []
    for item in priority:
        if item not in result:
            result.append(item)
    return result or ["tables", "clinical_text", "images"]
