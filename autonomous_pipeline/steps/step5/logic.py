from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from autonomous_pipeline.table_io import ensure_dir, write_json

Emit = Callable[[str], None] | None


def _confidence_threshold_default() -> float:
    raw = os.environ.get("STEP5_CONFIDENCE_THRESHOLD") or os.environ.get("STEP6_CONFIDENCE_THRESHOLD") or "0.70"
    try:
        return float(raw)
    except Exception:
        return 0.70


CONFIDENCE_THRESHOLD_DEFAULT = _confidence_threshold_default()
PATIENT_ID_CANDIDATES = [
    "patient_id",
    "record_index",
    "subject_id",
    "hadm_id",
    "patientid",
    "patient",
    "id",
    "pid",
]

STATIC_SKIP_COLS = {
    "ocr_text",
    "preprocessed_text",
    "file_path",
    "error",
    "relations",
    "temporal_info",
    "quantity_info",
    "impression",
    "indication",
}
META_COLS = frozenset({"patient_id", "filename", "file_path", "folder_category", "note_index", "note_type", "note_seq"})
LONG_TEXT_RATIO_THRESHOLD = 0.3
LONG_TEXT_LEN_THRESHOLD = 200


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def step_root(output_root: str | Path) -> Path:
    path = resolve_path(output_root)
    return ensure_dir(path if path.name == "step5_results" else path / "step5_results")


def meta_dir(output_root: str | Path) -> Path:
    return ensure_dir(step_root(output_root) / "_meta")


def _compact_trace_value(value: Any, limit: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def append_tool_trace(output_root: str | Path, tool: str, arguments: dict[str, Any], payload: dict[str, Any], started_at: float) -> str:
    path = meta_dir(output_root) / "tool_trace.json"
    try:
        trace = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        trace = []
    trace.append(
        {
            "tool": tool,
            "arguments": {k: _compact_trace_value(v) for k, v in arguments.items() if k != "reason"},
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "artifacts": payload.get("artifacts", {}),
            "issues": payload.get("issues", []),
            "duration_ms": int((time.time() - started_at) * 1000),
            "finished_at": datetime.now().isoformat(),
        }
    )
    return write_json(path, trace)


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "null", "na", "n/a"})


def _find_input_csv(path: Path) -> Path:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Step5 输入必须是 CSV 或包含 CSV 的目录: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Step5 输入路径不存在: {path}")
    candidates: list[Path] = []
    for rel in (
        "filtered.csv",
        "input.csv",
        "next_input/filtered.csv",
        "next_input/input.csv",
    ):
        candidate = path / rel
        if candidate.is_file():
            return candidate
    candidates.extend(sorted(path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
    candidates.extend(sorted(path.glob("next_input/*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
    if not candidates:
        raise FileNotFoundError(f"Step5 未找到可验证 CSV: {path}")
    return candidates[0]


def resolve_step5_input(input_path: str | Path) -> dict[str, Any]:
    csv_path = _find_input_csv(resolve_path(input_path))
    adjacent: dict[str, str] = {}
    for name in ("data_quality_report.json", "column_risk_report.json", "selection_report.json"):
        for candidate in (csv_path.parent / name, csv_path.parent.parent / name):
            if candidate.is_file():
                adjacent[name.removesuffix(".json")] = str(candidate)
                break
    return {
        "input_csv": str(csv_path),
        "input_dir": str(csv_path.parent),
        "adjacent_artifacts": adjacent,
    }


def _detect_patient_id_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for cand in PATIENT_ID_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in df.columns:
        if "id" in str(col).lower():
            return str(col)
    return str(df.columns[0])


def _build_skip_cols(df: pd.DataFrame) -> set[str]:
    skip_cols = set(STATIC_SKIP_COLS)
    for col in df.columns:
        if col in skip_cols:
            continue
        col_text = str(col)
        if col_text.endswith(("_抽取", "_标准化", "_extracted", "_normalized")):
            skip_cols.add(col)
            continue
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        long_ratio = float((sample.str.len() > LONG_TEXT_LEN_THRESHOLD).mean())
        if long_ratio >= LONG_TEXT_RATIO_THRESHOLD:
            skip_cols.add(col)
    return skip_cols


def _read_data(input_csv: str | Path, preferred_patient_col: str | None = None) -> tuple[pd.DataFrame, str, set[str]]:
    df = pd.read_csv(input_csv)
    if df.empty and len(df.columns) == 0:
        raise ValueError(f"Step5 输入 CSV 没有表头: {input_csv}")
    actual_id_col = _detect_patient_id_col(df, preferred_patient_col)
    if actual_id_col != "patient_id":
        df = df.rename(columns={actual_id_col: "patient_id"})
    skip_cols = _build_skip_cols(df)
    return df, actual_id_col, skip_cols


def _record_progress(emit: Emit, label: str, done: int, total: int) -> None:
    if not emit or total <= 0:
        return
    step = max(1, total // 20)
    if done == 1 or done == total or done % step == 0:
        emit(f"[Step5][Progress] {label}: {done}/{total} ({done / total * 100:.1f}%)")


def load_data_overview(input_csv: str, output_root: str, patient_id_col: str | None = None) -> dict[str, Any]:
    df, actual_id_col, skip_cols = _read_data(input_csv, patient_id_col)
    n_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0
    rec_counts = df.groupby("patient_id").size() if "patient_id" in df.columns else pd.Series(dtype=int)
    preview: dict[str, str] = {}
    if not df.empty:
        row = df.iloc[0]
        for col in df.columns:
            if col in skip_cols:
                continue
            value = row[col]
            if _is_nonempty(value):
                preview[str(col)] = str(value)[:120]
            if len(preview) >= 30:
                break
    overview = {
        "input_csv": str(resolve_path(input_csv)),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "patient_id_column": actual_id_col,
        "patient_count": n_patients,
        "skipped_text_columns": sorted(str(col) for col in skip_cols if col in df.columns),
        "record_count_distribution": {
            "min": int(rec_counts.min()) if not rec_counts.empty else 0,
            "max": int(rec_counts.max()) if not rec_counts.empty else 0,
            "mean": round(float(rec_counts.mean()), 2) if not rec_counts.empty else 0.0,
        },
        "sample_nonempty_fields": preview,
        "columns_preview": [str(col) for col in df.columns[:80]],
    }
    path = write_json(meta_dir(output_root) / "data_overview.json", overview)
    return {"overview_json": path, **overview}


def analyze_all_patients_consistency(
    input_csv: str,
    output_root: str,
    patient_id_col: str | None = None,
    threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    emit: Emit = None,
) -> dict[str, Any]:
    df, actual_id_col, skip_cols = _read_data(input_csv, patient_id_col)
    data_cols = [col for col in df.columns if col not in META_COLS and col not in skip_cols]
    if "patient_id" not in df.columns:
        raise ValueError("Step5 未能规范化出 patient_id 列")

    groups = list(df.groupby("patient_id", dropna=False))
    if emit:
        emit(f"[Step5][Runtime] patient consistency 开始: patients={len(groups)}, threshold={threshold}")

    results: list[dict[str, Any]] = []
    for idx, (pid, grp) in enumerate(groups, start=1):
        high = medium = low = 0
        examples: list[dict[str, Any]] = []
        for col in data_cols:
            values = grp[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            unique_values = list(dict.fromkeys(values.tolist()))
            if len(unique_values) <= 1:
                continue
            col_lower = str(col).lower()
            if any(key in col_lower for key in ("诊断", "diagnosis", "性别", "gender", "民族", "race")):
                high += 1
                severity = "HIGH"
            elif any(key in col_lower for key in ("年龄", "age", "dob", "birth")):
                medium += 1
                severity = "MEDIUM"
            else:
                low += 1
                severity = "LOW"
            if len(examples) < 8:
                examples.append({"column": str(col), "severity": severity, "values": unique_values[:5]})

        filled = sum(1 for col in data_cols if grp[col].apply(_is_nonempty).any())
        completeness = filled / len(data_cols) * 100 if data_cols else 100.0
        score = 1.0 - high * 0.20 - medium * 0.10 - low * 0.03
        if completeness < 30:
            score -= 0.10
        score = max(0.0, round(score, 3))
        results.append(
            {
                "patient_id": str(pid),
                "n_records": int(len(grp)),
                "high": high,
                "medium": medium,
                "low": low,
                "completeness": round(completeness, 1),
                "score": score,
                "passed": bool(score >= threshold),
                "conflict_examples": examples,
            }
        )
        _record_progress(emit, "patient consistency", idx, len(groups))

    passed_ids = [item["patient_id"] for item in results if item["passed"]]
    failed = [item for item in results if not item["passed"]]
    pass_scores = [float(item["score"]) for item in results if item["passed"]]
    fail_scores = [float(item["score"]) for item in failed]
    summary = {
        "input_csv": str(resolve_path(input_csv)),
        "patient_id_column": actual_id_col,
        "total_patients": len(results),
        "passed_patients": len(passed_ids),
        "failed_patients": len(failed),
        "confidence_threshold": threshold,
        "data_dimensions": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "checked_columns": len(data_cols),
        "skipped_text_columns": len(skip_cols),
        "passed_score_range": [min(pass_scores), max(pass_scores)] if pass_scores else [],
        "failed_score_range": [min(fail_scores), max(fail_scores)] if fail_scores else [],
    }
    payload = {
        "timestamp": datetime.now().isoformat(),
        "step": "STEP5_ConsistencyValidation",
        "summary": summary,
        "patients": results,
        "passed_patient_ids": passed_ids,
    }
    path = write_json(step_root(output_root) / f"patient_consistency_{_timestamp()}.json", payload)
    if emit:
        emit(f"[Step5][Runtime] patient consistency 完成: passed={len(passed_ids)}, failed={len(failed)}")
    return {"analysis_json": path, **summary, "passed_patient_ids": passed_ids[:20]}


def _render_report(analysis: dict[str, Any]) -> str:
    summary = analysis.get("summary", {})
    patients = analysis.get("patients", [])
    failed = [item for item in patients if not item.get("passed")]
    lines = [
        "=================================================================",
        "STEP 5: 患者数据一致性验证",
        "=================================================================",
        f"输入: {summary.get('input_csv', '')}",
        f"数据规模: {summary.get('data_dimensions', '')}",
        f"患者数: {summary.get('total_patients', 0)}",
        f"通过患者: {summary.get('passed_patients', 0)}",
        f"未通过患者: {summary.get('failed_patients', 0)}",
        f"置信度阈值: {summary.get('confidence_threshold', CONFIDENCE_THRESHOLD_DEFAULT)}",
        f"检查列数: {summary.get('checked_columns', 0)}",
        f"跳过长文本列数: {summary.get('skipped_text_columns', 0)}",
        "",
        "未通过患者概览（最多前30位）:",
    ]
    if not failed:
        lines.append("  无")
    else:
        lines.append(f"{'patient_id':<16} {'rows':>4} {'HIGH':>5} {'MED':>5} {'LOW':>5} {'完整性':>7} {'分数':>6}")
        lines.append("-" * 62)
        for item in failed[:30]:
            lines.append(
                f"{item['patient_id']:<16} {item['n_records']:>4} {item['high']:>5} {item['medium']:>5} "
                f"{item['low']:>5} {item['completeness']:>6.1f}% {item['score']:>6.3f}"
            )
        if len(failed) > 30:
            lines.append(f"  ... 还有 {len(failed) - 30} 位未通过患者")
    lines.extend(
        [
            "",
            "说明: HIGH 冲突主要来自诊断/性别/种族等关键字段；MEDIUM 来自年龄/出生日期类字段；LOW 来自其他非关键字段。",
            "PASSED_PATIENTS: <stored_in_step5_report_json>",
        ]
    )
    return "\n".join(lines)


def save_step5_report(input_csv: str, analysis_json_path: str, output_root: str, report_content: str | None = None, allow_auto_report: bool = False) -> dict[str, Any]:
    analysis = json.loads(resolve_path(analysis_json_path).read_text(encoding="utf-8"))
    passed_ids = [str(item) for item in analysis.get("passed_patient_ids") or []]
    if report_content and report_content.strip():
        report_text = report_content.strip()
    elif allow_auto_report:
        report_text = _render_report(analysis)
    else:
        raise ValueError("Step5 报告必须由 Agent/LLM 撰写并通过 report_content 传入。")
    if "PASSED_PATIENTS:" not in report_text:
        raise ValueError("Step5 报告缺少 PASSED_PATIENTS 标记。")
    summary = analysis.get("summary", {})

    root = step_root(output_root)
    ts = _timestamp()
    text_path = root / f"consistency_report_{ts}.txt"
    json_path = root / f"consistency_report_{ts}.json"
    passed_path = root / f"passed_patients_{ts}.json"
    next_dir = ensure_dir(root / "next_input")
    next_passed_path = next_dir / "passed_patients.json"
    next_report_path = next_dir / "consistency_report.json"
    next_input_csv = next_dir / "filtered.csv"

    text_path.write_text(report_text, encoding="utf-8")
    report_json = {
        "timestamp": datetime.now().isoformat(),
        "step": "STEP5_ConsistencyValidation",
        "data_file": str(resolve_path(input_csv)),
        "summary": summary,
        "report_text": report_text,
        "source_analysis_json": str(resolve_path(analysis_json_path)),
        "report_author": "agent_llm" if report_content and report_content.strip() else "auto_fallback",
    }
    passed_json = {
        "timestamp": datetime.now().isoformat(),
        "passed_count": len(passed_ids),
        "passed_patient_ids": passed_ids,
    }
    write_json(json_path, report_json)
    write_json(passed_path, passed_json)
    write_json(next_report_path, report_json)
    write_json(next_passed_path, passed_json)
    shutil.copy2(resolve_path(input_csv), next_input_csv)

    return {
        "text_report": str(text_path),
        "json_report": str(json_path),
        "passed_patients_json": str(passed_path),
        "next_input_dir": str(next_dir),
        "next_input_csv": str(next_input_csv),
        "next_passed_patients_json": str(next_passed_path),
        "next_consistency_report_json": str(next_report_path),
        "passed_patient_count": len(passed_ids),
        "total_patient_count": int(summary.get("total_patients") or 0),
    }


def validate_step5_output(
    json_report_path: str,
    passed_patients_json: str,
    next_input_dir: str | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    report_path = resolve_path(json_report_path)
    passed_path = resolve_path(passed_patients_json)
    if not report_path.is_file():
        issues.append(f"一致性报告 JSON 不存在: {report_path}")
    if not passed_path.is_file():
        issues.append(f"通过患者 JSON 不存在: {passed_path}")
        passed_payload = {}
    else:
        passed_payload = json.loads(passed_path.read_text(encoding="utf-8"))
        if not isinstance(passed_payload.get("passed_patient_ids"), list):
            issues.append("passed_patient_ids 不是列表")
    if next_input_dir:
        next_dir = resolve_path(next_input_dir)
        for name in ("filtered.csv", "passed_patients.json", "consistency_report.json"):
            if not (next_dir / name).is_file():
                issues.append(f"next_input 缺少 {name}")
    return {
        "passed": not issues,
        "issues": issues,
        "json_report_path": str(report_path),
        "passed_patients_json": str(passed_path),
        "passed_patient_count": len(passed_payload.get("passed_patient_ids") or []) if isinstance(passed_payload, dict) else 0,
        "next_input_dir": str(resolve_path(next_input_dir)) if next_input_dir else "",
    }
