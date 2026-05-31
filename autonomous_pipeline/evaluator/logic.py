from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_pipeline.gold import load_gold_standard, summarize_case_shape
from autonomous_pipeline.table_io import ensure_dir, read_json, write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _compact(value: Any, limit: int = 400) -> str:
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


def _load_latest_dataset(round_root: Path) -> dict[str, Any]:
    latest = round_root / "step6_results" / "next_input" / "latest_dataset.json"
    if latest.is_file():
        try:
            data = read_json(latest)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _score_patient_cases(patient_cases: list[dict[str, Any]], gold_cases: list[dict[str, Any]]) -> tuple[float, list[str], dict[str, Any]]:
    issues: list[str] = []
    if not patient_cases:
        return 0.0, ["缺少 Step4 patient_cases.jsonl，无法进行患者级病例内容评估"], {}
    score = 0.35
    if gold_cases:
        gold_shape = summarize_case_shape(gold_cases)
        gold_keys = {str(k) for k in gold_shape.get("top_keys", [])}
        candidate_blob = json.dumps(patient_cases[:20], ensure_ascii=False).lower()
        matched = [key for key in gold_keys if key and key.lower() in candidate_blob]
        if gold_keys:
            score += min(0.35, 0.35 * len(matched) / max(len(gold_keys), 1))
        if not matched:
            issues.append("患者病例与人工病例的显性区块/内容重合度偏低，需要改进病例汇总。")
    if any("sections" in row for row in patient_cases):
        score += 0.15
    if any("evidence" in json.dumps(row, ensure_ascii=False).lower() or "source" in row for row in patient_cases):
        score += 0.15
    return min(score, 1.0), issues, {"patient_case_count": len(patient_cases)}


def _score_ml_dataset(latest_dataset: dict[str, Any]) -> tuple[float, list[str], dict[str, Any]]:
    issues: list[str] = []
    dataset_csv = latest_dataset.get("dataset_csv")
    model_config_json = latest_dataset.get("model_config_json")
    details: dict[str, Any] = {"latest_dataset": latest_dataset}
    if not dataset_csv or not Path(str(dataset_csv)).is_file():
        return 0.0, ["缺少 Step6 ML dataset CSV。"], details
    try:
        df = pd.read_csv(dataset_csv)
    except Exception as exc:
        return 0.0, [f"Step6 ML dataset 无法读取: {exc}"], details
    score = 0.35
    details.update({"rows": int(len(df)), "columns": int(len(df.columns))})
    if "y" not in df.columns:
        issues.append("ML 数据集缺少 y 标签列。")
    else:
        classes = sorted(df["y"].dropna().astype(str).unique().tolist())
        details["classes"] = classes
        if len(classes) >= 2:
            score += 0.25
        else:
            issues.append("ML 数据集有效标签类别少于 2 类。")
    if len(df) >= 10:
        score += 0.15
    else:
        issues.append("ML 数据集样本数偏少。")
    if model_config_json and Path(str(model_config_json)).is_file():
        try:
            config = read_json(model_config_json)
            features = config.get("feature_names") or []
            details["feature_count"] = len(features)
            leakage = [col for col in features if any(token in str(col).lower() for token in ("icd", "label", "outcome"))]
            if leakage:
                issues.append(f"ML 特征存在潜在标签泄漏: {leakage[:5]}")
            else:
                score += 0.25
        except Exception as exc:
            issues.append(f"model_config 无法读取: {exc}")
    return min(score, 1.0), issues, details


def evaluate_round_outputs(round_root: str | Path, gold_dir: str | Path, evaluation_root: str | Path, previous_score: float = 0.0) -> dict[str, Any]:
    round_path = Path(round_root).expanduser().resolve()
    eval_path = ensure_dir(evaluation_root)
    gold = load_gold_standard(gold_dir, include_hidden=True)
    eval_cases = gold["hidden_cases"] if gold["hidden_cases"] else gold["visible_cases"]
    hidden_used = bool(gold["hidden_cases"])
    patient_cases_path = round_path / "step4_results" / "next_input" / "patient_cases.jsonl"
    patient_cases = _read_jsonl(patient_cases_path)
    latest_dataset = _load_latest_dataset(round_path)

    case_score, case_issues, case_details = _score_patient_cases(patient_cases, eval_cases)
    ml_score, ml_issues, ml_details = _score_ml_dataset(latest_dataset)
    score = round(case_score * 0.55 + ml_score * 0.45, 4)
    issues = case_issues + ml_issues
    if not hidden_used:
        issues.append("未提供 hidden 金标准，本轮不能验证未见病例泛化能力。")

    improved = score > float(previous_score or 0) + 0.01
    continue_recommended = score < 0.78 and improved
    if any("patient_cases" in item or "病例" in item for item in issues):
        restart_step = "step4"
    elif any("ML" in item or "标签" in item or "泄漏" in item for item in issues):
        restart_step = "step6"
    else:
        restart_step = "step3"

    private_report = {
        "status": "SUCCESS" if score >= 0.65 else "NEEDS_REPAIR",
        "round_root": str(round_path),
        "score": score,
        "previous_score": previous_score,
        "improved": improved,
        "hidden_used": hidden_used,
        "case_score": round(case_score, 4),
        "ml_score": round(ml_score, 4),
        "case_details": case_details,
        "ml_details": ml_details,
        "issues": issues,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    public_feedback = {
        "status": private_report["status"],
        "score": score,
        "improved": improved,
        "continue_recommended": continue_recommended,
        "restart_step": restart_step,
        "error_categories": _categorize_issues(issues),
        "optimization_directions": _directions_for_issues(issues)[:3],
        "examples": issues[:3],
    }
    private_path = write_json(eval_path / "private_report.json", private_report)
    public_path = write_json(eval_path / "public_feedback.json", public_feedback)
    return {
        "private_report_path": private_path,
        "public_feedback_path": public_path,
        "score": score,
        "improved": improved,
        "continue_recommended": continue_recommended,
        "restart_step": restart_step,
        "hidden_used": hidden_used,
        "issues": issues,
    }


def _categorize_issues(issues: list[str]) -> list[str]:
    categories: list[str] = []
    for issue in issues:
        if "病例" in issue or "patient_cases" in issue:
            categories.append("patient_case_content")
        elif "泄漏" in issue:
            categories.append("feature_leakage")
        elif "标签" in issue or "类别" in issue:
            categories.append("label_quality")
        elif "hidden" in issue or "泛化" in issue:
            categories.append("evaluation_coverage")
        else:
            categories.append("data_quality")
    result: list[str] = []
    for item in categories:
        if item not in result:
            result.append(item)
    return result


def _directions_for_issues(issues: list[str]) -> list[str]:
    directions: list[str] = []
    for issue in issues:
        if "patient_cases" in issue or "病例" in issue:
            directions.append("回到 Step4，增强患者级病例 JSONL 汇总，优先覆盖诊断、检查、用药和文本证据。")
        elif "泄漏" in issue:
            directions.append("回到 Step6，重新选择特征并排除标签来源、ICD 源列和 outcome 列。")
        elif "标签" in issue or "类别" in issue:
            directions.append("回到 Step6，调整标签构建策略或扩大阳性/阴性样本覆盖。")
        elif "hidden" in issue:
            directions.append("补充 hidden 金标准或明确当前仅为可见样例回放评估。")
        else:
            directions.append("回到 Planner，检查上游任务裁剪与清洗是否丢失关键信息。")
    return directions or ["当前结果可作为保留方案，下一轮只需做小范围优化。"]
