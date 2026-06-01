from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_pipeline.config import get_agent_config
from autonomous_pipeline.table_io import compact_value, ensure_dir, is_missing_like, read_rows, write_json, write_rows

TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls"}
VALID_MODALITIES = [
    "Identifiers/Keys", "Time", "Demographics", "Encounter", "Diagnosis", "Medication",
    "Labs", "Vitals", "Procedures", "Outcomes", "ClinicalNotes",
]

MIMIC_CORE_COLUMNS = {
    "subject_id", "hadm_id", "gender", "anchor_age", "dod", "admittime", "dischtime",
    "deathtime", "admission_type", "admission_location", "discharge_location", "insurance",
    "language", "marital_status", "race", "diagnoses_count", "diagnoses_icd_codes",
    "labevents_count", "labevents_abnormal_count", "prescriptions_count", "prescriptions_drugs",
    "icu_count", "icu_careunit", "impressions",
}

IDENTIFIER_COLUMN_TERMS = {
    "id", "patient_id", "subject_id", "hadm_id", "encounter_id", "visit_id", "record_id",
    "study_id", "case_id", "pid", "eid", "mrn", "患者id", "患者_id", "病历号", "住院号", "就诊号", "病例号",
}

BASIC_GENERAL_INFO_TERMS = {
    "性别", "年龄", "年龄单位", "出生年月", "民族", "文化程度", "工作性质", "职业", "婚姻", "婚育",
    "身高", "体重", "bmi", "sex", "gender", "age", "birth", "dob", "race", "ethnicity",
    "language", "marital", "occupation", "education",
}

BASIC_TIME_OR_ENCOUNTER_TERMS = {
    "visit_date", "visit_time", "encounter_date", "encounter_time", "admission_date", "admission_time",
    "discharge_date", "discharge_time", "就诊时间", "就诊日期", "入院时间", "入院日期", "出院时间", "出院日期",
}

GENERIC_COLUMN_RECALL_TERMS = {
    "diagnosis", "diagnoses", "finding", "findings", "status", "value", "values", "result",
    "results", "procedure", "procedures", "radiology", "discharge", "report", "clinical",
    "patient", "patients", "disease", "symptom", "symptoms", "onset", "trend", "etiology", "severity",
}

LAYERED_TERM_KEYS = [
    "direct_diagnosis_terms", "supportive_evidence_terms", "background_disease_terms",
    "imaging_finding_terms", "lab_marker_terms", "treatment_context_terms",
]


def step_root(output_root: str | Path) -> Path:
    return ensure_dir(Path(output_root) / "step3_results")


def meta_dir(output_root: str | Path) -> Path:
    return ensure_dir(step_root(output_root) / "_meta")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_tool_trace(output_root: str | Path, tool: str, arguments: dict[str, Any], payload: dict[str, Any], started_at: float) -> str:
    path = meta_dir(output_root) / "tool_trace.json"
    try:
        trace = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        trace = []
    trace.append({
        "tool": tool,
        "arguments": {k: compact_value(v, 300) for k, v in arguments.items() if k != "reason"},
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "artifacts": payload.get("artifacts", {}),
        "issues": payload.get("issues", []),
        "duration_ms": int((time.time() - started_at) * 1000),
        "finished_at": datetime.now().isoformat(),
    })
    return write_json(path, trace)


def resolve_input_csv(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if path.is_file():
        if path.suffix.lower() not in TABULAR_SUFFIXES:
            raise FileNotFoundError(f"输入文件不是支持的表格文件: {path}")
        input_csv = path
    elif path.is_dir():
        candidates = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_SUFFIXES]
        if not candidates:
            raise FileNotFoundError(f"在目录 {path} 中未找到 CSV/XLSX/XLS 文件。")
        by_name: dict[str, list[Path]] = {}
        for candidate in candidates:
            by_name.setdefault(candidate.name.lower(), []).append(candidate)
        input_csv = None
        for preferred in ("input.csv", "filtered.csv", "input.xlsx", "input.xls"):
            if preferred in by_name:
                input_csv = max(by_name[preferred], key=lambda p: p.stat().st_mtime)
                break
        input_csv = input_csv or max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        raise FileNotFoundError(f"输入路径不存在: {path}")

    columns, _ = read_rows(input_csv, limit=1)
    return {
        "input_path": str(path),
        "input_csv_path": str(input_csv),
        "suffix": input_csv.suffix.lower(),
        "file_size": input_csv.stat().st_size,
        "column_count": len(columns),
        "sample_columns": columns[:30],
    }


def resolve_task_text(task_text: str | None = None) -> dict[str, Any]:
    resolved = str(task_text or "").strip() or os.environ.get("AGENT4_TASK_TEXT", "").strip()
    if not resolved:
        raise ValueError("Step3 task_text 不能为空。")
    return {"task_text": resolved}


def _read_dataframe(input_csv: str | Path) -> pd.DataFrame:
    path = Path(input_csv)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input file type: {path.suffix}")


def _sample_dataframe(df: pd.DataFrame, max_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=seed)


def _is_operational_metadata_name(col_lower: str) -> bool:
    return col_lower in {"file", "filename", "file_name", "filepath", "file_path", "path", "source_file", "source_filename"}


def _is_task_supporting_identifier_name(col_lower: str) -> bool:
    if any(token in col_lower for token in ["mrn", "medical_record", "insurance", "policy", "account", "license", "passport", "ssn"]):
        return False
    if col_lower in {"id", "pid", "eid", "patient_id"}:
        return True
    return bool(re.search(r"(^|_)(patient|subject|study|record|encounter|visit|case|episode|stay|sample|specimen|accession|order)_?id$", col_lower))


def _is_potential_phi(col_name: str) -> bool:
    col_lower = col_name.lower()
    if _is_operational_metadata_name(col_lower) or _is_task_supporting_identifier_name(col_lower):
        return False
    patterns = [
        r"(^|[^a-z])name([^a-z]|$)", r"first.*name", r"last.*name", r"full.*name", r"fname", r"lname",
        r"address", r"phone", r"email", r"contact", r"ssn", r"mrn", r"medical.*record", r"insurance.*id",
        r"dob", r"date.*birth", r"birthday", r"birth.*date", r"account.*number", r"license", r"passport", r"zip.*code",
    ]
    return any(re.search(pattern, col_lower) for pattern in patterns)


def _infer_structured_table_modality(col_name: str) -> str | None:
    col_lower = col_name.lower()
    if col_lower.startswith("table_一般资料"):
        return "Demographics"
    if col_lower.startswith(("table_症状学", "table_初步诊断", "table_第一次复诊")):
        return "Diagnosis"
    if col_lower.startswith("table_床旁查体"):
        return "Procedures"
    if col_lower.startswith("table_辅助检查"):
        return "Labs"
    return None


def _infer_modality(col_name: str) -> str | None:
    low = col_name.lower()
    if _is_operational_metadata_name(low):
        return None
    structured = _infer_structured_table_modality(low)
    if structured:
        return structured
    if any(keyword in low for keyword in ["labresult", "labevent", "bilirubin", "albumin", "platelet", "inr", "alt", "ast"]):
        return "Labs"
    if any(keyword in low for keyword in ["diagnosis", "diagnoses", "icd", "cirrhosis", "hepatitis"]):
        return "Diagnosis"
    modality_keywords = {
        "Identifiers/Keys": ["id", "identifier", "patient_id", "study_id", "subject_id", "hadm_id", "stay_id", "患者id", "病历号", "住院号"],
        "Time": ["date", "time", "datetime", "admission", "discharge", "visit", "onset", "death", "birth", "日期", "时间", "入院", "出院"],
        "Demographics": ["sex", "gender", "race", "language", "marital", "age", "anchor_age", "一般资料", "性别", "年龄"],
        "Encounter": ["encounter", "visit", "admission", "discharge", "provider", "location", "icu", "careunit", "住院", "入院", "出院", "科室"],
        "Diagnosis": ["diagnosis", "dx", "icd", "condition", "symptom", "cirrhosis", "hepatitis", "症状", "诊断", "病史"],
        "Medication": ["medication", "drug", "prescription", "dose", "route", "pharmacy", "药物", "用药", "处方"],
        "Labs": ["lab", "test", "value", "result", "loinc", "units", "abnormal", "bilirubin", "albumin", "platelet", "辅助检查", "实验室", "检查", "检验"],
        "Vitals": ["vital", "bp", "heart_rate", "temperature", "oxygen", "respiratory"],
        "Procedures": ["procedure", "surgery", "operation", "intervention", "cpt", "手术", "操作", "治疗"],
        "Outcomes": ["outcome", "mortality", "death", "expire", "discharge_location", "readmission", "label"],
        "ClinicalNotes": ["note", "text", "narrative", "clinical_text", "report", "radiology", "discharge"],
    }
    for modality, keywords in modality_keywords.items():
        if any(keyword in low for keyword in keywords):
            return modality
    return None


def _safe_examples(series: pd.Series, is_phi: bool) -> list[str]:
    examples: list[str] = []
    for value in series.head(5):
        text = compact_value(value, 120)
        if is_phi:
            if re.match(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", text):
                examples.append("<DATE_REDACTED>")
            elif sum(ch.isdigit() for ch in text) / max(len(text), 1) > 0.7 and len(text) >= 3:
                examples.append("<ID_REDACTED>")
            else:
                examples.append("<TEXT_REDACTED>")
        else:
            examples.append(text)
    return examples


def profile_schema(input_csv: str, output_root: str | Path, max_rows: int = 1000) -> dict[str, Any]:
    df_full = _read_dataframe(input_csv)
    df = _sample_dataframe(df_full, max_rows=max_rows)
    columns: dict[str, Any] = {}
    for col in df.columns:
        sample = df[col]
        full = df_full[col]
        missing_rate = float(full.isna().sum() / max(len(full), 1))
        is_phi = _is_potential_phi(str(col))
        numeric_stats: dict[str, Any] = {}
        if pd.api.types.is_numeric_dtype(sample):
            numeric_col = pd.to_numeric(sample, errors="coerce")
            numeric_stats = {
                "mean": float(numeric_col.mean()) if not numeric_col.empty and not pd.isna(numeric_col.mean()) else None,
                "std": float(numeric_col.std()) if not numeric_col.empty and not pd.isna(numeric_col.std()) else None,
                "min": float(numeric_col.min()) if not numeric_col.empty and not pd.isna(numeric_col.min()) else None,
                "max": float(numeric_col.max()) if not numeric_col.empty and not pd.isna(numeric_col.max()) else None,
            }
        columns[str(col)] = {
            "name": str(col),
            "dtype_guess": str(sample.dtype),
            "missing_rate": missing_rate,
            "unique_ratio": float(full.nunique(dropna=False) / max(len(full), 1)),
            "examples": _safe_examples(sample.dropna(), is_phi),
            "numeric_stats": numeric_stats,
            "is_potential_phi": is_phi,
            "inferred_modality": _infer_modality(str(col)),
            "is_constant": bool(full.nunique(dropna=False) == 1) if len(full) > 0 else True,
        }
    profile = {
        "input_csv": input_csv,
        "dataset_info": {
            "total_rows": len(df_full),
            "sampled_rows": len(df),
            "total_columns": len(df_full.columns),
            "profile_generated_at": datetime.now().isoformat(),
        },
        "columns": columns,
    }
    path = write_json(step_root(output_root) / "schema_profile.json", profile)
    return {"schema_profile_path": path, "row_count": len(df_full), "column_count": len(df_full.columns)}


def _call_json_llm(prompt: str, temperature: float = 0.0) -> dict[str, Any] | None:
    cfg = get_agent_config("step3")
    if not cfg.has_credentials:
        return None
    try:
        import openai

        client = openai.OpenAI(api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY"), base_url=cfg.base_url, timeout=cfg.timeout)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = response.choices[0].message.content if response.choices else ""
        return _extract_json_object(text)
    except Exception:
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start():])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _require_json_llm(prompt: str, stage: str, temperature: float = 0.0) -> dict[str, Any]:
    parsed = _call_json_llm(prompt, temperature=temperature)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Step3 {stage} 必须调用 LLM，并且需要返回合法 JSON。请检查 config.yaml/API key/模型输出。")
    return parsed


def plan_task_spec(task_text: str, output_root: str | Path) -> dict[str, Any]:
    task_lower = (task_text or "").lower()
    fallback = {
        "task_type": "cohort_selection",
        "need_modalities": ["Identifiers/Keys", "Demographics", "Diagnosis", "Labs", "ClinicalNotes"],
        "index_event": task_text,
        "time_window": "",
        "uncertainty": 0.45,
    }
    if any(token in task_lower for token in ["死亡", "mortality", "death", "outcome", "预测"]):
        fallback["need_modalities"] = ["Identifiers/Keys", "Demographics", "Encounter", "Diagnosis", "Labs", "Outcomes", "Time"]
        fallback["task_type"] = "prediction"
    prompt = f"""Analyze this medical data task and return ONLY JSON with fields task_type, need_modalities, index_event, time_window, uncertainty.

Allowed modalities: {VALID_MODALITIES}
Task: {task_text}
"""
    parsed = _require_json_llm(prompt, "任务规划", temperature=0.0)
    fallback.update({k: parsed.get(k, fallback.get(k)) for k in fallback})
    if not isinstance(fallback.get("need_modalities"), list):
        fallback["need_modalities"] = ["Identifiers/Keys", "Diagnosis"]
    path = write_json(step_root(output_root) / "task_spec.json", {"task_text": task_text, "task_spec": fallback})
    return {"task_spec_path": path, "task_spec": fallback}


def _normalize_terms(values: Any) -> list[str]:
    if values is None:
        return []
    raw = re.split(r"[\n,，;；]+", values) if isinstance(values, str) else values if isinstance(values, list) else []
    out: list[str] = []
    for value in raw:
        term = str(value).strip().lower()
        if term and term not in GENERIC_COLUMN_RECALL_TERMS and term not in out:
            out.append(term)
    return out


def _fallback_terms(task_text: str) -> dict[str, Any]:
    text = (task_text or "").lower()
    terms = re.findall(r"[a-z][a-z0-9_]{1,}|[\u4e00-\u9fff]{2,}", text)
    liver_terms = ["肝", "liver", "hepatic", "hepatitis", "cirrhosis", "bilirubin", "albumin", "platelet", "inr", "alt", "ast", "icd", "diagnos"]
    if any(token in text for token in ["肝", "liver", "hepatic", "cirrhosis", "hepatitis"]):
        terms.extend(liver_terms)
    deduped = _normalize_terms(terms)
    return {
        "task_domain": task_text or "unknown_task",
        "trigger_terms": deduped,
        "column_terms": deduped,
        "direct_diagnosis_terms": [t for t in deduped if t in {"肝", "liver", "hepatic", "hepatitis", "cirrhosis", "diagnos", "icd"}],
        "supportive_evidence_terms": [t for t in deduped if t in {"bilirubin", "albumin", "platelet", "inr", "alt", "ast"}],
        "background_disease_terms": [],
        "imaging_finding_terms": ["radiology", "finding"],
        "lab_marker_terms": [t for t in deduped if t in {"bilirubin", "albumin", "platelet", "inr", "alt", "ast"}],
        "treatment_context_terms": [],
        "short_terms": [t for t in deduped if len(t) <= 3 and t.isascii()],
        "negative_terms": [],
        "reason": "fallback term profile",
    }


def plan_medical_terms(task_text: str, output_root: str | Path) -> dict[str, Any]:
    profile = _fallback_terms(task_text)
    prompt = f"""Generate task-specific medical recall terms for selecting columns from a medical wide table.
Return ONLY JSON with keys: task_domain, trigger_terms, column_terms, direct_diagnosis_terms, supportive_evidence_terms, background_disease_terms, imaging_finding_terms, lab_marker_terms, treatment_context_terms, short_terms, negative_terms, reason.
Task: {task_text}
"""
    parsed = _require_json_llm(prompt, "医学术语规划", temperature=0.1)
    for key in ["trigger_terms", "column_terms", "short_terms", "negative_terms"] + LAYERED_TERM_KEYS:
        parsed[key] = _normalize_terms(parsed.get(key))
    profile.update(parsed)
    path = write_json(step_root(output_root) / "term_profile.json", {"task_text": task_text, "term_profile": profile})
    return {"term_profile_path": path, "term_profile": profile}


def _tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", (text or "").lower()) if token}


def _is_identifier_column(col_name: str) -> bool:
    normalized = col_name.strip().lower()
    tokens = {token for token in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", normalized) if token}
    return normalized in IDENTIFIER_COLUMN_TERMS or bool(tokens & IDENTIFIER_COLUMN_TERMS)


def _is_basic_column(col_name: str) -> bool:
    low = col_name.strip().lower()
    tail = re.split(r"[-_/：:]+", low)[-1].strip()
    if low in MIMIC_CORE_COLUMNS or _is_identifier_column(col_name):
        return True
    if low.startswith("table_一般资料"):
        return True
    return bool({low, tail} & (BASIC_GENERAL_INFO_TERMS | BASIC_TIME_OR_ENCOUNTER_TERMS))


def _term_matches(col_name: str, term_profile: dict[str, Any]) -> tuple[bool, dict[str, list[str]]]:
    low = col_name.lower()
    if any(term and term in low for term in _normalize_terms(term_profile.get("negative_terms"))):
        return False, {}
    tokens = _tokenize(col_name)
    matches: dict[str, list[str]] = {}
    for key in ["trigger_terms", "column_terms"] + LAYERED_TERM_KEYS + ["short_terms"]:
        for term in _normalize_terms(term_profile.get(key)):
            term_tokens = _tokenize(term)
            if key == "short_terms" or (len(term) <= 3 and term.isascii()):
                matched = term in tokens
            elif len(term_tokens) > 1 and term.isascii():
                matched = all(token in tokens for token in term_tokens)
            else:
                matched = term in low
            if matched:
                matches.setdefault(key, []).append(term)
    return bool(matches), matches


def build_candidate_columns(schema_profile_path: str, task_spec_path: str, term_profile_path: str, output_root: str | Path, max_candidates: int | None = None) -> dict[str, Any]:
    schema = json.loads(Path(schema_profile_path).read_text(encoding="utf-8"))
    task_spec = json.loads(Path(task_spec_path).read_text(encoding="utf-8")).get("task_spec", {})
    term_profile = json.loads(Path(term_profile_path).read_text(encoding="utf-8")).get("term_profile", {})
    need_modalities = set(task_spec.get("need_modalities") or [])
    task_blob = json.dumps({"task_spec": task_spec, "term_profile": term_profile}, ensure_ascii=False).lower()
    is_diagnosis_task = any(token in task_blob for token in ("diagnos", "icd", "疾病", "诊断", "肝", "liver", "cirrhosis", "hepatitis"))
    candidate_columns: list[str] = []
    protected_columns: list[str] = []
    reason_by_column: dict[str, str] = {}
    match_layers: dict[str, Any] = {}
    for col, profile in schema.get("columns", {}).items():
        if profile.get("missing_rate") == 1.0:
            continue
        if profile.get("is_potential_phi"):
            continue
        col_low = str(col).lower()
        keep_diagnosis_code = is_diagnosis_task and (
            "diagnoses_icd" in col_low
            or col_low.endswith("icd_code")
            or col_low.endswith("icd_version")
            or col_low in {"icd_code", "icd_version", "diagnoses_icd_codes"}
        )
        matched_terms, layers = _term_matches(col, term_profile)
        modality = profile.get("inferred_modality")
        keep_basic = _is_basic_column(col)
        keep_modality = bool(modality and modality in need_modalities and len(schema.get("columns", {})) < 1000)
        if keep_basic or keep_diagnosis_code or matched_terms or keep_modality:
            candidate_columns.append(col)
            match_layers[col] = layers
            if keep_basic:
                protected_columns.append(col)
                reason_by_column[col] = "protected structured/core field"
            elif keep_diagnosis_code:
                protected_columns.append(col)
                reason_by_column[col] = "protected diagnosis ICD label-source field"
            elif matched_terms:
                reason_by_column[col] = "matched task term profile"
            else:
                reason_by_column[col] = f"matched needed modality: {modality}"
        if max_candidates and len(candidate_columns) >= max_candidates:
            break
    payload = {
        "candidate_columns": candidate_columns,
        "protected_columns": protected_columns,
        "reason_by_column": reason_by_column,
        "match_layers": match_layers,
        "candidate_count": len(candidate_columns),
        "original_column_count": len(schema.get("columns", {})),
    }
    path = write_json(step_root(output_root) / "candidate_columns.json", payload)
    return {"candidate_columns_path": path, **payload}


def select_columns(schema_profile_path: str, task_spec_path: str, term_profile_path: str, candidate_columns_path: str, output_root: str | Path) -> dict[str, Any]:
    schema = json.loads(Path(schema_profile_path).read_text(encoding="utf-8"))
    task_spec = json.loads(Path(task_spec_path).read_text(encoding="utf-8")).get("task_spec", {})
    term_profile = json.loads(Path(term_profile_path).read_text(encoding="utf-8")).get("term_profile", {})
    candidates = json.loads(Path(candidate_columns_path).read_text(encoding="utf-8"))
    selected = {
        "task_spec": task_spec,
        "term_profile": term_profile,
        "must_have_columns": list(candidates.get("protected_columns", [])),
        "useful_columns": [],
        "maybe_columns": [],
        "drop_columns": [],
        "reason_by_column": dict(candidates.get("reason_by_column", {})),
        "warnings": [],
    }
    candidate_cols = list(candidates.get("candidate_columns", []))
    if not candidate_cols:
        raise RuntimeError("Step3 候选列为空，无法调用 LLM 进行列选择。")
    compact_schema = {
        col: {
            "missing_rate": schema["columns"][col].get("missing_rate"),
            "modality": schema["columns"][col].get("inferred_modality"),
            "examples": schema["columns"][col].get("examples", [])[:3],
        }
        for col in candidate_cols[:500]
        if col in schema.get("columns", {})
    }
    prompt = f"""You are selecting columns for a medical ML/data task. Return ONLY JSON with arrays: must_have_columns, useful_columns, maybe_columns, drop_columns, reason_by_column, warnings.
Task spec: {json.dumps(task_spec, ensure_ascii=False)}
Term profile: {json.dumps(term_profile, ensure_ascii=False)}
Candidate schema: {json.dumps(compact_schema, ensure_ascii=False)}
"""
    parsed = _require_json_llm(prompt, "列选择", temperature=0.2)
    for key in ["must_have_columns", "useful_columns", "maybe_columns", "drop_columns", "warnings"]:
        if isinstance(parsed.get(key), list):
            selected[key] = [col for col in parsed[key] if isinstance(col, str)]
    if isinstance(parsed.get("reason_by_column"), dict):
        selected["reason_by_column"].update({str(k): str(v) for k, v in parsed["reason_by_column"].items()})
    path = write_json(step_root(output_root) / "raw_selection_result.json", selected)
    return {"raw_selection_path": path, "selected_count_before_gates": sum(len(selected.get(k, [])) for k in ["must_have_columns", "useful_columns", "maybe_columns"])}


def _dedupe(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def apply_hard_gates(raw_selection_path: str, schema_profile_path: str, output_root: str | Path, keep_free_text: bool = False, max_final_columns: int | None = None) -> dict[str, Any]:
    result = json.loads(Path(raw_selection_path).read_text(encoding="utf-8"))
    schema = json.loads(Path(schema_profile_path).read_text(encoding="utf-8"))
    overrides: list[dict[str, Any]] = []
    warnings: list[str] = list(result.get("warnings", []))
    for key in ["must_have_columns", "useful_columns", "maybe_columns", "drop_columns"]:
        result.setdefault(key, [])
    for col, profile in schema.get("columns", {}).items():
        if profile.get("is_potential_phi"):
            for key in ["must_have_columns", "useful_columns", "maybe_columns"]:
                if col in result[key]:
                    result[key].remove(col)
            if col not in result["drop_columns"]:
                result["drop_columns"].append(col)
                overrides.append({"column": col, "action": "removed_phi", "reason": "PHI protection"})
        if not keep_free_text and profile.get("inferred_modality") == "ClinicalNotes":
            for key in ["must_have_columns", "useful_columns"]:
                if col in result[key]:
                    result[key].remove(col)
                    result["maybe_columns"].append(col)
                    overrides.append({"column": col, "action": "downgraded_free_text", "reason": "Free text in non-keep mode"})
        if profile.get("is_constant") and profile.get("inferred_modality") != "Identifiers/Keys":
            for key in ["must_have_columns", "useful_columns", "maybe_columns"]:
                if col in result[key]:
                    result[key].remove(col)
                    if col not in result["drop_columns"]:
                        result["drop_columns"].append(col)
                    overrides.append({"column": col, "action": "removed_constant", "reason": "Constant value column"})
        if profile.get("missing_rate") == 1.0:
            for key in ["must_have_columns", "useful_columns", "maybe_columns"]:
                if col in result[key]:
                    result[key].remove(col)
                    if col not in result["drop_columns"]:
                        result["drop_columns"].append(col)
                    overrides.append({"column": col, "action": "removed_all_missing", "reason": "All missing values column"})
    selected = set(result["must_have_columns"] + result["useful_columns"] + result["maybe_columns"])
    final_columns = [col for col in schema.get("columns", {}) if col in selected]
    if isinstance(max_final_columns, int) and max_final_columns > 0 and len(final_columns) > max_final_columns:
        protected = [col for col in final_columns if col in result["must_have_columns"]]
        remaining = [col for col in final_columns if col not in protected]
        final_columns = protected + remaining[: max(0, max_final_columns - len(protected))]
        warnings.append(f"Trimmed final columns to max_final_columns={max_final_columns}")
    for key in ["must_have_columns", "useful_columns", "maybe_columns", "drop_columns"]:
        result[key] = _dedupe(result[key])
    result["final_columns"] = _dedupe(final_columns)
    result["overrides"] = overrides
    result["warnings"] = _dedupe(warnings)
    path = write_json(step_root(output_root) / "selection_report.json", result)
    return {"selection_report_path": path, "final_columns": result["final_columns"], "final_column_count": len(result["final_columns"]), "overrides": overrides, "warnings": result["warnings"]}


def export_filtered_table(input_csv: str, selection_report_path: str, output_root: str | Path) -> dict[str, Any]:
    df = _read_dataframe(input_csv)
    report = json.loads(Path(selection_report_path).read_text(encoding="utf-8"))
    final_columns = [col for col in report.get("final_columns", []) if col in df.columns]
    if not final_columns:
        raise ValueError("selection_report.final_columns 为空，无法导出裁剪表。")
    out = step_root(output_root)
    ts = timestamp()
    filtered_csv = out / f"input_filtered_{ts}.csv"
    df[final_columns].to_csv(filtered_csv, index=False, encoding="utf-8-sig")
    report["column_counts"] = {
        "original_column_count": len(df.columns),
        "final_column_count": len(final_columns),
        "selector_candidate_column_count": len(set(report.get("must_have_columns", []) + report.get("useful_columns", []) + report.get("maybe_columns", []))),
    }
    report["final_columns"] = final_columns
    report_path = out / f"input_selection_report_{ts}.json"
    write_json(report_path, report)
    next_dir = ensure_dir(out / "next_input")
    next_csv = next_dir / "filtered.csv"
    df[final_columns].to_csv(next_csv, index=False, encoding="utf-8-sig")
    next_report = write_json(next_dir / "selection_report.json", report)
    return {"filtered_csv": str(filtered_csv), "selection_report_path": str(report_path), "next_input_csv": str(next_csv), "next_selection_report": next_report, "rows": len(df), "columns": len(final_columns)}


def validate_output(filtered_csv_path: str, selection_report_path: str, next_input_csv: str | None = None, next_selection_report: str | None = None) -> dict[str, Any]:
    issues: list[str] = []
    details: dict[str, Any] = {"filtered_csv_path": filtered_csv_path, "selection_report_path": selection_report_path}
    filtered = Path(filtered_csv_path)
    report_path = Path(selection_report_path)
    if not filtered.is_file():
        issues.append(f"Step3 filtered CSV 不存在: {filtered}")
    if not report_path.is_file():
        issues.append(f"Step3 selection_report 不存在: {report_path}")
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"Step3 selection_report 不是合法 JSON: {exc}")
    output_columns: list[str] = []
    if filtered.is_file():
        try:
            output_columns, rows = read_rows(filtered, limit=2)
            details["output_column_count"] = len(output_columns)
            details["sample_output_columns"] = output_columns[:30]
            if not rows:
                issues.append("Step3 filtered CSV 无样本。")
        except Exception as exc:
            issues.append(f"Step3 filtered CSV 无法读取: {exc}")
    final_columns = report.get("final_columns") if isinstance(report.get("final_columns"), list) else []
    if report and not final_columns:
        issues.append("Step3 selection_report 缺少 final_columns 或 final_columns 为空。")
    if output_columns and final_columns and output_columns != final_columns:
        issues.append("Step3 filtered CSV 列顺序与 selection_report.final_columns 不一致。")
    for label, path in [("next_input_csv", next_input_csv), ("next_selection_report", next_selection_report)]:
        if path:
            details[label] = path
            if not Path(path).is_file():
                issues.append(f"Step3 {label} 不存在: {path}")
    return {"passed": not issues, "issues": sorted(set(issues)), "details": details}
