from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_pipeline.config import PROJECT_ROOT, get_agent_config
from autonomous_pipeline.table_io import ensure_dir, is_missing_like, split_codes, write_json, write_jsonl, write_rows

PATIENT_ID_CANDIDATES = ("patient_id", "record_index", "subject_id", "hadm_id", "patientid", "patient", "id", "pid")
LIVER_PREFIXES = ("K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77", "B18", "B19", "070", "155", "456", "570", "571", "572", "573", "C22")
NULL_THRESHOLD = 0.70
TIME_PATTERNS = ("admittime", "dischtime", "deathtime", "dod", "edregtime", "edouttime", "time", "date")
ID_PATTERNS = ("patient_id", "subject_id", "hadm_id", "stay_id", "transfer_id", "icustay_id", "record_index")
LEAKAGE_PATTERNS = ("icd", "diagnoses_icd", "_icd_label", "_notnull_label", "_label", "label", "outcome")
TEXT_SKIP_THRESHOLD = 200
DIAGNOSIS_FIELD_KEYWORDS = ("诊断", "diagnosis", "disease", "syndrome", "综合征", "icd", "condition", "impression", "indication")


def _first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


_LEGACY_STEP7_ROOT = Path("/Users/mkbk/PycharmProjects/new/step-7")
_STEP6_REFERENCE_ROOT = _first_existing_path([PROJECT_ROOT / "step-7", _LEGACY_STEP7_ROOT])
ICD10_XLSX = Path(os.environ.get("ICD10_XLSX", str(_first_existing_path([PROJECT_ROOT / "step-7" / "ICD-10.xlsx", _LEGACY_STEP7_ROOT / "ICD-10.xlsx"]))))
ICD10_VECTOR_DIR = Path(os.environ.get("ICD10_VECTOR_DIR", str(_first_existing_path([PROJECT_ROOT / "step-7" / "data" / "icd10_bert_index", _LEGACY_STEP7_ROOT / "data" / "icd10_bert_index"]))))
ICD10_BERT_MODEL = os.environ.get("ICD10_BERT_MODEL", str(_first_existing_path([PROJECT_ROOT / "step-7" / "models" / "bge-small-zh-v1.5", _LEGACY_STEP7_ROOT / "models" / "bge-small-zh-v1.5"])))
ICD10_USE_BERT = os.environ.get("ICD10_USE_BERT", "1").lower() not in {"0", "false", "no"}
ICD10_USE_LLM = os.environ.get("ICD10_USE_LLM", "1").lower() not in {"0", "false", "no"}
ICD10_BERT_MIN_SIM = float(os.environ.get("ICD10_BERT_MIN_SIM", "0.70") or "0.70")

_ICD10_CACHE: dict[str, tuple[str, str]] = {}
_ICD10_DF: pd.DataFrame | None = None
_ICD10_NAME_TO_ENTRY: dict[str, tuple[str, str]] | None = None
_ICD10_NORM_TO_ENTRY: dict[str, tuple[str, str]] | None = None
_ICD10_EMB_MATRIX: Any = None
_ICD10_EMB_CODES: Any = None
_ICD10_EMB_NAMES: Any = None
_ICD10_ST_MODEL: Any = None
_LLM_CN_NORMALIZE_CACHE: dict[str, str | None] = {}
_LLM_DISEASE_EXTRACT_CACHE: dict[tuple[str, str], str | None] = {}
_LLM_DIAG_FIELDS_CACHE: list[str] | None = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def step_root(output_root: str | Path) -> Path:
    path = resolve_path(output_root)
    return ensure_dir(path if path.name == "step6_results" else path / "step6_results")


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


def _find_csv(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Step6 输入路径不存在或不是 CSV/目录: {path}")
    for rel in ("filtered.csv", "input.csv", "next_input/filtered.csv", "next_input/input.csv"):
        candidate = path / rel
        if candidate.is_file():
            return candidate
    candidates = sorted(path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(path.glob("next_input/*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Step6 未找到输入 CSV: {path}")
    return candidates[0]


def _find_optional_json(input_path: Path, explicit: str | None, names: tuple[str, ...]) -> str:
    if explicit:
        path = resolve_path(explicit)
        if path.is_file():
            return str(path)
    search_roots = [input_path if input_path.is_dir() else input_path.parent]
    if not input_path.is_dir():
        search_roots.append(input_path.parent.parent)
    for root in search_roots:
        for name in names:
            for candidate in (root / name, root / "next_input" / name):
                if candidate.is_file():
                    return str(candidate)
    return ""


def resolve_step6_inputs(input_path: str, passed_patients_json: str | None = None, selection_report_path: str | None = None) -> dict[str, Any]:
    path = resolve_path(input_path)
    csv_path = _find_csv(path)
    passed_path = _find_optional_json(path, passed_patients_json, ("passed_patients.json",))
    report_path = _find_optional_json(path, selection_report_path, ("selection_report.json",))
    consistency_report = _find_optional_json(path, None, ("consistency_report.json",))
    return {
        "input_csv": str(csv_path),
        "passed_patients_json": passed_path,
        "selection_report_path": report_path,
        "consistency_report_path": consistency_report,
    }


def _detect_patient_col(df: pd.DataFrame) -> str:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in PATIENT_ID_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in df.columns:
        if "id" in str(col).lower():
            return str(col)
    return str(df.columns[0])


def _read_data(input_csv: str | Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_csv, dtype=object)
    if df.empty and len(df.columns) == 0:
        raise ValueError(f"Step6 输入 CSV 没有表头: {input_csv}")
    actual = _detect_patient_col(df)
    if actual != "patient_id":
        df = df.rename(columns={actual: "patient_id"})
    return df, actual


def _load_passed_ids(passed_patients_json: str | None) -> list[str]:
    if not passed_patients_json:
        return []
    path = resolve_path(passed_patients_json)
    if not path.is_file():
        raise FileNotFoundError(f"passed_patients JSON 不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("passed_patient_ids")
    if not isinstance(ids, list):
        raise ValueError(f"passed_patients JSON 缺少列表字段 passed_patient_ids: {path}")
    return [str(item) for item in ids]


def _filter_passed(df: pd.DataFrame, passed_ids: list[str]) -> pd.DataFrame:
    if not passed_ids:
        return df.reset_index(drop=True)
    return df[df["patient_id"].astype(str).isin(set(passed_ids))].reset_index(drop=True)


def _call_step6_llm(prompt: str, max_tokens: int = 800) -> str:
    if not ICD10_USE_LLM:
        return ""
    cfg = get_agent_config("step6")
    if not cfg.has_credentials:
        return ""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY"), base_url=cfg.base_url, timeout=cfg.timeout)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _normalize_diag(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\([A-Za-z0-9 \-]+\)$", "", text).strip()
    return text


def _load_icd10_local() -> tuple[pd.DataFrame, dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    global _ICD10_DF, _ICD10_NAME_TO_ENTRY, _ICD10_NORM_TO_ENTRY
    if _ICD10_DF is not None and _ICD10_NAME_TO_ENTRY is not None and _ICD10_NORM_TO_ENTRY is not None:
        return _ICD10_DF, _ICD10_NAME_TO_ENTRY, _ICD10_NORM_TO_ENTRY
    try:
        df = pd.read_excel(ICD10_XLSX, dtype=str)
        df = df.dropna(subset=["诊断名称", "诊断编码"])
        df["诊断名称"] = df["诊断名称"].astype(str).str.strip()
        df["诊断编码"] = df["诊断编码"].astype(str).str.strip()
        name_to_entry = {row["诊断名称"]: (row["诊断编码"], row["诊断名称"]) for _, row in df.iterrows()}
    except Exception:
        df = pd.DataFrame()
        name_to_entry = {}
    norm_to_entry = {_normalize_diag(name): entry for name, entry in name_to_entry.items()}
    _ICD10_DF = df
    _ICD10_NAME_TO_ENTRY = name_to_entry
    _ICD10_NORM_TO_ENTRY = norm_to_entry
    return df, name_to_entry, norm_to_entry


def _try_load_icd10_vector_store() -> bool:
    global _ICD10_EMB_MATRIX, _ICD10_EMB_CODES, _ICD10_EMB_NAMES
    if _ICD10_EMB_MATRIX is not None:
        return True
    emb_path = ICD10_VECTOR_DIR / "embeddings.npy"
    codes_path = ICD10_VECTOR_DIR / "icd_codes.npy"
    names_path = ICD10_VECTOR_DIR / "icd_names.npy"
    if not emb_path.is_file() or not codes_path.is_file() or not names_path.is_file():
        return False
    try:
        import numpy as np

        matrix = np.load(emb_path)
        matrix = np.asarray(matrix, dtype=np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-12, 1.0, norms)
        _ICD10_EMB_MATRIX = matrix / norms
        _ICD10_EMB_CODES = np.load(codes_path, allow_pickle=True)
        _ICD10_EMB_NAMES = np.load(names_path, allow_pickle=True)
        return int(_ICD10_EMB_MATRIX.shape[0]) == int(_ICD10_EMB_CODES.shape[0]) == int(_ICD10_EMB_NAMES.shape[0])
    except Exception:
        _ICD10_EMB_MATRIX = None
        return False


def _get_icd10_sentence_model() -> Any:
    global _ICD10_ST_MODEL
    if _ICD10_ST_MODEL is not None:
        return _ICD10_ST_MODEL
    from sentence_transformers import SentenceTransformer

    _ICD10_ST_MODEL = SentenceTransformer(ICD10_BERT_MODEL)
    return _ICD10_ST_MODEL


def _icd10_bert_search(query: str, top_k: int = 8) -> list[tuple[str, str, float]]:
    if not ICD10_USE_BERT or not _try_load_icd10_vector_store() or _ICD10_EMB_MATRIX is None:
        return []
    q = str(query or "").strip()
    if not q:
        return []
    try:
        import numpy as np

        model = _get_icd10_sentence_model()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        qv = np.asarray(qv, dtype=np.float32)
        qv = np.nan_to_num(qv, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(qv))
        if norm <= 1e-12:
            return []
        sims = np.nan_to_num(_ICD10_EMB_MATRIX @ (qv / norm), nan=0.0, posinf=0.0, neginf=0.0)
        k = min(top_k, int(sims.shape[0]))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(str(_ICD10_EMB_CODES[int(i)]), str(_ICD10_EMB_NAMES[int(i)]), float(sims[int(i)])) for i in idx]
    except Exception:
        return []


def _local_lookup_icd10_fuzzy(diagnosis: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    _, name_to_entry, _ = _load_icd10_local()
    if not name_to_entry:
        return []
    try:
        from rapidfuzz import fuzz as rfuzz, process

        candidates = process.extract(diagnosis, list(name_to_entry.keys()), scorer=rfuzz.WRatio, limit=30)
        reranked = sorted(
            [(name_to_entry[name][0], name, rfuzz.ratio(diagnosis, name)) for name, _, _ in candidates if name in name_to_entry],
            key=lambda item: -item[2],
        )
        return reranked[:top_k]
    except Exception:
        return []


def _llm_cn_normalize(term: str) -> str | None:
    term = str(term or "").strip()
    if not term:
        return None
    cn_chars = sum(1 for c in term if "\u4e00" <= c <= "\u9fff")
    if cn_chars / max(len(term), 1) > 0.6 and len(term) >= 4:
        return None
    if term in _LLM_CN_NORMALIZE_CACHE:
        return _LLM_CN_NORMALIZE_CACHE[term]
    prompt = (
        "你是医学术语专家。请将以下诊断术语转换为中国 ICD-10 标准库中对应的标准中文名称。\n"
        "规则：\n"
        "  1. 若是英文缩写（如 bppv、UPVD、PPPD），输出其对应的标准中文全称。\n"
        "  2. 若是英文全称，直接翻译为中文标准术语。\n"
        "  3. 若无法确定对应的标准中文术语，输出 NULL。\n"
        "  4. 只输出中文术语本身，不加引号、不加标点、不做解释。\n\n"
        f"诊断术语：{term}\n标准中文名称（或 NULL）："
    )
    raw = _call_step6_llm(prompt, max_tokens=60).strip().strip("\"'").strip()
    result = None if not raw or raw.upper() == "NULL" else raw
    _LLM_CN_NORMALIZE_CACHE[term] = result
    return result


def _llm_select_icd10_candidate(diagnosis: str, candidates: list[tuple[str, str, float]]) -> tuple[str, str] | None:
    if not candidates:
        return None
    options_text = "\n".join(f"  {idx + 1}. {name}（编码: {code}）" for idx, (code, name, _) in enumerate(candidates))
    prompt = (
        f"你是医疗编码专家。需要将以下诊断名称映射到 ICD-10 编码。\n\n待映射诊断：{diagnosis}\n\n"
        f"候选 ICD-10 条目（来自中国国家标准库）：\n{options_text}\n\n"
        f"规则：\n  1. 若某个候选与待映射诊断是同一疾病，返回对应序号（1-{len(candidates)}）。\n"
        "  2. 若所有候选均与待映射诊断不是同一疾病，返回 0。\n  3. 只返回数字，不加任何解释。\n\n选择（数字）："
    )
    raw = _call_step6_llm(prompt, max_tokens=10)
    match = re.search(r"\d+", raw or "")
    if not match:
        return None
    idx = int(match.group())
    if 1 <= idx <= len(candidates):
        code, name, _ = candidates[idx - 1]
        return code, name
    return None


def _merge_icd_candidates_for_llm(bert_cands: list[tuple[str, str, float]], fuzzy_cands: list[tuple[str, str, float]], max_total: int = 8) -> list[tuple[str, str, float]]:
    seen: set[str] = set()
    merged: list[tuple[str, str, float]] = []
    for code, name, sim in bert_cands:
        if code in seen:
            continue
        seen.add(code)
        merged.append((code, name, sim * 100.0))
    for code, name, ratio in fuzzy_cands:
        if code in seen:
            continue
        seen.add(code)
        merged.append((code, name, ratio))
        if len(merged) >= max_total:
            break
    return merged[:max_total]


def _local_icd10_lookup(diagnosis: str) -> tuple[str, str, str]:
    diag_clean = str(diagnosis or "").strip()
    if not diag_clean or diag_clean.lower() in {"nan", "none", "null"}:
        return "NOT_FOUND", "NOT_FOUND", ""
    if diag_clean in _ICD10_CACHE:
        code, name = _ICD10_CACHE[diag_clean]
        return "CACHE", code, name
    _, name_to_entry, norm_to_entry = _load_icd10_local()

    def store(src: str, code: str, name: str) -> tuple[str, str, str]:
        _ICD10_CACHE[diag_clean] = (code, name)
        return src, code, name

    if diag_clean in name_to_entry:
        code, name = name_to_entry[diag_clean]
        return store("EXACT", code, name)
    norm = _normalize_diag(diag_clean)
    if norm in norm_to_entry:
        code, name = norm_to_entry[norm]
        return store("NORM", code, name)

    cn_term = _llm_cn_normalize(diag_clean)
    if cn_term:
        if cn_term in name_to_entry:
            code, name = name_to_entry[cn_term]
            return store("CN_NORM", code, name)
        norm2 = _normalize_diag(cn_term)
        if norm2 in norm_to_entry:
            code, name = norm_to_entry[norm2]
            return store("CN_NORM", code, name)
    query = cn_term or diag_clean
    bert_cands = _icd10_bert_search(query, top_k=8)
    if bert_cands and bert_cands[0][2] >= ICD10_BERT_MIN_SIM:
        code, name, _ = bert_cands[0]
        return store("BERT", code, name)
    fuzzy_cands = _local_lookup_icd10_fuzzy(query, top_k=5)
    if fuzzy_cands and fuzzy_cands[0][2] >= 85:
        code, name, _ = fuzzy_cands[0]
        return store("FUZZY", code, name)
    selected = _llm_select_icd10_candidate(diag_clean, _merge_icd_candidates_for_llm(bert_cands, fuzzy_cands))
    if selected:
        code, name = selected
        return store("LLM_SELECT", code, name)
    return store("NOT_FOUND", "NOT_FOUND", "")


def _llm_identify_diagnosis_fields(columns_with_samples: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    for col, samples in columns_with_samples.items():
        clean = [str(s)[:50] for s in samples if str(s).strip() and str(s).strip().lower() != "nan"]
        if clean:
            lines.append(f"  - {col}：示例值=[{', '.join(clean[:3])}]")
    if not lines:
        return []
    prompt = (
        "你是医疗数据专家。以下是一份医疗数据集的字段名称和示例值。\n"
        "请识别出所有可能包含疾病诊断信息的字段，包括主诊断、初步诊断、最终诊断、出院诊断、可能诊断、综合征类型、疾病类型、病因诊断、定位诊断。\n"
        "请只返回字段名列表，每行一个完整字段名，不要解释，不要序号，不要其他内容。\n\n字段清单：\n"
        + "\n".join(lines)
        + "\n\n含诊断信息的字段（每行一个字段名）："
    )
    raw = _call_step6_llm(prompt, max_tokens=800)
    found: list[str] = []
    for line in raw.splitlines():
        field = line.strip().lstrip("-•0123456789.、 ").strip()
        if field in columns_with_samples and field not in found:
            found.append(field)
    return found


def _llm_extract_disease_content(field_name: str, field_value: str) -> str | None:
    value = str(field_value or "").strip()
    if not value or value.lower() in {"nan", "none", "null", "无", "未填写", "—", "-"}:
        return None
    cache_key = (field_name, value)
    if cache_key in _LLM_DISEASE_EXTRACT_CACHE:
        return _LLM_DISEASE_EXTRACT_CACHE[cache_key]
    prompt = (
        f"字段名称：{field_name}\n字段内容：{value}\n\n"
        "请从上面的字段内容中提取疾病/诊断名称。\n"
        "规则：\n"
        "  1. 若内容是一个疾病/综合征名称，直接返回该名称。\n"
        "  2. 若内容包含多个疾病名称，逐一提取每个疾病名称，用 || 分隔输出。\n"
        "  3. 若内容混有非疾病信息，只保留疾病名称部分。\n"
        "  4. 若内容完全不含疾病信息，返回 NULL。\n"
        "  5. 只输出疾病名称，不加引号、不加标点、不做解释。\n\n输出（单个疾病名称 或 疾病1||疾病2||... 或 NULL）："
    )
    raw = _call_step6_llm(prompt, max_tokens=200).strip().strip("\"'").strip()
    result = None if not raw or raw.upper() == "NULL" else raw
    if result is None and not ICD10_USE_LLM:
        result = value
    _LLM_DISEASE_EXTRACT_CACHE[cache_key] = result
    return result


def _is_long_text_col(series: pd.Series) -> bool:
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    return bool((sample.str.len() > TEXT_SKIP_THRESHOLD).mean() >= 0.3)


def _should_exclude_feature(col: str, df: pd.DataFrame, label_column: str | None = None, source_columns: list[str] | None = None) -> tuple[bool, str]:
    low = col.lower()
    source_set = {c.lower() for c in (source_columns or [])}
    if col == "patient_id" or low in ID_PATTERNS or any(low.endswith("__" + p) or low == p for p in ID_PATTERNS):
        return True, "identifier"
    if label_column and col == label_column:
        return True, "label_column"
    if low in source_set:
        return True, "label_source_column"
    if any(pattern in low for pattern in TIME_PATTERNS):
        return True, "time_or_date_leakage"
    if any(pattern in low for pattern in LEAKAGE_PATTERNS):
        return True, "label_leakage_risk"
    if col in df.columns and df[col].isna().mean() > NULL_THRESHOLD:
        return True, "high_missing_rate"
    if col in df.columns and _is_long_text_col(df[col]):
        return True, "long_text_column"
    return False, ""


def _auto_feature_columns(df: pd.DataFrame, label_column: str | None = None, source_columns: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    selected: list[str] = []
    excluded: dict[str, str] = {}
    for col in df.columns:
        should_exclude, reason = _should_exclude_feature(str(col), df, label_column=label_column, source_columns=source_columns)
        if should_exclude:
            excluded[str(col)] = reason
        else:
            selected.append(str(col))
    return selected, excluded


def _label_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for col in df.columns:
        if col == "patient_id":
            continue
        low = str(col).lower()
        non_null = df[col].dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        missing_rate = 1 - len(non_null) / max(len(df), 1)
        unique_values = sorted(non_null.unique().tolist()[:20])
        is_candidate = (
            "icd" in low
            or "diagnos" in low
            or "diagnosis" in low
            or "discharge_location" in low
            or low in {"dod", "deathtime"}
            or (missing_rate < 0.70 and 2 <= len(unique_values) <= 20)
        )
        if is_candidate:
            candidates.append(
                {
                    "column": str(col),
                    "missing_rate": round(float(missing_rate), 4),
                    "unique_count": int(len(set(non_null.tolist()))),
                    "examples": unique_values[:8],
                    "kind": "icd_code" if "icd" in low else "candidate_label",
                }
            )
    return candidates


def load_task_context(input_csv: str, output_root: str, task_text: str = "", passed_patients_json: str | None = None, selection_report_path: str | None = None) -> dict[str, Any]:
    df, patient_col = _read_data(input_csv)
    passed_ids = _load_passed_ids(passed_patients_json) if passed_patients_json else []
    filt = _filter_passed(df, passed_ids)
    feature_cols, excluded = _auto_feature_columns(filt)
    label_candidates = _label_candidates(filt)
    icd_code_columns = [item["column"] for item in label_candidates if item.get("kind") == "icd_code"]
    context = {
        "input_csv": str(resolve_path(input_csv)),
        "task_text": task_text,
        "patient_id_column": patient_col,
        "passed_patients_json": str(resolve_path(passed_patients_json)) if passed_patients_json else "",
        "passed_patient_count": len(passed_ids) if passed_ids else int(df["patient_id"].nunique()),
        "rows_after_patient_filter": int(len(filt)),
        "patients_after_filter": int(filt["patient_id"].nunique()) if "patient_id" in filt.columns else 0,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "excluded_columns": excluded,
        "label_candidates": label_candidates,
        "icd_code_columns": icd_code_columns,
        "selection_report_path": str(resolve_path(selection_report_path)) if selection_report_path else "",
        "recommended_liver_icd_prefixes": list(LIVER_PREFIXES),
    }
    path = write_json(meta_dir(output_root) / "task_context.json", context)
    return {"task_context_json": path, **context}


def discover_diagnosis_fields(input_csv: str, output_root: str, passed_patients_json: str | None = None, max_fields: int = 30) -> dict[str, Any]:
    global _LLM_DIAG_FIELDS_CACHE
    df, _ = _read_data(input_csv)
    passed_ids = _load_passed_ids(passed_patients_json) if passed_patients_json else []
    filt = _filter_passed(df, passed_ids)
    col_samples: dict[str, list[str]] = {}
    for col in filt.columns:
        if col == "patient_id":
            continue
        low = str(col).lower()
        if not any(keyword in low or keyword in str(col) for keyword in DIAGNOSIS_FIELD_KEYWORDS):
            continue
        values = [str(v).strip() for v in filt[col].dropna().unique().tolist() if str(v).strip() and str(v).strip().lower() != "nan"]
        if values:
            col_samples[str(col)] = values[:5]
    if 0 < len(col_samples) <= 50:
        diag_fields = _llm_identify_diagnosis_fields(col_samples)
        if not diag_fields:
            diag_fields = list(col_samples.keys())[:max_fields]
    elif len(col_samples) > 50:
        diag_fields = sorted(col_samples.keys(), key=lambda c: filt[c].notna().mean() if c in filt.columns else 0, reverse=True)[:max_fields]
    else:
        diag_fields = []
    _LLM_DIAG_FIELDS_CACHE = diag_fields
    payload = {
        "input_csv": str(resolve_path(input_csv)),
        "candidate_field_count": len(col_samples),
        "diagnosis_fields": diag_fields,
        "field_samples": {field: col_samples.get(field, []) for field in diag_fields},
    }
    path = write_json(meta_dir(output_root) / f"diagnosis_fields_{_timestamp()}.json", payload)
    return {"diagnosis_fields_json": path, **payload}


def extract_and_map_all_diagnosis_fields(
    input_csv: str,
    diagnosis_fields: list[str],
    output_root: str,
    passed_patients_json: str | None = None,
    max_unique_per_field: int = 200,
) -> dict[str, Any]:
    df, _ = _read_data(input_csv)
    passed_ids = _load_passed_ids(passed_patients_json) if passed_patients_json else []
    filt = _filter_passed(df, passed_ids)
    icd10_mapping: dict[str, str] = {}
    field_details: dict[str, list[dict[str, Any]]] = {}
    for field in diagnosis_fields:
        if field not in filt.columns:
            field_details[field] = []
            continue
        counts = filt[field].dropna().astype(str).str.strip()
        counts = counts[(counts != "") & (counts.str.lower() != "nan")].value_counts()
        records: list[dict[str, Any]] = []
        for value in counts.head(max_unique_per_field).index.tolist():
            disease_raw = _llm_extract_disease_content(field, value)
            if disease_raw is None:
                records.append({"original_value": value, "extracted_disease": None, "source": None, "icd10": None, "local_preferred": None})
                continue
            for disease in [part.strip() for part in disease_raw.split("||") if part.strip()]:
                source, code, preferred = _local_icd10_lookup(disease)
                icd10_mapping[disease] = code
                records.append({"original_value": value, "extracted_disease": disease, "source": source, "icd10": code, "local_preferred": preferred})
        field_details[field] = records
    payload = {
        "input_csv": str(resolve_path(input_csv)),
        "diagnosis_fields": diagnosis_fields,
        "icd10_mapping": icd10_mapping,
        "field_details": field_details,
    }
    path = write_json(meta_dir(output_root) / f"diagnosis_icd10_mapping_{_timestamp()}.json", payload)
    return {"icd10_mapping_json": path, "mapped_diagnosis_count": len(icd10_mapping), **payload}


def map_diagnoses_to_icd10(diagnosis_names: list[str], output_root: str) -> dict[str, Any]:
    mapping: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    for name in diagnosis_names:
        disease_raw = _llm_extract_disease_content("诊断字段", str(name))
        if disease_raw is None:
            details.append({"input": name, "extracted_disease": None, "source": None, "icd10": None, "local_preferred": None})
            continue
        for disease in [part.strip() for part in disease_raw.split("||") if part.strip()]:
            source, code, preferred = _local_icd10_lookup(disease)
            mapping[disease] = code
            details.append({"input": name, "extracted_disease": disease, "source": source, "icd10": code, "local_preferred": preferred})
    payload = {"icd10_mapping": mapping, "details": details}
    path = write_json(meta_dir(output_root) / f"manual_icd10_mapping_{_timestamp()}.json", payload)
    return {"icd10_mapping_json": path, "mapped_diagnosis_count": len(mapping), **payload}


def create_icd_binary_label(
    input_csv: str,
    icd_codes_column: str,
    output_root: str,
    positive_icd_prefixes: list[str] | None = None,
    label_name_positive: str = "liver_disease",
    label_name_negative: str = "no_liver_disease",
    passed_patients_json: str | None = None,
    task_name: str = "icd_binary",
) -> dict[str, Any]:
    df, _ = _read_data(input_csv)
    passed_ids = _load_passed_ids(passed_patients_json) if passed_patients_json else []
    filt = _filter_passed(df, passed_ids)
    if icd_codes_column not in filt.columns:
        raise ValueError(f"ICD 编码列不存在: {icd_codes_column}")
    prefixes = [p.strip().upper().replace(".", "") for p in (positive_icd_prefixes or list(LIVER_PREFIXES))]

    def is_positive(value: Any) -> bool:
        codes = split_codes(value)
        return any(any(code.startswith(prefix) for prefix in prefixes) for code in codes)

    filt["_icd_label"] = filt[icd_codes_column].apply(lambda value: label_name_positive if is_positive(value) else label_name_negative)
    label_mapping = {label_name_negative: 0, label_name_positive: 1}
    root = step_root(output_root)
    ts = _timestamp()
    labeled_csv = write_rows(root / f"labeled_{task_name}_{ts}.csv", filt.to_dict(orient="records"), list(filt.columns))
    label_config = {
        "label_column": "_icd_label",
        "label_mapping": label_mapping,
        "label_source_column": icd_codes_column,
        "positive_icd_prefixes": prefixes,
        "task_name": task_name,
    }
    label_config_json = write_json(root / f"label_config_{task_name}_{ts}.json", label_config)
    dist = filt["_icd_label"].value_counts().to_dict()
    return {
        "labeled_csv": labeled_csv,
        "label_config_json": label_config_json,
        "label_column": "_icd_label",
        "label_mapping": label_mapping,
        "label_source_column": icd_codes_column,
        "samples": int(len(filt)),
        "positive_count": int(dist.get(label_name_positive, 0)),
        "negative_count": int(dist.get(label_name_negative, 0)),
        "positive_rate": round(float(dist.get(label_name_positive, 0)) / max(len(filt), 1), 4),
    }


def create_notnull_binary_label(
    input_csv: str,
    source_column: str,
    output_root: str,
    positive_condition: str = "not_null",
    label_name_positive: str = "positive",
    label_name_negative: str = "negative",
    passed_patients_json: str | None = None,
    task_name: str = "notnull_binary",
) -> dict[str, Any]:
    df, _ = _read_data(input_csv)
    passed_ids = _load_passed_ids(passed_patients_json) if passed_patients_json else []
    filt = _filter_passed(df, passed_ids)
    if source_column not in filt.columns:
        raise ValueError(f"来源列不存在: {source_column}")
    col = filt[source_column]
    if positive_condition == "not_null":
        mask = col.notna() & (col.astype(str).str.strip() != "")
    elif positive_condition == "is_null":
        mask = col.isna() | (col.astype(str).str.strip() == "")
    elif positive_condition == "gt_0":
        numeric = pd.to_numeric(col, errors="coerce")
        mask = numeric.notna() & (numeric > 0)
    elif positive_condition == "eq_0":
        numeric = pd.to_numeric(col, errors="coerce")
        mask = numeric.notna() & (numeric == 0)
    else:
        raise ValueError("positive_condition 只支持 not_null/is_null/gt_0/eq_0")
    filt["_notnull_label"] = mask.map({True: label_name_positive, False: label_name_negative})
    label_mapping = {label_name_negative: 0, label_name_positive: 1}
    root = step_root(output_root)
    ts = _timestamp()
    labeled_csv = write_rows(root / f"labeled_{task_name}_{ts}.csv", filt.to_dict(orient="records"), list(filt.columns))
    label_config_json = write_json(
        root / f"label_config_{task_name}_{ts}.json",
        {"label_column": "_notnull_label", "label_mapping": label_mapping, "label_source_column": source_column, "positive_condition": positive_condition, "task_name": task_name},
    )
    dist = filt["_notnull_label"].value_counts().to_dict()
    return {
        "labeled_csv": labeled_csv,
        "label_config_json": label_config_json,
        "label_column": "_notnull_label",
        "label_mapping": label_mapping,
        "label_source_column": source_column,
        "samples": int(len(filt)),
        "positive_count": int(dist.get(label_name_positive, 0)),
        "negative_count": int(dist.get(label_name_negative, 0)),
        "positive_rate": round(float(dist.get(label_name_positive, 0)) / max(len(filt), 1), 4),
    }


def propose_label_mapping(input_csv: str, label_column: str, output_root: str, passed_patients_json: str | None = None, mode: str = "multiclass_enum", positive_values: list[str] | None = None) -> dict[str, Any]:
    df, _ = _read_data(input_csv)
    passed_ids = _load_passed_ids(passed_patients_json) if passed_patients_json else []
    filt = _filter_passed(df, passed_ids)
    if label_column not in filt.columns:
        raise ValueError(f"标签列不存在: {label_column}")
    values = [str(v).strip() for v in filt[label_column].dropna().tolist() if str(v).strip() and str(v).strip().lower() != "nan"]
    uniques = sorted(set(values))
    if len(uniques) < 2:
        raise ValueError(f"标签列有效类别数 < 2: {label_column}, uniques={uniques}")
    if mode == "binary_positive_vs_rest":
        positives = set(positive_values or [])
        if not positives:
            raise ValueError("binary_positive_vs_rest 需要 positive_values")
        mapping = {v: (1 if v in positives else 0) for v in uniques}
    else:
        mapping = {v: idx for idx, v in enumerate(uniques)}
    if len(set(mapping.values())) < 2:
        raise ValueError(f"标签映射后类别数 < 2: {mapping}")
    path = write_json(step_root(output_root) / f"label_mapping_{_timestamp()}.json", {"label_column": label_column, "label_mapping": mapping, "mode": mode})
    return {"label_mapping_json": path, "label_column": label_column, "label_mapping": mapping, "unique_values": uniques}


def build_label_series(input_csv: str, label_column: str, label_mapping: dict[str, int], output_root: str, passed_patients_json: str | None = None) -> dict[str, Any]:
    df, _ = _read_data(input_csv)
    passed_ids = _load_passed_ids(passed_patients_json) if passed_patients_json else []
    filt = _filter_passed(df, passed_ids)
    rows: list[dict[str, Any]] = []
    for idx, row in filt.iterrows():
        raw = row.get(label_column)
        label_raw = None if is_missing_like(raw) else str(raw).strip()
        rows.append({"sample_row_id": int(idx), "patient_id": str(row.get("patient_id", "")), "label_raw": label_raw, "y": label_mapping.get(label_raw) if label_raw is not None else None})
    path = write_json(step_root(output_root) / f"label_series_{_timestamp()}.json", rows)
    valid = [item for item in rows if item["y"] is not None]
    return {"label_series_json": path, "rows": len(rows), "valid_labels": len(valid), "classes": sorted({item["y"] for item in valid})}


def select_feature_columns(labeled_csv: str, output_root: str, label_column: str, label_source_column: str | None = None, max_missing_rate: float = NULL_THRESHOLD) -> dict[str, Any]:
    df, _ = _read_data(labeled_csv)
    global NULL_THRESHOLD
    old_threshold = NULL_THRESHOLD
    NULL_THRESHOLD = max_missing_rate
    try:
        features, excluded = _auto_feature_columns(df, label_column=label_column, source_columns=[label_source_column] if label_source_column else [])
    finally:
        NULL_THRESHOLD = old_threshold
    path = write_json(step_root(output_root) / f"feature_columns_{_timestamp()}.json", {"feature_columns": features, "excluded_columns": excluded, "label_column": label_column, "label_source_column": label_source_column})
    return {"feature_columns_json": path, "feature_count": len(features), "excluded_columns": excluded}


def _cell_value(value: Any) -> Any:
    if is_missing_like(value):
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        return round(float(numeric), 6)
    return str(value).strip()


def format_and_save_ml_dataset(
    labeled_csv: str,
    feature_columns_json: str,
    output_root: str,
    label_column: str,
    label_mapping: dict[str, int],
    task_name: str = "ml_dataset",
    icd10_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    df, _ = _read_data(labeled_csv)
    feature_payload = json.loads(resolve_path(feature_columns_json).read_text(encoding="utf-8"))
    features = [col for col in feature_payload.get("feature_columns", []) if col in df.columns and col != label_column]
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        label_name = None if is_missing_like(row.get(label_column)) else str(row.get(label_column)).strip()
        y = label_mapping.get(label_name) if label_name is not None else None
        if y is None:
            continue
        out = {"sample_row_id": int(idx), "patient_id": str(row.get("patient_id", ""))}
        for col in features:
            out[col] = _cell_value(row.get(col))
        out["label_name"] = label_name
        out["y"] = int(y)
        rows.append(out)
    classes = sorted({row["y"] for row in rows})
    if len(classes) < 2:
        raise ValueError(f"有效标签类别少于 2 类，无法保存 ML 数据集: classes={classes}, samples={len(rows)}")
    df_train = pd.DataFrame(rows)
    effective_icd_map = dict(icd10_mapping or {})

    diag_feature_cols = [
        col
        for col in features
        if col in df_train.columns
        and (
            col in (_LLM_DIAG_FIELDS_CACHE or [])
            or any(keyword in str(col).lower() or keyword in str(col) for keyword in DIAGNOSIS_FIELD_KEYWORDS)
        )
    ]

    def disease_and_icd(col: str, raw_value: Any) -> tuple[str | None, str | None]:
        if is_missing_like(raw_value):
            return None, None
        value = str(raw_value).strip()
        disease = _llm_extract_disease_content(col, value)
        if not disease:
            return None, None
        primary = disease.split("||")[0].strip()
        code = effective_icd_map.get(primary)
        if code is None:
            _, code, _ = _local_icd10_lookup(primary)
            effective_icd_map[primary] = code
        return primary, code if code and code != "NOT_FOUND" else None

    for col in diag_feature_cols:
        diseases: list[str | None] = []
        codes: list[str | None] = []
        for _, row in df_train.iterrows():
            disease, code = disease_and_icd(col, row.get(col))
            diseases.append(disease)
            codes.append(code)
        safe_col = str(col).split("-")[-1]
        df_train[f"{safe_col}_疾病名称"] = diseases
        df_train[f"{safe_col}_ICD10"] = codes

    def label_to_icd(label_name: Any) -> str | None:
        if is_missing_like(label_name):
            return None
        label = str(label_name).strip()
        code = effective_icd_map.get(label)
        if code is None:
            _, code, _ = _local_icd10_lookup(label)
            effective_icd_map[label] = code
        return code if code and code != "NOT_FOUND" else None

    df_train["label_ICD10"] = df_train["label_name"].apply(label_to_icd)
    rows = df_train.to_dict(orient="records")
    root = step_root(output_root)
    ts = _timestamp()
    safe_task = re.sub(r"[^A-Za-z0-9_\-]+", "_", task_name).strip("_") or "ml_dataset"
    csv_path = write_rows(root / f"ml_dataset_{safe_task}_{ts}.csv", rows)
    json_path = write_json(root / f"ml_dataset_{safe_task}_{ts}.json", {"samples": rows, "feature_names": features, "label_column": label_column, "label_mapping": label_mapping, "icd10_mapping": effective_icd_map})
    jsonl_path = write_jsonl(root / f"ml_dataset_{safe_task}_{ts}.jsonl", rows)
    label_dist = {str(k): sum(1 for row in rows if row["y"] == k) for k in classes}
    n_samples = len(rows)
    n_features = len(features)
    imbalance = max(label_dist.values()) / max(min(label_dist.values()), 1)
    model_config = {
        "timestamp": datetime.now().isoformat(),
        "dataset_summary": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": len(classes),
            "class_distribution": label_dist,
            "imbalance_ratio": round(float(imbalance), 2),
            "task_type": "binary_classification" if len(classes) == 2 else "multiclass_classification",
        },
        "feature_names": features,
        "label_column": label_column,
        "label_mapping": label_mapping,
        "icd10_mapping": effective_icd_map,
        "icd10_enhanced_columns": [col for col in df_train.columns if str(col).endswith("_疾病名称") or str(col).endswith("_ICD10") or col == "label_ICD10"],
        "recommended_models": [
            {"model": "LogisticRegression", "library": "sklearn.linear_model", "reason": "可解释性强，适合医疗表格基线", "params": {"class_weight": "balanced" if imbalance > 2 else None, "max_iter": 1000}},
            {"model": "SVC", "library": "sklearn.svm", "reason": "小样本高维场景可作为稳健基线", "params": {"class_weight": "balanced" if imbalance > 2 else None, "probability": True}},
            {"model": "RandomForestClassifier", "library": "sklearn.ensemble", "reason": "混合特征鲁棒，便于特征重要性分析", "params": {"class_weight": "balanced" if imbalance > 2 else None, "random_state": 42}},
        ],
        "data_files": {"csv": csv_path, "json": json_path, "jsonl": jsonl_path},
    }
    config_path = write_json(root / f"model_config_{safe_task}_{ts}.json", model_config)
    next_dir = ensure_dir(root / "next_input")
    write_json(next_dir / "latest_dataset.json", {"dataset_csv": csv_path, "dataset_json": json_path, "dataset_jsonl": jsonl_path, "model_config_json": config_path})
    return {"dataset_csv": csv_path, "dataset_json": json_path, "dataset_jsonl": jsonl_path, "model_config_json": config_path, "samples": n_samples, "features": n_features, "class_distribution": label_dist, "icd10_mapping": effective_icd_map}


def validate_dataset(dataset_csv: str, model_config_json: str) -> dict[str, Any]:
    df = pd.read_csv(dataset_csv, dtype=object)
    config = json.loads(resolve_path(model_config_json).read_text(encoding="utf-8"))
    issues: list[str] = []
    if "y" not in df.columns:
        issues.append("缺少标签列 y")
    if "label_name" not in df.columns:
        issues.append("缺少 label_name 列")
    classes = sorted(df["y"].dropna().astype(str).unique().tolist()) if "y" in df.columns else []
    if len(classes) < 2:
        issues.append("有效标签类别少于 2 类")
    features = config.get("feature_names") or config.get("features") or []
    for col in features:
        low = str(col).lower()
        if any(pattern in low for pattern in LEAKAGE_PATTERNS) or any(pattern in low for pattern in TIME_PATTERNS):
            issues.append(f"特征列存在标签泄漏/时间泄漏风险: {col}")
    return {"passed": not issues, "issues": issues, "rows": int(len(df)), "columns": int(len(df.columns)), "classes": classes, "feature_count": len(features)}
