import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentscope.tool import ToolResponse


REME_MEMORY_ROOT = Path(__file__).parent / "reme_memory"
STEP1_SCRIPT_CASES_PATH = REME_MEMORY_ROOT / "cases" / "step1_script_cases.jsonl"
REME_CASE_SCHEMA_VERSION = 1
MAX_SAMPLE_FILES = 20
MAX_SCHEMA_FILES = 8


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=json.dumps(payload, ensure_ascii=False, sort_keys=True))


def ensure_reme_memory_store() -> dict[str, Any]:
    STEP1_SCRIPT_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STEP1_SCRIPT_CASES_PATH.exists():
        STEP1_SCRIPT_CASES_PATH.write_text("", encoding="utf-8")
    return {
        "reme_memory_root": str(REME_MEMORY_ROOT),
        "cases_dir": str(STEP1_SCRIPT_CASES_PATH.parent),
        "step1_script_cases_path": str(STEP1_SCRIPT_CASES_PATH),
        "storage_exists": STEP1_SCRIPT_CASES_PATH.exists(),
    }


def initialize_reme_memory_store() -> ToolResponse:
    store = ensure_reme_memory_store()
    return _json_response({"status": "OK", "backend": "jsonl_fallback", **store})


def _normalize_suffix(path: Path) -> str:
    return path.suffix.lower() or "<no_suffix>"


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_visible_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    return [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and not any(part.startswith(".") for part in item.parts)
        and "reorganized_output" not in item.parts
    ]


def _relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return [str(item).strip() for item in next(reader, []) if str(item).strip()]
    except Exception:
        return []


def _build_schema_summary(files: list[Path], root: Path) -> dict[str, list[str]]:
    schema_summary: dict[str, list[str]] = {}
    for path in files:
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        header = _read_csv_header(path)
        if header:
            schema_summary[_relative_display(path, root)] = header[:50]
        if len(schema_summary) >= MAX_SCHEMA_FILES:
            break
    return schema_summary


def build_step1_input_profile(input_path: str) -> ToolResponse:
    path = Path(input_path).expanduser()
    exists = path.exists()
    path_type = "missing"
    if path.is_file():
        path_type = "file"
    elif path.is_dir():
        path_type = "directory"

    files = _iter_visible_files(path)
    suffix_counts: dict[str, int] = {}
    total_size = 0
    for file_path in files:
        suffix = _normalize_suffix(file_path)
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        try:
            total_size += file_path.stat().st_size
        except OSError:
            pass

    root = path if path.is_dir() else path.parent
    sample_files = [_relative_display(item, root) for item in files[:MAX_SAMPLE_FILES]]
    schema_summary = _build_schema_summary(files, root)
    profile = {
        "input_path": str(path),
        "exists": exists,
        "path_type": path_type,
        "file_count": len(files),
        "total_size": total_size,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "sample_files": sample_files,
        "schema_summary": schema_summary,
    }
    profile["profile_hash"] = _sha256_text(
        _stable_json(
            {
                "path_type": profile["path_type"],
                "file_count": profile["file_count"],
                "suffix_counts": profile["suffix_counts"],
                "schema_summary": profile["schema_summary"],
            }
        )
    )
    return _json_response(profile)


def _load_cases() -> list[dict[str, Any]]:
    if not STEP1_SCRIPT_CASES_PATH.exists():
        return []
    cases: list[dict[str, Any]] = []
    for line in STEP1_SCRIPT_CASES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            cases.append(item)
    return cases


def _append_case(case: dict[str, Any]) -> None:
    ensure_reme_memory_store()
    with STEP1_SCRIPT_CASES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(_stable_json(case) + "\n")


def _coerce_payload(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}
    return dict(payload or {})


def record_step1_script_case(case_payload: str | dict[str, Any]) -> ToolResponse:
    payload = _coerce_payload(case_payload)
    script_path = Path(str(payload.get("script_path") or "")).expanduser()
    input_profile = payload.get("input_profile")
    if not isinstance(input_profile, dict):
        input_profile = {}
    input_profile_hash = str(input_profile.get("profile_hash") or "") or _sha256_text(_stable_json(input_profile))

    script_hash = ""
    script_size = 0
    script_exists = script_path.exists() and script_path.is_file()
    if script_exists:
        script_hash = _sha256_file(script_path)
        script_size = script_path.stat().st_size

    case_identity = {
        "memory_type": "step1_script_case",
        "input_profile_hash": input_profile_hash,
        "script_path": str(script_path),
        "script_hash": script_hash,
    }
    case = {
        "case_id": "step1_" + _sha256_text(_stable_json(case_identity))[:16],
        "schema_version": REME_CASE_SCHEMA_VERSION,
        "memory_type": "step1_script_case",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_profile": input_profile,
        "script_path": str(script_path),
        "script_hash": script_hash,
        "script_size": script_size,
        "execution": payload.get("execution") if isinstance(payload.get("execution"), dict) else {},
        "summary": str(payload.get("summary") or "").strip(),
        "backend": "jsonl_fallback",
        "script_exists": script_exists,
    }
    existing = next((item for item in _load_cases() if item.get("case_id") == case["case_id"]), None)
    if existing:
        return _json_response({"status": "EXISTS", "case": existing})
    _append_case(case)
    return _json_response({"status": "RECORDED", "case": case})


def _suffix_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_counts = left.get("suffix_counts") if isinstance(left.get("suffix_counts"), dict) else {}
    right_counts = right.get("suffix_counts") if isinstance(right.get("suffix_counts"), dict) else {}
    suffixes = set(left_counts) | set(right_counts)
    if not suffixes:
        return 1.0
    overlap = sum(min(int(left_counts.get(s, 0)), int(right_counts.get(s, 0))) for s in suffixes)
    union = sum(max(int(left_counts.get(s, 0)), int(right_counts.get(s, 0))) for s in suffixes)
    return overlap / union if union else 0.0


def _file_count_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_count = int(left.get("file_count") or 0)
    right_count = int(right.get("file_count") or 0)
    if left_count == right_count:
        return 1.0
    high = max(left_count, right_count)
    low = min(left_count, right_count)
    return low / high if high else 0.0


def _schema_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_schema = left.get("schema_summary") if isinstance(left.get("schema_summary"), dict) else {}
    right_schema = right.get("schema_summary") if isinstance(right.get("schema_summary"), dict) else {}
    left_cols = {str(col).lower() for cols in left_schema.values() for col in (cols if isinstance(cols, list) else [])}
    right_cols = {str(col).lower() for cols in right_schema.values() for col in (cols if isinstance(cols, list) else [])}
    if not left_cols and not right_cols:
        return 1.0
    if not left_cols or not right_cols:
        return 0.0
    return len(left_cols & right_cols) / len(left_cols | right_cols)


def _case_similarity(query_profile: dict[str, Any], case: dict[str, Any]) -> float:
    case_profile = case.get("input_profile") if isinstance(case.get("input_profile"), dict) else {}
    path_type_score = 1.0 if query_profile.get("path_type") == case_profile.get("path_type") else 0.0
    score = (
        0.55 * _suffix_similarity(query_profile, case_profile)
        + 0.25 * _file_count_similarity(query_profile, case_profile)
        + 0.15 * _schema_similarity(query_profile, case_profile)
        + 0.05 * path_type_score
    )
    return round(min(max(score, 0.0), 1.0), 6)


def retrieve_step1_script_case(input_profile: str | dict[str, Any], limit: int = 3) -> ToolResponse:
    query_profile = _coerce_payload(input_profile)
    limit = max(1, int(limit or 3))
    ranked_cases = []
    for case in _load_cases():
        if case.get("memory_type") != "step1_script_case":
            continue
        candidate = dict(case)
        candidate["similarity_score"] = _case_similarity(query_profile, case)
        ranked_cases.append(candidate)
    ranked_cases.sort(key=lambda item: (item.get("similarity_score", 0.0), item.get("created_at", "")), reverse=True)
    return _json_response(
        {
            "status": "OK",
            "backend": "jsonl_fallback",
            "candidate_count": len(ranked_cases),
            "candidates": ranked_cases[:limit],
        }
    )


def list_step1_script_cases(limit: int = 20) -> ToolResponse:
    limit = max(1, int(limit or 20))
    cases = [case for case in _load_cases() if case.get("memory_type") == "step1_script_case"]
    cases.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return _json_response({"status": "OK", "backend": "jsonl_fallback", "cases": cases[:limit]})


def _real_reme_available() -> bool:
    return importlib.util.find_spec("reme_ai") is not None or importlib.util.find_spec("reme") is not None


def get_reme_memory_status() -> ToolResponse:
    store = ensure_reme_memory_store()
    cases = [case for case in _load_cases() if case.get("memory_type") == "step1_script_case"]
    real_reme_available = _real_reme_available()
    return _json_response(
        {
            "status": "OK",
            "backend": "jsonl_fallback",
            "real_reme_available": real_reme_available,
            "case_count": len(cases),
            **store,
        }
    )
