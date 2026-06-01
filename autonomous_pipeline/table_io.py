from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "unknown"}
PATIENT_COLUMN_CANDIDATES = ("patient_id", "subject_id", "hadm_id", "患者ID", "病人ID")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".xml"}
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_rows(path: str | Path, limit: int | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("读取 Excel 需要 pandas/openpyxl") from exc
        df = pd.read_excel(p)
        if limit is not None:
            df = df.head(limit)
        return list(df.columns.astype(str)), df.where(df.notna(), None).to_dict(orient="records")
    delimiter = "\t" if suffix == ".tsv" else ","
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            with p.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = []
                for i, row in enumerate(reader):
                    if limit is not None and i >= limit:
                        break
                    rows.append(dict(row))
                return list(reader.fieldnames or []), rows
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法读取表格: {p}")


def write_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(p)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(p)


def file_modality(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in TABLE_SUFFIXES:
        return "table"
    return "other"


def detect_patient_column(columns: Iterable[str]) -> str | None:
    columns = list(columns)
    lower_map = {str(c).lower(): str(c) for c in columns}
    for name in PATIENT_COLUMN_CANDIDATES:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for col in columns:
        low = str(col).lower()
        if "patient" in low or "subject" in low or "患者" in str(col):
            return str(col)
    return None


def is_missing_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def compact_value(value: Any, limit: int = 160) -> str:
    text = "" if value is None else re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit] + "..." if len(text) > limit else text


def split_codes(value: Any) -> list[str]:
    if is_missing_like(value):
        return []
    codes: list[str] = []
    for part in re.split(r"[|,;\s]+", str(value)):
        token = part.strip().upper()
        if not token:
            continue
        if re.fullmatch(r"\d+\.0+", token):
            token = token.split(".", 1)[0]
        else:
            token = token.replace(".", "")
        if token and token not in codes:
            codes.append(token)
    return codes


def safe_copy(src: str | Path, dst: str | Path) -> str:
    source = Path(src)
    target = Path(dst)
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return str(target)
