from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from autonomous_pipeline.config import get_agent_config
from autonomous_pipeline.table_io import detect_patient_column, ensure_dir, is_missing_like, write_json
from autonomous_pipeline.steps.step2.prompt import EXTRACT_PROMPT, ENTITY_FIELDS, ENTITY_SUBFIELDS, STANDARDIZE_PROMPT, build_fill_mapping_prompt

TEXT_COLUMN_HINTS = (
    "text",
    "report_text",
    "notes",
    "note",
    "finding",
    "findings",
    "impression",
    "description",
    "comment",
    "narrative",
    "indication",
    "conclusion",
    "summary",
    "discharge",
    "radiology",
    "pathology",
    "history",
    "assessment",
)
DIRECTORY_TEXT_COLUMNS = {"text", "report_text", "notes", "findings", "impression", "description"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}
DIRECTORY_HINTS = ("mimic", "目录", "结构化", "notes", "structured", "report_text", "不需要ocr", "跳过ocr")
OCR_HINTS = ("ocr", "图片", "回填", "模板", "填回表格", "rawdata")
_OCR_LOCAL = threading.local()
_OCR_INIT_LOCK = threading.Lock()


def step_root(output_root: str | Path) -> Path:
    return ensure_dir(Path(output_root) / "step2_results")


def meta_dir(output_root: str | Path) -> Path:
    return ensure_dir(step_root(output_root) / "_meta")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def results_dir(output_root: str | Path, existing_path: str | Path | None = None) -> Path:
    if existing_path:
        parent = Path(existing_path).parent
        if parent.name.startswith("results_"):
            return parent
    return ensure_dir(step_root(output_root) / f"results_{timestamp()}")


def emit(message: str) -> None:
    enabled = str(os.environ.get("STEP2_PROGRESS_ENABLED", "true")).strip().lower() not in {"0", "false", "no", "off"}
    if enabled:
        print(message, file=sys.stderr, flush=True)


def progress(label: str, completed: int, total: int, last_bucket: int) -> int:
    if total <= 0:
        return last_bucket
    percent = completed / total * 100
    bucket = int(percent // 5)
    if completed == 1 or completed == total or bucket > last_bucket:
        emit(f"[Step2][Progress] {label}: {completed}/{total} ({percent:.1f}%)")
        return bucket
    return last_bucket


def append_tool_trace(output_root: str | Path, tool: str, arguments: dict[str, Any], payload: dict[str, Any], started_at: float) -> str:
    path = meta_dir(output_root) / "tool_trace.json"
    try:
        trace = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        trace = []
    trace.append(
        {
            "tool": tool,
            "arguments": _compact_args(arguments),
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "artifacts": _compact_args(payload.get("artifacts", {})),
            "issues": payload.get("issues", []),
            "duration_ms": int((time.time() - started_at) * 1000),
        }
    )
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _compact_args(value: Any, limit: int = 600) -> Any:
    if isinstance(value, dict):
        return {k: _compact_args(v, limit) for k, v in value.items() if k != "reason"}
    if isinstance(value, list):
        return [_compact_args(v, limit) for v in value[:20]] + ([{"truncated": len(value) - 20}] if len(value) > 20 else [])
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


def read_csv_rows(path: str | Path, limit: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    suffix = Path(path).suffix.lower()
    delimiter = "\t" if suffix == ".tsv" else ","
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with Path(path).open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows: list[dict[str, str]] = []
                for idx, row in enumerate(reader):
                    if limit is not None and idx >= limit:
                        break
                    rows.append({str(k): "" if v is None else str(v) for k, v in row.items() if k is not None})
                return list(reader.fieldnames or []), rows
        except UnicodeDecodeError:
            continue
    return [], []


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    path = Path(path)
    ensure_dir(path.parent)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return str(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _is_table(path: Path) -> bool:
    return path.suffix.lower() in TABLE_SUFFIXES


def _safe_name(value: Any) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip())
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return text.strip("_") or "value"


def _row_hadm(row: dict[str, Any]) -> str:
    raw = str(row.get("hadm_id", "") or "").strip()
    return raw.split(".", 1)[0] if raw else ""


def _rows_for_hadm(rows: list[dict[str, str]], hadm_id: str) -> list[dict[str, str]]:
    if not hadm_id:
        return rows
    matched = [row for row in rows if _row_hadm(row) == hadm_id]
    return matched or [row for row in rows if not _row_hadm(row)]


def _join_unique(values: Iterable[Any], limit: int = 50000) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    total = 0
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if total + len(text) > limit:
            parts.append("<truncated>")
            break
        parts.append(text)
        total += len(text)
    return " | ".join(parts)


def _has_text_column(header: list[str]) -> bool:
    return any(_is_text_column(col) for col in header)


def _text_columns(header: list[str]) -> list[str]:
    return [col for col in header if _is_text_column(col)]


def _is_text_column(col: str) -> bool:
    name = str(col).strip().lower()
    if not name:
        return False
    if name in {"id", "row_id", "note_id", "subject_id", "patient_id", "hadm_id", "stay_id", "study_id", "dicom_id"}:
        return False
    if name.endswith("_id") or name.endswith("_code") or name.endswith("code"):
        return False
    return name in DIRECTORY_TEXT_COLUMNS


def resolve_input(input_path: str) -> dict[str, Any]:
    path = Path(str(input_path or "")).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Step2 输入不存在: {path}")
    if not path.is_dir():
        raise ValueError(f"Step2 只接受 Step1 输出目录输入，不接受单文件: {path}")
    return {"input_path": str(path), "is_dir": True, "data_type": "directory", "mode": "auto"}


def _patient_dirs(input_path: str | Path) -> list[Path]:
    root = Path(input_path)
    return [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith("_")]


def scan_input(input_path: str, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(input_path)
    if not root.is_dir():
        raise ValueError(f"不是目录: {input_path}")
    patients = _patient_dirs(root)
    image_count = 0
    notes_csv_count = 0
    structured_csv_count = 0
    text_csv_count = 0
    csv_file_count = 0
    sample_text_csvs: list[str] = []
    patient_summaries: dict[str, Any] = {}
    for patient_dir in patients:
        table_dir = patient_dir / "table"
        images = [p for p in patient_dir.rglob("*") if p.is_file() and _is_image(p)]
        image_count += len(images)
        table_files = [p for p in table_dir.rglob("*") if p.is_file() and _is_table(p)] if table_dir.is_dir() else []
        csv_file_count += len([p for p in table_files if p.suffix.lower() in {".csv", ".tsv"}])
        for csv_path in table_files:
            parts = {part.lower() for part in csv_path.relative_to(patient_dir).parts}
            if "notes" in parts:
                notes_csv_count += 1
            if "structured" in parts:
                structured_csv_count += 1
            header, _ = read_csv_rows(csv_path, limit=0)
            if _has_text_column(header):
                text_csv_count += 1
                if len(sample_text_csvs) < 5:
                    sample_text_csvs.append(str(csv_path))
        patient_summaries[patient_dir.name] = {
            "table_files": len(table_files),
            "image_files": len(images),
            "has_table_dir": table_dir.is_dir(),
        }
    return {
        "input_path": str(root),
        "patient_count": len(patients),
        "image_count": image_count,
        "csv_file_count": csv_file_count,
        "notes_csv_count": notes_csv_count,
        "structured_csv_count": structured_csv_count,
        "text_csv_count": text_csv_count,
        "sample_text_csvs": sample_text_csvs,
        "patients": patient_summaries,
    }


def select_pipeline(input_path: str, scan_artifacts: dict[str, Any] | None = None, requested_mode: str = "auto", user_hint: str = "") -> dict[str, Any]:
    requested = str(requested_mode or "auto").strip().lower()
    artifacts = scan_artifacts or scan_input(input_path)
    notes_csv_count = int(artifacts.get("notes_csv_count") or 0)
    structured_csv_count = int(artifacts.get("structured_csv_count") or 0)
    text_csv_count = int(artifacts.get("text_csv_count") or 0)
    csv_file_count = int(artifacts.get("csv_file_count") or 0)
    image_count = int(artifacts.get("image_count") or 0)
    has_direct_tables = bool(notes_csv_count or structured_csv_count or text_csv_count or csv_file_count)

    if requested in {"directory", "ocr_fill"}:
        if requested == "directory" and not has_direct_tables and image_count:
            return {
                "input_path": input_path,
                "requested_mode": requested,
                "resolved_mode": "ocr_fill",
                "mode_reason": "扫描结果显示没有可直接结构化的 CSV/text 表但存在图片，覆盖不兼容的 directory 请求，选择 OCR 回填模式。",
            }
        if requested == "ocr_fill" and has_direct_tables and not image_count:
            return {
                "input_path": input_path,
                "requested_mode": requested,
                "resolved_mode": "directory",
                "mode_reason": "扫描结果显示存在可直接结构化的表且没有图片，覆盖不兼容的 OCR 请求，选择目录结构化模式。",
            }
        return {"input_path": input_path, "requested_mode": requested, "resolved_mode": requested, "mode_reason": "用户显式指定模式。"}
    compact_hint = re.sub(r"[\s_\-—–]+", "", str(user_hint or "").lower())
    if any(token in compact_hint for token in DIRECTORY_HINTS) and has_direct_tables:
        return {"input_path": input_path, "requested_mode": requested, "resolved_mode": "directory", "mode_reason": "用户提示包含目录/结构化信息。"}
    if any(token in compact_hint for token in OCR_HINTS):
        return {"input_path": input_path, "requested_mode": requested, "resolved_mode": "ocr_fill", "mode_reason": "用户提示包含 OCR/回填信息。"}
    if has_direct_tables:
        return {
            "input_path": input_path,
            "requested_mode": requested,
            "resolved_mode": "directory",
            "mode_reason": "检测到 notes/structured 表或文本列，选择目录结构化模式。",
        }
    return {
        "input_path": input_path,
        "requested_mode": requested,
        "resolved_mode": "ocr_fill",
        "mode_reason": "未检测到可直接结构化的 notes/structured 表，选择 OCR 回填模式。",
    }


def _load_patient_table_data(patient_dir: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    table_dir = patient_dir / "table"
    structured: dict[str, list[dict[str, str]]] = {}
    text_sources: list[dict[str, Any]] = []
    if not table_dir.is_dir():
        return structured, text_sources
    for csv_path in sorted(p for p in table_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}):
        rel = csv_path.relative_to(table_dir)
        parts = [part.lower() for part in rel.parts]
        header, rows = read_csv_rows(csv_path)
        if not header:
            continue
        text_cols = _text_columns(header)
        if text_cols and "structured" not in parts:
            text_sources.append(
                {
                    "path": str(csv_path),
                    "relative_path": str(rel),
                    "source_category": "notes" if "notes" in parts else "table",
                    "text_columns": text_cols,
                    "rows": rows,
                }
            )
            continue
        key = csv_path.stem
        structured.setdefault(key, []).extend(rows)
    return structured, text_sources


def collect_directory_payloads(input_path: str, output_root: str | Path) -> dict[str, Any]:
    emit(f"[Step2][Runtime] directory payload 收集开始: {input_path}")
    patients = _patient_dirs(input_path)
    payloads: dict[str, Any] = {}
    text_tasks: list[dict[str, str]] = []
    completed = 0
    last_bucket = -1
    for patient_dir in patients:
        pid = patient_dir.name
        structured, text_sources = _load_patient_table_data(patient_dir)
        payloads[pid] = {"structured": structured, "table_dir": str(patient_dir / "table"), "text_tasks": []}
        for source in text_sources:
            source_file = Path(source["path"]).name
            for row_idx, row in enumerate(source["rows"]):
                hadm_id = _row_hadm(row)
                row_id = str(row.get("note_id") or row.get("dicom_id") or row.get("study_id") or row_idx)
                for col in source["text_columns"]:
                    text = str(row.get(col, "") or "").strip()
                    if len(text) < 50:
                        continue
                    task = {
                        "patient_id": pid,
                        "hadm_id": hadm_id,
                        "row_id": row_id,
                        "source_file": source_file,
                        "source_category": str(source["source_category"]),
                        "text_column": str(col),
                        "text": text,
                    }
                    text_tasks.append(task)
                    payloads[pid]["text_tasks"].append(task)
        completed += 1
        last_bucket = progress("directory payloads", completed, len(patients), last_bucket)
    out = meta_dir(output_root)
    patient_payloads_json = write_json(out / "patient_payloads.json", payloads)
    text_tasks_jsonl = write_jsonl(out / "text_tasks.jsonl", text_tasks)
    emit(f"[Step2][Runtime] directory payload 收集完成: patients={len(payloads)}, texts={len(text_tasks)}")
    return {
        "patient_payloads_json": patient_payloads_json,
        "text_tasks_jsonl": text_tasks_jsonl,
        "patient_count": len(payloads),
        "text_task_count": len(text_tasks),
    }


async def _close_async_client(client: Any) -> None:
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if not close:
        return
    result = close()
    if asyncio.iscoroutine(result):
        await result


async def _call_chat(prompt: str, client: Any | None = None) -> str | None:
    cfg = get_agent_config("step2")
    if not cfg.has_credentials:
        return None
    import openai

    owns_client = client is None
    client = client or openai.AsyncOpenAI(api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY"), base_url=cfg.base_url, timeout=cfg.timeout)
    try:
        response = await client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip() if response.choices else None
    finally:
        if owns_client:
            await _close_async_client(client)


def _parse_json_response(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    for marker in ("```json", "```"):
        if marker in content:
            try:
                return json.loads(content.split(marker, 1)[1].split("```", 1)[0].strip())
            except Exception:
                pass
    try:
        start, end = content.find("{"), content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        return None
    return None


NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _short_text(value: Any, max_words: int = 10) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _normalize_entity(raw: dict[str, Any]) -> dict[str, Any] | None:
    item = dict(raw)
    category = str(item.get("category") or "")
    value = item.get("value", "")
    if isinstance(value, str) and re.search(r"\d+\s*[~～\-]\s*\d+", value):
        name_lower = str(item.get("name", "")).lower()
        if any(keyword in name_lower for keyword in ("参考", "正常值", "reference", "normal range")):
            return None
    if category == "LabResult" and value is not None:
        if isinstance(value, (int, float)):
            item["value"] = str(value)
        elif isinstance(value, str) and value.strip():
            match = NUMERIC_RE.search(value)
            item["value"] = match.group(0) if match else value
    for field in ENTITY_SUBFIELDS.get(category, ()):  # old Step2-3 fills missing subfields with empty strings.
        if item.get(field) is None:
            item[field] = ""
        elif field not in item:
            item[field] = ""
    return item


async def extract_from_text(text: str, max_chars: int = 4000, client: Any | None = None) -> dict[str, Any]:
    if not text or not text.strip():
        return {"success": False, "entities": [], "impression": "", "indication": "", "error": "empty text"}
    parsed = _parse_json_response(await _call_chat(EXTRACT_PROMPT.format(text=text[:max_chars]), client=client))
    if not parsed:
        return {"success": False, "entities": [], "impression": "", "indication": "", "error": "LLM extraction failed"}
    entities = parsed.get("entities") if isinstance(parsed.get("entities"), list) else []
    cleaned: list[dict[str, Any]] = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        item = _normalize_entity(ent)
        if item:
            cleaned.append(item)
    return {
        "success": True,
        "entities": cleaned,
        "temporal_info": parsed.get("temporal_info", []),
        "quantity_info": parsed.get("quantity_info", []),
        "relations": parsed.get("relations", []),
        "impression": parsed.get("impression", ""),
        "indication": parsed.get("indication", ""),
        "entity_count": len(cleaned),
        "error": None,
    }


async def _extract_entities_async(text_tasks: list[dict[str, Any]], concurrency: int) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[str]]]:
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    lock = asyncio.Lock()
    entity_rows: list[dict[str, Any]] = []
    impressions: dict[tuple[str, str], list[str]] = defaultdict(list)
    completed = 0
    last_bucket = -1
    shared_client = None
    cfg = get_agent_config("step2")
    if cfg.has_credentials:
        import openai

        shared_client = openai.AsyncOpenAI(api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY"), base_url=cfg.base_url, timeout=cfg.timeout)

    async def _one(task: dict[str, Any]) -> None:
        nonlocal completed, last_bucket
        try:
            async with sem:
                try:
                    result = await extract_from_text(str(task.get("text") or ""), client=shared_client)
                except TypeError:
                    # Test doubles may still use the old extract_from_text(text, max_chars) signature.
                    result = await extract_from_text(str(task.get("text") or ""))
        except Exception as exc:
            result = {"success": False, "entities": [], "error": str(exc)}
        async with lock:
            key = (str(task.get("patient_id") or ""), str(task.get("hadm_id") or ""))
            if result.get("success"):
                for ent in result.get("entities") or []:
                    row = {
                        "patient_id": str(task.get("patient_id") or ""),
                        "hadm_id": str(task.get("hadm_id") or ""),
                        "source_file": str(task.get("source_file") or ""),
                        "source_category": str(task.get("source_category") or ""),
                        "row_id": str(task.get("row_id") or ""),
                        "text_column": str(task.get("text_column") or ""),
                    }
                    row.update(dict(ent))
                    entity_rows.append(row)
                if result.get("impression"):
                    impressions[key].append(str(result["impression"]))
            completed += 1
            last_bucket = progress("entity extraction", completed, len(text_tasks), last_bucket)

    try:
        await asyncio.gather(*[_one(task) for task in text_tasks])
    finally:
        if shared_client is not None:
            await _close_async_client(shared_client)
    return entity_rows, impressions


def extract_entities(text_tasks_jsonl: str, output_root: str | Path, source_kind: str = "directory", concurrency: int = 100, results_dir_path: str | None = None) -> dict[str, Any]:
    tasks = read_jsonl(text_tasks_jsonl)
    emit(f"[Step2][Runtime] LLM 抽取开始: texts={len(tasks)}, concurrency={concurrency}, source={source_kind}")
    entity_rows, _ = _run_async(_extract_entities_async(tasks, concurrency))
    out_dir = ensure_dir(Path(results_dir_path) if results_dir_path else results_dir(output_root))
    ts = timestamp()
    filename = f"entities_{ts}.csv" if source_kind == "ocr" else f"entities_long_{ts}.csv"
    entities_csv = out_dir / filename
    columns = ["patient_id", "hadm_id", "source_file", "source_category", "row_id", "text_column", "category", "name", "value", "unit"]
    for row in entity_rows:
        for col in row:
            if col not in columns:
                columns.append(col)
    write_csv_rows(entities_csv, entity_rows, columns)
    summary_json = write_json(out_dir / f"summary_extract_{ts}.json", {"text_task_count": len(tasks), "entity_count": len(entity_rows), "source_kind": source_kind})
    emit(f"[Step2][Runtime] LLM 抽取完成: entities={len(entity_rows)}")
    return {
        "results_dir": str(out_dir),
        "entities_csv": str(entities_csv),
        "entities_long_csv": str(entities_csv),
        "summary_json": summary_json,
        "text_task_count": len(tasks),
        "entity_count": len(entity_rows),
    }


def _parse_json_array_response(content: str | None) -> list[dict[str, Any]] | None:
    if not content:
        return None
    text = content.strip()
    for marker in ("```json", "```"):
        if marker in text:
            try:
                text = text.split(marker, 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else None
            except Exception:
                pass
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        pass
    start, end = text.find("["), text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None
    return None


async def _standardize_batch(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt = STANDARDIZE_PROMPT.format(entities_json=json.dumps(entities, ensure_ascii=False, indent=2))
    parsed = _parse_json_array_response(await _call_chat(prompt))
    return parsed if isinstance(parsed, list) else entities


def standardize_entities_csv(entities_csv: str, output_root: str | Path, batch_size: int = 20) -> dict[str, Any]:
    header, rows = read_csv_rows(entities_csv)
    if not rows:
        return {"standardized_entities_csv": entities_csv, "standardized_count": 0, "entity_count": 0}
    results: list[dict[str, Any]] = []
    for idx in range(0, len(rows), max(1, int(batch_size))):
        batch = rows[idx : idx + max(1, int(batch_size))]
        results.extend(_run_async(_standardize_batch(batch)))
    standardized_count = sum(1 for row in results if row.get("standard_code") or row.get("normalized_value") not in {None, ""})
    out_dir = ensure_dir(Path(entities_csv).parent if Path(entities_csv).parent.exists() else results_dir(output_root))
    standardized_csv = out_dir / "entities_standardized.csv"
    columns = list(header)
    for row in results:
        for col in row:
            if col not in columns:
                columns.append(col)
    write_csv_rows(standardized_csv, results, columns)
    return {"standardized_entities_csv": str(standardized_csv), "standardized_count": standardized_count, "entity_count": len(results)}


def get_embeddings(names: list[str]) -> list[list[float]] | None:
    cfg = get_agent_config("step2")
    emb = cfg.embedding or {}
    api_key = os.environ.get("OPENAI_API_KEY") or str(emb.get("api_key") or cfg.api_key or "")
    base_url = str(emb.get("base_url") or emb.get("api_base") or cfg.base_url)
    model = str(emb.get("model") or emb.get("model_name") or "text-embedding-3-small")
    if not api_key or not names:
        return None
    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=cfg.timeout)
        kwargs: dict[str, Any] = {"model": model, "input": names}
        if emb.get("dimensions"):
            kwargs["dimensions"] = emb["dimensions"]
        response = client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
    except Exception:
        return None


CLUSTER_DISTANCE_THRESHOLD = 0.25


def _cluster_by_embedding(names: list[str], embeddings: list[list[float]], freq: dict[str, int], threshold: float = CLUSTER_DISTANCE_THRESHOLD) -> dict[str, str]:
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    vecs = np.array(embeddings, dtype=np.float32)
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
    vecs = np.clip(vecs, -1e6, 1e6)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.nan_to_num(norms, nan=1.0, posinf=1.0, neginf=1.0)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    normalized = np.nan_to_num(vecs / norms, nan=0.0, posinf=0.0, neginf=0.0)
    sim = normalized @ normalized.T
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist), method="average"), t=threshold, criterion="distance")
    groups: dict[int, list[str]] = defaultdict(list)
    for name, label in zip(names, labels):
        groups[int(label)].append(name)
    mapping: dict[str, str] = {}
    for members in groups.values():
        canonical = min(members, key=lambda item: (-freq.get(item, 1), len(item), item))
        for member in members:
            mapping[member] = canonical
    return mapping


def cluster_entity_names(entities_csv: str, output_root: str | Path, threshold: float = CLUSTER_DISTANCE_THRESHOLD) -> dict[str, Any]:
    header, rows = read_csv_rows(entities_csv)
    if not header:
        header = ["patient_id", "hadm_id", "category", "name", "value"]
    names_by_category: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        name = str(row.get("name") or "").strip().lower()
        if name:
            names_by_category[str(row.get("category") or "entity")].append(name)
    mapping: dict[str, str] = {}
    for category, names in names_by_category.items():
        freq: dict[str, int] = defaultdict(int)
        for name in names:
            freq[name] += 1
        unique = sorted(freq)
        embeddings = get_embeddings(unique)
        if len(unique) == 1:
            mapping[unique[0]] = unique[0]
            continue
        if embeddings and len(embeddings) == len(unique):
            try:
                category_mapping = _cluster_by_embedding(unique, embeddings, freq, threshold=threshold)
            except Exception:
                category_mapping = {name: name for name in unique}
        else:
            category_mapping = {name: name for name in unique}
        mapping.update(category_mapping)
    for row in rows:
        raw_name = str(row.get("name") or "").strip()
        row["canonical_name"] = mapping.get(raw_name.lower(), raw_name)
    out_dir = ensure_dir(Path(entities_csv).parent if Path(entities_csv).parent.exists() else results_dir(output_root))
    clustered_csv = out_dir / "entities_clustered.csv"
    columns = list(dict.fromkeys(header + ["canonical_name"]))
    write_csv_rows(clustered_csv, rows, columns)
    before = len({f"{r.get('category')}::{r.get('name')}" for r in rows if r.get("name")})
    after = len({f"{r.get('category')}::{r.get('canonical_name')}" for r in rows if r.get("canonical_name")})
    emit(f"[Step2][Runtime] entity name clustering 完成: columns={before}->{after}")
    return {
        "clustered_entities_csv": str(clustered_csv),
        "entity_count": len(rows),
        "entity_column_count_before_clustering": before,
        "entity_column_count_after_clustering": after,
        "entity_column_reduction": before - after,
    }


def _structured_value_columns(rows: list[dict[str, str]]) -> list[str]:
    excluded = {"subject_id", "hadm_id", "stay_id", "note_id", "dicom_id", "study_id"}
    columns: list[str] = []
    for row in rows:
        for col in row:
            if col not in columns and col.lower() not in excluded:
                columns.append(col)
    return columns


def _legacy_entity_base_col(row: dict[str, Any]) -> str:
    stem = Path(str(row.get("source_file") or "")).stem
    prefix = stem.replace("_notes", "") if stem else "entity"
    category = str(row.get("category") or "")
    canonical = str(row.get("canonical_name") or row.get("name") or "").strip().lower()
    safe = canonical.replace(" ", "_").replace("/", "_").replace("-", "_") or "value"
    return f"{prefix}_{category}_{safe}"


def build_admission_wide(patient_payloads_json: str, clustered_entities_csv: str, output_root: str | Path) -> dict[str, Any]:
    payloads: dict[str, Any] = json.loads(Path(patient_payloads_json).read_text(encoding="utf-8"))
    _, entities = read_csv_rows(clustered_entities_csv)
    entities_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        entities_by_key[(str(entity.get("patient_id") or ""), str(entity.get("hadm_id") or ""))].append(entity)
    wide_rows: list[dict[str, Any]] = []
    for pid, payload in payloads.items():
        structured = payload.get("structured") or {}
        patient_base: dict[str, Any] = {"subject_id": pid}
        patients_rows = structured.get("patients") or []
        if patients_rows:
            for col, value in patients_rows[0].items():
                if str(value).strip():
                    patient_base[col] = value
        admissions = structured.get("admissions") or [{}]
        for admission in admissions:
            hadm_id = _row_hadm(admission)
            row = dict(patient_base)
            if hadm_id:
                row["hadm_id"] = hadm_id
            for col, value in admission.items():
                if str(value).strip():
                    row[col] = value
            for table_name, table_rows in structured.items():
                if table_name in {"patients", "admissions"}:
                    continue
                matched = _rows_for_hadm(table_rows, hadm_id)
                prefix = f"structured__{_safe_name(table_name)}"
                row[f"{prefix}__row_count"] = len(matched)
                for col in _structured_value_columns(matched):
                    row[f"{prefix}__{_safe_name(col)}"] = _join_unique(item.get(col, "") for item in matched)
            text_tasks = payload.get("text_tasks") or []
            patient_text = [task.get("text", "") for task in text_tasks if not hadm_id or str(task.get("hadm_id") or "") == hadm_id]
            if patient_text:
                row["clinical_text"] = _join_unique(patient_text, limit=80000)
                row["clinical_text_count"] = len(patient_text)
            key = (str(pid), str(hadm_id))
            for entity in entities_by_key.get(key, []):
                category = str(entity.get("category") or "")
                for field in ENTITY_SUBFIELDS.get(category, ("value",)):
                    sub_value = str(entity.get(field) or "").strip()
                    if sub_value:
                        sub_col = f"{_legacy_entity_base_col(entity)}_{field}"
                        row[sub_col] = _join_unique([row.get(sub_col, ""), sub_value])
            wide_rows.append(row)
    if not wide_rows:
        wide_rows = [{"subject_id": pid} for pid in payloads]
    columns: list[str] = []
    for preferred in ("subject_id", "hadm_id"):
        if any(preferred in row for row in wide_rows):
            columns.append(preferred)
    for row in wide_rows:
        for col in row:
            if col not in columns:
                columns.append(col)
    out_dir = results_dir(output_root, clustered_entities_csv)
    ts = timestamp()
    admission_wide_csv = out_dir / f"admission_wide_{ts}.csv"
    summary_json = out_dir / f"summary_{ts}.json"
    write_csv_rows(admission_wide_csv, wide_rows, columns)
    summary = {
        "success": True,
        "resolved_mode": "directory",
        "patient_count": len(payloads),
        "wide_row_count": len(wide_rows),
        "wide_column_count": len(columns),
        "entity_count": len(entities),
        "admission_wide_csv": str(admission_wide_csv),
        "entities_long_csv": clustered_entities_csv,
    }
    write_json(summary_json, summary)
    emit(f"[Step2][Runtime] directory wide table 完成: rows={len(wide_rows)}, columns={len(columns)}, entities={len(entities)}")
    return {"results_dir": str(out_dir), "admission_wide_csv": str(admission_wide_csv), "summary_json": str(summary_json), **summary}


def _get_ocr_engine() -> Any:
    engine = getattr(_OCR_LOCAL, "engine", None)
    if engine is not None:
        return engine
    with _OCR_INIT_LOCK:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR(
            det_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            rec_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            cls_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
        )
        _OCR_LOCAL.engine = engine
        return engine


def ocr_image(path: str) -> str:
    try:
        engine = _get_ocr_engine()
        result, _ = engine(path)
        if not result:
            return ""
        text = "\n".join(row[1] for row in result if len(row) >= 2)
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
    except Exception:
        return ""


def _image_tasks(input_path: str | Path) -> list[tuple[str, str, str]]:
    tasks: list[tuple[str, str, str]] = []
    for patient_dir in _patient_dirs(input_path):
        for image in sorted(p for p in patient_dir.rglob("*") if p.is_file() and _is_image(p)):
            rel = image.relative_to(patient_dir)
            if "table" in {part.lower() for part in rel.parts}:
                continue
            tasks.append((patient_dir.name, str(image), str(rel)))
    return tasks


def run_ocr(input_path: str, output_root: str | Path, ocr_workers: int = 64, cache_path: str | None = None) -> dict[str, Any]:
    out_dir = results_dir(output_root)
    cache_file = Path(cache_path) if cache_path else out_dir / "ocr_cache.json"
    existing: dict[str, dict[str, str]] = {}
    if cache_file.exists():
        existing = json.loads(cache_file.read_text(encoding="utf-8"))
    tasks = _image_tasks(input_path)
    done_keys = {(pid, rel) for pid, values in existing.items() for rel in values}
    pending = [task for task in tasks if (task[0], task[2]) not in done_keys]
    emit(f"[Step2][Runtime] OCR 开始: images={len(tasks)}, cached={len(done_keys)}, pending={len(pending)}, workers={ocr_workers}")
    cache: dict[str, dict[str, str]] = {str(pid): dict(values) for pid, values in existing.items()}
    completed = 0
    last_bucket = -1
    success = sum(1 for values in cache.values() for text in values.values() if text)
    failed = 0
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(ocr_workers))) as pool:
            futures = {pool.submit(ocr_image, path): (pid, rel) for pid, path, rel in pending}
            for future in concurrent.futures.as_completed(futures):
                pid, rel = futures[future]
                text = future.result()
                completed += 1
                if text:
                    cache.setdefault(str(pid), {})[str(rel)] = text
                    success += 1
                else:
                    failed += 1
                last_bucket = progress("OCR", completed, len(pending), last_bucket)
    ensure_dir(cache_file.parent)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"[Step2][Runtime] OCR 完成: success={success}, failed={failed}")
    return {
        "results_dir": str(out_dir),
        "ocr_cache_path": str(cache_file),
        "ocr_patient_count": len(cache),
        "total_images": len(tasks),
        "successful_ocr": success,
        "failed_ocr": failed,
        "cache_used": bool(existing),
    }


def extract_ocr_entities(ocr_cache_path: str, output_root: str | Path, results_dir_path: str | None = None, concurrency: int = 100) -> dict[str, Any]:
    cache: dict[str, dict[str, str]] = json.loads(Path(ocr_cache_path).read_text(encoding="utf-8"))
    tasks: list[dict[str, str]] = []
    for pid, texts in cache.items():
        for rel, text in texts.items():
            if text and len(text.strip()) >= 5:
                tasks.append({"patient_id": pid, "hadm_id": "", "source_file": rel, "source_category": "ocr", "row_id": rel, "text_column": "ocr_text", "text": text})
    out_dir = ensure_dir(Path(results_dir_path) if results_dir_path else results_dir(output_root, ocr_cache_path))
    text_tasks_jsonl = write_jsonl(out_dir / "ocr_text_tasks.jsonl", tasks)
    extracted = extract_entities(text_tasks_jsonl, output_root, source_kind="ocr", concurrency=concurrency, results_dir_path=str(out_dir))
    patients_csv = out_dir / f"patients_{timestamp()}.csv"
    patient_rows = [{"patient_id": pid, "ocr_file_count": len(texts)} for pid, texts in cache.items()]
    write_csv_rows(patients_csv, patient_rows, ["patient_id", "ocr_file_count"])
    extracted["patients_csv"] = str(patients_csv)
    return extracted


def _find_patient_template_csvs(input_path: str | Path) -> dict[str, Path]:
    templates: dict[str, Path] = {}
    for patient_dir in _patient_dirs(input_path):
        table_dir = patient_dir / "table"
        if not table_dir.is_dir():
            continue
        direct = [p for p in sorted(table_dir.iterdir()) if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}]
        if direct:
            templates[patient_dir.name] = direct[0]
    return templates


RULE_MAP = [
    ("扫视", "扫视试验", "value"),
    ("视跟踪", "跟踪试验", "value"),
    ("视动眼震", "视动眼震", "value"),
    ("增益(左)", "增益(左)", "value"),
    ("增益(右)", "增益(右)", "value"),
    ("视动增益-左", "增益(左)", "value"),
    ("视动增益-右", "增益(右)", "value"),
    ("左侧视动眼震", "增益(左)", "value"),
    ("右侧视动眼震", "增益(右)", "value"),
    ("视动眼震左侧", "增益(左)", "value"),
    ("视动眼震右侧", "增益(右)", "value"),
    ("凝视", "凝视试验", "value"),
    ("自发眼震", "自发眼震", "value"),
    ("摇头眼震", "摇头眼震", "value"),
    ("固视抑制", "固视抑制", "value"),
    ("Dix-Hallpike", "Dix-hallpike", "value"),
    ("Roll-test", "Roll试验", "value"),
    ("Roll试验", "Roll试验", "value"),
    ("深悬头位试验", "深悬头位", "value"),
    ("深悬头位", "深悬头位", "value"),
    ("动态位置试验", "动态位置试验", "value"),
    ("静态位置试验", "眼震视图等检查-静态位置试验", "value"),
    ("静态位置试验坐位", "坐位", "value"),
    ("静态位置试验仰卧位", "仰卧位", "value"),
    ("静态位置试验悬头左侧", "左悬头", "value"),
    ("静态位置试验悬头位", "悬头", "value"),
    ("静态位置试验悬头右侧", "右悬头", "value"),
    ("静态位置试验坐起", "坐起", "value"),
    ("坐位眼震", "坐位", "value"),
    ("左侧卧眼震", "左侧卧位", "value"),
    ("仰卧位眼震", "仰卧位", "value"),
    ("右侧卧眼震", "右侧卧位", "value"),
    ("悬头左侧眼震", "左侧悬头位", "value"),
    ("悬头右侧眼震", "右侧悬头位", "value"),
    ("悬头位眼震", "悬头", "value"),
    ("左侧卧位", "左侧卧位", "value"),
    ("右侧卧位", "右侧卧位", "value"),
    ("左侧卧", "左侧卧位", "value"),
    ("右侧卧", "右侧卧位", "value"),
    ("坐位", "坐位", "value"),
    ("仰卧位", "仰卧位", "value"),
    ("悬头左侧", "左侧悬头位", "value"),
    ("悬头右侧", "右侧悬头位", "value"),
    ("右侧悬头位", "右侧悬头位", "value"),
    ("左侧悬头位", "左侧悬头位", "value"),
    ("左侧50℃慢相速度", "左热", "value"),
    ("左侧热", "左热", "value"),
    ("右侧50℃慢相速度", "右热", "value"),
    ("右侧热", "右热", "value"),
    ("左侧24℃慢相速度", "左冷", "value"),
    ("左侧冷", "左冷", "value"),
    ("右侧24℃慢相速度", "右冷", "value"),
    ("右侧冷", "右冷", "value"),
    ("右侧 50℃", "右热", "value"),
    ("左侧 50℃", "左热", "value"),
    ("右侧50℃", "右热", "value"),
    ("左侧50℃", "左热", "value"),
    ("50℃右侧", "右热", "value"),
    ("50℃左侧", "左热", "value"),
    ("右侧 24℃", "右冷", "value"),
    ("左侧 24℃", "左冷", "value"),
    ("右侧24℃", "右冷", "value"),
    ("左侧24℃", "左冷", "value"),
    ("24℃右侧", "右冷", "value"),
    ("24℃左侧", "左冷", "value"),
    ("温度右侧慢向角速度", "右热", "value"),
    ("温度左侧慢向角速度", "左热", "value"),
    ("CP(R)", "CP值", "value"),
    ("CP（R）", "CP值", "value"),
    ("CP(L)", "CP值", "value"),
    ("CP（L）", "CP值", "value"),
    ("CP(R）", "CP值", "value"),
    ("校正值CP", "CP值", "value"),
    ("CP右侧", "CP值", "value"),
    ("CP左侧", "CP值", "value"),
    ("半规管轻瘫值(CP)", "CP值", "value"),
    ("半规管轻瘫", "CP值", "value"),
    ("风湿因子", "类风湿因子", "value"),
    ("类风湿因子", "类风湿因子", "value"),
    ("超敏C反应蛋白", "C反应蛋白", "value"),
    ("C反应蛋白", "C反应蛋白", "value"),
    ("甲状腺球蛋白抗体", "甲状腺球蛋白抗体", "value"),
    ("甲状腺过氧化物酶抗体", "甲状腺微粒抗体", "value"),
    ("抗心磷脂抗体IgM", "抗ACA", "value"),
    ("抗心磷脂抗体IgG", "抗ACA", "value"),
    ("抗心磷脂抗体IgA", "抗ACA", "value"),
    ("抗β2糖蛋白1抗体", "抗磷脂抗体", "value"),
    ("抗B2糖蛋白1抗体", "抗磷脂抗体", "value"),
]


def _rule_based_map(entities: list[dict[str, Any]], empty_cols: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        value = entity.get("value", "")
        if is_missing_like(value):
            continue
        for entity_keyword, column_keyword, field in RULE_MAP:
            if entity_keyword.lower() in name.lower():
                for col in empty_cols:
                    if column_keyword in col and col not in result:
                        result[col] = {
                            "value": str(entity.get(field, value)).strip(),
                            "entity_name": name,
                            "category": str(entity.get("category", "")),
                            "source_file": str(entity.get("source_file", "")),
                            "original_text": str(entity.get("original_text", "")),
                        }
                        break
    return result


def _is_unit_col(col: str) -> bool:
    return "单位" in col.rsplit("-", 1)[-1]


def _evaluable_fill_columns(empty_cols: list[str], entities: list[dict[str, Any]], rule_filled: set[str]) -> list[str]:
    prefixes = (
        "辅助检查-眼震视图等检查-",
        "辅助检查-实验室检查-",
        "辅助检查-耳科及眼科检查-",
        "床旁查体-",
        "第一次复诊-",
        "第二次复诊-",
        "第三次复诊-",
    )
    ent_keywords: set[str] = set()
    for entity in entities:
        for token in re.findall(r"[一-鿿]{2,}|[A-Za-z]{3,}", str(entity.get("name", ""))):
            ent_keywords.add(token.lower())
    candidates: list[str] = []
    seen_last: set[str] = set()
    for col in empty_cols:
        if col in rule_filled:
            continue
        if not any(col.startswith(prefix) for prefix in prefixes):
            continue
        last = re.sub(r"\.\d+$", "", col.rsplit("-", 1)[-1]).strip()
        if last in seen_last:
            continue
        col_keywords = set(re.findall(r"[一-鿿]{2,}|[A-Za-z]{3,}", last.lower()))
        if col_keywords & ent_keywords:
            candidates.append(col)
            seen_last.add(last)
    return candidates


async def _llm_map_entities_to_cols(patient_id: str, entities: list[dict[str, Any]], empty_cols: list[str], chunk_size: int = 80) -> dict[str, dict[str, str]]:
    if not entities or not empty_cols:
        return {}
    allowed = set(empty_cols)
    ent_index = {str(i): entity for i, entity in enumerate(entities)}
    mapping: dict[str, dict[str, str]] = {}
    for offset in range(0, len(empty_cols), chunk_size):
        batch_cols = empty_cols[offset:offset + chunk_size]
        prompt = build_fill_mapping_prompt(patient_id, entities, batch_cols)
        parsed = _parse_json_response(await _call_chat(prompt))
        if not isinstance(parsed, dict):
            continue
        rows = parsed.get("填写结果") or parsed.get("fill_results") or parsed.get("results") or []
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            col = str(item.get("列名") or item.get("column") or item.get("column_name") or "").strip()
            value = str(item.get("值") or item.get("value") or "").strip()
            if col not in allowed or not value or is_missing_like(value):
                continue
            entity_index = str(item.get("实体编号", ""))
            source_entity = ent_index.get(entity_index, {})
            mapping[col] = {
                "value": value,
                "entity_name": str(source_entity.get("name", "")),
                "category": str(source_entity.get("category", "")),
                "source_file": str(source_entity.get("source_file", "")),
                "original_text": str(source_entity.get("original_text", "")),
            }
    return mapping


def fill_table_from_entities(input_path: str, entities_csv: str, output_root: str | Path, use_llm: bool = True) -> dict[str, Any]:
    templates = _find_patient_template_csvs(input_path)
    _, entities = read_csv_rows(entities_csv)
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entities:
        by_patient[str(row.get("patient_id") or "")].append(row)
    output_dir = ensure_dir(step_root(output_root) / "filled_tables")
    merged_rows: list[dict[str, Any]] = []
    merged_cols: list[str] = []
    total_filled = 0
    llm_filled = 0
    filled_mappings: dict[str, Any] = {}
    completed = 0
    last_bucket = -1
    emit(f"[Step2][Runtime] 表格回填开始: patients={len(templates)}, use_llm={use_llm}")
    for pid, template in templates.items():
        cols, rows = read_csv_rows(template)
        if not rows:
            continue
        row = dict(rows[0])
        empty_cols = [
            col for col in cols
            if is_missing_like(row.get(col, ""))
            and not _is_unit_col(col)
            and col.lower() not in {"id", "subject_id", "patient_id"}
        ]
        entities_for_patient = by_patient.get(pid, [])
        mapping: dict[str, dict[str, str]] = {}
        llm_mapping: dict[str, dict[str, str]] = {}
        rule_mapping: dict[str, dict[str, str]] = {}
        if empty_cols and entities_for_patient:
            rule_mapping = _rule_based_map(entities_for_patient, empty_cols)
            if use_llm:
                candidate_cols = _evaluable_fill_columns(empty_cols, entities_for_patient, set(rule_mapping))
                if candidate_cols:
                    llm_mapping = _run_async(_llm_map_entities_to_cols(pid, entities_for_patient, candidate_cols))
            mapping = {**llm_mapping, **rule_mapping}

        filled = 0
        for col, info in mapping.items():
            if col in row and is_missing_like(row.get(col, "")) and not is_missing_like(info.get("value", "")):
                row[col] = info["value"]
                filled += 1
        llm_filled += len([col for col in llm_mapping if col in mapping and col not in rule_mapping])
        if mapping:
            filled_mappings[pid] = mapping
        patient_out = output_dir / pid / template.name
        write_csv_rows(patient_out, [row], cols)
        for col in cols:
            if col not in merged_cols:
                merged_cols.append(col)
        merged_rows.append(row)
        total_filled += filled
        completed += 1
        last_bucket = progress("table fill", completed, len(templates), last_bucket)
    merged_csv = output_dir / "filled_merged.csv"
    write_csv_rows(merged_csv, merged_rows, merged_cols)
    filled_mappings_json = write_json(output_dir / "filled_mappings.json", filled_mappings)
    emit(f"[Step2][Runtime] 表格回填完成: patients={len(merged_rows)}, filled={total_filled}, llm_filled={llm_filled}")
    return {
        "filled_tables_dir": str(output_dir),
        "filled_merged_csv": str(merged_csv),
        "filled_mappings_json": filled_mappings_json,
        "filled_patient_count": len(merged_rows),
        "filled_cell_count": total_filled,
        "llm_filled_cell_count": llm_filled,
    }


def normalize_outputs(output_root: str | Path, filled_merged_csv: str | None = None, fallback_csv: str | None = None, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    root = step_root(output_root)
    source: Path | None = None
    for candidate in (filled_merged_csv, fallback_csv):
        if candidate and Path(candidate).is_file():
            source = Path(candidate)
            break
    if source is None:
        candidates = list(root.glob("filled_tables/filled_merged.csv")) + list(root.glob("results_*/admission_wide_*.csv"))
        if candidates:
            source = max(candidates, key=lambda p: p.stat().st_mtime)
    if source is None or not source.is_file():
        raise FileNotFoundError("Step2 缺少可传给 Step3 的 CSV 产物。")
    next_dir = ensure_dir(root / "next_input")
    next_csv = next_dir / "input.csv"
    shutil.copy2(source, next_csv)
    header, rows = read_csv_rows(next_csv)
    summary_payload = dict(summary or {})
    summary_payload.update({"source_csv": str(source), "next_input_csv": str(next_csv), "rows": len(rows), "columns": len(header)})
    summary_path = write_json(next_dir / "summary.json", summary_payload)
    return {"next_input_dir": str(next_dir), "next_input_csv": str(next_csv), "next_summary_json": summary_path, "source_csv": str(source), "output_row_count": len(rows), "output_column_count": len(header)}


def validate_output(next_input_csv: str, next_summary_json: str | None = None) -> dict[str, Any]:
    path = Path(next_input_csv)
    issues: list[str] = []
    if not path.is_file():
        return {"passed": False, "issues": [f"next_input/input.csv 不存在: {path}"], "details": {"next_input_csv": str(path)}}
    header, rows = read_csv_rows(path, limit=5)
    if not header:
        issues.append("next_input/input.csv 无表头")
    if not rows:
        issues.append("next_input/input.csv 无数据行")
    if detect_patient_column(header) is None:
        issues.append("未发现 patient_id/subject_id 类患者列")
    non_patient_columns = [col for col in header if col != detect_patient_column(header)]
    if not non_patient_columns:
        issues.append("next_input/input.csv 只有患者 ID 列，缺少可交给 Step3 的临床特征列")
    if next_summary_json and not Path(next_summary_json).is_file():
        issues.append(f"summary.json 不存在: {next_summary_json}")
    return {"passed": not issues, "issues": issues, "details": {"next_input_csv": str(path), "next_summary_json": next_summary_json or "", "output_column_count": len(header), "sample_columns": header[:20]}}
