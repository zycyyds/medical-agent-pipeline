import json
import tempfile
from pathlib import Path


FIGURE_CLASS_ID = 3
TABLE_CLASS_ID = 5
TEXT_CLASS_ID = 1


def _get_layout_analysis_exports():
    from layout_analysis_tool import infer_and_save_layout

    return infer_and_save_layout


def _normalize_layout_modality(modality: str) -> str:
    normalized = str(modality or "ocr").lower()
    if normalized == "ocr+figure":
        return "figure"
    if normalized in {"figure", "ocr"}:
        return normalized
    return "ocr"


def _build_detection_stats(detections: list[dict]) -> dict:
    boxes = []
    figure_count = table_count = text_count = 0
    figure_area_ratio = table_area_ratio = text_area_ratio = 0.0

    for item in detections:
        label = str(item.get("label", "")).lower()
        confidence = float(item.get("confidence", 0.0) or 0.0)
        bbox = list(item.get("bbox", []))
        area_ratio = float(item.get("area_ratio", 0.0) or 0.0)
        boxes.append({"label": label, "confidence": confidence, "bbox": bbox})
        if label == "figure":
            figure_count += 1
            figure_area_ratio += area_ratio
        elif label == "table":
            table_count += 1
            table_area_ratio += area_ratio
        elif label == "text":
            text_count += 1
            text_area_ratio += area_ratio

    return {
        "figure_count": figure_count,
        "table_count": table_count,
        "text_count": text_count,
        "figure_area_ratio": round(figure_area_ratio, 6),
        "table_area_ratio": round(table_area_ratio, 6),
        "text_area_ratio": round(text_area_ratio, 6),
        "boxes": boxes,
    }


def _decide_page_modality(stats: dict) -> str:
    figure_count = int(stats.get("figure_count", 0) or 0)
    table_count = int(stats.get("table_count", 0) or 0)
    text_count = int(stats.get("text_count", 0) or 0)
    figure_area_ratio = float(stats.get("figure_area_ratio", 0.0) or 0.0)

    if figure_count > 0 and table_count == 0 and text_count == 0:
        return "figure"
    if figure_count == 0:
        return "ocr"
    if figure_area_ratio > 0.5:
        return "figure"
    return "ocr"


def _detection_stats_from_layout_modality(modality: str) -> dict:
    normalized = _normalize_layout_modality(modality)
    if normalized == "figure":
        return _build_detection_stats(
            [
                {
                    "label": "figure",
                    "confidence": 1.0,
                    "bbox": [0, 0, 100, 100],
                    "area_ratio": 1.0,
                }
            ]
        )
    return _build_detection_stats([])


def _build_stats_from_layout_detections(
    detections: list[dict],
    image_width: int | float | None,
    image_height: int | float | None,
) -> dict:
    try:
        width = float(image_width or 0)
        height = float(image_height or 0)
    except Exception:
        width = 0.0
        height = 0.0
    image_area = width * height if width > 0 and height > 0 else 0.0

    normalized_detections = []
    for item in detections or []:
        cls_id = int(item.get("class", -1))
        if cls_id == FIGURE_CLASS_ID:
            label = "figure"
        elif cls_id == TABLE_CLASS_ID:
            label = "table"
        elif cls_id == TEXT_CLASS_ID:
            label = "text"
        else:
            continue

        box = item.get("box") or {}
        x1 = float(box.get("x1", 0.0) or 0.0)
        y1 = float(box.get("y1", 0.0) or 0.0)
        x2 = float(box.get("x2", 0.0) or 0.0)
        y2 = float(box.get("y2", 0.0) or 0.0)
        width_span = max(0.0, x2 - x1)
        height_span = max(0.0, y2 - y1)
        area_ratio = 0.0
        if image_area > 0:
            area_ratio = (width_span * height_span) / image_area

        normalized_detections.append(
            {
                "label": label,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "bbox": [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 6),
            }
        )

    return _build_detection_stats(normalized_detections)


def _build_page_result(page_index: int, modality: str, detection_stats: dict | None = None) -> dict:
    normalized = _normalize_layout_modality(modality)
    stats = detection_stats or _detection_stats_from_layout_modality(normalized)
    derived_modality = _decide_page_modality(stats)
    return {
        "page_index": page_index,
        "classification_modality": derived_modality,
        "source_modality": normalized,
        "detection_stats": stats,
    }


def _analyze_layout_payload(payload: dict, page_index: int) -> dict:
    classification = payload.get("classification") or []
    raw_detections = payload.get("detections") or []
    image_width = payload.get("image_width")
    image_height = payload.get("image_height")
    if raw_detections:
        stats = _build_stats_from_layout_detections(raw_detections, image_width, image_height)
        modality = (classification[0] if classification else {}).get("modality", "ocr")
        return _build_page_result(page_index, modality, stats)
    if classification:
        return _build_page_result(page_index, classification[0].get("modality", "ocr"))
    return _build_page_result(page_index, "ocr")


def _render_pdf_pages(file_path: str | Path) -> list[Path]:
    import pymupdf

    path = Path(file_path)
    rendered_pages = []
    with tempfile.TemporaryDirectory(prefix="dataset_codegen_pdf_") as temp_dir:
        doc = pymupdf.open(str(path))
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            output_path = Path(temp_dir) / f"page_{page_index}.png"
            pix.save(str(output_path))
            rendered_pages.append(output_path)
        doc.close()
        persisted_pages = []
        for output_path in rendered_pages:
            persisted_path = path.parent / f".{path.stem}_{output_path.name}"
            persisted_path.write_bytes(output_path.read_bytes())
            persisted_pages.append(persisted_path)
        return persisted_pages


def _analyze_pdf_document(file_path: str | Path) -> list[dict]:
    infer_and_save_layout = _get_layout_analysis_exports()
    page_results = []
    for page_index, page_path in enumerate(_render_pdf_pages(file_path)):
        response = infer_and_save_layout(str(page_path), save_outputs=False)
        payload = json.loads(getattr(response, "content", "{}") or "{}")
        page_results.append(_analyze_layout_payload(payload, page_index))
    return page_results or [_build_page_result(0, "ocr")]


def analyze_visual_document(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return _analyze_pdf_document(path)

    infer_and_save_layout = _get_layout_analysis_exports()
    response = infer_and_save_layout(str(path), save_outputs=False)
    payload = json.loads(getattr(response, "content", "{}") or "{}")
    return [_analyze_layout_payload(payload, 0)]


__all__ = [
    "FIGURE_CLASS_ID",
    "TABLE_CLASS_ID",
    "TEXT_CLASS_ID",
    "analyze_visual_document",
    "_analyze_pdf_document",
    "_build_page_result",
    "_build_stats_from_layout_detections",
    "_detection_stats_from_layout_modality",
    "_normalize_layout_modality",
    "_render_pdf_pages",
]
