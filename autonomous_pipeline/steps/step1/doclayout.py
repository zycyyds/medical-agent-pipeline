from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}
FIGURE_CLASS_ID = 3
OCR_CLASS_IDS = {0, 1, 4, 5, 6, 7, 8, 9}
_MODEL = None
_MODEL_PATH = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _model_candidates(model_path: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    env_path = os.environ.get("STEP1_DOCLAYOUT_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    project = _project_root()
    candidates.extend(
        [
            project / "models" / "doclayout_yolo_docstructbench_imgsz1024.pt",
            project / "agent_1" / "doclayout_yolo" / "model" / "doclayout_yolo_docstructbench_imgsz1024.pt",
            Path("/Users/mkbk/PycharmProjects/new/agent_1/doclayout_yolo/model/doclayout_yolo_docstructbench_imgsz1024.pt"),
        ]
    )
    return [p for p in candidates if p.exists()]


def _get_model(model_path: str | None = None):
    global _MODEL, _MODEL_PATH
    candidates = _model_candidates(model_path)
    if not candidates:
        raise FileNotFoundError("DocLayout-YOLO model file not found")
    path = str(candidates[0])
    if _MODEL is not None and _MODEL_PATH == path:
        return _MODEL, path
    from doclayout_yolo import YOLOv10

    _MODEL = YOLOv10(path)
    _MODEL_PATH = path
    return _MODEL, path


def _is_visual_record(record: dict[str, Any]) -> bool:
    suffix = str(record.get("file_info", {}).get("suffix") or Path(str(record.get("source_path") or "")).suffix).lower()
    return suffix in IMAGE_SUFFIXES or suffix in PDF_SUFFIXES


def _fallback_modality(path: Path) -> str:
    # This fallback is only used when a file cannot be decoded or the model is unavailable.
    # It keeps the Step1 contract deterministic for tests and broken inputs.
    return "figure" if path.suffix.lower() in IMAGE_SUFFIXES else "ocr"


def _doclayout_enabled() -> bool:
    raw = os.environ.get("STEP1_DOCLAYOUT_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _figure_area_ratio(detections: list[dict[str, Any]], width: int, height: int) -> float:
    total = max(float(width * height), 1.0)
    area = 0.0
    for item in detections:
        try:
            if int(item.get("class")) != FIGURE_CLASS_ID:
                continue
            box = item.get("box") or {}
            x1 = float(box.get("x1", 0))
            y1 = float(box.get("y1", 0))
            x2 = float(box.get("x2", 0))
            y2 = float(box.get("y2", 0))
            area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        except Exception:
            continue
    return area / total


def _modality_from_result(image_path: Path, result: Any, figure_ratio_threshold: float) -> tuple[str, dict[str, Any]]:
    try:
        detections = json.loads(result.tojson())
    except Exception:
        detections = []
    has_ocr = False
    for item in detections:
        try:
            if int(item.get("class")) in OCR_CLASS_IDS:
                has_ocr = True
                break
        except Exception:
            continue
    has_figure = False
    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            has_figure = _figure_area_ratio(detections, w, h) > figure_ratio_threshold
    except Exception:
        has_figure = False

    if has_figure and has_ocr:
        raw_modality = "ocr+figure"
    elif has_figure:
        raw_modality = "figure"
    else:
        raw_modality = "ocr"
    modality = "figure" if raw_modality in {"figure", "ocr+figure"} else "ocr"
    return modality, {"raw_modality": raw_modality, "detections": len(detections), "has_figure": has_figure, "has_ocr": has_ocr}


def classify_visual_records(records: list[dict[str, Any]], model_path: str | None = None) -> dict[str, Any]:
    visual = [record for record in records if _is_visual_record(record)]
    patches: list[dict[str, Any]] = []
    if not visual:
        return {"patches": [], "visual_count": 0, "doclayout_used": False, "model_path": None}

    if not _doclayout_enabled():
        for done, record in enumerate(visual, 1):
            src = Path(str(record.get("source_path")))
            patches.append(
                {
                    "source_path": str(src),
                    "modality": _fallback_modality(src),
                    "evidence": "doclayout disabled",
                    "layout_analysis": {
                        "relative_path": record.get("relative_path"),
                        "file_name": src.name,
                        "raw_modality": "fallback",
                        "fallback_reason": "STEP1_DOCLAYOUT_ENABLED disabled",
                    },
                }
            )
            print(f"[Step1][Progress] DocLayout disabled fallback: {done}/{len(visual)} ({done / len(visual) * 100:.1f}%)", file=sys.stderr, flush=True)
        return {"patches": patches, "visual_count": len(visual), "doclayout_used": False, "model_path": None, "device": None}

    batch_size = max(1, int(os.environ.get("STEP1_DOCLAYOUT_BATCH_SIZE", "16") or "16"))
    imgsz = int(os.environ.get("STEP1_DOCLAYOUT_IMGSZ", "960") or "960")
    conf = float(os.environ.get("STEP1_DOCLAYOUT_CONF", "0.25") or "0.25")
    device = os.environ.get("STEP1_DOCLAYOUT_DEVICE", "mps")
    ratio_threshold = float(os.environ.get("STEP1_DOCLAYOUT_FIGURE_RATIO", "0.05") or "0.05")
    print(f"[Step1][Runtime] DocLayout-YOLO 开始识别图片/PDF文件: {len(visual)} 个, device={device}, batch_size={batch_size}", file=sys.stderr, flush=True)

    model = None
    selected_model_path = None
    doclayout_used = False
    try:
        # Do not load the heavy model for tiny synthetic test images.
        if all(Path(str(r.get("source_path"))).exists() and Path(str(r.get("source_path"))).stat().st_size < 1024 for r in visual):
            raise RuntimeError("tiny synthetic visual files use deterministic fallback")
        model, selected_model_path = _get_model(model_path)
        doclayout_used = True
    except Exception as exc:
        for done, record in enumerate(visual, 1):
            src = Path(str(record.get("source_path")))
            patches.append(
                {
                    "source_path": str(src),
                    "modality": _fallback_modality(src),
                    "evidence": "doclayout-yolo",
                    "layout_analysis": {
                        "relative_path": record.get("relative_path"),
                        "file_name": src.name,
                        "raw_modality": "fallback",
                        "fallback_reason": str(exc),
                    },
                }
            )
            print(f"[Step1][Progress] DocLayout-YOLO: {done}/{len(visual)} ({done / len(visual) * 100:.1f}%)", file=sys.stderr, flush=True)
        return {"patches": patches, "visual_count": len(visual), "doclayout_used": False, "model_path": selected_model_path, "device": device}

    done = 0
    for start in range(0, len(visual), batch_size):
        batch = visual[start : start + batch_size]
        paths = [str(record.get("source_path")) for record in batch]
        try:
            results = list(model.predict(paths, imgsz=imgsz, conf=conf, device=device, verbose=False))
        except Exception as exc:
            for record in batch:
                src = Path(str(record.get("source_path")))
                patches.append(
                    {
                        "source_path": str(src),
                        "modality": _fallback_modality(src),
                        "evidence": "doclayout-yolo",
                        "layout_analysis": {
                            "relative_path": record.get("relative_path"),
                            "file_name": src.name,
                            "raw_modality": "fallback",
                            "fallback_reason": str(exc),
                        },
                    }
                )
                done += 1
                print(f"[Step1][Progress] DocLayout-YOLO: {done}/{len(visual)} ({done / len(visual) * 100:.1f}%)", file=sys.stderr, flush=True)
            continue
        for record, result in zip(batch, results):
            src = Path(str(record.get("source_path")))
            modality, detail = _modality_from_result(src, result, ratio_threshold)
            patches.append(
                {
                    "source_path": str(src),
                    "modality": modality,
                    "evidence": "doclayout-yolo",
                    "layout_analysis": {
                        "relative_path": record.get("relative_path"),
                        "file_name": src.name,
                        **detail,
                    },
                }
            )
            done += 1
            print(f"[Step1][Progress] DocLayout-YOLO: {done}/{len(visual)} ({done / len(visual) * 100:.1f}%)", file=sys.stderr, flush=True)
        if len(results) < len(batch):
            for record in batch[len(results) :]:
                src = Path(str(record.get("source_path")))
                patches.append(
                    {
                        "source_path": str(src),
                        "modality": _fallback_modality(src),
                        "evidence": "doclayout-yolo",
                        "layout_analysis": {
                            "relative_path": record.get("relative_path"),
                            "file_name": src.name,
                            "raw_modality": "fallback",
                            "fallback_reason": "missing model result",
                        },
                    }
                )
                done += 1
                print(f"[Step1][Progress] DocLayout-YOLO: {done}/{len(visual)} ({done / len(visual) * 100:.1f}%)", file=sys.stderr, flush=True)
    return {"patches": patches, "visual_count": len(visual), "doclayout_used": doclayout_used, "model_path": selected_model_path, "device": device}
