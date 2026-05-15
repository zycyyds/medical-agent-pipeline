import json
import os

import cv2
import shutil
import numpy as np
from agentscope.tool import ToolResponse
from doclayout_yolo import YOLOv10
import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QWidget, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# 表格处理依赖
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("警告：pandas 未安装，表格处理功能将不可用。请运行：pip install pandas openpyxl")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
PROGRAM_OUTPUT_DIR = os.path.join(BASE_DIR, "program", "output")
PROGRAM_DATA_DIR = os.path.join(PROGRAM_OUTPUT_DIR, "step1_results")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "agent_1",
    "doclayout_yolo",
    "model",
    "doclayout_yolo_docstructbench_imgsz1024.pt",
)
DEFAULT_LAYOUT_IMGSZ = 960
FIGURE_CLASS_ID = 3
TABLE_CLASS_ID = 5
TEXT_CLASS_ID = 1
OCR_CLASS_IDS = {0, 1, 4, 5, 6, 7, 8, 9}
_MODEL_INSTANCE = None
_MODEL_PATH_IN_USE: str | None = None


def _get_model(model_path: str | None = None) -> YOLOv10:
    global _MODEL_INSTANCE, _MODEL_PATH_IN_USE
    if model_path and not os.path.isabs(model_path):
        path = os.path.join(BASE_DIR, model_path)
    elif model_path:
        path = model_path
    else:
        path = MODEL_PATH
    if _MODEL_INSTANCE is None or _MODEL_PATH_IN_USE != path:
        _MODEL_INSTANCE = YOLOv10(path)
        _MODEL_PATH_IN_USE = path
    return _MODEL_INSTANCE


def _get_doclayout_batch_size() -> int:
    raw = os.environ.get("STEP1_DOCLAYOUT_BATCH_SIZE", "16")
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def _rect_union_area(rects: list[tuple[float, float, float, float]]) -> float:
    cleaned = []
    xs = set()
    for x1, y1, x2, y2 in rects:
        if x2 <= x1 or y2 <= y1:
            continue
        cleaned.append((float(x1), float(y1), float(x2), float(y2)))
        xs.add(float(x1))
        xs.add(float(x2))

    if not cleaned:
        return 0.0

    xs_sorted = sorted(xs)
    area = 0.0

    for i in range(len(xs_sorted) - 1):
        x_left = xs_sorted[i]
        x_right = xs_sorted[i + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue

        intervals = []
        for rx1, ry1, rx2, ry2 in cleaned:
            if rx2 <= x_left or rx1 >= x_right:
                continue
            intervals.append((ry1, ry2))

        if not intervals:
            continue

        intervals.sort()
        merged = []
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        covered_y = 0.0
        for s, e in merged:
            if e > s:
                covered_y += e - s

        area += covered_y * dx

    return float(area)


def _figure_union_area_ratio(
    detections: list[dict],
    image_w: int,
    image_h: int,
    figure_class_id: int,
) -> float:
    try:
        w = int(image_w)
        h = int(image_h)
    except Exception:
        return 0.0

    image_area = float(w * h)
    if image_area <= 0:
        return 0.0

    rects = []
    for item in detections:
        try:
            cls_id = int(item.get("class"))
        except Exception:
            continue
        if cls_id != int(figure_class_id):
            continue
        box = item.get("box") or {}
        try:
            x1 = float(box.get("x1"))
            y1 = float(box.get("y1"))
            x2 = float(box.get("x2"))
            y2 = float(box.get("y2"))
        except Exception:
            continue

        x1 = max(0.0, min(x1, float(w)))
        x2 = max(0.0, min(x2, float(w)))
        y1 = max(0.0, min(y1, float(h)))
        y2 = max(0.0, min(y2, float(h)))
        if x2 > x1 and y2 > y1:
            rects.append((x1, y1, x2, y2))

    if not rects:
        return 0.0

    return float(_rect_union_area(rects) / image_area)


def _resolve_target_and_root(target_path: str, project_root: str) -> tuple[str, str]:
    raw_path = str(target_path)
    abs_target = os.path.abspath(raw_path)
    if not os.path.exists(abs_target):
        alt_path = os.path.join(project_root, raw_path)
        if os.path.exists(alt_path):
            abs_target = alt_path

    # Try to determine dataset_root intelligently
    dataset_root = None
    curr = abs_target
    while curr and curr != "/" and curr != project_root:
        if os.path.basename(curr) == "rawdata":
            dataset_root = curr
            break
        curr = os.path.dirname(curr)

    if dataset_root is None:
        if os.path.isdir(abs_target):
            base_for_root = abs_target
        elif os.path.isfile(abs_target):
            base_for_root = os.path.dirname(abs_target)
        else:
            # If path doesn't exist, we can't determine root easily, but let caller handle it
            # For now, raise as original code did or return what we have
            if not os.path.exists(abs_target):
                 raise FileNotFoundError(abs_target)
            base_for_root = os.path.dirname(abs_target)

        rel_to_project = os.path.relpath(base_for_root, project_root)
        if not rel_to_project.startswith(".."):
            top = rel_to_project.split(os.sep)[0]
            dataset_root = os.path.join(project_root, top)
        else:
            dataset_root = base_for_root
    return abs_target, dataset_root


def _collect_image_entries(target_path: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    out_img_dir = os.path.join(project_root, "output", "results")
    out_json_dir = os.path.join(project_root, "output", "result_json")
    
    abs_target, dataset_root = _resolve_target_and_root(target_path, project_root)
    
    file_entries: list[tuple[str, str]] = []
    if os.path.isdir(abs_target):
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        for dirpath, _, filenames in os.walk(abs_target):
            for fn in filenames:
                lower = fn.lower()
                if any(lower.endswith(e) for e in exts):
                    full = os.path.join(dirpath, fn)
                    if os.path.isfile(full):
                        rel = os.path.relpath(full, dataset_root)
                        file_entries.append((full, rel))
        file_entries.sort(key=lambda x: x[1])
    elif os.path.isfile(abs_target):
        rel = os.path.relpath(abs_target, dataset_root)
        file_entries.append((abs_target, rel))
    return abs_target, out_img_dir, out_json_dir, file_entries


def _save_figure_crops(
    img: np.ndarray | None,
    rel_path: str,
    stem: str,
    detections: list[dict],
    seg_root: str,
    w: int,
    h: int,
    figure_class_id: int,
) -> list[str]:
    saved_paths: list[str] = []
    if img is None:
        return saved_paths
    figure_boxes: list[tuple[int, int, int, int]] = []
    for item in detections:
        try:
            cls_id = int(item.get("class"))
        except Exception:
            continue
        if cls_id != figure_class_id:
            continue
        box = item.get("box") or {}
        try:
            x1 = float(box.get("x1"))
            y1 = float(box.get("y1"))
            x2 = float(box.get("x2"))
            y2 = float(box.get("y2"))
        except Exception:
            continue
        x1 = max(0.0, min(x1, float(w)))
        x2 = max(0.0, min(x2, float(w)))
        y1 = max(0.0, min(y1, float(h)))
        y2 = max(0.0, min(y2, float(h)))
        if x2 > x1 and y2 > y1:
            figure_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    if not figure_boxes:
        return saved_paths
    rel_dir_for_seg = os.path.dirname(rel_path)
    if rel_dir_for_seg and rel_dir_for_seg != ".":
        seg_base_dir = os.path.join(seg_root, rel_dir_for_seg, stem)
    else:
        seg_base_dir = os.path.join(seg_root, stem)
    os.makedirs(seg_base_dir, exist_ok=True)
    idx = 1
    for x1, y1, x2, y2 in figure_boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        seg_path = os.path.join(seg_base_dir, f"分割元素{idx}.jpg")
        cv2.imwrite(seg_path, crop)
        saved_paths.append(seg_path)
        idx += 1
    return saved_paths


def _run_inference_and_save(
    file_entries: list[tuple[str, str]],
    out_img_dir: str,
    out_json_dir: str,
    conf: float,
    imgsz: int,
    device: str,
    model_path: str | None,
    figure_class_id: int,
    figure_ratio_threshold: float,
    save_outputs: bool = True,
    progress_callback=None,
) -> tuple[list[dict], int]:
    try:
        model = _get_model(model_path=model_path)
    except Exception as e:
        raise RuntimeError(str(e))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    seg_root = os.path.join(project_root, "output", "分割")
    if save_outputs:
        if os.path.isdir(seg_root):
            shutil.rmtree(seg_root)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_json_dir, exist_ok=True)
        os.makedirs(seg_root, exist_ok=True)
    classification: list[dict] = []
    processed = 0
    batch_size = _get_doclayout_batch_size()

    def _report_progress() -> None:
        if progress_callback:
            progress_callback("DocLayout-YOLO", processed, len(file_entries))

    def _append_fallback(img_path: str, rel_path: str) -> None:
        nonlocal processed
        classification.append(
            {
                "file_name": os.path.basename(img_path),
                "relative_path": rel_path,
                "modality": "ocr",
            }
        )
        processed += 1
        _report_progress()

    def _append_result(img_path: str, rel_path: str, result) -> None:
        nonlocal processed
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if save_outputs:
            out_img_path = os.path.join(out_img_dir, rel_path)
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            annotated = result.plot(pil=True, line_width=5, font_size=20)
            try:
                import numpy as _np
                import cv2 as _cv2

                arr = _np.array(annotated)
                arr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
                _cv2.imwrite(out_img_path, arr)
            except Exception:
                pass
        rel_dir = os.path.dirname(rel_path)
        json_name = f"{stem}.json"
        if rel_dir and rel_dir != ".":
            out_json_path = os.path.join(out_json_dir, rel_dir, json_name)
        else:
            out_json_path = os.path.join(out_json_dir, json_name)
        det_json = result.tojson()
        if save_outputs:
            os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
            with open(out_json_path, "w", encoding="utf-8") as f:
                f.write(det_json)
        modality = "ocr"
        detections = []
        has_figure = False
        has_ocr = False
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                detections = json.loads(det_json)
                ratio = _figure_union_area_ratio(
                    detections,
                    w,
                    h,
                    figure_class_id=figure_class_id,
                )
                if ratio > float(figure_ratio_threshold):
                    has_figure = True
                for item in detections:
                    try:
                        cls_id = int(item.get("class"))
                    except Exception:
                        continue
                    if cls_id in OCR_CLASS_IDS:
                        has_ocr = True
                        break
                if has_figure and has_ocr:
                    modality = "ocr+figure"
                elif has_figure and not has_ocr:
                    modality = "figure"
                else:
                    modality = "ocr"
                if has_figure and save_outputs:
                    _save_figure_crops(
                        img=img,
                        rel_path=rel_path,
                        stem=stem,
                        detections=detections,
                        seg_root=seg_root,
                        w=w,
                        h=h,
                        figure_class_id=figure_class_id,
                    )
        except Exception:
            modality = "ocr"
        classification.append(
            {
                "file_name": os.path.basename(img_path),
                "relative_path": rel_path,
                "modality": modality,
            }
        )
        processed += 1
        _report_progress()

    for start in range(0, len(file_entries), batch_size):
        batch = file_entries[start : start + batch_size]
        image_paths = [item[0] for item in batch]
        try:
            det_res = model.predict(
                image_paths,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )
            results = list(det_res)
            for (img_path, rel_path), result in zip(batch, results):
                _append_result(img_path, rel_path, result)
            if len(results) < len(batch):
                for img_path, rel_path in batch[len(results) :]:
                    _append_fallback(img_path, rel_path)
        except Exception:
            for img_path, rel_path in batch:
                _append_fallback(img_path, rel_path)
    return classification, processed


def _extract_id(fname: str) -> str:
    """Helper to extract ID from filename or path."""
    stem = os.path.splitext(fname)[0]
    idx = stem.find("(")
    if idx != -1 and stem.endswith(")"):
        stem = stem[:idx]
    return stem


def _update_classification_summary(out_cls_path: str, classification: list[dict]) -> None:
    existing_data = []
    if os.path.exists(out_cls_path):
        try:
            with open(out_cls_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []
    by_rel: dict[str, dict] = {}
    for item in existing_data:
        rel_paths: list[tuple[str, str, str]] = []
        if "ID" in item:
            records = item.get("病历") or item.get("images") or []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                if "relative_path" in rec:
                    rel = rec.get("relative_path")
                    if rel:
                        rel_paths.append(
                            (
                                rel,
                                rec.get("modality") or "ocr",
                                rec.get("file_name") or os.path.basename(rel),
                            )
                        )
                else:
                    for key, imgs in rec.items():
                        if not isinstance(imgs, list):
                            continue
                        category = str(key)
                        for im in imgs:
                            if not isinstance(im, dict):
                                continue
                            fname = im.get("file_name")
                            if not fname:
                                continue
                            rel = im.get("relative_path") or os.path.join(category, fname)
                            rel_paths.append(
                                (
                                    rel,
                                    im.get("modality") or "ocr",
                                    fname,
                                )
                            )
        else:
            rel_single = item.get("relative_path")
            if rel_single:
                rel_paths.append(
                    (
                        rel_single,
                        item.get("modality") or "ocr",
                        item.get("file_name") or os.path.basename(rel_single),
                    )
                )
            rel_list = item.get("relative_paths")
            if isinstance(rel_list, list):
                for r in rel_list:
                    if r:
                        rel_paths.append(
                            (
                                r,
                                item.get("modality") or "ocr",
                                item.get("file_name") or os.path.basename(r),
                            )
                        )
        for rel, mod, fname in rel_paths:
            by_rel[rel] = {
                "file_name": fname,
                "relative_path": rel,
                "modality": mod,
            }
    for item in classification:
        rel = item.get("relative_path")
        if not rel:
            continue
        by_rel[rel] = {
            "file_name": item.get("file_name") or os.path.basename(rel),
            "relative_path": rel,
            "modality": item.get("modality") or "ocr",
        }

    grouped: dict[str, dict[str, list[dict]]] = {}
    for entry in by_rel.values():
        rel = entry.get("relative_path") or ""
        fname = entry.get("file_name") or os.path.basename(rel)
        id_key = _extract_id(fname)
        
        parts = rel.split(os.sep)
        category = parts[0] if parts and parts[0] else ""
        # 没有子目录时，category 保持为空，不使用 "default"
        mod = entry.get("modality") or "ocr"
        id_map = grouped.get(id_key)
        if id_map is None:
            id_map = {}
            grouped[id_key] = id_map
        arr = id_map.get(category)
        if arr is None:
            arr = []
            id_map[category] = arr
        arr.append(
            {
                "file_name": fname,
                "modality": mod,
            }
        )
    updated_data = []
    for id_key in sorted(grouped.keys()):
        id_map = grouped[id_key]
        records: list[dict] = []
        for category in sorted(id_map.keys()):
            imgs = sorted(id_map[category], key=lambda x: x.get("file_name") or "")
            records.append({category: imgs})
        updated_data.append({"ID": id_key, "病历": records})
    with open(out_cls_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)


def update_summary_with_validation(validation_results_path: str, classification_results_path: str, output_classification_path: str = None) -> None:
    """
    读取人工验证结果（ocr_figure_validation_results.json），将其中的 modality
    更新到总的 classification_results.json 中，并将最终合并后的结果保存为
    一个新的文件。

    Args:
        validation_results_path (str): 验证结果文件路径。
        classification_results_path (str): 原始分类结果文件路径 (源文件)。
        output_classification_path (str, optional): 输出的新文件路径。如果未指定，
                                                    默认会在源文件同目录下生成 classification_results_validated.json。
                                                    每次生成时都会覆盖该文件，确保内容是基于源文件和验证结果重新生成的。

    逻辑：
    1. 读取 validation_results（list[dict]），建立映射。
    2. 读取 classification_results.json（源文件）。
    3. 遍历源文件数据，找到匹配的文件并更新 modality。
    4. 将更新后的数据写入 output_classification_path（覆盖写入）。
    """
    if not os.path.exists(validation_results_path):
        print(f"Warning: Validation results file not found: {validation_results_path}")
        return

    if not os.path.exists(classification_results_path):
        print(f"Warning: Classification results file not found: {classification_results_path}")
        return

    # 确定输出路径
    if not output_classification_path:
        output_dir = os.path.dirname(classification_results_path)
        output_classification_path = os.path.join(output_dir, "classification_results_validated.json")

    try:
        # 1. 读取验证结果
        with open(validation_results_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        
        # 建立映射: (ID, category, file_name) -> modality
        val_map = {} # Key: (ID, category, file_name), Value: modality

        # 检查 val_data 是否为 list，并且判断其结构类型
        if isinstance(val_data, list) and len(val_data) > 0:
            first_item = val_data[0]
            if "relative_path" in first_item:
                # 这是一个扁平列表结构 (旧格式)
                for item in val_data:
                    rp = item.get("relative_path")
                    mod = item.get("modality")
                    if rp and mod:
                        val_map[rp] = mod 
            elif "ID" in first_item and "病历" in first_item:
                # 这是一个层级结构 (新格式，与 classification_results.json 一致)
                for record in val_data:
                    record_id = record.get("ID")
                    bingli_list = record.get("病历", [])
                    for cat_dict in bingli_list:
                        for category, file_list in cat_dict.items():
                            for file_info in file_list:
                                fname = file_info.get("file_name")
                                mod = file_info.get("modality")
                                if record_id and category and fname and mod:
                                    val_map[(record_id, category, fname)] = mod

        # 2. 始终读取原始的 classification_results.json (源文件)
        with open(classification_results_path, "r", encoding="utf-8") as f:
            cls_data = json.load(f)

        if not val_map:
            print("No validation updates found or unrecognized format. Copying source to output.")
        else:
            # 3. 遍历并更新
            updated_count = 0
            for record in cls_data:
                record_id = record.get("ID")
                bingli_list = record.get("病历", [])
                for cat_dict in bingli_list:
                    for category, file_list in cat_dict.items():
                        for file_info in file_list:
                            fname = file_info.get("file_name")
                            
                            # 使用 (ID, category, file_name) 作为 key 进行精确匹配
                            key = (record_id, category, fname)
                            
                            if key in val_map:
                                matched_modality = val_map[key]
                                if file_info.get("modality") != matched_modality:
                                    file_info["modality"] = matched_modality
                                    updated_count += 1
                            else:
                                pass
            
            print(f"Updated {updated_count} records based on validation.")

        # 4. 保存为新文件 (覆盖写入)
        with open(output_classification_path, "w", encoding="utf-8") as f:
            json.dump(cls_data, f, ensure_ascii=False, indent=2)
            
        print(f"Merged results saved to NEW file: {output_classification_path}")

    except Exception as e:
        print(f"Error merging validation results: {str(e)}")


def organize_dataset_by_modality(
    validated_json_path: str | None = None,
    output_root: str | None = None,
    with_segmentation: bool = False,
    result_json_dir: str = None,
    rawdata_root: str = None,
    min_ocr_area_ratio: float = 0.1,
    min_figure_area_ratio: float = 0.3,
) -> ToolResponse:
    """
    根据验证后的分类结果整理数据集。
    会自动尝试同步最新的分类结果和验证结果。

    Args:
        validated_json_path (str | None):
            验证后的 JSON 文件路径。如果未提供，将自动寻找并尝试重新生成。
        output_root (str | None):
            输出根目录，默认为项目下的 `program/output/data` 目录。
        with_segmentation (bool):
            是否同时执行 doclayout 分割操作，默认 False。
            如果为 True，将根据 YOLO 检测结果裁剪 figure/table/text 区域。
        result_json_dir (str | None):
            doclayout 检测结果 JSON 目录，仅在 with_segmentation=True 时使用，默认为 output/result_json
        rawdata_root (str | None):
            原始数据根目录，仅在 with_segmentation=True 时使用，默认为项目下的 rawdata
        min_ocr_area_ratio (float):
            OCR (table/text) 最小面积占比阈值，默认 0.1。
        min_figure_area_ratio (float):
            Figure (图表) 最小面积占比阈值，默认 0.3。
            裁剪区域面积小于其对应模态比例时，将跳过该裁剪。

    Returns:
        ToolResponse:
            content 为 JSON 字符串，包含整理统计信息。
            如果执行了分割，还会包含分割统计信息。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    
    # Try to sync latest results first
    raw_path = os.path.join(project_root, "output", "classification_results.json")
    val_res_path = os.path.join(project_root, "output", "ocr_figure_validation_results.json")
    auto_validated_path = os.path.join(project_root, "output", "classification_results_validated.json")
    
    if os.path.exists(raw_path):
        try:
            if os.path.exists(val_res_path):
                # If validation results exist, merge them
                update_summary_with_validation(val_res_path, raw_path, auto_validated_path)
            else:
                # If no validation results, just copy raw to validated
                shutil.copy2(raw_path, auto_validated_path)
        except Exception as e:
            # Log error but continue with what we have
            print(f"Warning: Failed to sync classification results: {e}")

    if validated_json_path and not validated_json_path.endswith("ocr_figure_validation_results.json"):
        # 清理路径中的空格
        input_path = validated_json_path.strip()
    else:
        if os.path.exists(auto_validated_path):
            input_path = auto_validated_path
        elif os.path.exists(raw_path):
            input_path = raw_path
        else:
            return ToolResponse(
                content=json.dumps(
                    {"error": {"code": "FILE_NOT_FOUND", "message": "No classification results found."}},
                    ensure_ascii=False,
                )
            )

    if not output_root:
        output_root = PROGRAM_DATA_DIR
    
    raw_root = os.path.join(project_root, "rawdata")
    
    if not os.path.exists(input_path):
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "FILE_NOT_FOUND", "message": f"File not found: {input_path}"}},
                ensure_ascii=False
            )
        )
        
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "INVALID_JSON", "message": f"Invalid JSON: {str(e)}"}},
                ensure_ascii=False
            )
        )
        
    stats = {"ocr": 0, "figure": 0, "ocr+figure": 0, "table": 0, "text": 0, "errors": 0}

    # 如果启用了分割模式，只执行分割逻辑（包含原图保存）
    # 如果没有启用分割模式，执行旧格式的原图复制
    if with_segmentation:
        # 执行分割逻辑（包含原图保存到 ocr/{category}/分割/原图/）
        segmentation_stats = _execute_segmentation(
            classification_results_path=input_path,
            result_json_dir=result_json_dir,
            rawdata_root=rawdata_root,
            output_root=output_root,
            min_ocr_area_ratio=min_ocr_area_ratio,
            min_figure_area_ratio=min_figure_area_ratio,
        )

        # 统计每个 modality 的数量
        if isinstance(data, list):
            for item in data:
                bingli = item.get("病历", [])
                for record in bingli:
                    if isinstance(record, dict):
                        for category, images in record.items():
                            if isinstance(images, list):
                                for img_info in images:
                                    modality = img_info.get("modality", "ocr")
                                    if modality == "ocr":
                                        stats["ocr"] += 1
                                    elif modality == "figure":
                                        stats["figure"] += 1
                                    elif modality == "ocr+figure":
                                        stats["ocr+figure"] += 1

        result = {
            "status": "success",
            "output_root": output_root,
            "source_file": input_path,
            "stats": stats,
            "segmentation": segmentation_stats,
            "message": "数据集整理完成，并已执行分割操作"
        }
    else:
        # 旧格式：仅复制原图到 program/output/data/{id}/{modality}/{category}/
        if isinstance(data, list):
            for item in data:
                record_id = item.get("ID", "unknown_id")
                bingli = item.get("病历", [])
                for record in bingli:
                    for category, images in record.items():
                        for img_info in images:
                            fname = img_info.get("file_name")
                            modality = img_info.get("modality")

                            if not fname or not modality:
                                continue

                            src_path = os.path.join(raw_root, category, fname)
                            if not os.path.exists(src_path):
                                continue

                            dst_dir = os.path.join(output_root, record_id, modality, category)
                            dst_path = os.path.join(dst_dir, fname)

                            os.makedirs(dst_dir, exist_ok=True)
                            try:
                                shutil.copy2(src_path, dst_path)
                                if modality == "ocr":
                                    stats["ocr"] += 1
                                elif modality == "figure":
                                    stats["figure"] += 1
                                elif modality == "ocr+figure":
                                    stats["ocr+figure"] += 1
                            except Exception as e:
                                print(f"Error copying {src_path} to {dst_path}: {e}")
                                stats["errors"] += 1

        segmentation_stats = None
        result = {
            "status": "success",
            "output_root": output_root,
            "source_file": input_path,
            "stats": stats,
            "message": "数据集整理完成（仅复制原图，未执行分割）"
        }

    # 注意：table 和 text 文件已经在 collect_image_files 中处理完成，
    # 此处不再重复调用 _organize_table_and_text_files，避免重复复制
    # stats 中的 table/text 计数已在 collect_image_files 中统计

    # 生成完全按 program/output/data 目录树级结构映射的 JSON（自动扁平化底层文件列表）
    def scan_dir_as_tree(path):
        res = {}
        files = []
        if not os.path.exists(path):
            return res
            
        for item in sorted(os.listdir(path)):
            if item.startswith('.'): 
                continue
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                res[item] = scan_dir_as_tree(item_path)
            else:
                files.append(item)
                
        # 排除自己的干扰
        if "classification.json" in files:
            files.remove("classification.json")
            
        # 如果当前文件夹下只有文件，没有子文件夹，直接返回数组，而不是 {"files": [...]}
        if files and not res:
            return files
        elif files:
            res["files"] = files
            
        return res

    try:
        final_structure = scan_dir_as_tree(output_root)
        
        # 存到 output 目录
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        classification_json_path = os.path.join(output_dir, "classification.json")
        
        with open(classification_json_path, "w", encoding="utf-8") as f:
            json.dump(final_structure, f, ensure_ascii=False, indent=2)
            
        result["classification_json_path"] = classification_json_path
    except Exception as e:
        print(f"Error generating classification.json: {e}")

    return ToolResponse(
        content=json.dumps(result, ensure_ascii=False, indent=2)
    )


organize_dataset_by_modality.__name__ = "Organize Dataset By Modality"


def _organize_table_and_text_files(
    raw_root: str,
    output_root: str,
) -> tuple[dict, dict]:
    """
    扫描 rawdata 目录，根据后缀名判断 table 和 text 文件，并整理到 `program/output/data` 目录。

    目录结构：
    - program/output/data/{id}/table/{category}/{filename}
    - program/output/data/{id}/text/{category}/{filename}

    Args:
        raw_root: rawdata 根目录
        output_root: 输出根目录（默认 `program/output/data`）

    Returns:
        tuple[dict, dict]: (table_stats, text_stats)
    """
    # 文件扩展名定义
    ext_table = {".csv", ".xls", ".xlsx", ".jsonl", ".json"}
    ext_text = {".txt", ".pdf", ".doc", ".docx", ".md", ".wps"}

    table_stats = {"count": 0, "errors": 0}
    text_stats = {"count": 0, "errors": 0}

    if not os.path.isdir(raw_root):
        return table_stats, text_stats

    def _process_file(full_path, category, fname):
        """处理单个文件"""
        ext = os.path.splitext(fname.lower())[1]

        # 判断文件类型
        if ext in ext_table:
            modality_type = "table"
        elif ext in ext_text:
            modality_type = "text"
        else:
            return False  # 跳过图片等其他文件类型

        # 从文件名提取 ID
        record_id = _extract_id(fname)

        # 构建目标路径：program/output/data/{id}/{modality_type}/{category}/{filename}
        dst_dir = os.path.join(output_root, record_id, modality_type, category)
        dst_path = os.path.join(dst_dir, fname)

        if modality_type == "table":
            # 处理表格文件（删除空列）
            if _process_table_file(full_path, dst_path):
                table_stats["count"] += 1
                return True
            else:
                table_stats["errors"] += 1
                return False
        else:
            # 文本文件直接复制（增加权限修复逻辑）
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(dst_path):
                import stat
                mode = os.stat(dst_path).st_mode
                if not mode & stat.S_IWUSR:
                    os.chmod(dst_path, mode | stat.S_IWUSR)
            try:
                shutil.copy2(full_path, dst_path)
                text_stats["count"] += 1
                return True
            except Exception as e:
                print(f"Error copying {modality_type} file {full_path} to {dst_path}: {e}")
                text_stats["errors"] += 1
                return False

    def _split_mimic_csv(src_path: str, table_root_path: str) -> bool:
        """
        专门处理 MIMIC 的四个大 CSV 文件，按 subject_id 拆分。
        """
        if not PANDAS_AVAILABLE:
            print(f"Warning: pandas not available, cannot split {src_path}")
            return False
            
        fname = os.path.basename(src_path)
        print(f"  正在按 subject_id 拆分超大表格文件: {fname}...")
        
        try:
            # 使用 chunksize 防止 2.8GB 的 radiology.csv 撑爆低配机器内存
            chunks = pd.read_csv(src_path, chunksize=100000, low_memory=False)
            for chunk in chunks:
                if "subject_id" not in chunk.columns:
                    print(f"  错误: {fname} 中找不到 subject_id 列，已跳过拆分。")
                    return False
                
                # 按 subject_id 分组
                for subj_id, group in chunk.groupby("subject_id"):
                    # 目标路径 data/<subject_id>/table/fname
                    dst_dir = os.path.join(table_root_path, str(subj_id), "table")
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, fname)
                    
                    if not os.path.exists(dst_path):
                        group.to_csv(dst_path, index=False, encoding="utf-8-sig")
                    else:
                        group.to_csv(dst_path, mode='a', header=False, index=False, encoding="utf-8-sig")
            print(f"  {fname} 拆分完成！")
            return True
        except Exception as e:
            print(f"Error splitting {fname}: {e}")
            return False

    # MIMIC 需要特殊拆分的大文件列表
    mimic_note_files = {"discharge.csv", "discharge_detail.csv", "radiology.csv", "radiology_detail.csv"}

    # 扫描 rawdata 目录下的所有文件和子目录
    for item in os.listdir(raw_root):
        item_path = os.path.join(raw_root, item)
        if os.path.isfile(item_path):
            if item in mimic_note_files:
                # 根目录下的文件如果是 MIMIC 大表，就不作为普通文件移动，而是执行拆分
                table_root = os.path.join(output_root, "table") # 这里 output_root 指向统一后的 program/output/data 路径，当前分支仍保留旧逻辑备注。
                # 旧版的 _organize_table_and_text_files 输出直接定位到单个病人层级，这里暂不调整其拆分策略。
                pass
            _process_file(item_path, "", item)
        elif os.path.isdir(item_path):
            category = item
            for fname in os.listdir(item_path):
                full_path = os.path.join(item_path, fname)
                if os.path.isfile(full_path):
                    _process_file(full_path, category, fname)

    return table_stats, text_stats


def _process_table_file(src_path: str, dst_path: str) -> bool:
    """
    处理表格文件：删除全为空的列，然后保存到目标位置。

    Args:
        src_path: 源文件路径
        dst_path: 目标文件路径

    Returns:
        bool: 处理是否成功
    """
    if not PANDAS_AVAILABLE:
        # 如果 pandas 不可用，直接复制文件
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            print(f"Error copying table file {src_path}: {e}")
            return False

    def _ensure_writable(path):
        """确保文件可写，解决 Errno 13 Permission denied 问题"""
        if os.path.exists(path):
            import stat
            mode = os.stat(path).st_mode
            if not mode & stat.S_IWUSR:
                os.chmod(path, mode | stat.S_IWUSR)

    ext = os.path.splitext(src_path.lower())[1]

    try:
        # 读取表格文件
        if ext == ".csv":
            df = pd.read_csv(src_path)
        elif ext in {".xls", ".xlsx"}:
            df = pd.read_excel(src_path)
        else:
            # 未知格式（如 .jsonl），直接复制
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            _ensure_writable(dst_path) # 核心修复
            shutil.copy2(src_path, dst_path)
            return True

        # 删除全为空的列（含空字符串/纯空白字符串）
        original_cols = len(df.columns)
        blank_mask = df.apply(
            lambda col: col.isna() | col.astype(str).str.strip().eq("")
        )
        df_cleaned = df.loc[:, ~blank_mask.all(axis=0)]
        dropped_cols = original_cols - len(df_cleaned.columns)

        if dropped_cols > 0:
            print(f"  已删除 {dropped_cols} 个空列：{os.path.basename(src_path)}")

        # 保存处理后的文件
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        _ensure_writable(dst_path) # 核心修复
        if ext == ".csv":
            df_cleaned.to_csv(dst_path, index=False, encoding="utf-8-sig")
        else:
            df_cleaned.to_excel(dst_path, index=False)

        return True
    except Exception as e:
        print(f"Error processing table file {src_path}: {e}")
        # 出错时尝试直接复制原文件
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            _ensure_writable(dst_path) # 核心修复
            shutil.copy2(src_path, dst_path)
        except:
            pass
        return False


def _execute_segmentation(
    classification_results_path: str,
    result_json_dir: str | None,
    rawdata_root: str | None,
    output_root: str,
    min_ocr_area_ratio: float = 0.1,
    min_figure_area_ratio: float = 0.3,
) -> dict:
    """
    执行 doclayout 分割操作（内部辅助函数）。

    根据 doclayout YOLO 检测结果，将分割后的图片按模态分类保存到 `program/output/data` 目录。

    目录结构:
    - program/output/data/{id}/picture/{category}/分割/图片/{id}_figure_{n}.jpg  (figure 类)
    - program/output/data/{id}/ocr/{category}/分割/图片/{id}_table_{n}.jpg      (table/text 类)
    - program/output/data/{id}/ocr/{category}/分割/原图/{原文件名}               (原始图片)

    Args:
        classification_results_path: 分类结果 JSON 路径
        result_json_dir: doclayout 检测结果 JSON 目录
        rawdata_root: 原始数据根目录
        output_root: 输出根目录
        min_ocr_area_ratio: OCR 最小面积占比阈值
        min_figure_area_ratio: Figure 最小面积占比阈值

    Returns:
        dict: 分割统计信息
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 默认路径
    if result_json_dir is None:
        result_json_dir = os.path.join(project_root, "output", "result_json")

    if rawdata_root is None:
        rawdata_root = os.path.join(project_root, "rawdata")

    # 加载分类结果
    try:
        with open(classification_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading classification results: {e}")
        return {"error": f"Failed to load JSON: {str(e)}"}

    # 统计信息
    stats = {
        "figure_crops": 0,
        "table_text_crops": 0,
        "original_images": 0,
        "errors": 0,
    }

    # DocLayout YOLO 类别 ID
    FIGURE_CLASS_ID = globals()["FIGURE_CLASS_ID"]
    TABLE_CLASS_ID = globals()["TABLE_CLASS_ID"]
    TEXT_CLASS_ID = globals()["TEXT_CLASS_ID"]

    # 遍历分类结果
    if isinstance(data, list):
        for item in data:
            record_id = item.get("ID", "unknown_id")
            bingli_list = item.get("病历", [])

            for record in bingli_list:
                if not isinstance(record, dict):
                    continue

                for category, images in record.items():
                    if not isinstance(images, list):
                        continue

                    for img_info in images:
                        fname = img_info.get("file_name")
                        modality = img_info.get("modality", "ocr")

                        if not fname:
                            continue

                        # 构建源文件路径（原图）
                        src_path = os.path.join(rawdata_root, category, fname)
                        if not os.path.exists(src_path):
                            # 尝试在 output/results 中查找
                            alt_src = os.path.join(project_root, "output", "results", category, fname)
                            if os.path.exists(alt_src):
                                src_path = alt_src
                            else:
                                stats["errors"] += 1
                                print(f"Warning: Source file not found: {src_path}")
                                continue

                        # 保存原图到 ocr 目录（去掉“分割”层级）
                        original_dst_dir = os.path.join(output_root, record_id, "ocr", category, "原图")
                        os.makedirs(original_dst_dir, exist_ok=True)
                        original_dst_path = os.path.join(original_dst_dir, fname)

                        try:
                            shutil.copy2(src_path, original_dst_path)
                            stats["original_images"] += 1
                        except Exception as e:
                            print(f"Error copying original image {src_path} to {original_dst_path}: {e}")
                            stats["errors"] += 1
                            continue

                        # 读取 doclayout 检测结果
                        stem = os.path.splitext(fname)[0]
                        result_json_path = os.path.join(result_json_dir, category, f"{stem}.json")

                        if not os.path.exists(result_json_path):
                            # 尝试带括号的文件名
                            result_json_path = os.path.join(result_json_dir, category, fname.replace(".jpg", ".json"))

                        detections = []
                        if os.path.exists(result_json_path):
                            try:
                                with open(result_json_path, "r", encoding="utf-8") as f:
                                    detections = json.load(f)
                            except Exception:
                                detections = []

                        # 根据 modality 和检测结果处理分割
                        img = cv2.imread(src_path)
                        if img is None:
                            continue

                        h, w = img.shape[:2]
                        img_area = h * w

                        # 所有图片：只要检测到 figure 区域，就保存到 figure 目录
                        # 先收集 figure 裁剪，有结果再创建目录，避免空目录
                        figure_crops = []
                        for det in detections:
                            cls_id = int(det.get("class", -1))
                            if cls_id != FIGURE_CLASS_ID:
                                continue

                            box = det.get("box", {})
                            x1 = int(max(0, box.get("x1", 0)))
                            y1 = int(max(0, box.get("y1", 0)))
                            x2 = int(min(w, box.get("x2", w)))
                            y2 = int(min(h, box.get("y2", h)))

                            if x2 <= x1 or y2 <= y1:
                                continue

                            # 计算面积占比
                            crop_area = (x2 - x1) * (y2 - y1)
                            if crop_area / img_area < min_figure_area_ratio:
                                continue  # 跳过面积占比过小的 figure 区域

                            crop = img[y1:y2, x1:x2]
                            figure_crops.append(crop)

                        # 只有在有 figure 裁剪或 modality 为 figure 时才创建目录并保存
                        if figure_crops or modality == "figure":
                            figure_dst_dir = os.path.join(output_root, record_id, "figure", category, "files")
                            os.makedirs(figure_dst_dir, exist_ok=True)

                            figure_count = 0
                            # 保存检测到的 figure 区域
                            for crop in figure_crops:
                                figure_count += 1
                                crop_name = f"{stem}_figure_{figure_count}.jpg"
                                crop_path = os.path.join(figure_dst_dir, crop_name)
                                try:
                                    cv2.imwrite(crop_path, crop)
                                    stats["figure_crops"] += 1
                                except Exception as e:
                                    print(f"Error saving figure crop {crop_path}: {e}")
                                    stats["errors"] += 1

                            # 如果没有检测到 figure 但 modality 是 figure，保存整张图
                            if figure_count == 0 and modality == "figure":
                                crop_name = f"{stem}_figure_1.jpg"
                                crop_path = os.path.join(figure_dst_dir, crop_name)
                                try:
                                    cv2.imwrite(crop_path, img)
                                    stats["figure_crops"] += 1
                                except Exception as e:
                                    print(f"Error saving full image as figure {crop_path}: {e}")
                                    stats["errors"] += 1

                        # ocr 类图片：保存 table/text 区域到 ocr 目录
                        if modality in ["ocr", "ocr+figure"]:
                            ocr_dst_dir = os.path.join(output_root, record_id, "ocr", category, "files")
                            os.makedirs(ocr_dst_dir, exist_ok=True)

                            table_text_count = 0
                            for det in detections:
                                cls_id = int(det.get("class", -1))
                                if cls_id not in [TABLE_CLASS_ID, TEXT_CLASS_ID]:
                                    continue

                                box = det.get("box", {})
                                x1 = int(max(0, box.get("x1", 0)))
                                y1 = int(max(0, box.get("y1", 0)))
                                x2 = int(min(w, box.get("x2", w)))
                                y2 = int(min(h, box.get("y2", h)))

                                if x2 <= x1 or y2 <= y1:
                                    continue

                                # 计算面积占比
                                crop_area = (x2 - x1) * (y2 - y1)
                                if crop_area / img_area < min_ocr_area_ratio:
                                    continue  # 跳过面积占比过小的 OCR 区域

                                table_text_count += 1
                                crop_name = f"{stem}_table_{table_text_count}.jpg"
                                crop_path = os.path.join(ocr_dst_dir, crop_name)

                                crop = img[y1:y2, x1:x2]
                                try:
                                    cv2.imwrite(crop_path, crop)
                                    stats["table_text_crops"] += 1
                                except Exception as e:
                                    print(f"Error saving table/text crop {crop_path}: {e}")
                                    stats["errors"] += 1

    return {
        "figure_images_saved": stats["figure_crops"],
        "table_text_crops_saved": stats["table_text_crops"],
        "original_images_saved": stats["original_images"],
        "errors": stats["errors"],
    }


def collect_image_files(target_path: str) -> ToolResponse:
    """
    收集指定路径下所有文件，根据后缀名进行初步分类（文本/表格/图片）。
    
    1. 图片文件：收集并返回列表，供后续流程使用。
    2. 文本文件 (.txt, .pdf, .doc, .docx, .md, .wps)：直接复制到 `program/output/data` 下的 text 结构目录。
    3. 表格文件 (.csv, .xls, .xlsx)：直接复制到 `program/output/data` 下的 table 结构目录。

    Args:
        target_path (str):
            根目录路径或文件路径。

    Returns:
        ToolResponse:
            content 为 JSON 字符串，包含 input_path、project_root、stats 和 files 列表（仅包含图片）。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    
    try:
        abs_target, dataset_root = _resolve_target_and_root(target_path, project_root)
    except FileNotFoundError as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "PATH_NOT_FOUND", "message": f"Path not found: {str(e)}"}},
                ensure_ascii=False,
            )
        )

    # Output directories
    data_root = PROGRAM_DATA_DIR
    text_root = os.path.join(data_root, "text")
    table_root = os.path.join(data_root, "table")
    
    # Extensions
    ext_img = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    ext_text = {".txt", ".pdf", ".doc", ".docx", ".md", ".wps"}
    ext_table = {".csv", ".xls", ".xlsx", ".jsonl", ".json"}

    file_entries: list[tuple[str, str]] = []
    stats = {"image": 0, "text": 0, "table": 0, "ignored": 0}

    def _process_file(full_path, relative_path):
        lower = full_path.lower()
        ext = os.path.splitext(lower)[1]

        # Determine category and ID for structured organization
        # relative_path usually is "Category/Filename" or just "Filename"
        parts = relative_path.split(os.sep)
        if len(parts) > 1:
            category = parts[0]
            fname = parts[-1]
        else:
            category = ""  # 没有子目录时，category 为空
            fname = relative_path

        record_id = _extract_id(fname)

        if ext in ext_img:
            file_entries.append((full_path, relative_path))
            stats["image"] += 1
        elif ext in ext_text:
            # Structure: program/output/data/{ID}/text/{category}/{filename}
            if category:
                dst = os.path.join(data_root, record_id, "text", category, fname)
            else:
                dst = os.path.join(data_root, record_id, "text", fname)
            
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if hasattr(_process_table_file, "_ensure_writable"):
                # 虽然是个局部函数，但这里简单重复逻辑或者把 helper 放外面
                pass 
            
            # 手动执行一次权限检查
            if os.path.exists(dst):
                import stat
                mode = os.stat(dst).st_mode
                if not mode & stat.S_IWUSR:
                    os.chmod(dst, mode | stat.S_IWUSR)

            try:
                shutil.copy2(full_path, dst)
                stats["text"] += 1
            except Exception as e:
                print(f"Error copying text file {full_path}: {e}")
        elif ext in ext_table:
            # Structure (for Step2_3 compatibility):
            # program/output/data/table/{category}/{filename}
            # Step2_3 的 table_reader 会在 <data_dir>/table 下递归查找 Excel。
            if category:
                dst = os.path.join(table_root, category, fname)
            else:
                dst = os.path.join(table_root, fname)

            # 处理表格文件（删除空列 & 权限修复）
            if _process_table_file(full_path, dst):
                stats["table"] += 1
        else:
            stats["ignored"] += 1

    # 新增专门的 MIMIC 大表拆分函数
    def _split_mimic_csv(src_path: str, fname: str):
        if not PANDAS_AVAILABLE:
            print(f"Warning: pandas not available, cannot split {fname}")
            return False
            
        print(f"正在按 subject_id 拆分巨型表格文件: {fname} (可能需要几分钟)...")
        try:
            # 分块读取，适配 radiology.csv 这种 2.8GB 的巨无霸
            chunks = pd.read_csv(src_path, chunksize=100000, low_memory=False)
            for chunk in chunks:
                if "subject_id" not in chunk.columns:
                    print(f"错误: {fname} 中找不到 subject_id 列，跳过拆分。")
                    return False
                
                # 把这一块大拼图按 subject_id 拆分给各自的患者
                for subj_id, group in chunk.groupby("subject_id"):
                    # user 要求: program/output/data/subject-id/table/文件
                    dst_dir = os.path.join(data_root, str(subj_id), "table")
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, fname)
                    
                    if not os.path.exists(dst_path):
                        group.to_csv(dst_path, index=False, encoding="utf-8-sig")
                    else:
                        group.to_csv(dst_path, mode='a', header=False, index=False, encoding="utf-8-sig")
            print(f"恭喜，{fname} 拆分并分发完成！")
            return True
        except Exception as e:
            print(f"Error splitting {fname}: {e}")
            return False

    mimic_special_files = {"discharge.csv", "discharge_detail.csv", "radiology.csv", "radiology_detail.csv"}

    if os.path.isdir(abs_target):
        for dirpath, _, filenames in os.walk(abs_target):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, dataset_root)
                
                # 拦截四个特殊的 MIMIC 大文件，不再走普通的文件按单 ID 复制逻辑
                if fn in mimic_special_files:
                    if _split_mimic_csv(full, fn):
                        stats["table"] += 1
                else:
                    _process_file(full, rel)
        file_entries.sort(key=lambda x: x[1])
    elif os.path.isfile(abs_target):
        rel = os.path.relpath(abs_target, dataset_root)
        fn = os.path.basename(abs_target)
        if fn in mimic_special_files:
            if _split_mimic_csv(abs_target, fn):
                stats["table"] += 1
        else:
            _process_file(abs_target, rel)

    items = []
    for img_path, rel_path in file_entries:
        items.append(
            {
                "img_path": img_path,
                "relative_path": rel_path,
            }
        )
        
    return ToolResponse(
        content=json.dumps(
            {
                "input_path": abs_target,
                "project_root": project_root,
                "stats": stats,
                "files": items,
            },
            ensure_ascii=False,
        )
    )


collect_image_files.__name__ = "Collect Image Files"


def infer_and_save_layout(
    target_path: str,
    conf: float = 0.2,
    imgsz: int = DEFAULT_LAYOUT_IMGSZ,
    device: str = "mps",
    model_path: str | None = None,
    figure_class_id: int = FIGURE_CLASS_ID,
    figure_ratio_threshold: float = 0.5,
    save_outputs: bool = True,
    progress_callback=None,
) -> ToolResponse:
    """
    对指定路径下的图片执行版面分析，保存可视化结果与检测 JSON，并返回分类结果。

    Args:
        target_path (str):
            图片目录路径或单张图片路径，可以是绝对路径或相对项目根目录的路径。
        conf (float):
            YOLO 模型的置信度阈值。
        imgsz (int):
            推理时的输入图像尺寸。
        device (str):
            推理设备，例如 \"mps\"、\"cpu\"。
        model_path (str | None):
            模型权重文件路径，为 None 时使用默认权重路径。
        figure_class_id (int):
            作为 figure 判定的检测类别 ID，例如 3。
        figure_ratio_threshold (float):
            将图片判定为 figure 的最小面积占比阈值。

    Returns:
        ToolResponse:
            content 为 JSON 字符串，包含输入路径、处理数量、输出目录以及 classification 列表。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    out_cls_path = os.path.join(project_root, "output", "classification_results.json")
    try:
        abs_target, out_img_dir, out_json_dir, file_entries = _collect_image_entries(target_path)
    except FileNotFoundError as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "PATH_NOT_FOUND", "message": f"Path not found: {str(e)}"}},
                ensure_ascii=False,
            )
        )
    try:
        classification, processed = _run_inference_and_save(
            file_entries=file_entries,
            out_img_dir=out_img_dir,
            out_json_dir=out_json_dir,
            conf=conf,
            imgsz=imgsz,
            device=device,
            model_path=model_path,
            figure_class_id=figure_class_id,
            figure_ratio_threshold=figure_ratio_threshold,
            save_outputs=save_outputs,
            progress_callback=progress_callback,
        )
    except RuntimeError as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "MODEL_INIT_ERROR", "message": str(e)}},
                ensure_ascii=False,
            )
        )
    try:
        if save_outputs:
            _update_classification_summary(out_cls_path, classification)
    except Exception as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "SUMMARY_UPDATE_ERROR", "message": str(e)}},
                ensure_ascii=False,
            )
        )
    return ToolResponse(
        content=json.dumps(
            {
                "input_path": abs_target,
                "processed": processed,
                "output_img_dir": out_img_dir,
                "output_json_dir": out_json_dir,
                "classification_results": out_cls_path,
                "model_path_in_use": _MODEL_PATH_IN_USE,
                "classification": classification,
            },
            ensure_ascii=False,
        )
    )


infer_and_save_layout.__name__ = "Infer And Save Layout"


def update_classification_summary_tool(classification_json: str) -> ToolResponse:
    """
    使用一批新的分类结果更新 classification_results.json 汇总文件。

    Args:
        classification_json (str):
            JSON 字符串形式的列表，每个元素包含 file_name、relative_path 与 modality 字段。

    Returns:
        ToolResponse:
            content 为 JSON 字符串，包含 classification_results 文件路径或错误信息。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    out_cls_path = os.path.join(project_root, "output", "classification_results.json")
    try:
        data = json.loads(classification_json)
    except Exception as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "BAD_INPUT", "message": str(e)}},
                ensure_ascii=False,
            )
        )
    if not isinstance(data, list):
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "BAD_INPUT", "message": "classification_json must be a JSON list"}},
                ensure_ascii=False,
            )
        )
    try:
        _update_classification_summary(out_cls_path, data)
    except Exception as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "SUMMARY_UPDATE_ERROR", "message": str(e)}},
                ensure_ascii=False,
            )
        )
    
    # Calculate stats to return
    stats = {"ocr": 0, "figure": 0, "ocr+figure": 0}
    for item in data:
        mod = item.get("modality")
        if mod in stats:
            stats[mod] += 1
            
    return ToolResponse(
        content=json.dumps(
            {
                "classification_results": out_cls_path,
                "stats": stats,
                "message": f"Updated summary. Found {stats.get('ocr+figure', 0)} cases requiring validation."
            },
            ensure_ascii=False,
        )
    )


update_classification_summary_tool.__name__ = "Update Classification Summary"


def prepare_and_launch_validation_gui(**kwargs) -> ToolResponse:
    """
    准备待验证图片并立即启动 OCR/Figure 可视化验证工具。
    合并了之前的 'Prepare OCR+Figure Validation' 和 'Launch Validation GUI' 功能。
    该工具会自动提取 modality 为 "ocr+figure" 的图片，准备验证环境，并启动 GUI 供用户进行人工验证。
    
    Args:
        **kwargs: 忽略任何传入的参数。

    Returns:
        ToolResponse: 包含执行结果的 JSON 字符串。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    cls_path = os.path.join(project_root, "output", "classification_results.json")
    out_dir = os.path.join(project_root, "output", "ocr_figure_validation")
    
    # --- Part 1: Prepare Validation Images ---
    # Don't delete existing directory to preserve previous work if any
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(cls_path):
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "FILE_NOT_FOUND", "message": f"{cls_path} not found."}},
                ensure_ascii=False
            )
        )
        
    try:
        with open(cls_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "JSON_ERROR", "message": str(e)}},
                ensure_ascii=False
            )
        )

    # Load already validated keys to avoid re-copying/validating
    validated_keys = set()
    validated_path = os.path.join(project_root, "output", "ocr_figure_validation_results.json")
    if os.path.exists(validated_path):
        try:
            with open(validated_path, "r", encoding="utf-8") as f:
                v_data = json.load(f)
            if isinstance(v_data, list):
                for item in v_data:
                    if not isinstance(item, dict): continue
                    records = item.get("病历") or []
                    if isinstance(records, list):
                        for rec in records:
                            if isinstance(rec, dict):
                                for cat, imgs in rec.items():
                                    if isinstance(imgs, list):
                                        for img in imgs:
                                            if isinstance(img, dict):
                                                fname = img.get("file_name")
                                                if fname: validated_keys.add((str(cat), fname))
        except Exception:
            pass
            
    raw_root = os.path.join(project_root, "rawdata")
    if not isinstance(data, list): data = []
    
    cases_count = 0
    for item in data:
        if not isinstance(item, dict): continue
        records = item.get("病历") or []
        if not isinstance(records, list): continue
        for rec in records:
            if not isinstance(rec, dict): continue
            for category, imgs in rec.items():
                if not isinstance(imgs, list): continue
                category_name = str(category)
                for info in imgs:
                    if not isinstance(info, dict): continue
                    if info.get("modality") != "ocr+figure": continue
                    fname = info.get("file_name")
                    if not fname: continue
                    if (category_name, fname) in validated_keys: continue
                    
                    # Copy file
                    src_path = os.path.join(raw_root, category_name, fname)
                    rel_dst = os.path.join(category_name, fname)
                    dst_path = os.path.join(out_dir, rel_dst)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    if os.path.exists(src_path):
                        try:
                            shutil.copy2(src_path, dst_path)
                            cases_count += 1
                        except:
                            pass

    # Check if we have any NEW files to validate
    # cases_count > 0 means we have new files that need validation
    if cases_count == 0:
        # No new files were copied, check if validation is already complete
        # by checking if validation result file exists and has data
        if os.path.exists(validated_path):
            try:
                with open(validated_path, "r", encoding="utf-8") as f:
                    v_data = json.load(f)
                if isinstance(v_data, list) and len(v_data) > 0:
                    # Validation already completed
                    return ToolResponse(
                        content=json.dumps(
                            {"status": "skipped", "message": "所有 ocr+figure 图片已完成验证，无需重复验证。请直接调用 Organize Dataset By Modality 工具整理数据集。"},
                            ensure_ascii=False
                        )
                    )
            except Exception:
                pass
        
        # Check if there are any unvalidated files in out_dir
        unvalidated_count = 0
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Check if this file is already validated
                    rel_dir = os.path.relpath(root, out_dir)
                    category = rel_dir if rel_dir != "." else ""
                    key = (category, file)
                    if key not in validated_keys:
                        unvalidated_count += 1
        
        if unvalidated_count == 0:
            return ToolResponse(
                content=json.dumps(
                    {"status": "skipped", "message": "所有 ocr+figure 图片已完成验证，无需重复验证。请直接调用 Organize Dataset By Modality 工具整理数据集。"},
                    ensure_ascii=False
                )
            )

    # --- Part 2: Launch GUI ---
    gui_script = os.path.abspath(__file__)
    try:
        print(f"Launching validation GUI script: {gui_script}")
        # Using subprocess to run the GUI in a separate process (blocking)
        subprocess.run([sys.executable, gui_script], check=True)
        
        output_json = os.path.join(project_root, "output", "ocr_figure_validation_results.json")
        if os.path.exists(output_json):
            return ToolResponse(
                content=json.dumps(
                    {
                        "status": "success",
                        "message": "Validation GUI completed successfully.",
                        "output_file": output_json,
                        "cases_prepared": cases_count
                    },
                    ensure_ascii=False,
                )
            )
        else:
             return ToolResponse(
                content=json.dumps(
                    {"status": "warning", "message": "Validation GUI closed but output file not found."},
                    ensure_ascii=False,
                )
            )
    except subprocess.CalledProcessError as e:
        return ToolResponse(
            content=json.dumps(
                {"error": {"code": "GUI_EXECUTION_ERROR", "message": f"GUI script failed with return code {e.returncode}"}},
                ensure_ascii=False,
            )
        )
    except Exception as e:
         return ToolResponse(
            content=json.dumps(
                {"error": {"code": "UNKNOWN_ERROR", "message": str(e)}},
                ensure_ascii=False,
            )
        )

prepare_and_launch_validation_gui.__name__ = "prepare_and_launch_validation_gui"


def launch_gradio_app():
    import gradio as gr
    import threading
    import time
    
    project_root = BASE_DIR
    validation_dir = os.path.join(project_root, "output", "ocr_figure_validation")
    output_json_path = os.path.join(project_root, "output", "ocr_figure_validation_results.json")
    
    image_files = []
    if os.path.exists(validation_dir):
        for root, dirs, files in os.walk(validation_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, validation_dir)
                    image_files.append({
                        "full_path": full_path,
                        "relative_path": rel_path,
                        "file_name": file
                    })
    image_files.sort(key=lambda x: x['relative_path'])
    
    if not image_files:
        print("No images found for validation.")
        sys.exit(0)

    with gr.Blocks(title="OCR/Figure Validation Tool") as demo:
        gr.Markdown("## 👁️ OCR / Figure 模态人工验证服务\n请根据下方出图，判断它是纯文字（OCR）还是含图表（Figure）。验证完成后会自动继续任务。")
        
        state_imgs = gr.State(image_files)
        state_idx = gr.State(0)
        state_results = gr.State([])
        
        with gr.Column(visible=True) as main_ui:
            progress_md = gr.Markdown("### Loading...")
            img_disp = gr.Image(type="filepath", interactive=False, height=500)
            
            with gr.Row():
                ocr_btn = gr.Button("📝 纯文本 (OCR)", variant="primary", size="lg")
                fig_btn = gr.Button("📊 图表 (Figure)", variant="primary", size="lg")
                
        finish_md = gr.Markdown(visible=False)
        
        def load_initial(imgs, idx):
            if not imgs:
                return gr.update(value="没有找到验证图片。")
            img_data = imgs[idx]
            prog = f"### 进度: {idx + 1} / {len(imgs)} \n **当前文件:** {img_data['relative_path']}"
            return gr.update(value=prog), gr.update(value=img_data["full_path"])
            
        demo.load(load_initial, inputs=[state_imgs, state_idx], outputs=[progress_md, img_disp])
        
        def process_choice(choice_type, imgs, idx, results):
            img_data = imgs[idx]
            rel_path = img_data["relative_path"]
            parts = rel_path.split(os.sep)
            category = parts[0] if len(parts) > 1 else ""
            fname = img_data["file_name"]
            id_key = _extract_id(fname)
            
            results.append({
                "ID": id_key,
                "category": category,
                "file_name": fname,
                "relative_path": rel_path,
                "modality": "figure" if choice_type == "Figure" else "ocr"
            })
            
            idx += 1
            if idx < len(imgs):
                next_img = imgs[idx]
                prog = f"### 进度: {idx + 1} / {len(imgs)} \n **当前文件:** {next_img['relative_path']}"
                return (
                    idx, results, 
                    gr.update(value=prog), gr.update(value=next_img["full_path"]), 
                    gr.update(visible=True), gr.update(visible=False)
                )
            else:
                try:
                    # 分层保存结构
                    grouped = {}
                    for res in results:
                        ik = res.get("ID", "unknown")
                        cat = res.get("category", "")
                        if ik not in grouped: grouped[ik] = {}
                        if cat not in grouped[ik]: grouped[ik][cat] = []
                        grouped[ik][cat].append({"file_name": res["file_name"], "modality": res["modality"]})
                        
                    output_data = []
                    for ik in sorted(grouped.keys()):
                        records = []
                        for cat in sorted(grouped[ik].keys()):
                            records.append({cat: grouped[ik][cat]})
                        output_data.append({"ID": ik, "病历": records})
                        
                    with open(output_json_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                        
                    # 合并结果
                    classification_json_path = os.path.join(project_root, "output", "classification_results.json")
                    update_summary_with_validation(output_json_path, classification_json_path)
                    
                    msg = f"### ✅ 验证全部完成！\n结果已保存并合并。**当前网页可关闭，服务器 Agent 会自动继续下一阶段的任务。**"
                    
                    def delayed_exit():
                        time.sleep(1.5)
                        os._exit(0)
                    threading.Thread(target=delayed_exit).start()
                    
                except Exception as e:
                    msg = f"### ❌ 保存失败: {e}"
                    
                return (
                    idx, results, 
                    gr.update(), gr.update(), 
                    gr.update(visible=False), gr.update(value=msg, visible=True)
                )

        ocr_btn.click(
            process_choice, 
            inputs=[gr.State("OCR"), state_imgs, state_idx, state_results], 
            outputs=[state_idx, state_results, progress_md, img_disp, main_ui, finish_md]
        )
        fig_btn.click(
            process_choice, 
            inputs=[gr.State("Figure"), state_imgs, state_idx, state_results], 
            outputs=[state_idx, state_results, progress_md, img_disp, main_ui, finish_md]
        )

    print("\n" + "="*50)
    print("🚀 [Web 验证启动] 请在浏览器中打开以下链接（如: http://<服务器IP>:7860）进行人工验证。")
    print("完成后 Agent 将自动继续执行。")
    print("="*50 + "\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[validation_dir])

if __name__ == "__main__":
    try:
        import gradio as gr
    except ImportError:
        print("错误：未找到 gradio 包。请在终端运行: pip install gradio", file=sys.stderr)
        sys.exit(1)
        
    launch_gradio_app()
