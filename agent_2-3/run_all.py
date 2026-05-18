#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一体化 OCR + LLM 抽取 + 回填

Step 1: 多线程 OCR（~4.5小时，吞吐 ~1张/s）
Step 2: LLM 并发抽取（~5分钟）
Step 3: 回填表格

用法：
  python run_all.py --input 归档/output/step1_results --output results-all/ --ocr-workers 64 --concurrency 100

  # 如果已有 ocr_cache.json，跳过 OCR 直接抽取：
  python run_all.py --input 归档/output/step1_results --output results-all/ --cache ocr_cache.json --concurrency 100
"""
import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tools_v2.explore_tools import scan_directory
from tools_v2.extract_tools import extract_from_text
from tools_v2.io_tools import save_rows_to_csv, save_result_json

import tqdm as _tqdm_module

# ---------------------------------------------------------------------------
# OCR 引擎（全局单例，线程共享）
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR(
            det_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            rec_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            cls_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
        )
    return _engine


def _do_ocr(item: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """(pid, path, filename) -> (pid, filename, cleaned_text)"""
    pid, path, filename = item
    try:
        engine = _get_engine()
        result, _ = engine(path)
        if not result:
            return pid, filename, ""
        text = "\n".join(r[1] for r in result if len(r) >= 2)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return pid, filename, text
    except Exception:
        return pid, filename, ""


# ---------------------------------------------------------------------------
# Step 1: OCR
# ---------------------------------------------------------------------------

def run_ocr(all_imgs: List[Tuple[str, str, str]], workers: int, cache_path: str) -> Dict[str, Dict[str, str]]:
    """多线程 OCR，返回 {pid: {filename: text}}，同时保存到 cache_path。"""
    print(f"\n{'='*60}")
    print(f"Step 1: OCR ({len(all_imgs)} 张图片, {workers} 线程)")
    print(f"{'='*60}")

    # 预热引擎
    print("预热 OCR 引擎...", flush=True)
    _get_engine()

    results: Dict[str, Dict[str, str]] = {}
    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for pid, filename, text in _tqdm_module.tqdm(
            pool.map(_do_ocr, all_imgs),
            total=len(all_imgs),
            desc="OCR进度",
            unit="张",
            dynamic_ncols=True,
        ):
            if text:
                results.setdefault(pid, {})[filename] = text
                success_count += 1
            else:
                fail_count += 1

    # 保存缓存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"OCR 完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"缓存保存到: {cache_path}")
    return results


# ---------------------------------------------------------------------------
# Step 2: LLM 抽取
# ---------------------------------------------------------------------------

async def run_extract(
    ocr_cache: Dict[str, Dict[str, str]],
    concurrency: int,
    out_dir: str,
    ts: str,
) -> str:
    """LLM 并发抽取，返回 entities_csv 路径。"""
    total_texts = sum(len(v) for v in ocr_cache.values())
    print(f"\n{'='*60}")
    print(f"Step 2: LLM 抽取 ({total_texts} 条文本, {concurrency} 并发)")
    print(f"{'='*60}")

    sem = asyncio.Semaphore(concurrency)

    # 构建任务
    all_tasks: List[Dict] = []
    for pid, texts in ocr_cache.items():
        for filename, text in texts.items():
            if text and len(text.strip()) >= 50:
                all_tasks.append({"pid": pid, "filename": filename, "text": text})

    print(f"有效文本: {len(all_tasks)} 条")

    entities_by_patient: Dict[str, List[Dict]] = {}
    _lock = asyncio.Lock()
    stats = {"success": 0, "failed": 0}

    async def _extract_one(task: Dict) -> None:
        pid = task["pid"]
        filename = task["filename"]
        text = task["text"]
        async with sem:
            result = await extract_from_text(text)
        if result.get("success") and result.get("entities"):
            ents = []
            for ent in result["entities"]:
                ent["source_file"] = filename
                ent["source_category"] = "ocr"
                ents.append(ent)
            async with _lock:
                entities_by_patient.setdefault(pid, []).extend(ents)
                stats["success"] += 1
        else:
            async with _lock:
                stats["failed"] += 1

    bar = _tqdm_module.tqdm(total=len(all_tasks), desc="LLM抽取", unit="条", dynamic_ncols=True)
    futs = [asyncio.ensure_future(_extract_one(t)) for t in all_tasks]
    for fut in asyncio.as_completed(futs):
        await fut
        bar.update(1)
    bar.close()

    total_entities = sum(len(v) for v in entities_by_patient.values())
    print(f"抽取完成: 成功 {stats['success']}, 失败 {stats['failed']}, 总实体 {total_entities}")

    # 保存 entities CSV
    entities_csv = os.path.join(out_dir, f"entities_{ts}.csv")
    ent_rows = []
    for pid, ents in entities_by_patient.items():
        for ent in ents:
            row = {"patient_id": pid}
            row.update(ent)
            ent_rows.append(row)

    if ent_rows:
        ent_cols = ["patient_id", "name", "category", "value", "unit",
                    "source_file", "source_category"]
        all_keys = set()
        for r in ent_rows:
            all_keys.update(r.keys())
        for k in all_keys:
            if k not in ent_cols:
                ent_cols.append(k)
        save_rows_to_csv(ent_rows, entities_csv, columns=ent_cols)
        print(f"实体 CSV: {entities_csv}")

    # 保存 patients CSV
    patients_csv = os.path.join(out_dir, f"patients_{ts}.csv")
    pat_rows = []
    for pid in ocr_cache.keys():
        ents = entities_by_patient.get(pid, [])
        pat_rows.append({
            "patient_id": pid,
            "ocr_file_count": len(ocr_cache[pid]),
            "entity_count": len(ents),
            "categories": "; ".join(set(e.get("category", "") for e in ents)),
        })
    save_rows_to_csv(pat_rows, patients_csv)
    print(f"患者 CSV: {patients_csv}")

    # 保存汇总
    summary = {
        "success": True,
        "processed_at": datetime.now().isoformat(),
        "statistics": {
            "total_patients": len(ocr_cache),
            "total_ocr_texts": total_texts,
            "extracted_texts": stats["success"],
            "failed_texts": stats["failed"],
            "total_entities": total_entities,
        },
        "output_dir": out_dir,
        "entities_csv": entities_csv,
        "patients_csv": patients_csv,
    }
    summary_path = os.path.join(out_dir, f"summary_{ts}.json")
    save_result_json(summary, summary_path)

    return entities_csv


# ---------------------------------------------------------------------------
# Step 3: 回填
# ---------------------------------------------------------------------------

async def run_fill(input_dir: str, out_dir: str):
    """回填表格。"""
    print(f"\n{'='*60}")
    print(f"Step 3: 回填表格")
    print(f"{'='*60}")

    from fill_table import run as fill_run
    filled_dir = os.path.join(out_dir, "filled_tables")
    await fill_run(
        reorg_dir=input_dir,
        results_dir=out_dir,
        output_dir=filled_dir,
        use_llm=False,
        dry_run=False,
        verbose=False,
    )
    print(f"回填完成: {filled_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    parser = argparse.ArgumentParser(description="一体化 OCR + LLM 抽取 + 回填")
    parser.add_argument("--input", "-i", required=True, help="输入目录")
    parser.add_argument("--output", "-o", default="results-all/", help="输出目录")
    parser.add_argument("--cache", "-c", default=None, help="已有 OCR 缓存（跳过 Step 1）")
    parser.add_argument("--ocr-workers", type=int, default=64, help="OCR 线程数（默认: 64）")
    parser.add_argument("--concurrency", type=int, default=100, help="LLM 并发数（默认: 100）")
    parser.add_argument("--no-fill", action="store_true", help="跳过回填步骤")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"results_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    cache_path = args.cache or os.path.join(out_dir, "ocr_cache.json")

    # Step 1: OCR
    if args.cache and os.path.exists(args.cache):
        print(f"使用已有 OCR 缓存: {args.cache}")
        with open(args.cache, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)
        print(f"加载 {len(ocr_cache)} 个病人的 OCR 结果")
    else:
        print(f"扫描目录: {args.input}")
        info = scan_directory(args.input)
        patients = info.get("patients", {})
        all_imgs = [
            (pid, f["path"], f["filename"])
            for pid, pdata in patients.items()
            for f in pdata.get("files", [])
            if f.get("file_type") == "image"
        ]
        ocr_cache = run_ocr(all_imgs, args.ocr_workers, cache_path)

    # Step 2: LLM 抽取
    entities_csv = await run_extract(ocr_cache, args.concurrency, out_dir, ts)

    # Step 3: 回填
    if not args.no_fill and entities_csv:
        await run_fill(args.input, out_dir)

    print(f"\n{'='*60}")
    print(f"全部完成! 输出目录: {out_dir}")
    print(f"{'='*60}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
