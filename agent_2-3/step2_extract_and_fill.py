#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: 基于 OCR 缓存做 LLM 信息抽取 + 回填表格

读取 step1 产出的 ocr_cache.json，跳过 OCR 阶段，直接并发调用 LLM 抽取实体，
然后回填到模板表格。

用法：
  python step2_extract_and_fill.py --input 归档/output/step1_results --cache ocr_cache.json --output results-all/ --concurrency 100
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tools_v2.extract_tools import extract_from_text
from tools_v2.io_tools import save_rows_to_csv, save_result_json, load_excel_table
from tools_v2.explore_tools import scan_directory

import tqdm as _tqdm_module


async def main():
    parser = argparse.ArgumentParser(description="基于 OCR 缓存做 LLM 抽取 + 回填")
    parser.add_argument("--input", "-i", required=True, help="输入目录（如 归档/output/step1_results）")
    parser.add_argument("--cache", "-c", default="ocr_cache.json", help="OCR 缓存 JSON（step1 产出）")
    parser.add_argument("--output", "-o", default="results-all/", help="输出目录")
    parser.add_argument("--concurrency", type=int, default=100, help="LLM 并发数（默认: 100）")
    parser.add_argument("--fill", action="store_true", help="抽取完成后自动回填表格")
    args = parser.parse_args()

    # 读取 OCR 缓存
    print(f"读取 OCR 缓存: {args.cache}")
    with open(args.cache, "r", encoding="utf-8") as f:
        ocr_cache: Dict[str, Dict[str, str]] = json.load(f)

    total_texts = sum(len(v) for v in ocr_cache.values())
    print(f"共 {len(ocr_cache)} 个病人, {total_texts} 条 OCR 文本")

    # 扫描目录获取病人信息
    info = scan_directory(args.input)
    patients = info.get("patients", {})

    # 准备输出目录
    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"results_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # LLM 并发信号量
    sem = asyncio.Semaphore(args.concurrency)

    # 构建所有抽取任务：(pid, filename, text)
    all_tasks: List[Dict] = []
    for pid, texts in ocr_cache.items():
        for filename, text in texts.items():
            if text and len(text.strip()) >= 50:
                all_tasks.append({"pid": pid, "filename": filename, "text": text})

    print(f"需要 LLM 抽取: {len(all_tasks)} 条文本")

    # 并发抽取
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

    # 用 as_completed 实时更新进度条
    bar = _tqdm_module.tqdm(total=len(all_tasks), desc="LLM抽取进度", unit="条", dynamic_ncols=True)
    futs = [asyncio.ensure_future(_extract_one(t)) for t in all_tasks]
    for fut in asyncio.as_completed(futs):
        await fut
        bar.update(1)
    bar.close()

    print(f"\n抽取完成: 成功 {stats['success']}, 失败 {stats['failed']}")
    total_entities = sum(len(v) for v in entities_by_patient.values())
    print(f"总实体数: {total_entities}")

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
        # 补全所有可能的列
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

    # 保存汇总 JSON
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
    print(f"汇总: {summary_path}")

    # 自动回填
    if args.fill and ent_rows:
        print("\n开始回填表格...")
        from fill_table import run as fill_run
        filled_dir = os.path.join(out_dir, "filled_tables")
        await fill_run(
            reorg_dir=args.input,
            results_dir=out_dir,
            output_dir=filled_dir,
            use_llm=False,
            dry_run=False,
            verbose=False,
        )
        print(f"回填完成: {filled_dir}")


if __name__ == "__main__":
    asyncio.run(main())
