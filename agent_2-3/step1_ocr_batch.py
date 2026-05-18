#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 批量 OCR（单进程多线程，利用 onnxruntime 释放 GIL）

用法：
  python step1_ocr_batch.py --input 归档/output/step1_results --output ocr_cache.json --workers 64
"""
import argparse
import concurrent.futures
import json
import os
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tools_v2.explore_tools import scan_directory


# 全局 engine，线程共享（onnxruntime session 是线程安全的）
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
        # 简单清洗
        import re
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return pid, filename, text
    except Exception:
        return pid, filename, ""


def main():
    parser = argparse.ArgumentParser(description="批量 OCR（单进程多线程）")
    parser.add_argument("--input", "-i", required=True, help="输入目录")
    parser.add_argument("--output", "-o", default="ocr_cache.json", help="输出 JSON 路径")
    parser.add_argument("--workers", "-w", type=int, default=64, help="并发线程数（默认: 64）")
    args = parser.parse_args()

    print(f"扫描目录: {args.input}")
    info = scan_directory(args.input)
    if not info.get("success"):
        print(f"扫描失败: {info.get('error')}")
        sys.exit(1)

    patients = info.get("patients", {})
    all_imgs: List[Tuple[str, str, str]] = []
    for pid, pdata in patients.items():
        for f in pdata.get("files", []):
            if f.get("file_type") == "image":
                all_imgs.append((pid, f["path"], f["filename"]))

    print(f"共 {len(all_imgs)} 张图片，使用 {args.workers} 线程并发 OCR")
    print(f"预计耗时: ~{len(all_imgs) / max(args.workers * 0.016, 1) / 60:.0f} 分钟")

    # 预热引擎
    print("预热 OCR 引擎...", flush=True)
    _get_engine()

    results: Dict[str, Dict[str, str]] = {}
    success_count = 0
    fail_count = 0

    from tqdm import tqdm

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for pid, filename, text in tqdm(
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

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"\nOCR 完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"结果保存到: {args.output}")


if __name__ == "__main__":
    main()
