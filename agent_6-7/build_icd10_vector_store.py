# -*- coding: utf-8 -*-
"""
从 ICD-10.xlsx 构建中文 BERT 向量库（一次性离线运行，供 Step 6/7 管道复用）

输出目录（默认）: agent_6-7/data/icd10_bert_index/
  - embeddings.npy   float32 (N, D)，L2 归一化，便于点积=余弦相似度
  - icd_codes.npy      诊断编码
  - icd_names.npy      诊断名称（与 xlsx 一致）
  - meta.json          模型名、维度、条数、源文件 mtime

依赖:
  pip install sentence-transformers torch openpyxl pandas numpy

用法:
  cd agent_6-7 && python build_icd10_vector_store.py
  ICD10_BERT_MODEL=BAAI/bge-small-zh-v1.5 python build_icd10_vector_store.py

若无法访问 HuggingFace，可先从魔搭下载到本地再建库:
  pip install -r requirements_modelscope.txt
  python download_bert_modelscope.py
  export ICD10_BERT_MODEL=$PWD/models/bge-small-zh-v1.5
  python build_icd10_vector_store.py
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# 与 run_step6_7_ml_pipeline.py 一致的路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")


def _resolve_icd10_xlsx() -> str:
    env_path = os.environ.get("ICD10_XLSX", "").strip()
    candidates = [
        env_path,
        os.path.join(_SCRIPT_DIR, "ICD-10.xlsx"),
        os.path.join(_SCRIPT_DIR, "..", "ICD-10.xlsx"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "ICD-10.xlsx"),
    ]
    for path in candidates:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return abs_path
    return os.path.abspath(os.path.join(_SCRIPT_DIR, "ICD-10.xlsx"))


_ICD10_XLSX = _resolve_icd10_xlsx()
_DEFAULT_OUT = os.path.join(_DATA_DIR, "icd10_bert_index")
_DEFAULT_MODEL = os.environ.get("ICD10_BERT_MODEL", "BAAI/bge-small-zh-v1.5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ICD-10 BERT vector index from xlsx")
    parser.add_argument("--xlsx", default=_ICD10_XLSX, help="Path to ICD-10.xlsx")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="Output directory for index")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="sentence-transformers model id")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    xlsx = os.path.abspath(args.xlsx)
    if not os.path.isfile(xlsx):
        raise FileNotFoundError(f"找不到 ICD-10 文件: {xlsx}")

    print(f"[1/4] 读取 {xlsx} …")
    df = pd.read_excel(xlsx, dtype=str)
    df = df.dropna(subset=["诊断名称", "诊断编码"])
    df["诊断名称"] = df["诊断名称"].str.strip()
    df["诊断编码"] = df["诊断编码"].str.strip()
    names = df["诊断名称"].tolist()
    codes = df["诊断编码"].tolist()
    n = len(names)
    print(f"      共 {n} 条诊断条目")

    print(f"[2/4] 加载模型 {args.model} …")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    print(f"[3/4] 编码（batch={args.batch_size}，normalize_embeddings=True）…")
    t0 = time.time()
    emb = model.encode(
        names,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"      完成，形状 {emb.shape}，耗时 {elapsed:.1f}s")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), emb)
    np.save(os.path.join(out_dir, "icd_codes.npy"), np.array(codes, dtype=object))
    np.save(os.path.join(out_dir, "icd_names.npy"), np.array(names, dtype=object))

    mtime = os.path.getmtime(xlsx)
    meta = {
        "model": args.model,
        "dim": int(dim),
        "n_vectors": int(n),
        "xlsx_path": xlsx,
        "xlsx_mtime": mtime,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[4/4] 已写入: {out_dir}")
    print("      embeddings.npy | icd_codes.npy | icd_names.npy | meta.json")


if __name__ == "__main__":
    main()
