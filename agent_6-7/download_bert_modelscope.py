# -*- coding: utf-8 -*-
"""
从魔搭 ModelScope 下载中文句向量模型（供 ICD-10 BERT 向量库使用）

默认模型: BAAI/bge-small-zh-v1.5（与 build_icd10_vector_store.py / run_step6_7_ml_pipeline.py 默认一致）

依赖:
  pip install modelscope

用法:
  cd agent_6-7
  python download_bert_modelscope.py
  python download_bert_modelscope.py --model BAAI/bge-small-zh-v1.5 --out ./models/bge-small-zh

下载完成后，构建向量库前设置本地路径（不要用 HuggingFace 拉取）:
  export ICD10_BERT_MODEL=/绝对路径/agent_6-7/models/bge-small-zh
  python build_icd10_vector_store.py

说明:
  - 若网络仅可访问魔搭、不可访问 HuggingFace，请先用本脚本下载，再指向 local_dir。
  - ModelScope 账号 Token 一般公开模型不需要；若需私有模型可设置环境变量 MODELSCOPE_API_TOKEN。
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "models", "bge-small-zh-v1.5")
_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def _snapshot_download(model_id: str, local_dir: str, revision: str | None) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            print("请先安装: pip install modelscope", file=sys.stderr)
            sys.exit(1)

    parent = os.path.dirname(os.path.abspath(local_dir))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # 魔搭文档：local_dir 指定后直接落到该目录（便于 SentenceTransformer 加载）
    if revision:
        path = snapshot_download(model_id, local_dir=local_dir, revision=revision)
    else:
        path = snapshot_download(model_id, local_dir=local_dir)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="从魔搭 ModelScope 下载 BERT/句向量模型")
    parser.add_argument(
        "--model",
        default=os.environ.get("ICD10_BERT_MODELSCOPE_ID", _DEFAULT_MODEL),
        help=f"魔搭模型 ID（默认: {_DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help=f"保存到本地目录（默认: {_DEFAULT_OUT}）",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选：模型版本/commit，如 master 或具体 revision",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)

    print(f"模型 ID: {args.model}")
    print(f"保存目录: {out_dir}")
    print("开始从 ModelScope 下载…")

    path = _snapshot_download(args.model, out_dir, args.revision)

    print("\n下载完成。")
    print(f"本地路径: {path}")
    print("\n后续命令示例:")
    print(f"  export ICD10_BERT_MODEL={path}")
    print("  python build_icd10_vector_store.py")


if __name__ == "__main__":
    main()
