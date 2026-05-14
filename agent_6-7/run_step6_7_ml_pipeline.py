# -*- coding: utf-8 -*-
"""
STEP 6 + STEP 7 ML Pipeline: 一致性验证 → ML训练数据生成
==========================================================
架构理念
────────
工具（Tools）只负责"数据访问"，不内置任何领域知识：
  - Step 6 Agent 自己探索 schema → 识别关键字段 → 制定检查规则 → 评分
  - Step 7 Agent 自己探索列分布 → 决定特征列 → 决定标签列 → 生成数据集

所有领域判断（哪些列有意义、如何聚合、标签如何映射）均由 LLM 推理完成。

ICD 标准化增强（新增）
────────────────────
  - LLM 自动扫描全部字段，识别哪些列含有诊断信息（不限于主诊断字段）
  - 对每个诊断字段的每个值，LLM 过滤非疾病内容，只保留疾病名称
  - 提取后再走 ICD-10 标准化流程

ICD-10 查询链路
────────────────
  1. 精确 / 规范化匹配（ICD-10.xlsx）
  2. LLM 将英文缩写等转为标准中文后重试精确匹配
  3. BERT 向量检索：预建向量库 data/icd10_bert_index/（见 build_icd10_vector_store.py）
  4. 字符串模糊匹配（rapidfuzz）兜底
  5. LLM 在 BERT+模糊 Top 候选中择优

环境变量（可选）:
  ICD10_USE_BERT=1          启用 BERT 检索（默认 1）
  ICD10_BERT_MODEL=...      默认 BAAI/bge-small-zh-v1.5（须与建库时一致）
  ICD10_BERT_MIN_SIM=0.70   余弦相似度 ≥ 此值则直接采纳 BERT Top1

数据来源
────────
  data_input/input.csv 或 data_input/filtered.csv
  data_input/selection_report.json

输出
────
  ../program/output/step6_results/consistency_report_<ts>.txt/.json
  ../program/output/step7_results/ml_dataset_<ts>.csv/.jsonl/.json

运行方式
────────
  python run_step6_7_ml_pipeline.py

向量库构建（首次或 xlsx/模型变更后）:
  cd agent_6-7 && pip install sentence-transformers torch
  python build_icd10_vector_store.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import unicodedata
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

try:
    from configs.loader import get_agent_config
except Exception:
    get_agent_config = None

# ══════════════════════════════════════════════════════════════════
# 0. 全局配置
# ══════════════════════════════════════════════════════════════════

AGENT_CFG = get_agent_config("agent_6_7") if get_agent_config else {}
OPENAI_API_KEY = AGENT_CFG.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = (
    AGENT_CFG.get("base_url")
    or AGENT_CFG.get("api_base")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or "https://api.openai.com/v1"
)
MODEL_NAME = AGENT_CFG.get("model") or AGENT_CFG.get("model_name") or os.environ.get("MODEL_NAME", "gpt-4.1-mini")


def _strip_thinking_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

# ── 数据源切换：DATA_SOURCE=mimic 启用 MIMIC 模式，默认 original ──────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_project_root() -> str:
    return os.path.dirname(_SCRIPT_DIR)


def get_default_input_dir() -> str:
    return os.path.join(_SCRIPT_DIR, "data_input")


def get_program_output_root() -> str:
    return os.path.join(get_project_root(), "program", "output")


def get_default_output_step6_dir() -> str:
    return os.path.join(get_program_output_root(), "step6_results")


def get_default_output_step7_dir() -> str:
    return os.path.join(get_program_output_root(), "step7_results")


def _find_latest_in_dir(search_dir: str, prefix: str = "", suffix: str = "") -> str:
    if not search_dir or not os.path.isdir(search_dir):
        return ""
    candidates = [
        os.path.join(search_dir, name)
        for name in os.listdir(search_dir)
        if (not prefix or name.startswith(prefix)) and (not suffix or name.endswith(suffix))
    ]
    if not candidates:
        return ""
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_default_filtered_csv(input_dir: str) -> str:
    for name in ("filtered.csv", "input.csv"):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            return path
    csv_candidates = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(".csv")
    ] if os.path.isdir(input_dir) else []
    csv_candidates.sort(key=os.path.getmtime, reverse=True)
    return csv_candidates[0] if csv_candidates else ""


def _resolve_default_selection_report(input_dir: str) -> str:
    path = os.path.join(input_dir, "selection_report.json")
    if os.path.isfile(path):
        return path
    latest = _find_latest_in_dir(input_dir, suffix="_selection_report.json")
    if latest:
        return latest
    return _find_latest_in_dir(input_dir, prefix="selection_report", suffix=".json")

# 自动检测：若 data/mimic/ 下存在筛选后 CSV，则默认使用 MIMIC 模式
_MIMIC_DATA_DIR = get_default_input_dir()
_mimic_files_exist = os.path.isdir(_MIMIC_DATA_DIR) and any(
    f.startswith("liver_notes_extracted_filtered") and f.endswith(".csv")
    for f in os.listdir(_MIMIC_DATA_DIR)
)
_default_source = "mimic" if _mimic_files_exist else "original"
DATA_SOURCE = os.environ.get("DATA_SOURCE", _default_source).lower()
_IS_MIMIC = DATA_SOURCE == "mimic"

if _IS_MIMIC:
    DATA_DIR = get_default_input_dir()
    PATIENT_ID_COL = "record_index"   # MIMIC 用 record_index 标识患者/住院
else:
    DATA_DIR = get_default_input_dir()
    PATIENT_ID_COL = "patient_id"

_PATIENT_ID_CANDIDATES = (
    "patient_id",
    PATIENT_ID_COL,
    "subject_id",
    "record_index",
    "hadm_id",
    "id",
)
_ID_ALIAS_COLS = {"subject_id", "record_index", "hadm_id", "id", "note_index"}


# ICD-10 本地库路径（可由 ICD10_XLSX 环境变量覆盖）
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
    fallback = os.path.abspath(candidates[1])
    print(f"[ICD-10本地库] 未找到 ICD-10.xlsx，当前默认路径: {fallback}。可通过 ICD10_XLSX 环境变量指定。")
    return fallback


ICD10_XLSX = _resolve_icd10_xlsx()
# BERT 向量库目录（由 build_icd10_vector_store.py 生成）
ICD10_VECTOR_DIR = os.path.join(_SCRIPT_DIR, "data", "icd10_bert_index")
ICD10_USE_BERT = os.environ.get("ICD10_USE_BERT", "1").lower() not in ("0", "false", "no")
ICD10_BERT_MODEL = os.environ.get("ICD10_BERT_MODEL", "BAAI/bge-small-zh-v1.5")
ICD10_BERT_MIN_SIM = float(os.environ.get("ICD10_BERT_MIN_SIM", "0.70"))


def _find_latest(prefix: str, suffix: str, search_dir: str | None = None) -> str:
    d = search_dir or DATA_DIR
    candidates = [f for f in os.listdir(d)
                  if f.startswith(prefix) and f.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"找不到 {prefix}*{suffix}（目录：{d}）")
    return os.path.join(d, sorted(candidates)[-1])


def _safe_find_latest(prefix: str, suffix: str, search_dir: str | None = None) -> str:
    try:
        return _find_latest(prefix, suffix, search_dir)
    except Exception:
        return ""


def _resolve_default_runtime_paths() -> dict[str, str]:
    input_dir = get_default_input_dir()
    filtered_csv = _resolve_default_filtered_csv(input_dir)
    raw_csv = os.path.join(input_dir, "raw.csv")
    if not os.path.isfile(raw_csv):
        raw_csv = filtered_csv
    selection_report = _resolve_default_selection_report(input_dir)
    output_step6 = get_default_output_step6_dir()
    output_step7 = get_default_output_step7_dir()

    return {
        "raw_csv": raw_csv,
        "filtered_csv": filtered_csv,
        "selection_report": selection_report,
        "output_step6": output_step6,
        "output_step7": output_step7,
    }


def configure_runtime(runtime_overrides: dict[str, Any] | None = None) -> dict[str, str]:
    global RAW_CSV, FILTERED_CSV, SELECTION_REPORT, OUTPUT_STEP6, OUTPUT_STEP7

    defaults = _resolve_default_runtime_paths()
    overrides = runtime_overrides or {}

    raw_csv = str(overrides.get("raw_csv") or defaults.get("raw_csv") or "").strip()
    filtered_csv = str(overrides.get("filtered_csv") or defaults.get("filtered_csv") or "").strip()
    selection_report = str(overrides.get("selection_report") or defaults.get("selection_report") or "").strip()
    output_step6 = str(overrides.get("output_step6_dir") or defaults.get("output_step6") or "").strip()
    output_step7 = str(overrides.get("output_step7_dir") or defaults.get("output_step7") or "").strip()

    if filtered_csv:
        filtered_csv = os.path.abspath(filtered_csv)
    if raw_csv:
        raw_csv = os.path.abspath(raw_csv)
    if selection_report:
        selection_report = os.path.abspath(selection_report)
    if output_step6:
        output_step6 = os.path.abspath(output_step6)
    if output_step7:
        output_step7 = os.path.abspath(output_step7)

    if filtered_csv and (not raw_csv or not os.path.isfile(raw_csv)):
        raw_csv = filtered_csv

    RAW_CSV = raw_csv
    FILTERED_CSV = filtered_csv
    SELECTION_REPORT = selection_report
    OUTPUT_STEP6 = output_step6
    OUTPUT_STEP7 = output_step7

    if OUTPUT_STEP6:
        os.makedirs(OUTPUT_STEP6, exist_ok=True)
    if OUTPUT_STEP7:
        os.makedirs(OUTPUT_STEP7, exist_ok=True)

    return {
        "raw_csv": RAW_CSV,
        "filtered_csv": FILTERED_CSV,
        "selection_report": SELECTION_REPORT,
        "output_step6": OUTPUT_STEP6,
        "output_step7": OUTPUT_STEP7,
    }


def _validate_runtime_inputs(require_selection_report: bool = False) -> None:
    if not FILTERED_CSV or not os.path.isfile(FILTERED_CSV):
        raise FileNotFoundError(f"Step6/7 输入 filtered_csv 不存在: {FILTERED_CSV}")
    if not RAW_CSV or not os.path.isfile(RAW_CSV):
        raise FileNotFoundError(f"Step6 输入 raw_csv 不存在: {RAW_CSV}")
    if require_selection_report and (not SELECTION_REPORT or not os.path.isfile(SELECTION_REPORT)):
        raise FileNotFoundError(f"Step7 输入 selection_report 不存在: {SELECTION_REPORT}")


_DEFAULT_RUNTIME_PATHS = configure_runtime(None)

CONFIDENCE_THRESHOLD = 0.70

# Step 6 → Step 7 传递通过患者列表
_passed_patient_ids: list[str] = []

# 长文本列黑名单：这些列包含 OCR 原文/NLP 抽取全文，不适合输出给 LLM
_SKIP_COLS = {
    "ocr_text", "preprocessed_text", "file_path", "error",
    "Test_抽取", "Test_标准化", "Disease_抽取", "Disease_标准化",
    "Drug_抽取", "Drug_标准化", "Symptom_抽取", "Symptom_标准化",
    "Treatment_抽取", "Treatment_标准化", "Anatomy_抽取", "Anatomy_标准化",
    "LabValue_抽取", "LabValue_标准化", "Finding_抽取", "Finding_标准化",
    "Other_抽取", "Other_标准化", "relations", "temporal_info",
    "quantity_info",
}

# MIMIC 数据特有的长文本列（仅在 MIMIC 模式下追加到黑名单）
_MIMIC_EXTRA_SKIP_COLS = {
    "所有用药", "实验室检验_摘要", "所有手术操作_titles", "所有手术操作_codes",
    "所有诊断_icd_codes", "impression", "indication",
    "抽取_Disease", "抽取_Drug", "抽取_Symptom", "抽取_Test",
    "抽取_Treatment", "抽取_LabValue", "抽取_Finding",
}
if _IS_MIMIC:
    _SKIP_COLS = _SKIP_COLS | _MIMIC_EXTRA_SKIP_COLS


def _trunc(val: Any, n: int = 80) -> str:
    s = str(val)
    return s if len(s) <= n else s[:n] + "…"


def _should_skip(col: str) -> bool:
    return col in _SKIP_COLS or col in _ID_ALIAS_COLS


def _read_data(path: str) -> "pd.DataFrame":
    """
    读取 CSV 并规范化患者 ID 列名为 'patient_id'。

    在 MIMIC 模式下，PATIENT_ID_COL='record_index'，会自动重命名为 'patient_id'。
    若上游输出使用通用 id 列，也会统一映射为 'patient_id'。
    """
    df = pd.read_csv(path, low_memory=False)
    if "patient_id" in df.columns:
        return df
    for col in _PATIENT_ID_CANDIDATES:
        if col != "patient_id" and col in df.columns:
            return df.rename(columns={col: "patient_id"})
    available = ", ".join(map(str, df.columns[:12]))
    raise KeyError(
        "Input CSV must contain a patient identifier column. "
        "Expected one of: "
        + ", ".join(dict.fromkeys(_PATIENT_ID_CANDIDATES))
        + f". First columns: {available}"
    )


def _report_columns(report: dict[str, Any]) -> list[str]:
    """Return selected feature columns from known Step5 report schemas."""
    for key in ("final_columns", "selected_columns", "columns"):
        cols = report.get(key)
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols]
    return []


def _report_task_text(report: dict[str, Any]) -> str:
    for key in ("task_text", "task_en", "task"):
        val = str(report.get(key) or "").strip()
        if val:
            return val
    return "N/A"


_DIAGNOSIS_COLUMN_KEYWORDS = (
    "诊断", "综合征", "疾病", "病因",
    "diagnosis", "diagnoses", "icd", "etiology",
    "hcc", "hepatocellular", "carcinoma", "cancer",
    "cirrhosis", "hepatitis", "metastasis", "metastatic",
    "abscess", "thrombus", "thrombosis", "lesion", "mass",
)


def _is_diagnosis_like_column(col: str) -> bool:
    col_l = str(col).lower()
    if col_l.endswith("_count") or col_l in {"diagnoses_count"}:
        return False
    return any(k.lower() in col_l for k in _DIAGNOSIS_COLUMN_KEYWORDS)


# ══════════════════════════════════════════════════════════════════
# LLM 辅助缓存
# ══════════════════════════════════════════════════════════════════

# LLM 诊断字段识别缓存
_LLM_DIAG_FIELDS_CACHE: list[str] | None = None

# LLM 疾病内容提取缓存 {(field_name, value) -> disease_or_None}
_LLM_DISEASE_EXTRACT_CACHE: dict[tuple[str, str], str | None] = {}

# ICD-10 查询结果缓存 {中文诊断名 -> (code, name)}
_ICD10_CACHE: dict[str, tuple[str, str]] = {}


# ══════════════════════════════════════════════════════════════════
# 本地 ICD-10 查询库（精确 + BERT 向量 + 模糊 + LLM）
# ══════════════════════════════════════════════════════════════════
#
# 向量库：data/icd10_bert_index/（python build_icd10_vector_store.py）
# 查询顺序见 _local_icd10_lookup 文档字符串。
# ══════════════════════════════════════════════════════════════════

def _load_icd10_local() -> tuple["pd.DataFrame", dict[str, tuple[str, str]]]:
    """
    加载 ICD-10.xlsx，返回 DataFrame 和 {诊断名称 → (诊断编码, 诊断名称)} 字典。
    若文件不存在，返回空结构并打印警告。
    """
    try:
        df = pd.read_excel(ICD10_XLSX, dtype=str)
        df = df.dropna(subset=["诊断名称", "诊断编码"])
        df["诊断名称"] = df["诊断名称"].str.strip()
        df["诊断编码"] = df["诊断编码"].str.strip()
        name_to_entry: dict[str, tuple[str, str]] = {
            row["诊断名称"]: (row["诊断编码"], row["诊断名称"])
            for _, row in df.iterrows()
        }
        print(f"[ICD-10本地库] 已加载 {len(name_to_entry)} 条诊断记录（{ICD10_XLSX}）")
        return df, name_to_entry
    except Exception as exc:
        print(f"[ICD-10本地库] 加载失败: {exc}，ICD 查询将全部返回 NOT_FOUND")
        return pd.DataFrame(), {}


_ICD10_DF, _ICD10_NAME_TO_ENTRY = _load_icd10_local()
# 供模糊匹配使用的名称列表（按加载顺序固定，避免每次重建）
_ICD10_NAMES_LIST: list[str] = list(_ICD10_NAME_TO_ENTRY.keys())

# ── BERT 向量库（懒加载）──────────────────────────────────────────
_ICD10_EMB_MATRIX: np.ndarray | None = None
_ICD10_EMB_CODES: np.ndarray | None = None
_ICD10_EMB_NAMES: np.ndarray | None = None
_ICD10_EMB_META: dict[str, Any] | None = None
_ICD10_ST_MODEL: Any = None
_ICD10_VECTOR_WARNED: bool = False


def _try_load_icd10_vector_store() -> bool:
    """
    从 data/icd10_bert_index/ 加载预计算的 embeddings 与编码/名称表。
    若文件不存在或损坏，返回 False（后续仅使用字符串模糊匹配）。
    """
    global _ICD10_EMB_MATRIX, _ICD10_EMB_CODES, _ICD10_EMB_NAMES, _ICD10_EMB_META, _ICD10_VECTOR_WARNED

    if _ICD10_EMB_MATRIX is not None:
        return True

    emb_path = os.path.join(ICD10_VECTOR_DIR, "embeddings.npy")
    codes_path = os.path.join(ICD10_VECTOR_DIR, "icd_codes.npy")
    names_path = os.path.join(ICD10_VECTOR_DIR, "icd_names.npy")
    meta_path = os.path.join(ICD10_VECTOR_DIR, "meta.json")

    if not os.path.isfile(emb_path):
        if not _ICD10_VECTOR_WARNED:
            print(
                f"[ICD-BERT] 未找到向量库 {emb_path}，跳过 BERT 检索。"
                f"请运行: python {os.path.join(os.path.dirname(__file__), 'build_icd10_vector_store.py')}"
            )
            _ICD10_VECTOR_WARNED = True
        return False

    try:
        _ICD10_EMB_MATRIX = np.load(emb_path)
        _ICD10_EMB_CODES = np.load(codes_path, allow_pickle=True)
        _ICD10_EMB_NAMES = np.load(names_path, allow_pickle=True)
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                _ICD10_EMB_META = json.load(f)
            built_model = (_ICD10_EMB_META or {}).get("model", "")
            if built_model and built_model != ICD10_BERT_MODEL:
                print(
                    f"[ICD-BERT] 警告：向量库模型为「{built_model}」，"
                    f"当前 ICD10_BERT_MODEL=「{ICD10_BERT_MODEL}」。请重建向量库或对齐环境变量。"
                )
        else:
            _ICD10_EMB_META = {}
        n = int(_ICD10_EMB_MATRIX.shape[0])
        if _ICD10_EMB_CODES.shape[0] != n or _ICD10_EMB_NAMES.shape[0] != n:
            raise ValueError("embeddings 与 codes/names 行数不一致")
        print(
            f"[ICD-BERT] 已加载向量库 {n} 条 × dim={_ICD10_EMB_MATRIX.shape[1]}（{ICD10_VECTOR_DIR}）"
        )
        return True
    except Exception as exc:
        _ICD10_EMB_MATRIX = None
        print(f"[ICD-BERT] 向量库加载失败: {exc}")
        return False


def _get_icd10_sentence_model() -> Any:
    """懒加载 sentence-transformers 模型（与建库时 ICD10_BERT_MODEL 一致）。"""
    global _ICD10_ST_MODEL
    if _ICD10_ST_MODEL is not None:
        return _ICD10_ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "使用 ICD BERT 检索需要安装: pip install sentence-transformers torch"
        ) from exc
    _ICD10_ST_MODEL = SentenceTransformer(ICD10_BERT_MODEL)
    return _ICD10_ST_MODEL


def _icd10_bert_search(query: str, top_k: int = 8) -> list[tuple[str, str, float]]:
    """
    用 BERT 句向量在预建库上做余弦相似度 Top-K 检索。

    Returns:
        [(icd_code, icd_name, cosine_sim), ...] 按相似度降序
    """
    if not ICD10_USE_BERT or not _try_load_icd10_vector_store():
        return []
    if _ICD10_EMB_MATRIX is None:
        return []

    q = str(query).strip()
    if not q:
        return []

    try:
        model = _get_icd10_sentence_model()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        qv = np.asarray(qv, dtype=np.float32)
        # 矩阵已 L2 归一化时点积即余弦相似度
        sims = _ICD10_EMB_MATRIX @ qv
        k = min(top_k, int(sims.shape[0]))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out: list[tuple[str, str, float]] = []
        for i in idx:
            code = str(_ICD10_EMB_CODES[int(i)])
            name = str(_ICD10_EMB_NAMES[int(i)])
            out.append((code, name, float(sims[int(i)])))
        return out
    except Exception as exc:
        print(f"    [ICD-BERT-异常] {exc}")
        return []


def _merge_icd_candidates_for_llm(
    bert_cands: list[tuple[str, str, float]],
    fuzzy_cands: list[tuple[str, str, float]],
    max_total: int = 8,
) -> list[tuple[str, str, float]]:
    """合并 BERT 与模糊候选，去重（按编码），BERT 相似度映射到 0–100 供 LLM 提示。"""
    seen: set[str] = set()
    merged: list[tuple[str, str, float]] = []
    for code, name, sim in bert_cands:
        if code in seen:
            continue
        seen.add(code)
        merged.append((code, name, sim * 100.0))
    for code, name, ratio in fuzzy_cands:
        if code in seen:
            continue
        seen.add(code)
        merged.append((code, name, ratio))
        if len(merged) >= max_total:
            break
    return merged[:max_total]


def _normalize_diag(text: str) -> str:
    """
    规范化诊断文本：转全角为半角、去除空格与常见标点。
    用于规范化精确匹配，使"良性阵发性位置性眩晕（BPPV）"与
    "良性阵发性位置性眩晕(BPPV)"均能命中。
    """
    # 全角 → 半角
    text = unicodedata.normalize("NFKC", text)
    # 去除空格与括号内英文缩写（保留括号中的中文）
    text = re.sub(r'\s+', '', text)
    # 去除尾部的括号英文标注，例如 (BPPV)
    text = re.sub(r'\([A-Za-z0-9 \-]+\)$', '', text).strip()
    return text


# 预先建立规范化索引，用于规范化精确匹配
_ICD10_NORM_TO_ENTRY: dict[str, tuple[str, str]] = {
    _normalize_diag(name): entry
    for name, entry in _ICD10_NAME_TO_ENTRY.items()
}


def _local_lookup_icd10_fuzzy(diagnosis: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    """
    使用 rapidfuzz 对诊断名称进行模糊匹配。

    策略：
      1. 用 WRatio（召回率高）取 top-30 候选
      2. 用 fuzz.ratio（严格字符级比对，防止子串虚高）对候选重新排序
      3. 返回 ratio 排序后的 top_k 结果

    这样避免了 WRatio 因子串命中导致的误匹配
    （如"损伤"2字 vs "可逆性毛细胞损伤"9字 得到 WRatio=90 的假高分）。

    Args:
        diagnosis: 待查询诊断名称（中文）
        top_k: 返回候选数量

    Returns:
        [(icd10_code, name, ratio_score), ...] 按 ratio 分数降序排列，score ∈ [0, 100]
    """
    try:
        from rapidfuzz import process, fuzz as rfuzz
        # 第一步：WRatio 取宽泛候选（高召回）
        wide_candidates = process.extract(
            diagnosis,
            _ICD10_NAMES_LIST,
            scorer=rfuzz.WRatio,
            limit=30,
        )
        # 第二步：用严格 ratio 重新评分并排序（防止子串膨胀）
        reranked = sorted(
            [
                (_ICD10_NAME_TO_ENTRY[name][0], name, rfuzz.ratio(diagnosis, name))
                for name, _, _ in wide_candidates
                if name in _ICD10_NAME_TO_ENTRY
            ],
            key=lambda x: -x[2],
        )
        return reranked[:top_k]
    except ImportError:
        print("    [模糊匹配] rapidfuzz 未安装，跳过模糊匹配步骤")
        return []
    except Exception as exc:
        print(f"    [模糊匹配-异常] {exc}")
        return []


def _llm_select_icd10_candidate(
    diagnosis: str,
    candidates: list[tuple[str, str, float]],
) -> tuple[str, str] | None:
    """
    当模糊匹配无高置信度结果时，让 LLM 从候选列表中选择最合适的 ICD-10 条目。

    Args:
        diagnosis: 原始诊断名称
        candidates: [(code, name, score), ...]

    Returns:
        (icd10_code, name) 或 None（LLM 判定无合适候选）
    """
    if not candidates:
        return None

    options_text = "\n".join(
        f"  {i+1}. {name}（编码: {code}）"
        for i, (code, name, _) in enumerate(candidates)
    )
    prompt = (
        f"你是医疗编码专家。需要将以下诊断名称映射到 ICD-10 编码。\n\n"
        f"待映射诊断：{diagnosis}\n\n"
        f"候选 ICD-10 条目（来自中国国家标准库）：\n{options_text}\n\n"
        f"规则：\n"
        f"  1. 若某个候选与待映射诊断是同一疾病（允许叫法/缩写不同），返回对应序号（1-{len(candidates)}）。\n"
        f"  2. 若所有候选均与待映射诊断不是同一疾病，返回 0。\n"
        f"  3. 只返回数字，不加任何解释。\n\n"
        f"选择（数字）："
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = _strip_thinking_text(response.choices[0].message.content)
        idx = int(re.search(r'\d+', raw).group())
        if 1 <= idx <= len(candidates):
            code, name, _ = candidates[idx - 1]
            print(f"    [LLM候选选择] '{diagnosis}' → 选择第{idx}项: {name}（{code}）")
            return code, name
        print(f"    [LLM候选选择] '{diagnosis}' → LLM 返回 {idx}，无合适候选")
        return None
    except Exception as exc:
        print(f"    [LLM候选选择-失败] {exc}")
        return None


_LLM_CN_NORMALIZE_CACHE: dict[str, str] = {}


def _llm_cn_normalize(term: str) -> str | None:
    """
    对含有英文缩写、英文单词或混合语言的诊断术语，调用 LLM 转换为
    中国 ICD-10 标准库中对应的中文名称，以便后续精确/模糊查找。

    典型场景：
      "bppv"  → "良性阵发性位置性眩晕"
      "UPVD"  → "单侧原发前庭病变"
      "PPPD"  → "持续性姿势感知性头晕"

    如果术语已经是纯中文且长度 > 3，直接返回 None（无需标准化）。

    Args:
        term: 原始诊断术语

    Returns:
        标准中文诊断术语；若无法标准化或不需要则返回 None。
    """
    if not term or not term.strip():
        return None

    cn_chars = sum(1 for c in term if '\u4e00' <= c <= '\u9fff')
    # 主要是中文（超过 60%）且长度足够，无需标准化
    if cn_chars / max(len(term.strip()), 1) > 0.6 and len(term.strip()) >= 4:
        return None

    if term in _LLM_CN_NORMALIZE_CACHE:
        return _LLM_CN_NORMALIZE_CACHE[term]

    prompt = (
        "你是医学术语专家。请将以下诊断术语转换为中国 ICD-10 标准库中对应的标准中文名称。\n"
        "规则：\n"
        "  1. 若是英文缩写（如 bppv、UPVD、PPPD），输出其对应的标准中文全称。\n"
        "  2. 若是英文全称，直接翻译为中文标准术语。\n"
        "  3. 若无法确定对应的标准中文术语，输出 NULL。\n"
        "  4. 只输出中文术语本身，不加引号、不加标点、不做解释。\n\n"
        f"诊断术语：{term}\n"
        "标准中文名称（或 NULL）："
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        raw = _strip_thinking_text(response.choices[0].message.content).strip('"\'').strip()
        result = None if raw.upper() == "NULL" or not raw else raw
        _LLM_CN_NORMALIZE_CACHE[term] = result
        if result:
            print(f"    [LLM标准化] '{term}' → '{result}'")
        return result
    except Exception as exc:
        print(f"    [LLM标准化-失败] '{term}': {exc}")
        return None


def _local_icd10_lookup(diagnosis: str) -> tuple[str, str, str]:
    """
    本地 ICD-10 查询主入口（精确匹配 + BERT 向量检索 + 模糊 + LLM）。

    查询链路（按优先级顺序）：
      1. 精确匹配         — 字典 O(1)
      2. 规范化精确匹配   — 去除全角/标点后再查
      3. LLM 中文标准化   — 对英文缩写(bppv/UPVD)等先转为标准中文，再重试 1–2
      4. BERT 向量检索    — 预建库 Top1 余弦相似度 ≥ ICD10_BERT_MIN_SIM 则采纳
      5. 模糊匹配         — WRatio 召回 + ratio 精排，≥ 85 自动命中
      6. LLM 候选选择     — BERT Top-K 与模糊候选合并后择优

    Args:
        diagnosis: 诊断名称（中文 / 英文缩写 / 混合均可）

    Returns:
        (source, icd10_code, preferred_name)
          source: "EXACT"|"NORM"|"CN_NORM"|"BERT"|"FUZZY"|"LLM_SELECT"|"NOT_FOUND"|"ERROR"
    """
    diag_clean = str(diagnosis).strip()
    if not diag_clean or diag_clean.lower() in ("nan", "none", "null"):
        return "NOT_FOUND", "NOT_FOUND", ""

    # ── 缓存命中 ──────────────────────────────────────────────────
    if diag_clean in _ICD10_CACHE:
        cached = _ICD10_CACHE[diag_clean]
        print(f"    [ICD-缓存] '{diag_clean}' → {cached[0]}")
        return "CACHE", cached[0], cached[1]

    try:
        def _store(src, code, name):
            _ICD10_CACHE[diag_clean] = (code, name)
            return src, code, name

        # ── 1. 精确匹配 ──────────────────────────────────────────
        if diag_clean in _ICD10_NAME_TO_ENTRY:
            code, name = _ICD10_NAME_TO_ENTRY[diag_clean]
            print(f"    [ICD-精确] '{diag_clean}' → {code}")
            return _store("EXACT", code, name)

        # ── 2. 规范化精确匹配 ────────────────────────────────────
        norm = _normalize_diag(diag_clean)
        if norm in _ICD10_NORM_TO_ENTRY:
            code, name = _ICD10_NORM_TO_ENTRY[norm]
            print(f"    [ICD-规范化] '{diag_clean}' → {code}（{name}）")
            return _store("NORM", code, name)

        # ── 3. LLM 中文标准化（处理英文缩写等非中文术语）────────
        cn_term = _llm_cn_normalize(diag_clean)
        if cn_term:
            if cn_term in _ICD10_NAME_TO_ENTRY:
                code, name = _ICD10_NAME_TO_ENTRY[cn_term]
                print(f"    [ICD-CN标准化] '{diag_clean}' → '{cn_term}' → {code}")
                return _store("CN_NORM", code, name)
            norm2 = _normalize_diag(cn_term)
            if norm2 in _ICD10_NORM_TO_ENTRY:
                code, name = _ICD10_NORM_TO_ENTRY[norm2]
                print(f"    [ICD-CN标准化+规范化] '{diag_clean}' → '{cn_term}' → {code}")
                return _store("CN_NORM", code, name)

        # 检索查询串：优先 LLM 标准化后的中文，否则原始诊断
        query_for_match = (cn_term if cn_term else diag_clean).strip()

        # ── 4. BERT 向量检索（预建库，语义最近邻）────────────────
        bert_cands = _icd10_bert_search(query_for_match, top_k=8)
        # 若标准化词与原文不同，对原文再检一次，取相似度更高的一组
        if cn_term and cn_term.strip() != diag_clean.strip():
            bert_alt = _icd10_bert_search(diag_clean.strip(), top_k=8)
            if bert_alt and (
                not bert_cands or bert_alt[0][2] > bert_cands[0][2]
            ):
                bert_cands = bert_alt

        if bert_cands:
            b_code, b_name, b_sim = bert_cands[0]
            if b_sim >= ICD10_BERT_MIN_SIM:
                print(
                    f"    [ICD-BERT] '{diag_clean}' → {b_code}（{b_name}）"
                    f" cos_sim={b_sim:.3f}"
                )
                return _store("BERT", b_code, b_name)

        # ── 5. 模糊匹配（WRatio 召回 + ratio 精排）──────────────
        candidates = _local_lookup_icd10_fuzzy(query_for_match, top_k=5)
        if not candidates and cn_term and cn_term != diag_clean:
            candidates = _local_lookup_icd10_fuzzy(diag_clean, top_k=5)

        if candidates:
            best_code, best_name, best_score = candidates[0]
            if best_score >= 85:
                print(f"    [ICD-模糊] '{diag_clean}' → {best_code}（{best_name}）ratio={best_score:.1f}")
                return _store("FUZZY", best_code, best_name)

        # ── 6. LLM：合并 BERT + 模糊候选 ────────────────────────
        merged = _merge_icd_candidates_for_llm(
            bert_cands or [],
            candidates or [],
            max_total=8,
        )
        if merged:
            _bs = bert_cands[0][2] if bert_cands else 0.0
            print(
                f"    [ICD-LLM选择] BERT最高={_bs:.3f}，"
                f"合并候选 {len(merged)} 条，调用 LLM…"
            )
            result = _llm_select_icd10_candidate(diag_clean, merged)
            if result:
                code, name = result
                return _store("LLM_SELECT", code, name)

        print(f"    [ICD-未找到] '{diag_clean}' 在本地库中无匹配")
        return _store("NOT_FOUND", "NOT_FOUND", "")

    except Exception as exc:
        print(f"    [ICD-严重错误] '{diag_clean}': {exc}")
        return "ERROR", f"ERROR: {exc}", ""


# ══════════════════════════════════════════════════════════════════
# LLM 辅助函数①：识别含诊断信息的字段
# ══════════════════════════════════════════════════════════════════

def _llm_identify_diagnosis_fields(columns_with_samples: dict[str, list]) -> list[str]:
    """
    调用 LLM 从数据集所有字段中识别哪些列可能含有疾病/诊断信息。

    策略：将字段名 + 最多 3 个样本值拼成提示词，让 LLM 返回疑似
    诊断字段的名称列表。对识别结果做合法性过滤（返回列名必须在
    原字段集合中存在）。

    Args:
        columns_with_samples (dict): {列名: [样本值, ...]} 字典，
            长文本列已在调用前过滤。

    Returns:
        list[str]: 识别出的可能含诊断信息的字段名列表。
    """
    col_info_lines = []
    for col, samples in columns_with_samples.items():
        clean = [str(s)[:50] for s in samples if pd.notna(s) and str(s).strip() not in ("", "nan")]
        if clean:
            col_info_lines.append(f"  - {col}：示例值=[{', '.join(clean[:3])}]")

    if not col_info_lines:
        return []

    prompt = (
        "你是医疗数据专家。以下是一份医疗数据集的字段名称和示例值。\n"
        "请识别出所有可能包含疾病诊断信息的字段，包括但不限于：\n"
        "  - 主诊断、初步诊断、最终诊断、出院诊断\n"
        "  - 可能诊断（如可能诊断1、可能诊断2）\n"
        "  - 综合征类型、疾病类型\n"
        "  - 病因诊断、定位诊断\n"
        "请只返回字段名列表，每行一个完整字段名，不要解释，不要序号，不要其他内容。\n\n"
        "字段清单：\n"
        + "\n".join(col_info_lines)
        + "\n\n含诊断信息的字段（每行一个字段名）："
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )
        raw_output = _strip_thinking_text(response.choices[0].message.content)
        diag_fields = []
        for line in raw_output.splitlines():
            field = line.strip().lstrip("-•").strip()
            if field and field in columns_with_samples:
                diag_fields.append(field)
        print(f"    [LLM识别诊断字段] 识别出 {len(diag_fields)} 个: {diag_fields}")
        return diag_fields
    except Exception as exc:
        print(f"    [LLM识别诊断字段-失败] {exc}，返回空列表")
        return []


# ══════════════════════════════════════════════════════════════════
# LLM 辅助函数②：从字段值中提取纯疾病内容
# ══════════════════════════════════════════════════════════════════

def _llm_extract_disease_content(field_name: str, field_value: str) -> str | None:
    """
    调用 LLM 从字段值中提取纯疾病/诊断名称，过滤非疾病信息。

    某些诊断字段（如"综合征类型"、"可能诊断2"）的值可能混有：
      - 状态描述（"无"、"未明确"、"待排除"）
      - 检查结论（"眼震方向向左"）
      - 纯数值或编码
    此函数使用 LLM 判断并提取其中的疾病名称部分。

    缓存键为 (field_name, field_value) 二元组，避免重复 API 调用。

    Args:
        field_name  (str): 字段名称，提供上下文帮助 LLM 判断。
        field_value (str): 字段原始值。

    Returns:
        str | None: 提取出的疾病名称；若字段值不含疾病信息则返回 None。
    """
    val_clean = str(field_value).strip()
    if not val_clean or val_clean.lower() in ("nan", "none", "null", "无", "未填写", "—", "-"):
        return None

    cache_key = (field_name, val_clean)
    if cache_key in _LLM_DISEASE_EXTRACT_CACHE:
        result = _LLM_DISEASE_EXTRACT_CACHE[cache_key]
        print(f"    [LLM疾病提取-缓存] {field_name}='{val_clean}' → {result}")
        return result

    prompt = (
        f"字段名称：{field_name}\n"
        f"字段内容：{val_clean}\n\n"
        "请从上面的字段内容中提取疾病/诊断名称。\n"
        "规则：\n"
        "  1. 若内容本身就是一个明确的疾病/综合征名称，直接返回该名称。\n"
        "  2. 若内容混有多种信息，只提取疾病/诊断部分（去除状态描述、数值等）。\n"
        "  3. 若内容完全不含疾病信息（如纯数字、日期、'无'、'正常'、检查结论），返回 NULL。\n"
        "  4. 只输出疾病名称本身，不加引号、不加标点、不做解释。\n\n"
        "输出（疾病名称 或 NULL）："
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
        )
        raw = _strip_thinking_text(response.choices[0].message.content).strip('"\'').strip()
        result = None if raw.upper() == "NULL" or not raw else raw
        _LLM_DISEASE_EXTRACT_CACHE[cache_key] = result
        print(f"    [LLM疾病提取] {field_name}='{val_clean[:40]}' → {result}")
        return result
    except Exception as exc:
        print(f"    [LLM疾病提取-失败] {exc}，返回原值降级")
        fallback = val_clean if val_clean else None
        _LLM_DISEASE_EXTRACT_CACHE[cache_key] = fallback
        return fallback


# _umls_lookup_icd10 已被 _local_icd10_lookup 替代（见上方），
# 保留此别名以兼容代码其他位置的调用（如需可直接调用 _local_icd10_lookup）。
_umls_lookup_icd10 = _local_icd10_lookup


# ══════════════════════════════════════════════════════════════════
# 1. @register_tool 装饰器工厂
# ══════════════════════════════════════════════════════════════════

def register_tool(toolkit: Toolkit):
    def decorator(func):
        toolkit.register_tool_function(func)
        return func
    return decorator


# ══════════════════════════════════════════════════════════════════
# 2. Step 6 工具集
# ══════════════════════════════════════════════════════════════════

step6_toolkit = Toolkit()


@register_tool(step6_toolkit)
def load_data_overview() -> ToolResponse:
    """
    加载数据集，返回整体概况：总行数、患者数、列数、每位患者记录数。
    同时展示第1位患者第1条记录的非空字段，帮助 Agent 理解字段结构。
    """
    raw = _read_data(RAW_CSV)
    lines = [
        "=== 数据集概况 ===",
        f"原始CSV:  {raw.shape[0]} 行 × {raw.shape[1]} 列",
        f"患者数:   {raw['patient_id'].nunique()}",
        "",
        "每位患者的记录数:",
    ]
    for pid, cnt in raw.groupby("patient_id").size().items():
        lines.append(f"  patient_id={pid}: {cnt} 条记录")

    lines.append("\n字段预览（第1条记录，长文本列已跳过）:")
    row = raw.iloc[0]
    for col in raw.columns:
        if _should_skip(col):
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip() not in ("", "nan"):
            lines.append(f"  {col}: {_trunc(val)}")

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step6_toolkit)
def get_patient_list() -> ToolResponse:
    """
    返回所有患者 ID 列表及其记录数。
    Step 6 分析时应对每位患者依次调用 analyze_patient_consistency。
    """
    raw  = _read_data(RAW_CSV)
    pids = raw.groupby("patient_id").size().reset_index()
    pids.columns = ["patient_id", "n_records"]

    lines = [f"=== 患者列表（共 {len(pids)} 位）===\n"]
    for _, r in pids.iterrows():
        lines.append(f"  patient_id={r['patient_id']}  记录数={r['n_records']}")
    lines.append(f"\n请对每位患者依次调用 analyze_patient_consistency(patient_id)。")
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step6_toolkit)
def analyze_patient_consistency(patient_id: str) -> ToolResponse:
    """
    对指定患者的所有记录进行全面一致性分析（Step 6 核心工具）。

    分析内容：
      1. 记录概况：共几条记录、来自哪些文件/类别
      2. 字段完整性：关键字段的填写率
      3. 跨记录不一致字段：同一患者不同记录中值不同的字段（含具体差异值）
      4. 数值字段波动：数值类字段跨记录的最大波动幅度
      5. 诊断一致性：诊断相关字段是否跨记录一致

    Args:
        patient_id (str): 患者 ID。
    """
    raw  = _read_data(RAW_CSV)
    mask = raw["patient_id"].astype(str) == str(patient_id)
    rows = raw[mask]

    if rows.empty:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"未找到 patient_id={patient_id}。")])

    n_rec = len(rows)
    lines = [
        f"╔══ 患者 {patient_id} 一致性分析报告 ══",
        f"║  记录数: {n_rec}",
        f"║  来源文件: {rows.get('filename', rows.index).tolist()[:5]}",
        f"║  记录类别: {rows['folder_category'].unique().tolist() if 'folder_category' in rows.columns else 'N/A'}",
        "║",
    ]

    inconsistent_fields: list[tuple[str, list, str]] = []
    for col in raw.columns:
        if col in ("patient_id", "filename", "file_path", "folder_category") \
                or _should_skip(col):
            continue
        uniq = rows[col].dropna().unique()
        if len(uniq) > 1:
            if any(k in col for k in ("诊断", "性别", "民族")):
                sev = "HIGH"
            elif any(k in col for k in ("年龄",)):
                sev = "MEDIUM"
            else:
                sev = "LOW"
            inconsistent_fields.append((col, [_trunc(v, 50) for v in uniq.tolist()], sev))

    high   = [(c, v) for c, v, s in inconsistent_fields if s == "HIGH"]
    medium = [(c, v) for c, v, s in inconsistent_fields if s == "MEDIUM"]
    low    = [(c, v) for c, v, s in inconsistent_fields if s == "LOW"]

    lines.append(f"║ 【跨记录不一致字段】共 {len(inconsistent_fields)} 个")
    if high:
        lines.append(f"║   HIGH（{len(high)} 个）—— 诊断/性别等关键字段：")
        for col, vals in high:
            lines.append(f"║     {col}")
            lines.append(f"║       → {vals}")
    if medium:
        lines.append(f"║   MEDIUM（{len(medium)} 个）—— 人口学字段：")
        for col, vals in medium:
            lines.append(f"║     {col} → {vals}")
    if low:
        lines.append(f"║   LOW（{len(low)} 个）—— 其他字段（仅列名，不展开）：")
        for col, vals in low[:10]:
            lines.append(f"║     {col}")
        if len(low) > 10:
            lines.append(f"║     ... 还有 {len(low)-10} 个")
    lines.append("║")

    numeric_var: list[tuple[str, float, float, float]] = []
    for col in raw.columns:
        if _should_skip(col):
            continue
        series = pd.to_numeric(rows[col], errors="coerce").astype(float).dropna()
        if len(series) >= 2:
            rng = float(series.max() - series.min())
            mn  = float(series.mean())
            pct = rng / abs(mn) * 100 if mn != 0 else float("inf")
            if rng > 0:
                numeric_var.append((col, round(mn, 3), round(rng, 3), round(pct, 1)))

    if numeric_var:
        lines.append(f"║ 【数值字段跨记录波动】共 {len(numeric_var)} 个有变化的数值字段：")
        for col, mn, rng, pct in sorted(numeric_var, key=lambda x: -x[3])[:8]:
            lines.append(f"║   {_trunc(col, 55)} 均值={mn}  波动范围={rng}  波动率={pct}%")
    else:
        lines.append("║ 【数值字段】所有数值字段跨记录一致或仅有单条记录")
    lines.append("║")

    data_cols = [c for c in raw.columns
                 if c not in ("patient_id", "filename", "file_path", "folder_category")
                 and not _should_skip(c)]
    filled = sum(1 for c in data_cols if rows[c].notna().any())
    completeness = filled / len(data_cols) * 100 if data_cols else 0

    lines.append(f"║ 【字段完整性】{filled}/{len(data_cols)} 个字段有值（{completeness:.1f}%）")
    lines.append("║")
    lines.append("║ 【一致性摘要】")
    lines.append(f"║   HIGH 级别不一致: {len(high)} 处")
    lines.append(f"║   MEDIUM 级别不一致: {len(medium)} 处")
    lines.append(f"║   LOW 级别不一致: {len(low)} 处")
    lines.append(f"║   数值波动字段: {len(numeric_var)} 个")
    lines.append(f"║   完整性: {completeness:.1f}%")
    lines.append(f"╚══ 请根据以上信息为患者 {patient_id} 打置信度分数并决定是否通过验证 ══")

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step6_toolkit)
def save_step6_report(report_content: str) -> ToolResponse:
    """
    保存 Step 6 一致性验证报告（文本版 + JSON 版）。

    Args:
        report_content (str): 完整的报告文本。
    """
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path  = os.path.join(OUTPUT_STEP6, f"consistency_report_{ts}.txt")
    json_path = os.path.join(OUTPUT_STEP6, f"consistency_report_{ts}.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    raw = _read_data(RAW_CSV)
    struct = {
        "timestamp": datetime.now().isoformat(),
        "step":      "STEP6_ConsistencyValidation",
        "data_file": RAW_CSV,
        "summary": {
            "total_patients":       int(raw["patient_id"].nunique()),
            "total_records":        int(len(raw)),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        "report_text": report_content,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(struct, f, ensure_ascii=False, indent=2)

    return ToolResponse(content=[TextBlock(type="text",
        text=f"Step 6 报告已保存:\n  文本: {txt_path}\n  JSON: {json_path}")])


# ══════════════════════════════════════════════════════════════════
# 3. Step 7 工具集
# ══════════════════════════════════════════════════════════════════

step7_toolkit = Toolkit()


def _get_auto_feature_cols() -> list[str]:
    """从 final_columns 中取出所有可用的非长文本特征列。"""
    try:
        with open(SELECTION_REPORT, encoding="utf-8") as f:
            report = json.load(f)
        final_cols = _report_columns(report)
    except Exception:
        filt = _read_data(FILTERED_CSV)
        final_cols = filt.columns.tolist()
    return [c for c in final_cols
            if c not in ("patient_id", PATIENT_ID_COL) and not _should_skip(c)]


@register_tool(step7_toolkit)
def load_task_context() -> ToolResponse:
    """
    读取列筛选报告，返回：
      1. 原始任务描述（task_text）
      2. 所有已筛选特征列及其入选原因
      3. 可能用于标签的候选列（诊断相关列）及每位患者的取值分布
      4. 关键数值指标的每位患者聚合均值
    """
    with open(SELECTION_REPORT, encoding="utf-8") as f:
        report = json.load(f)

    task_text  = _report_task_text(report)
    final_cols = _report_columns(report)
    reasons    = report.get("agent_decisions", {}).get("reason_by_column", {})

    filt      = _read_data(FILTERED_CSV)
    feat_cols = _get_auto_feature_cols()

    lines = [
        "=" * 65,
        "  Step 7 任务上下文",
        "=" * 65,
        f"  任务描述: {task_text}",
        f"  筛选后数据: {filt.shape[0]} 行 × {filt.shape[1]} 列 / {filt['patient_id'].nunique()} 位患者",
        f"  自动特征列: {len(feat_cols)} 列",
        "",
        "── 自动特征列清单 ──────────────────────────────────────",
    ]
    for col in feat_cols:
        miss_pct = filt[col].isnull().mean() * 100
        reason   = reasons.get(col, "")[:80]
        lines.append(f"  {col:<60}  缺失={miss_pct:.0f}%  原因={reason}")

    diag_cols = [c for c in final_cols if _is_diagnosis_like_column(c) and not _should_skip(c)]
    lines += [
        "",
        "── 标签候选列（含[诊断]字样，展示每位患者的取值）───────",
    ]
    for col in diag_cols:
        if col not in filt.columns:
            continue
        per_pt = filt.groupby("patient_id")[col].apply(
            lambda x: list(x.dropna().unique())
        ).to_dict()
        lines.append(f"\n  {col}:")
        for pid, vals in per_pt.items():
            lines.append(f"    患者{pid}: {vals}")

    numeric_cols = [c for c in feat_cols
                    if filt[c].apply(pd.to_numeric, errors="coerce").notna().sum() > 0
                    and ("偏差" in c or "标准差" in c or "翻滚角" in c)
                    and c.endswith("_value")]
    if numeric_cols:
        lines += ["", "── 关键数值指标（每位患者聚合均值）──────────────────"]
        for col in numeric_cols:
            series = filt[["patient_id", col]].copy()
            series[col] = pd.to_numeric(series[col], errors="coerce")
            per_pt = series.groupby("patient_id")[col].agg(
                ["mean", "max", "min", "count"]
            ).round(3)
            lines.append(f"\n  {col}:")
            lines.append(f"  {'患者ID':<10} {'均值':>8} {'最大值':>8} {'最小值':>8} {'记录数':>6}")
            for pid, row in per_pt.iterrows():
                lines.append(f"  {str(pid):<10} {row['mean']:>8.3f} {row['max']:>8.3f} "
                              f"{row['min']:>8.3f} {int(row['count']):>6}")

    lines += [
        "",
        "=" * 65,
        "  建议下一步：先调用 discover_diagnosis_fields() 识别所有诊断字段，",
        "  再调用 extract_and_map_all_diagnosis_fields() 完成 ICD 标准化，",
        "  然后调用 propose_label_mapping() 自动生成 label_mapping（禁止向用户追问），",
        "  再 build_label_series → format_and_save_ml_dataset 保存数据集。",
        "=" * 65,
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def get_column_distribution(column_name: str) -> ToolResponse:
    """
    获取筛选后数据集中指定列的完整分布信息。

    Args:
        column_name (str): 列名（必须精确匹配）。
    """
    filt = _read_data(FILTERED_CSV)
    if column_name not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"列 '{column_name}' 不在筛选数据中。请参考 load_task_context 中的列清单。")])

    col   = filt[column_name]
    lines = [
        f"=== 列分布: {column_name} ===",
        f"  缺失: {col.isnull().sum()}/{len(col)} ({col.isnull().mean()*100:.1f}%)",
        f"  唯一值: {col.nunique()}",
        "  值分布（行级别）:",
    ]
    for val, cnt in col.value_counts(dropna=False).head(15).items():
        lines.append(f"    {str(val):<55} → {cnt} 行")

    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().sum() >= 3:
        lines.append(f"  数值统计: min={numeric.min():.3f}  "
                     f"max={numeric.max():.3f}  mean={numeric.mean():.3f}  "
                     f"std={numeric.std():.3f}")

    filt2 = filt[["patient_id", column_name]].copy()
    filt2[column_name] = pd.to_numeric(filt2[column_name], errors="coerce")
    per_patient_num = filt2.groupby("patient_id")[column_name].mean().dropna()
    if len(per_patient_num) > 0:
        lines.append(f"\n  按患者聚合均值（数值）:")
        for pid, val in per_patient_num.items():
            lines.append(f"    患者 {pid}: {val:.3f}")
    else:
        per_patient_cat = filt.groupby("patient_id")[column_name].apply(
            lambda x: list(x.dropna().unique())
        ).to_dict()
        lines.append(f"\n  按患者取值（分类）:")
        for pid, vals in per_patient_cat.items():
            lines.append(f"    患者 {pid}: {vals}")

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# Tool①：LLM 自动识别所有诊断相关字段
# ══════════════════════════════════════════════════════════════════

@register_tool(step7_toolkit)
def discover_diagnosis_fields() -> ToolResponse:
    """
    使用 LLM 自动扫描原始数据集与筛选数据集的全部字段，
    识别哪些列可能包含疾病/诊断信息。

    识别范围包括但不限于：
      - 主诊断、初步诊断、出院诊断
      - 可能诊断1/2/3（如 Table_初步诊断-初步诊断-可能诊断2）
      - 综合征类型（如 Table_初步诊断-初步诊断-综合征类型）
      - 病因诊断、定位诊断

    识别结果会展示每个字段的所有唯一值，供后续
    extract_and_map_all_diagnosis_fields 使用。

    返回内容：
      - 识别出的字段名列表
      - 每个字段的唯一值（原始值，可能含非疾病内容）
      - LLM 将在 extract_and_map_all_diagnosis_fields 中进一步提取纯疾病内容
    """
    global _LLM_DIAG_FIELDS_CACHE

    raw  = _read_data(RAW_CSV)
    filt = _read_data(FILTERED_CSV)

    # 合并两个数据集的列名，去重保序
    all_cols = list(dict.fromkeys(raw.columns.tolist() + filt.columns.tolist()))

    # 构建 {列名: 样本值} 字典，跳过黑名单列
    col_samples: dict[str, list] = {}
    for col in all_cols:
        if _should_skip(col) or col in ("patient_id", "filename", "file_path",
                                         "folder_category"):
            continue
        src     = raw if col in raw.columns else filt
        samples = [v for v in src[col].dropna().unique()[:5].tolist()
                   if str(v).strip() not in ("", "nan")]
        if samples:
            col_samples[col] = samples

    heuristic_diag_fields = [
        col for col in col_samples
        if _is_diagnosis_like_column(col)
    ]

    # 调用 LLM 识别；再合并列名启发式，避免英文 MIMIC 列名被漏掉。
    llm_diag_fields = _llm_identify_diagnosis_fields(col_samples)
    diag_fields = list(dict.fromkeys(llm_diag_fields + heuristic_diag_fields))
    _LLM_DIAG_FIELDS_CACHE = diag_fields

    lines = [
        "=" * 65,
        "  LLM 自动识别诊断相关字段",
        "=" * 65,
        f"  扫描字段总数:        {len(col_samples)}",
        f"  识别出的诊断字段数:  {len(diag_fields)}",
        f"  LLM 识别字段数:       {len(llm_diag_fields)}",
        f"  列名启发式字段数:     {len(heuristic_diag_fields)}",
        "",
        "── 识别结果（字段名 + 所有唯一值）────────────────────────",
    ]

    for field in diag_fields:
        src = raw if field in raw.columns else filt if field in filt.columns else None
        if src is None:
            lines.append(f"\n  [{field}]  ← 字段不存在，跳过")
            continue
        unique_vals = [str(v) for v in src[field].dropna().unique().tolist()
                       if str(v).strip() not in ("", "nan")]
        lines.append(f"\n  [{field}]  ({len(unique_vals)} 个唯一值)")
        for v in unique_vals:
            lines.append(f"    • {v[:80]}")

    lines += [
        "",
        "  ⬇ 下一步：调用 extract_and_map_all_diagnosis_fields(",
        f"      diagnosis_fields={diag_fields}",
        "  ) 完成疾病内容提取与 ICD-10 标准化。",
        "=" * 65,
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# Tool②：多字段诊断提取 + ICD-10 标准化
# ══════════════════════════════════════════════════════════════════

@register_tool(step7_toolkit)
def extract_and_map_all_diagnosis_fields(
    diagnosis_fields: list[str],
) -> ToolResponse:
    """
    对指定的所有诊断相关字段进行两阶段处理：

    阶段 1 — LLM 疾病内容过滤
      对每个字段的每个唯一值，调用 LLM 判断是否为疾病名称，
      并提取其中的疾病内容（过滤状态描述、数值、非疾病词等）。

    阶段 2 — ICD-10 本地库标准化（无需翻译，无需外部 API）
      对过滤后的纯疾病名称，查询本地 ICD-10.xlsx 库：
        精确匹配 → 规范化精确匹配 → 模糊匹配（rapidfuzz）
                → LLM 从候选中选择

    输出：
      - 每个字段的逐值处理结果（原始值 → 提取疾病 → ICD-10 编码）
      - 汇总 icd10_mapping（诊断名称 → ICD-10 编码），
        可直接传入 format_and_save_ml_dataset 的 icd10_mapping 参数

    Args:
        diagnosis_fields (list[str]): 含诊断信息的字段名列表，
            通常由 discover_diagnosis_fields 的返回结果提供。
    """
    raw  = _read_data(RAW_CSV)
    filt = _read_data(FILTERED_CSV)

    # 汇总 icd10_mapping：{疾病名称 → ICD-10 编码}
    icd10_mapping: dict[str, str] = {}
    # 字段级详细结果：{字段名 → [{original, disease, source, icd10, umls_name}]}
    field_details: dict[str, list[dict]] = {}

    lines = [
        "=" * 65,
        "  多字段诊断提取 + ICD-10 标准化",
        "=" * 65,
        f"  处理字段数: {len(diagnosis_fields)}",
        "",
    ]

    for field in diagnosis_fields:
        src = raw if field in raw.columns else filt if field in filt.columns else None
        if src is None:
            lines.append(f"[跳过] 字段 '{field}' 在两个数据集中均不存在")
            continue

        unique_vals = [str(v) for v in src[field].dropna().unique().tolist()
                       if str(v).strip() not in ("", "nan")]
        lines.append(f"\n── 字段: {field}  ({len(unique_vals)} 个唯一值) ──")

        field_records: list[dict] = []

        for val in unique_vals:
            # ── 阶段 1：LLM 提取疾病内容 ─────────────────────────
            disease = _llm_extract_disease_content(field, val)

            if disease is None:
                lines.append(
                    f"  [非疾病] {val[:55]:<57} → 跳过 ICD 查询"
                )
                field_records.append({
                    "original_value":    val,
                    "extracted_disease": None,
                    "source":            None,
                    "icd10":             None,
                    "umls_preferred":    None,
                })
                continue

            # ── 阶段 2：ICD-10 标准化（NLM 主路 + UMLS 降级）────
            source, icd10_code, pref_name = _umls_lookup_icd10(disease)
            icd10_mapping[disease] = icd10_code

            lines.append(
                f"  {val[:40]:<42}  →疾病: {disease:<28}  "
                f"ICD-10: {icd10_code:<12}  来源: {source:<12}  "
                f"名称: {pref_name[:25]}"
            )
            field_records.append({
                "original_value":    val,
                "extracted_disease": disease,
                "source":            source,
                "icd10":             icd10_code,
                "local_preferred":   pref_name,
            })

        field_details[field] = field_records

    # ── 汇总输出 ──────────────────────────────────────────────────
    lines += [
        "",
        "=" * 65,
        f"  汇总：共提取 {len(icd10_mapping)} 个疾病名称并完成 ICD-10 标准化",
        "=" * 65,
        "",
        "  icd10_mapping（可直接传入 format_and_save_ml_dataset）:",
        json.dumps(icd10_mapping, ensure_ascii=False, indent=2),
        "",
        "  完整字段级结果（field_details）:",
        json.dumps(field_details, ensure_ascii=False, indent=2),
    ]

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# Tool③：单列诊断名称 → ICD-10（已知是疾病名称时使用）
# ══════════════════════════════════════════════════════════════════

@register_tool(step7_toolkit)
def map_diagnoses_to_icd10(diagnosis_names: list[str]) -> ToolResponse:
    """
    将诊断名称列表映射为 ICD-10 编码（本地库查询，无需翻译，无需外部 API）。

    与 extract_and_map_all_diagnosis_fields 的区别：
      - 本工具适用于"已知是疾病名称"的情况（如标签列的唯一值）。
      - 会先调用 LLM 过滤每个名称，确认是否为疾病内容，
        非疾病内容（如"无"、"待查"）自动跳过，不发起 ICD 查询。
      - ICD-10 查询使用本地库（精确→模糊→LLM候选选择）。

    结果中的 icd10_mapping 可直接传入 format_and_save_ml_dataset。

    Args:
        diagnosis_names (list[str]): 需要标准化的诊断名称列表。
    """
    lines = [
        "=== ICD-10 标准化映射（本地库：精确→模糊→LLM候选选择）===",
        f"查询 {len(diagnosis_names)} 个诊断名称...\n",
        f"{'诊断名称':<30} {'来源':<12} {'ICD-10 编码':<15} {'本地库首选名称'}",
        "-" * 90,
    ]

    mapping: dict[str, str] = {}

    for name in diagnosis_names:
        # ── LLM 疾病内容提取（过滤非疾病条目）─────────────────
        disease = _llm_extract_disease_content("诊断字段", name)
        if disease is None:
            lines.append(
                f"{name:<30} {'—':<8} {'[非疾病跳过]':<15} —"
            )
            continue

        # 若 LLM 提取后名称有变化，以提取后名称查询
        query_name = disease if disease != name else name
        source, code, preferred = _umls_lookup_icd10(query_name)
        mapping[name] = code  # 键保留原始输入名称，便于与标签值对应

        lines.append(
            f"{name:<30} {source:<8} {code:<15} {preferred}"
        )

    lines += [
        "-" * 90,
        "",
        "icd10_mapping（可直接传入 format_and_save_ml_dataset）:",
        json.dumps(mapping, ensure_ascii=False, indent=2),
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# Tool：自动生成 label_mapping（避免 Agent 向用户追问）
# ══════════════════════════════════════════════════════════════════

@register_tool(step7_toolkit)
def propose_label_mapping(
    label_column: str,
    mode: str = "multiclass_enum",
    positive_values: list[str] | None = None,
) -> ToolResponse:
    """
    根据筛选后数据集中标签列的**实际非空取值**，自动生成 label_mapping，
    可直接传入 build_label_series 与 format_and_save_ml_dataset。

    无需人工提供映射；Pipeline 中 Agent 应优先调用本工具，**禁止**以「请用户提供
    label_mapping」结束任务。

    Args:
        label_column (str): 作为 ML 标签的列名（如「可能诊断2」对应的长列名）。
        mode (str):
          - "multiclass_enum"（默认）：每个不同取值按字典序映射为 0,1,2,…
          - "binary_positive_vs_rest"：positive_values 中出现的取值 → 1，其余非空 → 0
        positive_values (list[str] | None): 仅 mode=binary_positive_vs_rest 时必填，
            列出应标为 1 的原始标签字符串（须与单元格内文字完全一致）。

    Returns:
        文本中包含 JSON 格式的 label_mapping 及类别说明。
    """
    filt = _read_data(FILTERED_CSV)
    pids_to_use = _passed_patient_ids if _passed_patient_ids else \
                  [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids_to_use)]

    if label_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"❌ 列 '{label_column}' 不在筛选数据中。")])

    def _norm(v: Any) -> str | None:
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return None
        return s

    raw_vals = [_norm(v) for v in filt[label_column].tolist()]
    uniques = sorted({v for v in raw_vals if v is not None})

    if len(uniques) < 2:
        return ToolResponse(content=[TextBlock(type="text",
            text=(
                f"❌ 列 '{label_column}' 在通过 Step6 的子集中有效类别数 < 2。\n"
                f"  非空唯一值: {uniques}\n"
                f"  请换用其他标签列或检查筛选数据。"
            ))])

    mapping: dict[str, int] = {}
    mode_l = (mode or "multiclass_enum").strip().lower()

    if mode_l == "binary_positive_vs_rest":
        pos = set(positive_values or [])
        if not pos:
            return ToolResponse(content=[TextBlock(type="text",
                text=(
                    "❌ mode=binary_positive_vs_rest 时必须提供 positive_values（非空列表），"
                    f"当前数据中的可取值为: {uniques}"
                ))])
        for u in uniques:
            mapping[u] = 1 if u in pos else 0
        if len(set(mapping.values())) < 2:
            return ToolResponse(content=[TextBlock(type="text",
                text=(
                    "❌ 按 positive_values 划分后只有 1 类，无法二分类。"
                    f" positive_values={sorted(pos)} 与数据交集请检查。"
                ))])
    else:
        mapping = {u: i for i, u in enumerate(uniques)}

    lines = [
        "=== propose_label_mapping 结果 ===",
        f"  标签列: {label_column}",
        f"  模式:   {mode_l}",
        f"  非空类别数: {len(uniques)}",
        "  各类别 → 整数:",
    ]
    for k, v in sorted(mapping.items(), key=lambda x: (x[1], x[0])):
        lines.append(f"    {repr(k)} → {v}")
    lines += [
        "",
        "  请将下方 JSON 原样用于 build_label_series 与 format_and_save_ml_dataset：",
        "",
        "label_mapping =",
        json.dumps(mapping, ensure_ascii=False, indent=2),
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# Tool④：标签验证
# ══════════════════════════════════════════════════════════════════

@register_tool(step7_toolkit)
def build_label_series(label_column: str, label_mapping: dict[str, int]) -> ToolResponse:
    """
    验证标签生成效果：按**每条筛选记录**（每行）用标签列与映射生成 y，不保存。
    与 format_and_save_ml_dataset 一致：同一患者多条记录会对应多行验证结果。

    Args:
        label_column (str): 标签列名。
        label_mapping (dict[str, int]): 类别值 → 整数标签。
    """
    filt = _read_data(FILTERED_CSV)
    pids_to_use = _passed_patient_ids if _passed_patient_ids else \
                  [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids_to_use)].reset_index(drop=True)

    if label_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"列 '{label_column}' 不在筛选数据中。")])

    rows_out = []
    for idx, row in filt.iterrows():
        lv = row[label_column]
        if pd.isna(lv):
            label_raw, y = None, None
        else:
            label_raw = str(lv).strip()
            y         = label_mapping.get(label_raw)
        rows_out.append({
            "sample_row_id": int(idx),
            "patient_id":    str(row["patient_id"]),
            "label_raw":     label_raw,
            "y":             y,
        })

    n_valid   = sum(1 for r in rows_out if r["y"] is not None)
    n_invalid = len(rows_out) - n_valid
    dist: dict[str, int] = {}
    for r in rows_out:
        k = str(r["y"])
        dist[k] = dist.get(k, 0) + 1

    lines = [
        f"=== 标签验证（列: {label_column}，按记录逐行）===\n",
        f"{'行号':<8} {'患者ID':<12} {'原始标签':<35} {'y':>4}",
        "-" * 65,
    ]
    for r in rows_out:
        y_str = str(r["y"]) if r["y"] is not None else "—(无法映射)"
        lines.append(
            f"{r['sample_row_id']:<8} {r['patient_id']:<12} {str(r['label_raw']):<35} {y_str:>4}"
        )
    lines += [
        "-" * 65,
        f"总记录数: {len(rows_out)}  有效标签: {n_valid}  无法映射: {n_invalid}",
        f"标签分布: {dist}",
    ]

    unique_y = {r["y"] for r in rows_out if r["y"] is not None}
    if len(unique_y) < 2:
        lines.append(
            "\n⚠ [质量警告] 有效标签中只有 1 种类别值，该列不适合作为分类标签！"
            "\n  请重新思考任务定义，选择能区分不同患者状态的列或重新设计映射。"
        )
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# Tool⑤：生成并保存完整 ML 数据集
# ══════════════════════════════════════════════════════════════════

@register_tool(step7_toolkit)
def format_and_save_ml_dataset(
    label_column: str,
    label_mapping: dict[str, int],
    icd10_mapping: dict[str, str] | None = None,
) -> ToolResponse:
    """
    使用全量特征 + Agent 确定的标签生成完整 ML 数据集并保存。

    **粒度：筛选后 CSV 的每一行 = 一条 ML 样本**（同一患者多条检查/多条记录
    会保留为多行，不再按患者做均值/众数聚合）。

    特征列自动来源于筛选报告的 final_columns（去掉长文本列）。
    icd10_mapping 建议由 extract_and_map_all_diagnosis_fields 或
    map_diagnoses_to_icd10 的结果提供，将写入 JSON 输出文件。

    每行附带 ``sample_row_id``（0..N-1，对应当前筛选子集内的行序），便于追溯。

    Args:
        label_column (str): 标签列名。
        label_mapping (dict[str, int]): 类别值 → 整数标签映射。
        icd10_mapping (dict[str, str] | None): 诊断名称 → ICD-10 编码映射。
    """
    filt = _read_data(FILTERED_CSV)
    pids_to_use = _passed_patient_ids if _passed_patient_ids else \
                  [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids_to_use)].reset_index(drop=True)

    feat_cols = _get_auto_feature_cols()
    feat_cols = [c for c in feat_cols if c in filt.columns and c != label_column]

    if label_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"❌ 列 '{label_column}' 不在筛选数据中，请重新确认列名。")])

    def _cell_value(v: Any) -> Any:
        """单元格转 Python 原生类型：数值尽量为 float，否则为字符串或 None。"""
        if pd.isna(v):
            return None
        num = pd.to_numeric(v, errors="coerce")
        if pd.notna(num):
            return round(float(num), 6)
        s = str(v).strip()
        return s if s and s.lower() != "nan" else None

    ml_rows: list[dict[str, Any]] = []
    for idx, row in filt.iterrows():
        rec: dict[str, Any] = {
            "sample_row_id": int(idx),
            "patient_id":    str(row["patient_id"]),
        }
        for col in feat_cols:
            rec[col] = _cell_value(row[col])
        lv = row[label_column]
        if pd.isna(lv):
            rec["label_name"], rec["y"] = None, None
        else:
            rec["label_name"] = str(lv).strip()
            rec["y"] = label_mapping.get(rec["label_name"])
        ml_rows.append(rec)

    df_ml = pd.DataFrame(ml_rows)
    df_train = df_ml.dropna(subset=["y"]).copy()

    unique_y = df_train["y"].nunique() if not df_train.empty else 0
    if unique_y < 2:
        preview = ""
        if not df_ml.empty and label_column in filt.columns:
            preview = df_ml[["sample_row_id", "patient_id", "label_name"]].head(15).to_string(index=False)
        err_lines = [
            "❌ [错误] 数据集质量不合格，已中止保存。",
            f"  label_column='{label_column}' 有效样本中只有 {unique_y} 种类别值。",
            f"  当前 label_mapping={label_mapping}",
            f"  筛选后总记录数: {len(df_ml)}，有标签记录数: {len(df_train)}",
            "",
            "  前若干条记录的 label_name 预览:",
            preview or "（无）",
            "",
            "  请重新思考标签策略后再调用此工具。",
        ]
        return ToolResponse(content=[TextBlock(type="text", text="\n".join(err_lines))])

    df_train["y"] = df_train["y"].astype(int)
    for col in feat_cols:
        if pd.api.types.is_numeric_dtype(df_train[col]):
            col_mean = df_train[col].mean()
            df_train[col] = df_train[col].fillna(col_mean)

    # ── 诊断字段增强：提取疾病名称 + ICD 编码 ──────────────────────
    # 识别诊断相关列：从 LLM 识别缓存 或 列名启发式（含"诊断"/"综合征"）
    _DIAG_KEYWORDS = ("诊断", "综合征", "疾病类型", "病因")
    diag_feat_cols = [
        c for c in feat_cols
        if (c in (_LLM_DIAG_FIELDS_CACHE or []))
        or _is_diagnosis_like_column(c)
    ]

    # 合并 icd10_mapping + 实时查询 兜底
    _effective_icd_map: dict[str, str] = dict(icd10_mapping or {})

    def _get_disease_and_icd(col: str, raw_val: Any) -> tuple[str | None, str | None]:
        """对单个字段值：LLM 提取疾病名称 + ICD 编码查询"""
        if pd.isna(raw_val) or str(raw_val).strip() in ("", "nan", "None"):
            return None, None
        val_str = str(raw_val).strip()
        # 优先用缓存（extract_and_map_all_diagnosis_fields 已填充）
        disease = _LLM_DISEASE_EXTRACT_CACHE.get((col, val_str))
        if disease is None:
            disease = _llm_extract_disease_content(col, val_str)
        if not disease:
            return None, None
        # ICD 编码：先查合并后的映射，再实时查询本地库
        code = _effective_icd_map.get(disease)
        if code is None:
            _, code, pref = _local_icd10_lookup(disease)
            if code and code != "NOT_FOUND":
                _effective_icd_map[disease] = code
        return disease, (code if code and code != "NOT_FOUND" else None)

    icd_summary_lines: list[str] = []
    for col in diag_feat_cols:
        disease_vals: list[str | None] = []
        icd_vals: list[str | None] = []
        for _, row in df_train.iterrows():
            d, c = _get_disease_and_icd(col, row.get(col))
            disease_vals.append(d)
            icd_vals.append(c)
        col_short = col.split("-")[-1] if "-" in col else col
        df_train[f"{col_short}_疾病名称"] = disease_vals
        df_train[f"{col_short}_ICD10"]   = icd_vals
        icd_summary_lines.append(f"  {col}")
        for d, c in zip(disease_vals, icd_vals):
            if d:
                icd_summary_lines.append(f"    → {d} | {c or 'NOT_FOUND'}")

    # 标签列的 ICD 编码
    def _label_to_icd(label_name: Any) -> str | None:
        if pd.isna(label_name) or str(label_name).strip() in ("", "nan"):
            return None
        lbl = str(label_name).strip()
        code = _effective_icd_map.get(lbl)
        if code is None:
            _, code, _ = _local_icd10_lookup(lbl)
        return code if code and code != "NOT_FOUND" else None

    df_train["label_ICD10"] = df_train["label_name"].apply(_label_to_icd)
    # ── 诊断字段增强 结束 ──────────────────────────────────────────

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(OUTPUT_STEP7, f"ml_dataset_{ts}.csv")
    df_train.to_csv(csv_path, index=False, encoding="utf-8-sig")

    jsonl_path = os.path.join(OUTPUT_STEP7, f"ml_dataset_{ts}.jsonl")
    def _json_safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, (np.integer, np.floating)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df_train.iterrows():
            sample = {
                "sample_row_id": int(row["sample_row_id"]),
                "patient_id":    str(row["patient_id"]),
                "features":      {c: _json_safe(row[c]) for c in feat_cols},
                "label":         int(row["y"]),
                "label_name":    row.get("label_name", ""),
                "label_ICD10":   _json_safe(row.get("label_ICD10")),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    json_path = os.path.join(OUTPUT_STEP7, f"ml_dataset_{ts}.json")
    with open(SELECTION_REPORT, encoding="utf-8") as f:
        sel_report = json.load(f)
    pytorch = {
        "timestamp":      datetime.now().isoformat(),
        "step":           "STEP7_ML_DataPrep",
        "granularity":    "record",  # 每行一条原始筛选记录，非按患者聚合
        "task_text":      _report_task_text(sel_report),
        "feature_names":  feat_cols,
        "n_samples":      int(len(df_train)),
        "n_features":     int(len(feat_cols)),
        "label_column":   label_column,
        "label_map":      label_mapping,
        "icd10_mapping":  _effective_icd_map,
        "X": [[_json_safe(row[c]) for c in feat_cols] for _, row in df_train.iterrows()],
        "y": df_train["y"].tolist(),
        "sample_row_ids": df_train["sample_row_id"].tolist(),
        "patient_ids":    df_train["patient_id"].tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pytorch, f, ensure_ascii=False, indent=2)

    label_dist = df_train["y"].value_counts().to_dict()
    inv_map    = {v: k for k, v in label_mapping.items()}
    dist_str   = "  ".join(
        f"y={k}({inv_map.get(k,'?')}): {v}例"
        for k, v in sorted(label_dist.items())
    )
    num_feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_train[c])]

    # 新增的诊断增强列
    added_cols = [c for c in df_train.columns
                  if c.endswith("_疾病名称") or c.endswith("_ICD10")]

    lines = [
        "=" * 65,
        "  ML 训练数据集（按记录逐行 + ICD 编码增强）",
        "=" * 65,
        f"  任务: {pytorch['task_text']}",
        f"  训练样本: {len(df_train)} 条（筛选后共 {len(df_ml)} 行记录，"
        f"{len(df_ml) - len(df_train)} 行因标签无法映射已排除）",
        f"  涉及患者数: {df_train['patient_id'].nunique()}",
        f"  特征维数: {len(feat_cols)}（自动提取）",
        f"  标签列:  {label_column}",
        f"  标签分布: {dist_str}",
        f"  新增ICD增强列: {added_cols}",
    ]
    if icd_summary_lines:
        lines.append("\n── 诊断字段 ICD 标准化结果 ────────────────────────────")
        lines.extend(icd_summary_lines)
    lines += [
        "",
        "── 诊断标签 ICD 编码 ─────────────────────────────────────",
        df_train[["sample_row_id", "patient_id", "label_name", "label_ICD10", "y"]].to_string(index=False),
        "",
        "── 完整数据集（记录 × 特征+标签，前10列）──────────────────",
        df_train[["sample_row_id", "patient_id"] + feat_cols[:10] + ["y", "label_name", "label_ICD10"]].to_string(index=False),
        "",
        "── 文件保存路径 ───────────────────────────────────────────",
        f"  CSV:   {csv_path}",
        f"  JSONL: {jsonl_path}",
        f"  JSON:  {json_path}",
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# 4. 模型与 Agent
# ══════════════════════════════════════════════════════════════════

class ThinkingSafeOpenAIChatFormatter(OpenAIChatFormatter):
    """在送入 OpenAI formatter 前过滤 thinking block，避免告警日志。"""

    async def _format(self, msgs: list[Msg]) -> list[dict]:
        sanitized_msgs = [_strip_thinking_from_msg(msg) for msg in msgs]
        formatted = await super()._format(sanitized_msgs)
        for item in formatted:
            if item.get("role") == "user":
                item.pop("name", None)
        return formatted


def _strip_thinking_from_msg(msg: Msg) -> Msg:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return msg
    content_blocks = [
        block
        for block in msg.get_content_blocks()
        if str(block.get("type") or "") not in ("thinking")
    ]
    sanitized_msg = Msg(
        name=msg.name,
        content=content_blocks,
        role=msg.role,
        metadata=msg.metadata,
        timestamp=msg.timestamp,
        invocation_id=msg.invocation_id,
    )
    sanitized_msg.id = msg.id
    return sanitized_msg


def _register_no_thinking_print_hook(agent: ReActAgent) -> ReActAgent:
    def _hook(_agent: ReActAgent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        msg = kwargs.get("msg")
        if isinstance(msg, Msg):
            kwargs["msg"] = _strip_thinking_from_msg(msg)
        return kwargs

    agent.register_instance_hook("pre_print", "strip_thinking_for_console", _hook)
    return agent


def create_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        stream=True,
        client_kwargs={"base_url": OPENAI_BASE_URL},
        generate_kwargs={"temperature": 0.2, "max_tokens": 4096},
    )


def create_consistency_agent() -> ReActAgent:
    if _IS_MIMIC:
        _dataset_desc = (
            "MIMIC 住院患者医疗数据集（肝硬化/肝衰竭队列），"
            "每条记录对应一次住院（record_index），可能包含多条临床笔记（note_index）。"
        )
        _analysis_hint = (
            "注意：MIMIC 数据中一个 record_index 可能对应多条不同类型的临床笔记记录，"
            "请检查同一 record_index 下不同笔记之间的关键临床字段是否一致。"
        )
    else:
        _dataset_desc = "前庭功能检查数据集，每位患者有多条检查记录。"
        _analysis_hint = "注意：同一患者可能来自不同检查批次，请重点关注诊断、性别等关键字段的一致性。"

    agent = ReActAgent(
        name="ConsistencyAgent",
        sys_prompt=f"""你是医疗数据质量工程师，负责对{_dataset_desc}执行 Step 6 一致性验证。
分析粒度是**患者**：每位患者有多条记录，你需要逐一检查每位患者内部记录是否一致。
{_analysis_hint}

## 关键要求
**你必须对数据集中每一位患者（1000位）调用 analyze_patient_consistency 进行分析，不得遗漏任何一位。**
不允许"根据模式推断"或"估算"，每位患者的结果都必须通过实际调用工具获得。

## 工作流程

### 第一步：了解数据
1. 调用 load_data_overview   了解数据集整体情况和字段结构
2. 调用 get_patient_list     获取所有患者 ID 列表

### 第二步：逐患者分析（核心）
3. 对每位患者**逐一**调用 analyze_patient_consistency(patient_id)
   - 该工具会返回该患者的跨记录不一致字段、数值波动、完整性等详细信息
   - 每分析完 100 位患者，立即输出当时的 PASSED_PATIENTS 列表：
     PASSED_PATIENTS: [id1, id2, id3, ...]
     这样可以避免最后一次性输出导致遗漏

### 第三步：评分与决策
4. 根据每位患者的分析结果，独立给出置信度分数（0.0–1.0）：
     HIGH 不一致  × 个 → 每个扣 0.20
     MEDIUM 不一致 × 个 → 每个扣 0.10
     LOW 不一致   × 个 → 每个扣 0.03
     完整性 < 30% → 额外扣 0.10
   是否通过验证（置信度 ≥ {CONFIDENCE_THRESHOLD}）

### 第四步：报告
5. 分析完全部 1000 位患者后，撰写完整中文报告，包含每位患者的分析结果和汇总
6. 调用 save_step6_report 保存报告

报告最后必须包含（供 Step 7 读取）：
PASSED_PATIENTS: [patient_id_1, patient_id_2, ...]
""",
        model=create_model(),
        formatter=ThinkingSafeOpenAIChatFormatter(),
        toolkit=step6_toolkit,
        memory=InMemoryMemory(),
        max_iters=400000,
    )
    return _register_no_thinking_print_hook(agent)


def create_dataprep_agent() -> ReActAgent:
    if _IS_MIMIC:
        _diag_field_hint = (
            "   - 不限于主诊断字段，会发现如 主诊断_icd_title 等含诊断信息的列\n"
            "   - MIMIC 数据中诊断字段以英文为主，ICD-10 查询会先尝试 LLM 转中文再匹配本地库，\n"
            "     部分英文术语可能 NOT_FOUND"
        )
        _label_hint = (
            "   - MIMIC 任务聚焦于死亡风险与住院时长；建议优先考虑死亡时间（是否为空 → 是否死亡）\n"
            "     或住院天数分桶作为标签，而非仅依赖 ICD 诊断列"
        )
    else:
        _diag_field_hint = (
            "   - 不限于主诊断字段，会发现综合征类型、可能诊断2等隐含诊断字段"
        )
        _label_hint = (
            "   - 根据 task_text 与诊断列分布选定最合适的标签列"
        )

    agent = ReActAgent(
        name="DataPrepAgent",
        sys_prompt=f"""你是一位 ML 数据工程师，负责将高置信度医疗数据转化为机器学习训练集。
特征列已由筛选报告自动确定，你只需聚焦于：
  ① 理解任务目标
  ② 识别所有含诊断信息的字段并完成 ICD 标准化
  ③ 确定最合适的标签策略

ICD-10 标准化说明（本地库 + BERT 向量）：
  - 使用本地 ICD-10.xlsx；语义匹配优先用预建 BERT 向量库 data/icd10_bert_index/
  - 查询链路：精确/规范化 → LLM 英文缩写转中文 → BERT 最近邻 → 模糊字符串 → LLM 在合并候选中择优
  - 若未运行 build_icd10_vector_store.py 建库，则自动跳过 BERT，仅用模糊+LLM
  - 部分新兴术语在标准库中无条目时可能为 NOT_FOUND

## 工作流程（严格按顺序执行）

### 第一阶段：加载任务上下文
1. 调用 load_task_context
   - 获取任务描述（task_text）
   - 查看已自动选定的特征列清单
   - 查看标签候选列的每位患者取值

### 第二阶段：识别诊断字段并完成 ICD 标准化（必须执行）
2. 调用 discover_diagnosis_fields()
   - LLM 自动扫描所有字段，识别含诊断信息的列
{_diag_field_hint}
3. 调用 extract_and_map_all_diagnosis_fields(diagnosis_fields=[...])
   - 传入上一步识别出的全部诊断字段名
   - LLM 逐值过滤非疾病内容（如"无"、数值、状态描述）
   - 对提取出的纯疾病名称查询本地 ICD-10 库完成编码
   - 保存返回的 icd10_mapping 供后续步骤使用

### 第三阶段：确定标签策略（禁止向用户追问）
4. 根据 task_text 与 load_task_context 中的诊断列分布，选定 **label_column**
   - 可调用 get_column_distribution 查看候选列分布（可选）
{_label_hint}
5. **必须**调用 propose_label_mapping(label_column=..., mode=...)
   - 默认使用 mode="multiclass_enum"：按字典序把各取值映射为 0,1,2,…（最简单、总能得到合法多分类）
   - 若 task_text 明确要求二分类且你能从数据中明确列出「阳性」原始字符串，可用
     mode="binary_positive_vs_rest" 并传入 positive_values=[...]（须与单元格文本完全一致）
   - **禁止**以「请用户提供 label_mapping」「请告知如何映射」等话术结束；映射只能由你
     通过工具生成或基于工具输出抄写
6. 调用 build_label_series(label_column, label_mapping) 验证标签质量

### 第四阶段：保存数据集
7. 调用 format_and_save_ml_dataset(
       label_column=...,
       label_mapping=<与上一步完全相同>,
       icd10_mapping=<第2步 extract 的 icd10_mapping>
   )

### 第五阶段：撰写报告
8. 简要说明：标签列、label_mapping 含义、ICD 覆盖情况、数据集局限性

⚠ 关键提醒：
  - 步骤 2-3（诊断字段识别 + ICD 标准化）为必须步骤，不可跳过
  - 字段值中可能混有非疾病内容，LLM 会自动过滤，无需人工干预
  - 标签必须有 ≥2 个不同类别；若某列所有记录值相同则不能作为标签
  - format_and_save_ml_dataset 按**每条筛选记录一行**输出（同一患者可有多行）
  - **工作流必须闭环**：最后一次工具调用应是 format_and_save_ml_dataset 成功保存，不得停在追问用户
""",
        model=create_model(),
        formatter=ThinkingSafeOpenAIChatFormatter(),
        toolkit=step7_toolkit,
        memory=InMemoryMemory(),
        max_iters=35,
    )
    return _register_no_thinking_print_hook(agent)


# ══════════════════════════════════════════════════════════════════
# 5. Pipeline
# ══════════════════════════════════════════════════════════════════

def _extract_msg_text(msg: Msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _parse_passed_ids(text: str) -> list[str] | None:
    """Parse the explicit PASSED_PATIENTS list from Step 6 output.

    Returns None when the marker is missing. An explicit empty list is returned
    as [], so Step 7 will not silently treat all patients as validated.
    """
    m = re.search(r"PASSED_PATIENTS:\s*\[([^\]]*)\]", text)
    if not m:
        return None
    return re.findall(r"[\w.-]+", m.group(1))


def _latest_output_file(directory: str, prefix: str, suffix: str) -> str:
    if not directory or not os.path.isdir(directory):
        return ""
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    if not candidates:
        return ""
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


async def run_step6_only(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    global _passed_patient_ids

    paths = configure_runtime(runtime_overrides)
    _validate_runtime_inputs(require_selection_report=False)

    _data_label = "MIMIC" if _IS_MIMIC else "Original"
    print("=" * 65)
    print("  STEP 6 一致性验证")
    print("=" * 65)
    print(f"  模式:    {_data_label}（DATA_SOURCE={DATA_SOURCE}）")
    print(f"  模型:    {MODEL_NAME}")
    print(f"  数据:    {RAW_CSV}")
    print(f"  筛选后:  {FILTERED_CSV}")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print("=" * 65)

    agent6 = create_consistency_agent()

    if _IS_MIMIC:
        _step6_data_hint = (
            "数据为 MIMIC 住院患者队列（肝硬化/肝衰竭），"
            f"患者 ID 对应原始字段 {PATIENT_ID_COL}（已自动重命名为 patient_id）。\n"
            "一条记录可能含多条临床笔记，请检查同一 patient_id 下字段的一致性。\n"
        )
    else:
        _step6_data_hint = "请先通过工具自主探索数据结构，再进行一致性分析。\n"

    msg6 = Msg(
        name="Pipeline",
        role="user",
        content=(
            "请对以下医疗数据集执行完整的 Step 6 一致性验证。\n\n"
            f"数据文件路径：{RAW_CSV}\n\n"
            + _step6_data_hint
            + f"置信度阈值：{CONFIDENCE_THRESHOLD}（低于此值的患者将被过滤）\n"
            "完成后请保存报告，并在末尾严格按以下格式输出通过验证的患者 ID 列表：\n"
            "PASSED_PATIENTS: [id1, id2, id3, ...]\n"
            "注意：必须包含所有通过验证的患者 ID，不得省略任何一项。"
        ),
    )

    print("\n[Pipeline] ▶ 启动 ConsistencyAgent (Step 6)...")
    result6 = await agent6(msg6)
    text6 = _extract_msg_text(result6)

    print("\n" + "=" * 65)
    print("  Step 6 完成")
    print("=" * 65)
    print(text6)

    parsed_passed_ids = _parse_passed_ids(text6)
    print(f"\n[Debug] parsed_passed_ids from text6: {parsed_passed_ids}")
    if parsed_passed_ids is None:
        # Try reading from the saved report file
        latest_txt = _latest_output_file(OUTPUT_STEP6, "consistency_report_", ".txt")
        if latest_txt and os.path.isfile(latest_txt):
            with open(latest_txt, "r", encoding="utf-8") as f:
                report_text = f.read()
            parsed_passed_ids = _parse_passed_ids(report_text)
            print(f"[Debug] parsed_passed_ids from report file: {parsed_passed_ids}")
    if parsed_passed_ids is None:
        err = "Step 6 output missing required PASSED_PATIENTS: [...] marker; refusing to pass all patients implicitly."
        print(f"\n[Pipeline] Step 6 解析失败: {err}")
        return {
            "success": False,
            "error": err,
            "step6_text": text6,
            "passed_patient_ids": [],
            "step6_report_txt": _latest_output_file(OUTPUT_STEP6, "consistency_report_", ".txt"),
            "step6_report_json": _latest_output_file(OUTPUT_STEP6, "consistency_report_", ".json"),
            "raw_csv": RAW_CSV,
            "filtered_csv": FILTERED_CSV,
            "selection_report": SELECTION_REPORT,
            "output_step6": OUTPUT_STEP6,
            "output_step7": OUTPUT_STEP7,
            "runtime_paths": paths,
        }

    _passed_patient_ids = parsed_passed_ids
    print(f"\n[Pipeline] Step 6 通过患者 ID: {_passed_patient_ids}")

    step6_report_txt = _latest_output_file(OUTPUT_STEP6, "consistency_report_", ".txt")
    step6_report_json = _latest_output_file(OUTPUT_STEP6, "consistency_report_", ".json")

    return {
        "success": bool(text6),
        "error": "",
        "step6_text": text6,
        "passed_patient_ids": _passed_patient_ids,
        "step6_report_txt": step6_report_txt,
        "step6_report_json": step6_report_json,
        "raw_csv": RAW_CSV,
        "filtered_csv": FILTERED_CSV,
        "selection_report": SELECTION_REPORT,
        "output_step6": OUTPUT_STEP6,
        "output_step7": OUTPUT_STEP7,
        "runtime_paths": paths,
    }


async def run_step7_only(
    passed_patient_ids: list[str] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global _passed_patient_ids

    configure_runtime(runtime_overrides)
    _validate_runtime_inputs(require_selection_report=True)

    if passed_patient_ids is not None:
        _passed_patient_ids = [str(pid) for pid in passed_patient_ids]
    elif not _passed_patient_ids:
        filt = _read_data(FILTERED_CSV)
        _passed_patient_ids = [str(p) for p in filt["patient_id"].unique()]

    agent7 = create_dataprep_agent()

    if _IS_MIMIC:
        _step7_diag_hint = (
            "诊断信息主要在 主诊断_icd_title 等字段，均为英文，"
            "ICD 查询会先 LLM 转中文再匹配本地库。\n"
            "标签建议：死亡时间是否为空（二分类，mortality）或住院天数分桶。\n"
        )
    else:
        _step7_diag_hint = (
            "诊断信息可能分布在多个字段中\n"
            "（如 Table_初步诊断-初步诊断-综合征类型、Table_初步诊断-初步诊断-可能诊断2 等）。\n"
        )

    msg7 = Msg(
        name="ConsistencyAgent",
        role="user",
        content=(
            "Step 6 一致性验证已完成。\n\n"
            f"通过验证的患者数：{len(_passed_patient_ids)}\n\n"
            "请执行 Step 7：基于任务目标生成 ML 训练数据集。\n\n"
            "工作流程：\n"
            "1. discover_diagnosis_fields() 识别诊断字段\n"
            "2. extract_and_map_all_diagnosis_fields() 完成 ICD 标准化\n"
            "3. propose_label_mapping(label_column=..., mode=\"multiclass_enum\") 生成标签映射\n"
            "4. build_label_series\n"
            "5. format_and_save_ml_dataset 保存\n\n"
            f"数据路径：{FILTERED_CSV}\n"
            f"筛选报告：{SELECTION_REPORT}\n"
            "ICD 库：本地 ICD-10.xlsx（35,725 条）"
        ),
    )

    run_started_at = datetime.now().timestamp()
    print("\n[Pipeline] ▶ 启动 DataPrepAgent (Step 7)...")
    result7 = await agent7(msg7)
    text7 = _extract_msg_text(result7)

    print("\n" + "=" * 65)
    print("  Step 7 完成")
    print("=" * 65)
    print(text7)

    step7_csv = _latest_output_file(OUTPUT_STEP7, "ml_dataset_", ".csv")
    step7_jsonl = _latest_output_file(OUTPUT_STEP7, "ml_dataset_", ".jsonl")
    step7_json = _latest_output_file(OUTPUT_STEP7, "ml_dataset_", ".json")
    dataset_saved = all(
        os.path.isfile(p) and os.path.getmtime(p) >= run_started_at
        for p in (step7_csv, step7_jsonl, step7_json)
    )
    step7_error = "" if dataset_saved else (
        "Step 7 did not save ml_dataset CSV/JSONL/JSON; treating this run as failed."
    )

    return {
        "success": bool(text7) and dataset_saved,
        "error": step7_error,
        "step7_text": text7,
        "passed_patient_ids": list(_passed_patient_ids),
        "step7_dataset_csv": step7_csv,
        "step7_dataset_jsonl": step7_jsonl,
        "step7_dataset_json": step7_json,
        "raw_csv": RAW_CSV,
        "filtered_csv": FILTERED_CSV,
        "selection_report": SELECTION_REPORT,
        "output_step6": OUTPUT_STEP6,
        "output_step7": OUTPUT_STEP7,
    }


async def run_pipeline(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    configure_runtime(runtime_overrides)

    step6_result = await run_step6_only(runtime_overrides=runtime_overrides)
    if not step6_result.get("success"):
        return {
            "success": False,
            "error": str(step6_result.get("error") or "Step6 执行失败"),
            "step6": step6_result,
            "step7": {},
        }

    step7_result = await run_step7_only(
        passed_patient_ids=step6_result.get("passed_patient_ids") or [],
        runtime_overrides=runtime_overrides,
    )
    if not step7_result.get("success"):
        return {
            "success": False,
            "error": str(step7_result.get("error") or "Step7 执行失败"),
            "step6": step6_result,
            "step7": step7_result,
        }

    print("\n" + "=" * 65)
    print("  Pipeline 全部完成")
    print(f"  Step 6 报告 → {OUTPUT_STEP6}/")
    print(f"  Step 7 数据 → {OUTPUT_STEP7}/")
    print("=" * 65)

    return {
        "success": True,
        "error": "",
        "step6": step6_result,
        "step7": step7_result,
        "output_step6": OUTPUT_STEP6,
        "output_step7": OUTPUT_STEP7,
    }


if __name__ == "__main__":
    asyncio.run(run_pipeline())
