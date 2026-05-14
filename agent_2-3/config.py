# -*- coding: utf-8 -*-
"""
统一配置文件

所有路径、API 配置、枚举类型、处理管线定义集中在此处。
"""
import os
from enum import Enum
from typing import Dict, List, Set

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from configs.loader import get_agent_config
except Exception:
    get_agent_config = None


# ---------------------------------------------------------------------------
# 数据类型枚举
# ---------------------------------------------------------------------------

class DataType(Enum):
    IMAGE     = "image"
    TEXT      = "text"
    CSV       = "csv"
    EXCEL     = "excel"
    JSONL     = "jsonl"
    DIRECTORY = "directory"
    UNKNOWN   = "unknown"


# ---------------------------------------------------------------------------
# 文件扩展名
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"
}
TEXT_EXTENSIONS: Set[str] = {".txt", ".md", ".json", ".xml"}
CSV_EXTENSIONS:  Set[str] = {".csv", ".tsv"}
EXCEL_EXTENSIONS: Set[str] = {".xlsx", ".xls"}
JSONL_EXTENSIONS: Set[str] = {".jsonl"}


# ---------------------------------------------------------------------------
# 医学列名关键词（用于自动识别 CSV 列）
# ---------------------------------------------------------------------------

MEDICAL_COL_KEYWORDS = {
    "standardize": [
        "test_name", "org_name", "ab_name", "spec_type_desc",
        "diagnosis", "drug", "medication", "disease", "symptom",
        "诊断", "药品", "疾病", "症状", "检验项目", "检查项目",
        "icd", "loinc", "snomed", "atc",
    ],
    "unit_normalize": [
        "value", "result", "quantity", "dilution_value", "amount",
        "数值", "结果", "剂量", "浓度",
        "concentration", "dose", "measurement",
    ],
    "unit_col": [
        "unit", "units", "单位", "dilution_text", "dilution_comparison",
    ],
    "extract_text": [
        "text", "note", "report", "description", "comment", "narrative",
        "findings", "impression", "indication", "conclusion", "summary",
        "discharge", "radiology", "pathology", "history", "assessment",
    ],
}

# 遍历目录时跳过的文件夹名
DEFAULT_SKIP_FOLDERS: Set[str] = {"figure", "分割"}


def get_skip_folders() -> Set[str]:
    env = os.environ.get("SKIP_FOLDERS")
    if env is not None:
        if env.strip() == "":
            return set()
        return {n.strip() for n in env.split(",") if n.strip()}
    return DEFAULT_SKIP_FOLDERS.copy()


# ---------------------------------------------------------------------------
# API 配置
# ---------------------------------------------------------------------------

def get_api_config() -> Dict:
    agent_cfg = get_agent_config("agent_2_3") if get_agent_config else {}
    embedding_cfg = agent_cfg.get("embedding") or {}

    chat_api_key = os.environ.get("AGENT23_CHAT_API_KEY") or agent_cfg.get("api_key", "")
    chat_api_keys_raw = os.environ.get("AGENT23_CHAT_API_KEYS") or agent_cfg.get("api_keys") or chat_api_key
    if isinstance(chat_api_keys_raw, list):
        chat_api_keys = [str(k).strip() for k in chat_api_keys_raw if str(k).strip()]
    else:
        chat_api_keys = [k.strip() for k in str(chat_api_keys_raw or "").split(",") if k.strip()]
    if not chat_api_keys and chat_api_key:
        chat_api_keys = [chat_api_key]
    chat_api_key = chat_api_keys[0] if chat_api_keys else ""
    chat_api_base = os.environ.get("AGENT23_CHAT_API_BASE") or agent_cfg.get("base_url") or "https://api.openai.com/v1"
    chat_model = os.environ.get("AGENT23_CHAT_MODEL") or agent_cfg.get("model") or "gpt-4.1-mini"

    embed_api_key = os.environ.get("AGENT23_EMBED_API_KEY") or embedding_cfg.get("api_key", "")
    embed_api_base = os.environ.get("AGENT23_EMBED_API_BASE") or embedding_cfg.get("base_url") or chat_api_base
    embed_model = os.environ.get("AGENT23_EMBED_MODEL") or embedding_cfg.get("model") or "text-embedding-3-small"

    return {
        # Chat/extraction API for Step2-3 only.
        "api_key":    chat_api_key,
        "api_base":   chat_api_base,
        "model_name": chat_model,
        "chat_api_key": chat_api_key,
        "chat_api_keys": chat_api_keys,
        "chat_api_base": chat_api_base,
        "chat_model": chat_model,

        # Embedding API for Step2-3 only. Keep this separate from chat because
        # some OpenAI-compatible chat providers do not support embeddings.
        "embedding_api_key": embed_api_key,
        "embedding_api_base": embed_api_base,
        "embedding_model": embed_model,

        "timeout":    int(os.environ.get("OPENAI_TIMEOUT", "120")),
        "use_umls":   os.environ.get("USE_UMLS", "true").lower() == "true",
        "verbose":    os.environ.get("VERBOSE", "false").lower() == "true",
    }
