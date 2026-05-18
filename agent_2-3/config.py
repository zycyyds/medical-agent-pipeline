# -*- coding: utf-8 -*-
"""
统一配置文件

所有路径、API 配置、枚举类型、处理管线定义集中在此处。
"""
import os
import sys
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from configs.loader import get_agent_config
except Exception:  # pragma: no cover - keeps the legacy standalone path usable
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

def _agent23_config() -> Dict[str, Any]:
    if get_agent_config is None:
        return {}
    return dict(get_agent_config("agent_2_3") or {})


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_api_config() -> Dict[str, Any]:
    """Chat/LLM config for Agent2-3.

    YAML is the primary source. Legacy environment variables are kept as
    fallback so old standalone scripts do not silently break.
    """
    cfg = _agent23_config()
    return {
        "api_key": (
            os.environ.get("AGENT_2_3_API_KEY")
            or cfg.get("api_key")
            or os.environ.get("OPENAI_API_KEYS", "")
        ),
        "api_base": (
            os.environ.get("AGENT_2_3_BASE_URL")
            or cfg.get("api_base")
            or cfg.get("base_url")
            or os.environ.get("YUNWU_BASE_URL", "")
        ),
        "timeout": _int_value(
            os.environ.get("AGENT_2_3_TIMEOUT") or cfg.get("timeout") or os.environ.get("OPENAI_TIMEOUT"),
            120,
        ),
        "model_name": (
            os.environ.get("AGENT_2_3_MODEL")
            or os.environ.get("MODEL_NAME")
            or cfg.get("model_name")
            or cfg.get("model")
            or "gpt-5.4-nano"
        ),
        "temperature": float(cfg.get("temperature", 0.0) or 0.0),
        "seed": cfg.get("seed", 666),
        "use_umls": os.environ.get("USE_UMLS", "true").lower() == "true",
        "verbose": os.environ.get("VERBOSE", "false").lower() == "true",
    }


def get_embedding_config() -> Dict[str, Any]:
    """Embedding config for Agent2-3 semantic clustering.

    This intentionally does not reuse the chat model endpoint because many
    OpenAI-compatible chat providers do not expose embeddings.
    """
    cfg = _agent23_config()
    emb = dict(cfg.get("embedding") or {})
    return {
        "api_key": (
            os.environ.get("AGENT_2_3_EMBEDDING_API_KEY")
            or emb.get("api_key")
            or ""
        ),
        "api_base": (
            os.environ.get("AGENT_2_3_EMBEDDING_BASE_URL")
            or emb.get("api_base")
            or emb.get("base_url")
            or ""
        ),
        "timeout": _int_value(
            os.environ.get("AGENT_2_3_EMBEDDING_TIMEOUT") or emb.get("timeout"),
            120,
        ),
        "model_name": (
            os.environ.get("AGENT_2_3_EMBEDDING_MODEL")
            or emb.get("model_name")
            or emb.get("model")
            or "text-embedding-3-small"
        ),
        "dimensions": emb.get("dimensions"),
    }
