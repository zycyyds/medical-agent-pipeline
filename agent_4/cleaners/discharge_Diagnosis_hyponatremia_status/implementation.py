import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "discharge_Diagnosis_hyponatremia_status"


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_match(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_hyponatremia_status(value: str) -> str:
    if value is None:
        return value

    original = str(value)
    trimmed = _collapse_spaces(original)

    if trimmed == "":
        return ""

    norm = _normalize_for_match(trimmed)

    # 保留缺失
    if norm in {"", "nan", "null"}:
        return ""

    # 明确未知
    if norm in {
        "unknown",
        "unclear",
        "unsure",
        "indeterminate",
        "not documented",
        "not specified",
        "undocumented",
    }:
        return "unknown"

    # 历史/既往，避免误判为当前已确诊
    history_patterns = [
        r"\bhistory of\b",
        r"\bhx of\b",
        r"\bh/o\b",
        r"\bprior\b",
        r"\bprevious\b",
        r"\bpast\b",
        r"\bremote\b",
    ]
    if any(re.search(p, norm) for p in history_patterns) and (
        "hyponatremia" in norm or "low sodium" in norm or "low na" in norm
    ):
        return "history_of"

    # 已恢复/纠正
    resolved_patterns = [
        r"\bresolved\b",
        r"\bcorrected\b",
        r"\bcorrection of\b",
    ]
    if any(re.search(p, norm) for p in resolved_patterns) and (
        "hyponatremia" in norm or "low sodium" in norm or "low na" in norm
    ):
        return "resolved"

    # 明确否定
    absent_patterns = [
        r"\bno hyponatremia\b",
        r"\bwithout hyponatremia\b",
        r"\bnegative for hyponatremia\b",
        r"\bhyponatremia absent\b",
        r"\babsence of hyponatremia\b",
        r"\bnot hyponatremia\b",
        r"\bno low sodium\b",
    ]
    if norm in {"absent", "negative", "none"}:
        return "absent"
    if any(re.search(p, norm) for p in absent_patterns):
        return "absent"

    # 明确阳性/存在
    present_patterns = [
        r"\bhyponatremia\b",
        r"\blow sodium\b",
        r"\blow na\b",
        r"\bhypo natremia\b",
    ]
    if norm in {"present", "positive", "yes"}:
        return "present"
    if any(re.search(p, norm) for p in present_patterns):
        return "present"

    # 不确定时尽量保留原值，仅做轻量空白规范化
    return trimmed


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_clean_hyponatremia_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed for {TARGET_COLUMN}.",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")