import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "discharge_Diagnosis_atrial_fibrillation_status"


def _clean_af_status(value: str) -> str:
    if pd.isna(value):
        return value

    original = value
    text = str(value).strip()

    if text == "":
        return original

    # 轻量规范化
    normalized_space = re.sub(r"\s+", " ", text)
    key = normalized_space.lower().strip()
    key = key.replace("–", "-").replace("—", "-")
    key = re.sub(r"\s*-\s*", "-", key)
    key = re.sub(r"\s*/\s*", "/", key)
    key = re.sub(r"\s+", " ", key).strip()

    # 缺失/未知
    if key in {
        "unknown", "unk", "unclear", "not known", "n/a", "na", "not available"
    }:
        return "unknown"

    # 否定
    if key in {
        "no", "none", "negative", "absent",
        "no af", "no a-fib", "no afib",
        "no atrial fibrillation", "without atrial fibrillation"
    }:
        return "no atrial fibrillation"

    # 一般阳性
    if key in {
        "yes", "present", "positive",
        "af", "a-fib", "afib", "a fib",
        "atrial fib", "atrial fibrillation"
    }:
        return "atrial fibrillation"

    # 病史
    if key in {
        "history of af", "h/o af", "hx of af", "history af",
        "history of afib", "h/o afib", "hx of afib",
        "history of atrial fibrillation", "h/o atrial fibrillation", "hx of atrial fibrillation"
    }:
        return "history of atrial fibrillation"

    # 新发
    if key in {
        "new onset af", "new-onset af", "new onset afib", "new-onset afib",
        "new onset a-fib", "new-onset a-fib",
        "new onset atrial fibrillation", "new-onset atrial fibrillation"
    }:
        return "new-onset atrial fibrillation"

    # 阵发性
    if key in {
        "paroxysmal af", "paroxysmal afib", "paroxysmal a-fib",
        "paroxysmal atrial fibrillation", "paf"
    }:
        return "paroxysmal atrial fibrillation"

    # 持续性
    if key in {
        "persistent af", "persistent afib", "persistent a-fib",
        "persistent atrial fibrillation"
    }:
        return "persistent atrial fibrillation"

    # 长期/慢性
    if key in {
        "chronic af", "chronic afib", "chronic a-fib",
        "chronic atrial fibrillation"
    }:
        return "chronic atrial fibrillation"

    # 永久性
    if key in {
        "permanent af", "permanent afib", "permanent a-fib",
        "permanent atrial fibrillation"
    }:
        return "permanent atrial fibrillation"

    # 对明显包含关系做保守标准化
    if re.fullmatch(r"(history of|h/o|hx of)\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "history of atrial fibrillation"

    if re.fullmatch(r"new-?onset\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "new-onset atrial fibrillation"

    if re.fullmatch(r"paroxysmal\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "paroxysmal atrial fibrillation"

    if re.fullmatch(r"persistent\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "persistent atrial fibrillation"

    if re.fullmatch(r"chronic\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "chronic atrial fibrillation"

    if re.fullmatch(r"permanent\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "permanent atrial fibrillation"

    if re.fullmatch(r"(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "atrial fibrillation"

    if re.fullmatch(r"(no|none|negative|absent)\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "no atrial fibrillation"

    if re.fullmatch(r"(unknown|unclear|not known)\s+(atrial fibrillation|afib|a-fib|a fib|af)", key):
        return "unknown"

    # 不确定则保留原值，但清理多余空白
    return normalized_space


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_clean_af_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COL}` and saved cleaned file.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")