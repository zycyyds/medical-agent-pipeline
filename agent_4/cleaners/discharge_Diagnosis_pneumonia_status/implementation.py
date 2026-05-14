import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_pneumonia_status(value: str) -> str:
    original = value
    text = _normalize_text(value)

    if text == "":
        return original

    key = text.lower()

    mapping = {
        # present
        "present": "present",
        "positive": "present",
        "yes": "present",
        "y": "present",
        "pneumonia": "present",
        "with pneumonia": "present",
        "有": "present",
        "是": "present",
        "存在": "present",
        "肺炎": "present",
        "阳性": "present",

        # absent
        "absent": "absent",
        "negative": "absent",
        "no": "absent",
        "n": "absent",
        "none": "absent",
        "no pneumonia": "absent",
        "without pneumonia": "absent",
        "无": "absent",
        "否": "absent",
        "未见": "absent",
        "无肺炎": "absent",
        "阴性": "absent",

        # resolved
        "resolved": "resolved",
        "resolution": "resolved",
        "resolved pneumonia": "resolved",
        "已吸收": "resolved",
        "吸收": "resolved",
        "痊愈": "resolved",
        "治愈": "resolved",
        "已恢复": "resolved",

        # improved
        "improved": "improved",
        "improving": "improved",
        "improvement": "improved",
        "好转": "improved",
        "改善": "improved",
        "部分吸收": "improved",
        "吸收好转": "improved",

        # worsened
        "worsened": "worsened",
        "worse": "worsened",
        "progressed": "worsened",
        "aggravated": "worsened",
        "加重": "worsened",
        "恶化": "worsened",
        "进展": "worsened",

        # active
        "active": "active",
        "ongoing": "active",
        "persistent": "active",
        "活动性": "active",
        "持续存在": "active",
        "未愈": "active",

        # suspected
        "suspected": "suspected",
        "possible": "suspected",
        "query": "suspected",
        "疑似": "suspected",
        "可疑": "suspected",
        "待排": "suspected",

        # unknown
        "unknown": "unknown",
        "unclear": "unknown",
        "not sure": "unknown",
        "不详": "unknown",
        "未知": "unknown",
        "不明确": "unknown",
    }

    if key in mapping:
        return mapping[key]

    return text


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        target_col = "discharge_Diagnosis_pneumonia_status"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_pneumonia_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned CSV saved to: {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")