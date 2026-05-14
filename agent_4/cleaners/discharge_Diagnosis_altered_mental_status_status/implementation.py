import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "discharge_Diagnosis_altered_mental_status_status"


def _normalize_status_value(value):
    if value is None:
        return value

    text = str(value)
    if text == "":
        return ""

    original = text
    text = text.strip()
    if text == "":
        return ""

    lowered = text.lower()
    lowered = re.sub(r"[_\-\/]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()

    confirmed_patterns = [
        r"^confirmed$",
        r"^confirm$",
        r"^conformed$",
        r"^positive$",
        r"^present$",
        r"^diagnosed$",
        r"^dx$",
        r"^yes$",
        r"^true$",
        r"^known$",
        r"^active$",
        r"^probable confirmed$",
        r"^clinically confirmed$",
    ]

    suspected_patterns = [
        r"^suspected$",
        r"^suspect$",
        r"^possible$",
        r"^probable$",
        r"^presumed$",
        r"^likely$",
        r"^query$",
        r"^questionable$",
        r"^concern for$",
        r"^cannot rule out$",
        r"^r o$",
        r"^rule out$",
    ]

    ruled_out_patterns = [
        r"^ruled out$",
        r"^rule out negative$",
        r"^not present$",
        r"^negative$",
        r"^absent$",
        r"^excluded$",
        r"^no$",
        r"^false$",
        r"^denied$",
        r"^resolved no longer present$",
    ]

    for pattern in confirmed_patterns:
        if re.fullmatch(pattern, lowered):
            return "confirmed"

    for pattern in suspected_patterns:
        if re.fullmatch(pattern, lowered):
            return "suspected"

    for pattern in ruled_out_patterns:
        if re.fullmatch(pattern, lowered):
            return "ruled_out"

    # 对明显由状态词组成但大小写/空格不统一的值做保守标准化
    if lowered in {"confirmed", "suspected", "ruled out", "ruled_out"}:
        if lowered == "confirmed":
            return "confirmed"
        if lowered == "suspected":
            return "suspected"
        return "ruled_out"

    # 不确定时保留原值，仅做首尾空格清理
    return original.strip()


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_normalize_status_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COLUMN}` and saved output to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")