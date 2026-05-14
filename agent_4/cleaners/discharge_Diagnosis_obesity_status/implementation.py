import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_obesity_status(value: str) -> str:
    text = _normalize_text(value)
    if text == "":
        return ""

    lowered = text.lower()
    lowered = re.sub(r"[_\-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()

    missing_like = {
        "na", "n/a", "nan", "null", "nil", "missing", "not available"
    }
    unknown_like = {
        "unknown", "unk", "unspecified", "not specified", "not documented"
    }

    if lowered in missing_like:
        return ""
    if lowered in unknown_like:
        return "unknown"

    mapping = {
        "obese": "obesity",
        "obesity": "obesity",
        "morbid obesity": "morbid obesity",
        "severe obesity": "morbid obesity",
        "class iii obesity": "morbid obesity",
        "class 3 obesity": "morbid obesity",
        "overweight": "overweight",
        "non obese": "no obesity",
        "nonobese": "no obesity",
        "not obese": "no obesity",
        "without obesity": "no obesity",
        "no obesity": "no obesity",
        "obesity negative": "no obesity",
    }

    if lowered in mapping:
        return mapping[lowered]

    return lowered


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_obesity_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_obesity_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned `{target_col}` and saved output to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")