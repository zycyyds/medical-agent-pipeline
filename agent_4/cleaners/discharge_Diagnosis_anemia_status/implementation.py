import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    text = text.lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_anemia_status(value: str) -> str:
    original = value
    norm = _normalize_text(value)

    if norm == "":
        return original

    unknown_values = {
        "unknown", "unk", "unclear", "not clear", "n/a", "na", "none stated",
        "not documented", "not specified", "unspecified", "pending"
    }
    if norm in unknown_values:
        return "unknown"

    no_anemia_patterns = [
        r"^no anemia$",
        r"^non anemia$",
        r"^without anemia$",
        r"^anemia absent$",
        r"^negative for anemia$",
        r"^not anemic$",
        r"^no evidence of anemia$",
        r"^absent$",
        r"^none$",
        r"^normal$",
    ]
    for pattern in no_anemia_patterns:
        if re.fullmatch(pattern, norm):
            return "no anemia"

    anemia_patterns = [
        r"^anemia$",
        r"^anaemia$",
        r"^anemic$",
        r"^anaemic$",
        r"^with anemia$",
        r"^positive for anemia$",
        r"^present$",
    ]
    for pattern in anemia_patterns:
        if re.fullmatch(pattern, norm):
            return "anemia"

    # Conservative phrase-based normalization
    if "anemia" in norm or "anaemia" in norm:
        if re.search(r"\b(no|without|not)\b", norm):
            return "no anemia"
        return "anemia"

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_anemia_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_anemia_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Data cleaning completed. Output saved to: {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")