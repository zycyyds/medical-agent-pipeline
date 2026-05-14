import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_match(text: str) -> str:
    text = _collapse_spaces(text).lower()
    text = re.sub(r"[，,;；]+", " ", text)
    text = _collapse_spaces(text)
    return text


def _clean_copd_status(value):
    if pd.isna(value):
        return value

    original = str(value)
    cleaned = _collapse_spaces(original)
    if cleaned == "":
        return value

    norm = _normalize_for_match(cleaned)

    direct_map = {
        "copd": "COPD",
        "chronic obstructive pulmonary disease": "COPD",
        "chronic obstructive pulmonary dz": "COPD",
        "chronic obstructive airway disease": "COPD",
        "coad": "COPD",
        "aecopd": "COPD exacerbation",
        "ae copd": "COPD exacerbation",
        "copd exacerbation": "COPD exacerbation",
        "acute copd exacerbation": "COPD exacerbation",
        "acute exacerbation of copd": "COPD exacerbation",
        "acute exacerbation copd": "COPD exacerbation",
        "acute exacerbation of chronic obstructive pulmonary disease": "COPD exacerbation",
        "acute on chronic copd exacerbation": "COPD exacerbation",
        "history of copd": "History of COPD",
        "hx of copd": "History of COPD",
        "h/o copd": "History of COPD",
        "ho copd": "History of COPD",
        "pmh copd": "History of COPD",
        "no copd": "No COPD",
        "without copd": "No COPD",
        "copd negative": "No COPD",
        "negative for copd": "No COPD",
        "denies copd": "No COPD",
        "unknown": "Unknown",
        "unk": "Unknown",
        "not documented": "Unknown",
        "not specified": "Unknown",
        "n/a": "Unknown",
        "na": "Unknown",
    }

    if norm in direct_map:
        return direct_map[norm]

    if re.fullmatch(r"copd[\.\-]?", norm):
        return "COPD"

    if re.fullmatch(r"(hx|h/o|ho)\s+(of\s+)?copd", norm):
        return "History of COPD"

    if re.fullmatch(r"(no|without)\s+copd", norm):
        return "No COPD"

    if re.fullmatch(r"(acute\s+)?(exacerbation\s+of\s+)?copd", norm):
        return "COPD exacerbation"

    return cleaned


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_copd_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_copd_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned CSV saved to: {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")