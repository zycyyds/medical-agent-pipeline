import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "radiology_Diagnosis_respiratory_failure_status"


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.strip()
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[/|;,:]+\s*", " / ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonicalize_status(value: str) -> str:
    original = _normalize_text(value)
    if original == "":
        return ""

    norm = original.lower()
    norm = norm.replace("-", " ")
    norm = norm.replace("_", " ")
    norm = re.sub(r"[(){}\[\]]", " ", norm)
    norm = re.sub(r"[^a-z0-9/ ]+", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()

    present_terms = {
        "present",
        "positive",
        "yes",
        "y",
        "respiratory failure",
        "rf",
        "in respiratory failure",
        "with respiratory failure",
    }
    absent_terms = {
        "absent",
        "negative",
        "no",
        "n",
        "none",
        "without respiratory failure",
        "no respiratory failure",
        "not in respiratory failure",
    }
    unknown_terms = {
        "unknown",
        "unk",
        "unclear",
        "indeterminate",
        "not specified",
        "unspecified",
        "na",
        "n a",
    }
    history_terms = {
        "history of respiratory failure",
        "hx of respiratory failure",
        "h/o respiratory failure",
        "prior respiratory failure",
        "previous respiratory failure",
        "resolved respiratory failure",
    }

    if norm in present_terms:
        return "present"
    if norm in absent_terms:
        return "absent"
    if norm in unknown_terms:
        return "unknown"
    if norm in history_terms:
        return "history of"

    if re.search(r"\bacute on chronic\b", norm):
        return "acute on chronic"
    if re.search(r"\bacute\b", norm) and re.search(r"\bchronic\b", norm):
        return "acute on chronic"
    if re.search(r"\bchronic\b", norm):
        return "chronic"
    if re.search(r"\bacute\b", norm):
        return "acute"

    if re.search(r"\bhistory of\b", norm) and re.search(r"\brespiratory failure\b", norm):
        return "history of"

    if re.fullmatch(r"respiratory failure status unknown", norm):
        return "unknown"

    if re.search(r"\brespiratory failure\b", norm):
        return "present"

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_canonicalize_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COLUMN}` and saved output to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")