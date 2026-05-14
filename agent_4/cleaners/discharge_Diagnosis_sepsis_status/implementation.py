import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_sepsis_status(value: str) -> str:
    original = value
    text = _normalize_text(value)

    if text == "":
        return ""

    lowered = text.lower()
    lowered = re.sub(r"[_\-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()

    null_like = {
        "na", "n/a", "none", "null", "nil", "unknown", "unk", "not documented", "not recorded"
    }
    if lowered in null_like:
        return ""

    confirmed_patterns = {
        "confirmed",
        "confirm",
        "sepsis confirmed",
        "confirmed sepsis",
        "positive",
        "yes",
        "present",
    }
    suspected_patterns = {
        "suspected",
        "suspect",
        "possible",
        "probable",
        "query",
        "r/o",
        "rule out",
        "sepsis suspected",
        "suspected sepsis",
    }
    history_patterns = {
        "history of",
        "hx of",
        "h/o",
        "ho",
        "past history of",
        "prior history of",
        "history",
        "sepsis history",
        "history of sepsis",
    }

    if lowered in confirmed_patterns:
        return "confirmed"
    if lowered in suspected_patterns:
        return "suspected"
    if lowered in history_patterns:
        return "history of"

    if re.fullmatch(r"(confirmed|confirm)\s+(sepsis)?", lowered):
        return "confirmed"
    if re.fullmatch(r"(suspected|suspect|possible|probable)\s+(sepsis)?", lowered):
        return "suspected"
    if re.fullmatch(r"(history of|hx of|h/o|past history of|prior history of)\s+(sepsis)?", lowered):
        return "history of"

    if re.fullmatch(r"sepsis\s+(confirmed|confirm)", lowered):
        return "confirmed"
    if re.fullmatch(r"sepsis\s+(suspected|suspect|possible|probable)", lowered):
        return "suspected"
    if re.fullmatch(r"sepsis\s+(history|history of)", lowered):
        return "history of"

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_sepsis_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_sepsis_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed for column: {target_col}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")