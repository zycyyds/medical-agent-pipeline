import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_esrd_status(value: str) -> str:
    if value is None:
        return ""

    original = value
    trimmed = re.sub(r"\s+", " ", str(original).strip())

    if trimmed == "":
        return ""

    key = trimmed.lower()
    key = key.replace("_", " ")
    key = re.sub(r"\s*-\s*", "-", key)
    key = re.sub(r"\s+", " ", key).strip()

    esrd_positive = {
        "esrd",
        "yes",
        "y",
        "true",
        "1",
        "end stage renal disease",
        "end-stage renal disease",
        "endstage renal disease",
        "patient has esrd",
        "with esrd",
        "esrd present",
    }

    esrd_negative = {
        "no",
        "n",
        "false",
        "0",
        "no esrd",
        "non esrd",
        "non-esrd",
        "not esrd",
        "without esrd",
        "esrd absent",
        "negative for esrd",
    }

    esrd_unknown = {
        "unknown",
        "unk",
        "undetermined",
        "not documented",
        "not specified",
        "unspecified",
        "unable to determine",
        "pending",
        "n/a",
        "na",
    }

    if key in esrd_positive:
        return "ESRD"
    if key in esrd_negative:
        return "No ESRD"
    if key in esrd_unknown:
        return "Unknown"

    return trimmed


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_esrd_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_esrd_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed for {os.path.basename(input_path)}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")