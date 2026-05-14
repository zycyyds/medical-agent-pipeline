import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_hypotension_status(value: str) -> str:
    if value is None:
        return value

    original = value
    text = str(value).strip()
    if text == "":
        return ""

    text = re.sub(r"\s+", " ", text)
    lowered = text.lower()

    match_key = re.sub(r"[\s_\-\/]+", " ", lowered).strip()

    yes_values = {
        "yes",
        "y",
        "true",
        "1",
        "positive",
        "pos",
        "present",
    }

    no_values = {
        "no",
        "n",
        "false",
        "0",
        "negative",
        "neg",
        "absent",
        "none",
        "not present",
    }

    if match_key in yes_values:
        return "yes"
    if match_key in no_values:
        return "no"

    return text


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        target_col = "discharge_Diagnosis_hypotension_status"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_hypotension_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned CSV saved to: {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")