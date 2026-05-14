import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "radiology_Diagnosis_pneumonia_status"


def _clean_single_value(value: str) -> str:
    if value is None:
        return ""

    original = str(value)
    stripped = original.strip()
    if stripped == "":
        return ""

    lightweight = re.sub(r"\s+", " ", stripped)
    normalized = lightweight.lower()
    normalized = re.sub(r"[\-_\/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    mapping = {
        # confirmed / present
        "confirmed": "confirmed",
        "present": "confirmed",
        "positive": "confirmed",
        "yes": "confirmed",
        "y": "confirmed",
        "pneumonia": "confirmed",
        "with pneumonia": "confirmed",
        "has pneumonia": "confirmed",
        "evidence of pneumonia": "confirmed",
        # suspected / possible
        "suspected": "suspected",
        "possible": "suspected",
        "probable": "suspected",
        "query pneumonia": "suspected",
        "? pneumonia": "suspected",
        "?pneumonia": "suspected",
        "cannot exclude pneumonia": "suspected",
        "concerning for pneumonia": "suspected",
        # ruled out / negative
        "ruled out": "ruled_out",
        "ruledout": "ruled_out",
        "negative": "ruled_out",
        "none": "ruled_out",
        "no": "ruled_out",
        "n": "ruled_out",
        "no pneumonia": "ruled_out",
        "without pneumonia": "ruled_out",
        "absent": "ruled_out",
        "not present": "ruled_out",
        "no evidence of pneumonia": "ruled_out",
        # indeterminate
        "indeterminate": "indeterminate",
        "equivocal": "indeterminate",
        # unknown
        "unknown": "unknown",
        "unclear": "unknown",
        "undetermined": "unknown",
        "na": "unknown",
        "n a": "unknown",
        "n/a": "unknown",
    }

    if normalized in mapping:
        return mapping[normalized]

    return lightweight


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_clean_single_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Successfully cleaned column `{TARGET_COLUMN}` and saved cleaned file.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")