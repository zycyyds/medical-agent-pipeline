import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return value
    s = str(value)

    # Trim and normalize whitespace
    s = s.strip()
    s = re.sub(r"\s+", " ", s)

    # Normalize common separators lightly
    s = re.sub(r"\s*[/|]+\s*", "/", s)
    s = re.sub(r"\s*-\s*", "-", s)

    return s


def _canonicalize_dm2_status(value: str) -> str:
    original = value
    s = _normalize_text(value)

    if s == "":
        return s

    lower = s.lower()

    # Normalize punctuation/spaces for matching only
    key = lower
    key = key.replace("’", "'")
    key = re.sub(r"[,_;:]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()

    # Very light standardization for common type 2 DM status expressions
    mapping = {
        "yes": "present",
        "y": "present",
        "present": "present",
        "positive": "present",
        "diagnosed": "present",
        "known": "present",
        "has": "present",
        "with": "present",
        "dm2": "type 2 diabetes mellitus",
        "t2dm": "type 2 diabetes mellitus",
        "type 2 dm": "type 2 diabetes mellitus",
        "type ii dm": "type 2 diabetes mellitus",
        "type 2 diabetes": "type 2 diabetes mellitus",
        "diabetes mellitus type 2": "type 2 diabetes mellitus",
        "type 2 diabetes mellitus": "type 2 diabetes mellitus",
        "niddm": "type 2 diabetes mellitus",
        "non insulin dependent diabetes mellitus": "type 2 diabetes mellitus",
        "non-insulin-dependent diabetes mellitus": "type 2 diabetes mellitus",
        "no": "absent",
        "n": "absent",
        "absent": "absent",
        "negative": "absent",
        "none": "absent",
        "without": "absent",
        "denies": "absent",
        "unknown": "unknown",
        "unk": "unknown",
        "unclear": "unknown",
        "not known": "unknown",
        "not sure": "unknown",
        "history of type 2 diabetes mellitus": "history of type 2 diabetes mellitus",
        "hx of type 2 diabetes mellitus": "history of type 2 diabetes mellitus",
        "h/o type 2 diabetes mellitus": "history of type 2 diabetes mellitus",
        "history of t2dm": "history of type 2 diabetes mellitus",
        "hx of t2dm": "history of type 2 diabetes mellitus",
        "h/o t2dm": "history of type 2 diabetes mellitus",
        "new onset type 2 diabetes mellitus": "new onset type 2 diabetes mellitus",
        "newly diagnosed type 2 diabetes mellitus": "new onset type 2 diabetes mellitus",
        "uncontrolled type 2 diabetes mellitus": "uncontrolled type 2 diabetes mellitus",
        "controlled type 2 diabetes mellitus": "controlled type 2 diabetes mellitus",
    }

    if key in mapping:
        return mapping[key]

    # Pattern-based normalization while preserving distinct medical status
    if re.fullmatch(r"(hx|h/o|history of)\s+(dm2|t2dm|type 2 dm|type 2 diabetes|type 2 diabetes mellitus)", key):
        return "history of type 2 diabetes mellitus"

    if re.fullmatch(r"(new onset|newly diagnosed)\s+(dm2|t2dm|type 2 dm|type 2 diabetes|type 2 diabetes mellitus)", key):
        return "new onset type 2 diabetes mellitus"

    if re.fullmatch(r"(controlled)\s+(dm2|t2dm|type 2 dm|type 2 diabetes|type 2 diabetes mellitus)", key):
        return "controlled type 2 diabetes mellitus"

    if re.fullmatch(r"(uncontrolled)\s+(dm2|t2dm|type 2 dm|type 2 diabetes|type 2 diabetes mellitus)", key):
        return "uncontrolled type 2 diabetes mellitus"

    if re.fullmatch(r"(dm2|t2dm|type 2 dm|type ii dm|type 2 diabetes|diabetes mellitus type 2|type 2 diabetes mellitus)", key):
        return "type 2 diabetes mellitus"

    # If not confidently recognized, keep lightly normalized original text
    return s if s != "" else original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_diabetes_mellitus_type_2_status"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_canonicalize_dm2_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed for {target_col}.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")