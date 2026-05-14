import pandas as pd
import os
import re
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return value
    s = str(value)

    # Trim and normalize whitespace
    s = s.strip()
    s = re.sub(r"\s+", " ", s)

    # Normalize common separators/spaces around punctuation
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*_\s*", "_", s)

    return s


def _map_cardiogenic_shock_status(value: str) -> str:
    original = value
    s = _normalize_text(value)

    if s == "":
        return s

    key = s.lower()
    key = re.sub(r"[ \t\r\n]+", " ", key)

    # Missing/unknown-like values
    unknown_map = {
        "unknown": "Unknown",
        "unk": "Unknown",
        "unclear": "Unknown",
        "not known": "Unknown",
        "n/k": "Unknown",
        "na": "Unknown",
        "n/a": "Unknown",
        "not available": "Unknown",
        "not documented": "Unknown",
        "not specified": "Unknown",
        "unspecified": "Unknown",
        "unable to determine": "Unknown",
        "?": "Unknown",
    }
    if key in unknown_map:
        return unknown_map[key]

    # Positive/present/diagnosed
    positive_patterns = {
        "yes": "Present",
        "y": "Present",
        "present": "Present",
        "positive": "Present",
        "pos": "Present",
        "diagnosed": "Present",
        "with cardiogenic shock": "Present",
        "cardiogenic shock": "Present",
        "cs": "Present",
    }
    if key in positive_patterns:
        return positive_patterns[key]

    # Negative/absent
    negative_patterns = {
        "no": "Absent",
        "n": "Absent",
        "absent": "Absent",
        "negative": "Absent",
        "neg": "Absent",
        "without cardiogenic shock": "Absent",
        "no cardiogenic shock": "Absent",
        "denied": "Absent",
        "none": "Absent",
    }
    if key in negative_patterns:
        return negative_patterns[key]

    # More explicit phrase normalization
    key_compact = re.sub(r"[^a-z0-9/+\- ]", "", key).strip()

    explicit_map = {
        "present on discharge": "Present",
        "diagnosis present": "Present",
        "confirmed": "Present",
        "confirmed cardiogenic shock": "Present",
        "resolved": "Resolved",
        "improved": "Improved",
        "persistent": "Persistent",
        "ongoing": "Persistent",
        "active": "Persistent",
        "ruled out": "Ruled out",
        "rule out": "Ruled out",
        "excluded": "Ruled out",
        "not present": "Absent",
    }
    if key_compact in explicit_map:
        return explicit_map[key_compact]

    # Preserve original normalized text if uncertain
    return s if s != "" else original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_cardiogenic_shock_status"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_map_cardiogenic_shock_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed for column `{target_col}`.",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")