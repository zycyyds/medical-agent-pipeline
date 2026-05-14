import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _clean_gi_bleeding_status(value):
    if value is None:
        return value

    text = str(value)
    if text == "":
        return text

    original = text

    # Normalize whitespace
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    if text == "":
        return ""

    lower = text.lower()

    # Preserve explicit missing-like values as empty string
    if lower in {"na", "n/a", "nan", "none", "null", "unknown", "unk", "not sure"}:
        return ""

    # Remove surrounding punctuation noise for matching only
    norm = re.sub(r"[\s\-_/]+", " ", lower).strip()

    # Conservative standardization for common gastrointestinal bleeding status expressions
    positive_patterns = {
        "gi bleed": "present",
        "gib": "present",
        "gastrointestinal bleeding": "present",
        "gastro intestinal bleeding": "present",
        "gastrointestinal bleed": "present",
        "gastro intestinal bleed": "present",
        "upper gi bleed": "upper gi bleeding",
        "ugi bleed": "upper gi bleeding",
        "upper gastrointestinal bleeding": "upper gi bleeding",
        "lower gi bleed": "lower gi bleeding",
        "lgi bleed": "lower gi bleeding",
        "lower gastrointestinal bleeding": "lower gi bleeding",
        "melena": "melena",
        "hematemesis": "hematemesis",
        "haematemesis": "hematemesis",
        "hematochezia": "hematochezia",
        "haematochezia": "hematochezia",
        "occult gi bleed": "occult gi bleeding",
        "occult gastrointestinal bleeding": "occult gi bleeding",
        "active gi bleed": "active gi bleeding",
        "active gastrointestinal bleeding": "active gi bleeding",
        "recent gi bleed": "recent gi bleeding",
        "recent gastrointestinal bleeding": "recent gi bleeding",
    }

    negative_patterns = {
        "no gi bleed": "absent",
        "no gib": "absent",
        "no gastrointestinal bleeding": "absent",
        "without gi bleed": "absent",
        "without gastrointestinal bleeding": "absent",
        "negative gi bleed": "absent",
        "negative for gi bleed": "absent",
        "negative for gastrointestinal bleeding": "absent",
        "denies gi bleed": "absent",
        "denies gastrointestinal bleeding": "absent",
        "gi bleed absent": "absent",
        "gastrointestinal bleeding absent": "absent",
        "not active gi bleed": "not active gi bleeding",
        "not active gastrointestinal bleeding": "not active gi bleeding",
        "resolved gi bleed": "resolved gi bleeding",
        "resolved gastrointestinal bleeding": "resolved gi bleeding",
    }

    # Exact matching first
    if norm in positive_patterns:
        return positive_patterns[norm]
    if norm in negative_patterns:
        return negative_patterns[norm]

    # Conservative phrase-based normalization
    # Do not infer "history of" as current diagnosis
    if re.fullmatch(r"(present|positive|yes)", norm):
        return "present"
    if re.fullmatch(r"(absent|negative|no|none)", norm):
        return "absent"

    # Normalize common casing/spacing only when not otherwise mapped
    cleaned = text.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Keep uncertain or complex descriptions after light normalization
    return cleaned if cleaned != "" else original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_gastrointestinal_bleeding_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_gi_bleeding_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content="Cleaning completed successfully.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content="Cleaning failed: " + str(e))