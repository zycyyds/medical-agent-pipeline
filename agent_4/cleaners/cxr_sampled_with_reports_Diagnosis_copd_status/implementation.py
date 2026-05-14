import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "cxr_sampled_with_reports_Diagnosis_copd_status"


def _normalize_copd_status(value: str) -> str:
    if value is None:
        return value

    original = value
    text = str(value).strip()

    if text == "":
        return ""

    # Normalize whitespace and case for matching
    normalized = re.sub(r"\s+", " ", text).strip().lower()

    # Preserve explicit missing / unknown style states
    if normalized in {
        "unknown", "unk", "unclear", "indeterminate", "undetermined",
        "not sure", "n/a", "na", "none", "null", "missing"
    }:
        return "unknown"

    # Negative / absent COPD
    if normalized in {
        "no copd", "copd negative", "negative for copd", "without copd",
        "copd absent", "absent", "none detected"
    }:
        return "no copd"

    # Positive / confirmed COPD
    if normalized in {
        "copd", "has copd", "with copd", "copd present", "positive for copd",
        "confirmed copd", "known copd"
    }:
        return "copd"

    # History should not be merged into confirmed COPD
    if normalized in {
        "history of copd", "hx of copd", "h/o copd", "ho copd", "past copd"
    }:
        return "history of copd"

    # Suspected / possible should remain uncertain and distinct
    if normalized in {
        "suspected copd", "possible copd", "probable copd", "query copd",
        "?copd", "r/o copd", "rule out copd", "cannot exclude copd"
    }:
        return "suspected copd"

    # Mild normalization for common punctuation variants
    compact = re.sub(r"[^\w\s/]+", "", normalized).strip()

    if compact in {
        "no copd", "copd negative", "negative for copd", "without copd",
        "copd absent"
    }:
        return "no copd"

    if compact in {
        "copd", "has copd", "with copd", "copd present", "positive for copd",
        "confirmed copd", "known copd"
    }:
        return "copd"

    if compact in {
        "history of copd", "hx of copd", "h/o copd", "ho copd", "past copd"
    }:
        return "history of copd"

    if compact in {
        "suspected copd", "possible copd", "probable copd", "query copd",
        "r/o copd", "rule out copd", "cannot exclude copd"
    }:
        return "suspected copd"

    if compact in {
        "unknown", "unk", "unclear", "indeterminate", "undetermined",
        "not sure", "na", "none", "null", "missing"
    }:
        return "unknown"

    # If only whitespace/casing changed, return normalized spacing version
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text if cleaned_text != "" else original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_normalize_copd_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed for column: {TARGET_COLUMN}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")