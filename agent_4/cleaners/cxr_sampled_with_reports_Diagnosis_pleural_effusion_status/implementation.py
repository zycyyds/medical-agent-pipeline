import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "cxr_sampled_with_reports_Diagnosis_pleural_effusion_status"


def _normalize_text(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _canonical_key(value: str) -> str:
    text = _normalize_text(value).lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_pleural_effusion_status(value: str) -> str:
    original = str(value)
    normalized = _normalize_text(original)

    if normalized == "":
        return "missing"

    key = _canonical_key(normalized)

    missing_tokens = {
        "na", "n a", "n/a", "none", "null", "nil", "nan", "missing", "not available",
        "not recorded", "not reported", "blank"
    }
    if key in missing_tokens:
        return "missing"

    positive_tokens = {
        "positive", "present", "effusion", "pleural effusion", "yes", "y",
        "with pleural effusion", "has pleural effusion", "pe present", "pos"
    }
    if key in positive_tokens:
        return "positive"

    negative_tokens = {
        "negative", "absent", "no", "n", "no pleural effusion", "without pleural effusion",
        "no effusion", "none seen", "pe absent", "neg"
    }
    if key in negative_tokens:
        return "negative"

    unknown_tokens = {
        "unknown", "uncertain", "indeterminate", "equivocal", "cannot determine",
        "unable to determine", "undetermined", "not sure"
    }
    if key in unknown_tokens:
        return "unknown"

    if re.fullmatch(r"0+(\.0+)?", key):
        return "negative"
    if re.fullmatch(r"1+(\.0+)?", key):
        return "positive"

    return normalized


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_clean_pleural_effusion_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned file saved to {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")