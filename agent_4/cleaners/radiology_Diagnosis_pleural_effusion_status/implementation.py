import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_pleural_effusion_status(value: str) -> str:
    if value is None:
        return value

    original = value
    text = str(value)

    if text == "":
        return text

    stripped = text.strip()
    if stripped == "":
        return stripped

    lowered = stripped.lower()
    lowered = re.sub(r"[\s_/\\\-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()

    ruled_out_patterns = [
        r"^ruled out$",
        r"^rule out$",
        r"^r/o$",
        r"^no pleural effusion$",
        r"^without pleural effusion$",
        r"^negative$",
        r"^none$",
        r"^absent$",
        r"^not seen$",
        r"^not present$",
        r"^excluded$",
    ]

    confirmed_patterns = [
        r"^confirmed$",
        r"^present$",
        r"^positive$",
        r"^pleural effusion$",
        r"^effusion present$",
        r"^with pleural effusion$",
    ]

    suspected_patterns = [
        r"^suspected$",
        r"^possible$",
        r"^probable$",
        r"^likely$",
        r"^questionable$",
        r"^cannot exclude$",
        r"^suspicious$",
        r"^equivocal$",
        r"^indeterminate$",
    ]

    for pattern in ruled_out_patterns:
        if re.fullmatch(pattern, lowered):
            return "ruled out"

    for pattern in confirmed_patterns:
        if re.fullmatch(pattern, lowered):
            return "confirmed"

    for pattern in suspected_patterns:
        if re.fullmatch(pattern, lowered):
            return "suspected"

    if "pleural effusion" in lowered:
        if any(token in lowered for token in ["no ", "without ", "absent", "negative", "ruled out", "rule out", "excluded", "not seen", "not present"]):
            return "ruled out"
        if any(token in lowered for token in ["possible", "probable", "likely", "questionable", "cannot exclude", "suspicious", "equivocal", "indeterminate", "suspected"]):
            return "suspected"
        if any(token in lowered for token in ["present", "confirmed", "positive"]):
            return "confirmed"

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "radiology_Diagnosis_pleural_effusion_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_pleural_effusion_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed. Output saved to {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")