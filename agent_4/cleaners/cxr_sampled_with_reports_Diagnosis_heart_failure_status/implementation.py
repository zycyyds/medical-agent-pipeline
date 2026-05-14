import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "cxr_sampled_with_reports_Diagnosis_heart_failure_status"


def _normalize_heart_failure_status(value):
    if value is None:
        return value

    text = str(value)

    # Preserve true missing values represented as empty string after stripping
    stripped = text.strip()
    if stripped == "":
        return ""

    # Lightweight normalization only
    normalized = stripped.lower()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[\/|]+", " / ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Missing-like tokens -> keep as empty
    if normalized in {
        "na", "n/a", "none", "null", "nan", "unknown", "not available",
        "missing", "unspecified", "undetermined", "?"
    }:
        return ""

    # Explicit positive / present / diagnosed
    positive_patterns = [
        r"^positive$",
        r"^present$",
        r"^yes$",
        r"^y$",
        r"^hf$",
        r"^chf$",
        r"^heart failure$",
        r"^congestive heart failure$",
        r"^diagnosed heart failure$",
        r"^heart failure present$",
        r"^heart failure positive$",
    ]
    for pattern in positive_patterns:
        if re.fullmatch(pattern, normalized):
            return "heart failure"

    # Explicit negative / absent / no diagnosis
    negative_patterns = [
        r"^negative$",
        r"^absent$",
        r"^no$",
        r"^n$",
        r"^none$",
        r"^no heart failure$",
        r"^without heart failure$",
        r"^heart failure absent$",
        r"^heart failure negative$",
        r"^no evidence of heart failure$",
        r"^no chf$",
        r"^no hf$",
    ]
    for pattern in negative_patterns:
        if re.fullmatch(pattern, normalized):
            return "no heart failure"

    # Do not over-merge uncertain or historical expressions; preserve normalized text
    return normalized


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_normalize_heart_failure_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COLUMN}` and saved output to `{output_path}`.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")