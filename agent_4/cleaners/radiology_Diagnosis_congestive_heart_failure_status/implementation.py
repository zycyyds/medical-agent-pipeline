import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_chf_status(value: str) -> str:
    if value is None:
        return ""

    original = value
    s = str(value).strip()
    if s == "":
        return ""

    s = re.sub(r"\s+", " ", s)
    s_lower = s.lower()

    # Common missing / null-like tokens
    if s_lower in {
        "na", "n/a", "nan", "none", "null", "unknown", "unk", "not available",
        "not documented", "missing", "-"
    }:
        return ""

    # Normalize punctuation lightly for matching
    s_compact = re.sub(r"[\s_\-\/]+", " ", s_lower)
    s_compact = re.sub(r"[^\w\s\?]", "", s_compact).strip()

    # Explicit indeterminate / unclear
    if s_compact in {
        "indeterminate", "equivocal", "unclear", "inconclusive", "undetermined"
    }:
        return "indeterminate"

    # Suspected / possible, kept distinct from confirmed
    suspected_patterns = [
        r"^suspected$",
        r"^possible$",
        r"^probable$",
        r"^query$",
        r"^\?$",
        r"^cannot exclude$",
        r"^not excluded$",
        r"^likely$",
        r"^suggestive of$",
        r"^concerning for$",
        r"^suspect(ed)? chf$",
        r"^possible chf$",
        r"^probable chf$",
        r"^query chf$",
        r"^suspect(ed)? congestive heart failure$",
        r"^possible congestive heart failure$",
        r"^probable congestive heart failure$",
    ]
    for pat in suspected_patterns:
        if re.fullmatch(pat, s_compact):
            return "suspected"

    # Negative / absent
    absent_patterns = [
        r"^absent$",
        r"^negative$",
        r"^no$",
        r"^none$",
        r"^not present$",
        r"^without$",
        r"^no chf$",
        r"^without chf$",
        r"^no evidence of chf$",
        r"^no congestive heart failure$",
        r"^without congestive heart failure$",
        r"^no evidence of congestive heart failure$",
        r"^heart failure absent$",
        r"^congestive heart failure absent$",
    ]
    for pat in absent_patterns:
        if re.fullmatch(pat, s_compact):
            return "absent"

    # Positive / confirmed
    confirmed_patterns = [
        r"^confirmed$",
        r"^present$",
        r"^positive$",
        r"^yes$",
        r"^chf$",
        r"^congestive heart failure$",
        r"^heart failure$",
        r"^cardiac failure$",
        r"^consistent with chf$",
        r"^findings consistent with chf$",
        r"^evidence of chf$",
        r"^diagnosed chf$",
        r"^confirmed chf$",
        r"^consistent with congestive heart failure$",
        r"^evidence of congestive heart failure$",
        r"^diagnosed congestive heart failure$",
        r"^confirmed congestive heart failure$",
    ]
    for pat in confirmed_patterns:
        if re.fullmatch(pat, s_compact):
            return "confirmed"

    # Keep lightly normalized original text if not confidently mappable
    return s


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "radiology_Diagnosis_congestive_heart_failure_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_chf_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{target_col}` and saved cleaned CSV to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")