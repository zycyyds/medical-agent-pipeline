import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "discharge_Diagnosis_congestive_heart_failure_status"


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_chf_status(value: str) -> str:
    original = value
    s = _normalize_text(value)

    if s == "":
        return original

    # Missing-like tokens: normalize lightly to empty string
    if s in {
        "na", "n a", "n/a", "none", "null", "nil", "unknown", "unk",
        "not known", "not documented", "not specified", "unspecified"
    }:
        return ""

    # History should not be merged into confirmed
    history_patterns = [
        r"\bhistory of\b",
        r"\bhx of\b",
        r"\bhx\b",
        r"\bpmh\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\bprevious\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\bprior\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\bold\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\b(chf|congestive heart failure|heart failure)\b.*\bhistory\b",
    ]
    for p in history_patterns:
        if re.search(p, s):
            return "history_of"

    # Suspected / possible / rule-out
    suspected_patterns = [
        r"\bsuspected\b",
        r"\bpossible\b",
        r"\bprobable\b",
        r"\blikely\b",
        r"\bquery\b",
        r"\b?\b",  # harmless fallback, guarded below
        r"\bcannot rule out\b",
        r"\brule out\b",
        r"\br o\b",
        r"\bunder evaluation\b",
        r"\bconcern for\b",
    ]
    for p in suspected_patterns:
        if p == r"\b?\b":
            if "?" in s and re.search(r"\b(chf|congestive heart failure|heart failure)\b", s):
                return "suspected"
        elif re.search(p, s):
            return "suspected"

    # Explicit negatives
    negative_patterns = [
        r"\bno\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\bwithout\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\bdenies\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\bnegative for\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\b(chf|congestive heart failure|heart failure)\b.*\bruled out\b",
        r"\bruled out\b.*\b(chf|congestive heart failure|heart failure)\b",
        r"\bnot present\b",
        r"\babsent\b",
        r"\bnone\b",
    ]
    if s in {"no", "not present", "absent", "negative"}:
        return "absent"
    for p in negative_patterns:
        if re.search(p, s):
            return "absent"

    # Confirmed / present
    heart_failure_terms = r"(chf|c\.?h\.?f\.?|congestive heart failure|congestive cardiac failure|heart failure|hf)"
    confirmed_patterns = [
        rf"^{heart_failure_terms}$",
        rf"\bdiagnosed\b.*\b{heart_failure_terms}\b",
        rf"\b{heart_failure_terms}\b.*\bdiagnosed\b",
        rf"\bconfirmed\b",
        rf"\bpresent\b",
        rf"\bactive\b",
        rf"\bacute\b.*\b{heart_failure_terms}\b",
        rf"\bchronic\b.*\b{heart_failure_terms}\b",
        rf"\bdecompensated\b.*\b{heart_failure_terms}\b",
        rf"\b{heart_failure_terms}\b.*\bexacerbation\b",
    ]
    for p in confirmed_patterns:
        if re.search(p, s):
            return "confirmed"

    # Canonical direct mappings
    direct_map = {
        "confirmed": "confirmed",
        "suspected": "suspected",
        "possible": "suspected",
        "probable": "suspected",
        "history of": "history_of",
        "history_of": "history_of",
        "hx of chf": "history_of",
        "hx chf": "history_of",
        "absent": "absent",
        "negative": "absent",
        "ruled out": "absent",
        "chf": "confirmed",
        "congestive heart failure": "confirmed",
        "heart failure": "confirmed",
        "hf": "confirmed",
    }
    if s in direct_map:
        return direct_map[s]

    # If the content references CHF/HF but status is unclear, preserve original
    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_clean_chf_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COL}` and saved output to `{output_path}`.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")