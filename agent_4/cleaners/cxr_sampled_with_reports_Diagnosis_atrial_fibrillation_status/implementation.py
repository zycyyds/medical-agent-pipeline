import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "cxr_sampled_with_reports_Diagnosis_atrial_fibrillation_status"


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    text = text.lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[;/|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_af_status(value: str) -> str:
    original = value
    norm = _normalize_text(value)

    if norm == "":
        return original

    # Direct unknown / indeterminate labels
    if norm in {
        "unknown",
        "unk",
        "unclear",
        "indeterminate",
        "not specified",
        "nonspecific",
        "equivocal",
    }:
        return "unknown"

    # History should not be promoted to confirmed/present
    history_patterns = [
        r"\bhistory of atrial fibrillation\b",
        r"\bhx of atrial fibrillation\b",
        r"\bh/o atrial fibrillation\b",
        r"\baf history\b",
        r"\batrial fibrillation history\b",
        r"\bchronic atrial fibrillation\b",
        r"\bparoxysmal atrial fibrillation\b",
        r"\bpersistent atrial fibrillation\b",
        r"\bpermanent atrial fibrillation\b",
        r"\bprior atrial fibrillation\b",
        r"\bprevious atrial fibrillation\b",
        r"\bknown atrial fibrillation\b",
        r"\bafib history\b",
        r"\ba fib history\b",
    ]
    for pat in history_patterns:
        if re.search(pat, norm):
            return "history_of"

    # Cannot exclude / possible / suspected
    if (
        re.search(r"\bcannot exclude\b", norm)
        or re.search(r"\bpossible\b", norm)
        or re.search(r"\bpossibly\b", norm)
        or re.search(r"\bsuspected\b", norm)
        or re.search(r"\bquestion of\b", norm)
        or re.search(r"\bmaybe\b", norm)
        or re.search(r"\blikely\b", norm)
    ) and (
        "atrial fibrillation" in norm
        or "afib" in norm
        or "a fib" in norm
        or norm == "af"
    ):
        return "cannot_exclude"

    # Negative / absent
    negative_patterns = [
        r"\bno atrial fibrillation\b",
        r"\bno evidence of atrial fibrillation\b",
        r"\bwithout atrial fibrillation\b",
        r"\bnegative for atrial fibrillation\b",
        r"\babsence of atrial fibrillation\b",
        r"\bnot atrial fibrillation\b",
        r"\bafib not present\b",
        r"\ba fib not present\b",
        r"\bno afib\b",
        r"\bno a fib\b",
        r"\baf absent\b",
        r"\bnone\b",
        r"\babsent\b",
        r"\bnegative\b",
        r"\bnormal sinus rhythm\b",
        r"\bsinus rhythm\b",
        r"\bnsr\b",
    ]
    for pat in negative_patterns:
        if re.search(pat, norm):
            return "absent"

    # Positive / present
    positive_patterns = [
        r"\batrial fibrillation\b",
        r"\bafib\b",
        r"\ba fib\b",
        r"\baf\b",
        r"\bin atrial fibrillation\b",
        r"\bwith atrial fibrillation\b",
        r"\bnew onset atrial fibrillation\b",
        r"\bcurrent atrial fibrillation\b",
    ]
    for pat in positive_patterns:
        if re.search(pat, norm):
            return "present"

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_clean_af_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        if output_path == input_path:
            root, ext = os.path.splitext(input_path)
            output_path = root + "_cleaned" + ext

        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Data cleaning completed. Output saved to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")