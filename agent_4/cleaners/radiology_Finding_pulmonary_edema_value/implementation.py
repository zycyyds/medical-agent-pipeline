import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "radiology_Finding_pulmonary_edema_value"


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if s == "":
        return ""

    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"[|/]+", "; ", s)
    s = re.sub(r"\s*[,;:]+\s*", "; ", s)
    s = re.sub(r"\s+", " ", s).strip(" ;,:")

    # Light synonym normalization only
    replacements = [
        (r"\bpulm\.?\s*edema\b", "pulmonary edema"),
        (r"\boedema\b", "edema"),
        (r"\bchf\b", "congestive heart failure"),
        (r"\binterstitial edema\b", "interstitial pulmonary edema"),
        (r"\balveolar edema\b", "alveolar pulmonary edema"),
        (r"\bedematous changes\b", "edema"),
        (r"\bfluid overload\b", "pulmonary edema"),
        (r"\bvascular congestion\b", "pulmonary vascular congestion"),
        (r"\bcephalization\b", "pulmonary vascular congestion"),
    ]
    lowered = s.lower()
    for pat, repl in replacements:
        lowered = re.sub(pat, repl, lowered, flags=re.IGNORECASE)

    lowered = re.sub(r"\s+", " ", lowered).strip(" ;,:")
    return lowered


def _detect_status(text: str) -> str:
    t = text.lower()

    uncertain_patterns = [
        r"\bcannot\s+exclude\b",
        r"\bpossible\b",
        r"\bpossibly\b",
        r"\bmay\s+represent\b",
        r"\bmight\s+represent\b",
        r"\bsuggest(?:ing|ive)?\s+of\b",
        r"\bconcerning\s+for\b",
        r"\bquestion\s+of\b",
        r"\bequivocal\b",
        r"\bindeterminate\b",
        r"\bunlikely\b",
        r"\bless\s+likely\b",
        r"\bnot\s+excluded\b",
    ]
    negation_patterns = [
        r"\bno\b.{0,30}\bpulmonary edema\b",
        r"\bwithout\b.{0,30}\bpulmonary edema\b",
        r"\bnegative\s+for\b.{0,20}\bpulmonary edema\b",
        r"\bno\s+evidence\s+of\b.{0,20}\bpulmonary edema\b",
        r"\bno\s+focal\b.{0,20}\bedema\b",
        r"\bresolved\b.{0,20}\bpulmonary edema\b",
        r"\babsence\s+of\b.{0,20}\bpulmonary edema\b",
        r"\bfree\s+of\b.{0,20}\bpulmonary edema\b",
    ]
    present_patterns = [
        r"\bpulmonary edema\b",
        r"\bedema\b",
        r"\binterstitial pulmonary edema\b",
        r"\balveolar pulmonary edema\b",
        r"\bcardiogenic edema\b",
    ]

    has_uncertain = any(re.search(p, t, flags=re.IGNORECASE) for p in uncertain_patterns)
    has_negative = any(re.search(p, t, flags=re.IGNORECASE) for p in negation_patterns)
    has_present = any(re.search(p, t, flags=re.IGNORECASE) for p in present_patterns)

    if has_negative:
        return "absent"
    if has_uncertain and has_present:
        return "uncertain"
    if has_present:
        return "present"
    if has_uncertain:
        return "uncertain"
    return "unknown"


def _detect_severity(text: str, status: str) -> str:
    if status == "absent":
        return "none"

    t = text.lower()

    severe_patterns = [
        r"\bsevere\b",
        r"\bmarked\b",
        r"\bflorid\b",
        r"\bextensive\b",
        r"\bpronounced\b",
        r"\bmassive\b",
    ]
    moderate_patterns = [
        r"\bmoderate\b",
        r"\bmoderately\b",
    ]
    mild_patterns = [
        r"\bmild\b",
        r"\bminimal\b",
        r"\bslight\b",
        r"\btrace\b",
        r"\bsmall\b",
        r"\bearly\b",
    ]

    if any(re.search(p, t, flags=re.IGNORECASE) for p in severe_patterns):
        return "severe"
    if any(re.search(p, t, flags=re.IGNORECASE) for p in moderate_patterns):
        return "moderate"
    if any(re.search(p, t, flags=re.IGNORECASE) for p in mild_patterns):
        return "mild"
    return "unknown"


def _detect_trend(text: str) -> str:
    t = text.lower()

    improved_patterns = [
        r"\bimprov(?:ed|ing|ement)\b",
        r"\bdecreas(?:ed|ing|e)\b",
        r"\bless\b.{0,20}\bedema\b",
        r"\bresolved\b",
        r"\bnear\s+resolution\b",
        r"\bbetter\b",
        r"\bclearing\b",
    ]
    worsened_patterns = [
        r"\bwors(?:e|ened|ening)\b",
        r"\bincreas(?:ed|ing|e)\b",
        r"\bprogress(?:ed|ion|ive)\b",
        r"\bmore\b.{0,20}\bedema\b",
        r"\bdeveloped\b",
        r"\bnew\b.{0,20}\bedema\b",
    ]
    stable_patterns = [
        r"\bstable\b",
        r"\bunchanged\b",
        r"\bno\s+significant\s+change\b",
        r"\bsimilar\b",
        r"\bpersistent\b",
    ]

    if any(re.search(p, t, flags=re.IGNORECASE) for p in improved_patterns):
        return "improved"
    if any(re.search(p, t, flags=re.IGNORECASE) for p in worsened_patterns):
        return "worsened"
    if any(re.search(p, t, flags=re.IGNORECASE) for p in stable_patterns):
        return "stable"
    return "unknown"


def _clean_value(value: str) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if raw == "":
        return ""

    normalized_text = _normalize_text(raw)
    status = _detect_status(normalized_text)
    severity = _detect_severity(normalized_text, status)
    trend = _detect_trend(normalized_text)

    return f"status={status}; severity={severity}; trend={trend}; text={normalized_text}"


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_clean_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COL}` and saved file to `{output_path}`.",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")