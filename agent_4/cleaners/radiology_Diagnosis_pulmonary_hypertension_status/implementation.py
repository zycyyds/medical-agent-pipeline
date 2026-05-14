import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "radiology_Diagnosis_pulmonary_hypertension_status"


def _normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,;:/()])\s*", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_ph_status(value: str) -> str:
    if value is None:
        return value

    original = value
    text = str(value)

    if text == "":
        return text

    text = _normalize_whitespace(text)
    lowered = text.lower()

    # Normalize common abbreviations and synonymous disease mentions
    normalized = lowered

    # Standardize pulmonary hypertension mentions
    normalized = re.sub(r"\bphtn\b", "pulmonary hypertension", normalized)
    normalized = re.sub(r"\bph\b", "pulmonary hypertension", normalized)
    normalized = re.sub(r"\bpulmonary\s+htn\b", "pulmonary hypertension", normalized)
    normalized = re.sub(r"\bpulm\s+htn\b", "pulmonary hypertension", normalized)
    normalized = re.sub(r"\bpulmonary\s+arterial\s+hypertension\b", "pulmonary hypertension", normalized)
    normalized = re.sub(r"\bpa[h]?\b", "pulmonary hypertension", normalized)
    normalized = re.sub(r"\bpulmonary\s+artery\s+hypertension\b", "pulmonary hypertension", normalized)

    # Normalize common qualifiers while preserving uncertainty/history/negation
    normalized = re.sub(r"\bh\/o\b", "history of", normalized)
    normalized = re.sub(r"\bhx of\b", "history of", normalized)
    normalized = re.sub(r"\bhistory\s+for\b", "history of", normalized)
    normalized = re.sub(r"\bquestion\s+of\b", "possible", normalized)
    normalized = re.sub(r"\bc\/w\b", "consistent with", normalized)
    normalized = re.sub(r"\bs\/o\b", "suggestive of", normalized)
    normalized = re.sub(r"\bprobable\b", "possible", normalized)
    normalized = re.sub(r"\blikely\b", "possible", normalized)
    normalized = re.sub(r"\bcannot exclude\b", "possible", normalized)
    normalized = re.sub(r"\bno\s+definite\b", "no", normalized)
    normalized = re.sub(r"\bwithout evidence of\b", "no evidence of", normalized)
    normalized = re.sub(r"\bnegative for\b", "no evidence of", normalized)
    normalized = re.sub(r"\babsent\b", "no evidence of", normalized)

    normalized = _normalize_whitespace(normalized)

    # Canonical mappings for common single-status values
    canonical_map = {
        "pulmonary hypertension": "pulmonary hypertension",
        "possible pulmonary hypertension": "possible pulmonary hypertension",
        "suspected pulmonary hypertension": "suspected pulmonary hypertension",
        "suggestive of pulmonary hypertension": "suggestive of pulmonary hypertension",
        "consistent with pulmonary hypertension": "consistent with pulmonary hypertension",
        "history of pulmonary hypertension": "history of pulmonary hypertension",
        "no pulmonary hypertension": "no pulmonary hypertension",
        "no evidence of pulmonary hypertension": "no evidence of pulmonary hypertension",
    }

    if normalized in canonical_map:
        return canonical_map[normalized]

    # Pattern-based normalization preserving status semantics
    patterns = [
        (r"^history of pulmonary hypertension$", "history of pulmonary hypertension"),
        (r"^possible pulmonary hypertension$", "possible pulmonary hypertension"),
        (r"^suspected pulmonary hypertension$", "suspected pulmonary hypertension"),
        (r"^suggestive of pulmonary hypertension$", "suggestive of pulmonary hypertension"),
        (r"^consistent with pulmonary hypertension$", "consistent with pulmonary hypertension"),
        (r"^no evidence of pulmonary hypertension$", "no evidence of pulmonary hypertension"),
        (r"^no pulmonary hypertension$", "no pulmonary hypertension"),
    ]
    for pattern, replacement in patterns:
        if re.fullmatch(pattern, normalized):
            return replacement

    # If phrase contains pulmonary hypertension and a clear qualifier, standardize lightly
    if "pulmonary hypertension" in normalized:
        if re.search(r"\bhistory of\b", normalized):
            return "history of pulmonary hypertension"
        if re.search(r"\b(suspected|suspicious for)\b", normalized):
            return "suspected pulmonary hypertension"
        if re.search(r"\b(possible|may represent|cannot rule out)\b", normalized):
            return "possible pulmonary hypertension"
        if re.search(r"\b(suggestive of)\b", normalized):
            return "suggestive of pulmonary hypertension"
        if re.search(r"\b(consistent with)\b", normalized):
            return "consistent with pulmonary hypertension"
        if re.search(r"\b(no evidence of|no|without)\b", normalized):
            return "no evidence of pulmonary hypertension"
        return "pulmonary hypertension"

    # Preserve original semantics when uncertain; just return normalized text
    if normalized != "":
        return normalized

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_clean_ph_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COLUMN}` and saved output to `{output_path}`.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")