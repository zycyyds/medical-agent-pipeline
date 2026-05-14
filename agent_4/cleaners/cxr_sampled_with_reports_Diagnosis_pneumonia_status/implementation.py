import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "cxr_sampled_with_reports_Diagnosis_pneumonia_status"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _canonicalize_token(token: str) -> str:
    original = _normalize_space(token)
    if original == "":
        return ""

    t = original.lower()
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # Remove wrapping punctuation only
    t = re.sub(r"^[\[\(\{]+|[\]\)\}]+$", "", t).strip()

    # Strong negative / absent patterns
    negative_patterns = [
        r"^no pneumonia$",
        r"^without pneumonia$",
        r"^pneumonia absent$",
        r"^absence of pneumonia$",
        r"^absent$",
        r"^negative$",
        r"^not present$",
        r"^none identified$",
        r"^none seen$",
        r"^no acute pneumonia$",
    ]
    for p in negative_patterns:
        if re.fullmatch(p, t):
            return "absent"

    # Strong positive / present patterns
    positive_patterns = [
        r"^pneumonia$",
        r"^present$",
        r"^positive$",
        r"^consistent with pneumonia$",
        r"^compatible with pneumonia$",
        r"^evidence of pneumonia$",
        r"^findings of pneumonia$",
        r"^multifocal pneumonia$",
        r"^lobar pneumonia$",
        r"^bilateral pneumonia$",
        r"^left lower lobe pneumonia$",
        r"^right lower lobe pneumonia$",
    ]
    for p in positive_patterns:
        if re.fullmatch(p, t):
            return "present"

    # Possible / uncertain patterns
    possible_patterns = [
        r"^possible$",
        r"^probable$",
        r"^suspected$",
        r"^questionable$",
        r"^equivocal$",
        r"^indeterminate$",
        r"^uncertain$",
        r"^cannot exclude$",
        r"^cannot rule out$",
        r"^rule out$",
        r"^likely$",
        r"^concerning for pneumonia$",
        r"^possible pneumonia$",
        r"^probable pneumonia$",
        r"^suspected pneumonia$",
        r"^questionable pneumonia$",
        r"^equivocal pneumonia$",
        r"^indeterminate for pneumonia$",
        r"^cannot exclude pneumonia$",
        r"^cannot rule out pneumonia$",
        r"^rule out pneumonia$",
        r"^likely pneumonia$",
    ]
    for p in possible_patterns:
        if re.fullmatch(p, t):
            return "possible"

    # Resolving / improving patterns
    resolving_patterns = [
        r"^resolving$",
        r"^improving$",
        r"^improved$",
        r"^partial resolution$",
        r"^resolving pneumonia$",
        r"^improving pneumonia$",
        r"^improved pneumonia$",
    ]
    for p in resolving_patterns:
        if re.fullmatch(p, t):
            return "resolving"

    # Worsening / progression patterns
    worsening_patterns = [
        r"^worsening$",
        r"^worse$",
        r"^progressing$",
        r"^progressive$",
        r"^increasing$",
        r"^worsening pneumonia$",
        r"^progressing pneumonia$",
        r"^progressive pneumonia$",
    ]
    for p in worsening_patterns:
        if re.fullmatch(p, t):
            return "worsening"

    # Stable patterns
    stable_patterns = [
        r"^stable$",
        r"^unchanged$",
        r"^stable pneumonia$",
        r"^unchanged pneumonia$",
    ]
    for p in stable_patterns:
        if re.fullmatch(p, t):
            return "stable"

    # If phrase clearly contains pneumonia with a polarity cue, standardize conservatively
    if "pneumonia" in t:
        if re.search(r"\b(no|without|absent|negative)\b", t):
            return "absent"
        if re.search(r"\b(possible|probable|suspected|questionable|equivocal|indeterminate|likely)\b", t):
            return "possible"
        if re.search(r"\b(resolving|improving|improved)\b", t):
            return "resolving"
        if re.search(r"\b(worsening|worse|progressing|progressive|increasing)\b", t):
            return "worsening"
        if re.search(r"\b(stable|unchanged)\b", t):
            return "stable"
        if re.search(r"\b(present|positive|consistent with|compatible with|evidence of)\b", t):
            return "present"

    return original


def _clean_pneumonia_status(value: str) -> str:
    if value is None:
        return value

    raw = str(value)
    if raw == "":
        return raw

    text = raw.strip()
    if text == "":
        return ""

    # Normalize common delimiters while preserving multi-label order
    normalized = text
    normalized = normalized.replace("；", ";").replace("，", ",").replace("、", ",")
    normalized = normalized.replace("｜", "|").replace("/", "|").replace("\\", "|")
    normalized = re.sub(r"\s*(?:;|\||,|\band\b|\&|\+)\s*", "|", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\|{2,}", "|", normalized).strip("| ").strip()

    if normalized == "":
        return ""

    parts = [p for p in normalized.split("|")]
    cleaned_parts = []
    seen = set()

    for part in parts:
        token = _canonicalize_token(part)
        if token == "":
            continue
        key = token.lower()
        if key not in seen:
            cleaned_parts.append(token)
            seen.add(key)

    if not cleaned_parts:
        return ""

    return " | ".join(cleaned_parts)


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_clean_pneumonia_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned file saved to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")