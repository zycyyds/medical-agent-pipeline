import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\[\]\(\)\{\},;:|]+", " ", text)
    text = re.sub(r"[/_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_modifier(text: str):
    modifiers = []

    if re.search(r"\b(no change|unchanged|stable|similar|same)\b", text):
        modifiers.append("stable")

    if re.search(r"\b(improved|improvement|decreased|decrease|reduced|less)\b", text):
        modifiers.append("improved")

    if re.search(r"\b(worsened|worsening|worse|increased|increase|progressed|progressive)\b", text):
        modifiers.append("worsened")

    if re.search(r"\bnew\b", text):
        modifiers.append("new")

    ordered = []
    for item in ["stable", "improved", "worsened", "new"]:
        if item in modifiers:
            ordered.append(item)
    return ordered


def _extract_base_status(text: str):
    absent_patterns = [
        r"\bno cardiomegaly\b",
        r"\bwithout cardiomegaly\b",
        r"\bcardiomegaly absent\b",
        r"\babsence of cardiomegaly\b",
        r"\bnegative for cardiomegaly\b",
        r"\bheart size (?:is )?normal\b",
        r"\bcardiac silhouette (?:is )?normal\b",
        r"\bcardiomediastinal silhouette (?:is )?normal\b",
        r"\bno cardiac enlargement\b",
    ]
    for pat in absent_patterns:
        if re.search(pat, text):
            return "absent"

    if re.search(r"\b(severe|marked|pronounced)\b", text) and re.search(
        r"\b(cardiomegaly|cardiac enlargement|enlarged heart|enlarged cardiac silhouette)\b", text
    ):
        return "severe"

    if re.search(r"\bmoderate(?:ly)?\b", text) and re.search(
        r"\b(cardiomegaly|cardiac enlargement|enlarged heart|enlarged cardiac silhouette)\b", text
    ):
        return "moderate"

    if re.search(r"\b(mild|mildly|slight|minimal|borderline)\b", text) and re.search(
        r"\b(cardiomegaly|cardiac enlargement|enlarged heart|enlarged cardiac silhouette)\b", text
    ):
        return "mild"

    if re.search(r"\b(cardiomegaly|cardiac enlargement|enlarged heart|enlarged cardiac silhouette)\b", text):
        return "present"

    if text in {"absent", "present", "mild", "moderate", "severe"}:
        return text

    return None


def _clean_cardiomegaly_value(value: str) -> str:
    original = value
    if value is None:
        return value

    raw = str(value)
    if raw.strip() == "":
        return raw

    text = _normalize_text(raw)
    if text == "":
        return raw

    base = _extract_base_status(text)
    modifiers = _extract_modifier(text)

    if base is None:
        return original

    if modifiers:
        return base + " " + " ".join(modifiers)
    return base


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "radiology_Finding_cardiomegaly_value"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_cardiomegaly_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned CSV saved to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")