import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "cxr_sampled_with_reports_Diagnosis_cardiomegaly_status"


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.strip()
    if text == "":
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = text.replace("cardiomegally", "cardiomegaly")
    text = re.sub(r"\s*([,;:/\-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonicalize_cardiomegaly_status(value: str) -> str:
    original = value
    normalized = _normalize_text(value)

    if normalized == "":
        return original

    exact_map = {
        "cardiomegaly": "cardiomegaly",
        "cardiomegaly present": "cardiomegaly",
        "enlarged cardiac silhouette": "cardiomegaly",
        "enlarged cardiomediastinal silhouette": "cardiomegaly",
        "cardiac silhouette enlarged": "cardiomegaly",
        "heart size enlarged": "cardiomegaly",
        "cardiac enlargement": "cardiomegaly",
        "enlargement of the cardiac silhouette": "cardiomegaly",
        "enlargement of cardiac silhouette": "cardiomegaly",
        "enlargement of the cardiomediastinal silhouette": "cardiomegaly",
        "enlargement of cardiomediastinal silhouette": "cardiomegaly",

        "no cardiomegaly": "no cardiomegaly",
        "without cardiomegaly": "no cardiomegaly",
        "negative for cardiomegaly": "no cardiomegaly",
        "cardiomegaly absent": "no cardiomegaly",
        "absent": "no cardiomegaly",
        "none": "no cardiomegaly",
        "normal heart size": "no cardiomegaly",
        "heart size normal": "no cardiomegaly",
        "cardiac silhouette within normal limits": "no cardiomegaly",
        "cardiomediastinal silhouette within normal limits": "no cardiomegaly",
        "cardiomediastinal silhouette is within normal limits": "no cardiomegaly",
        "cardiac silhouette is within normal limits": "no cardiomegaly",

        "borderline enlarged cardiac silhouette": "borderline cardiomegaly",
        "borderline enlarged cardiomediastinal silhouette": "borderline cardiomegaly",
        "borderline cardiac enlargement": "borderline cardiomegaly",
        "borderline cardiomegaly": "borderline cardiomegaly",

        "stable cardiomegaly": "stable cardiomegaly",
        "unchanged cardiomegaly": "stable cardiomegaly",
        "persistent cardiomegaly": "stable cardiomegaly",

        "possible cardiomegaly": "possible cardiomegaly",
        "suspected cardiomegaly": "possible cardiomegaly",
        "questionable cardiomegaly": "possible cardiomegaly",
        "cannot exclude cardiomegaly": "possible cardiomegaly",
    }

    if normalized in exact_map:
        return exact_map[normalized]

    severity_patterns = [
        (r"^(mild|mildly|slight|slightly|minimal|minimally)\s+(cardiomegaly|cardiac enlargement|heart enlargement)$", "mild cardiomegaly"),
        (r"^(mild|mildly|slight|slightly|minimal|minimally)\s+(enlarged|enlargement of)\s+(the )?(cardiac|cardiomediastinal)\s+silhouette$", "mild cardiomegaly"),
        (r"^moderate\s+(cardiomegaly|cardiac enlargement|heart enlargement)$", "moderate cardiomegaly"),
        (r"^moderate\s+(enlarged|enlargement of)\s+(the )?(cardiac|cardiomediastinal)\s+silhouette$", "moderate cardiomegaly"),
        (r"^(marked|severe)\s+(cardiomegaly|cardiac enlargement|heart enlargement)$", "severe cardiomegaly"),
        (r"^(marked|severe)\s+(enlarged|enlargement of)\s+(the )?(cardiac|cardiomediastinal)\s+silhouette$", "severe cardiomegaly"),
    ]
    for pattern, replacement in severity_patterns:
        if re.match(pattern, normalized):
            return replacement

    if re.match(r"^(stable|unchanged|persistent)\s+(mild|moderate|severe|marked|borderline)\s+cardiomegaly$", normalized):
        normalized = normalized.replace("unchanged", "stable").replace("persistent", "stable").replace("marked", "severe")
        return normalized

    if re.match(r"^(possible|suspected|questionable)\s+(mild|moderate|severe|borderline)?\s*cardiomegaly$", normalized):
        normalized = normalized.replace("suspected", "possible").replace("questionable", "possible")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    return normalized


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_canonicalize_cardiomegaly_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COLUMN}` and saved file to `{output_path}`.",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")