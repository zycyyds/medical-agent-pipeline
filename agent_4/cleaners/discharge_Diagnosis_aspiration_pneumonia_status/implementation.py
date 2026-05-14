import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "discharge_Diagnosis_aspiration_pneumonia_status"


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_status(value: str) -> str:
    if value is None:
        return value

    original = value
    cleaned = _collapse_spaces(str(original))

    if cleaned == "":
        return ""

    lowered = cleaned.lower()

    unknown_patterns = {
        "unknown", "unk", "unclear", "undetermined", "not specified",
        "unspecified", "pending", "n/a", "na"
    }
    if lowered in unknown_patterns:
        return "unknown"

    history_patterns = {
        "history of aspiration pneumonia",
        "hx of aspiration pneumonia",
        "h/o aspiration pneumonia",
        "prior aspiration pneumonia",
        "previous aspiration pneumonia",
        "past aspiration pneumonia",
        "remote aspiration pneumonia",
    }
    if lowered in history_patterns:
        return "history_of"

    ruled_out_patterns = {
        "ruled out",
        "rule out aspiration pneumonia",
        "ruled out aspiration pneumonia",
        "exclude aspiration pneumonia",
        "excluded aspiration pneumonia",
        "aspiration pneumonia ruled out",
        "no evidence of aspiration pneumonia",
    }
    if lowered in ruled_out_patterns:
        return "ruled_out"

    suspected_patterns = {
        "suspected",
        "possible",
        "probable",
        "query aspiration pneumonia",
        "? aspiration pneumonia",
        "suspected aspiration pneumonia",
        "possible aspiration pneumonia",
        "probable aspiration pneumonia",
        "concern for aspiration pneumonia",
        "cannot rule out aspiration pneumonia",
    }
    if lowered in suspected_patterns:
        return "suspected"

    absent_patterns = {
        "absent",
        "negative",
        "no",
        "n",
        "none",
        "nil",
        "without aspiration pneumonia",
        "no aspiration pneumonia",
        "not present",
        "aspiration pneumonia absent",
        "aspiration pneumonia negative",
    }
    if lowered in absent_patterns:
        return "absent"

    present_patterns = {
        "present",
        "positive",
        "yes",
        "y",
        "diagnosed",
        "confirmed",
        "aspiration pneumonia",
        "with aspiration pneumonia",
        "has aspiration pneumonia",
        "aspiration pneumonia present",
        "aspiration pneumonia positive",
        "confirmed aspiration pneumonia",
        "diagnosis of aspiration pneumonia",
    }
    if lowered in present_patterns:
        return "present"

    if re.fullmatch(r"aspiration pneumonia[,;:/ -]*(present|positive|confirmed)", lowered):
        return "present"
    if re.fullmatch(r"(no|without|negative for)[,;:/ -]*aspiration pneumonia", lowered):
        return "absent"
    if re.fullmatch(r"(possible|probable|suspected)[,;:/ -]*aspiration pneumonia", lowered):
        return "suspected"
    if re.fullmatch(r"(history of|hx of|h/o|prior|previous)[,;:/ -]*aspiration pneumonia", lowered):
        return "history_of"

    return cleaned


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_normalize_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column: {TARGET_COLUMN}. Output saved to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")