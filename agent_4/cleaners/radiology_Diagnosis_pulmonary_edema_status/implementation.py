import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[-]+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n.;,:")
    return text


def _clean_pulmonary_edema_status(value):
    if pd.isna(value):
        return value

    original = str(value)
    if original.strip() == "":
        return value

    norm = _normalize_text(original)

    mapping = {
        # absent
        "absent": "absent",
        "no pulmonary edema": "absent",
        "without pulmonary edema": "absent",
        "pulmonary edema absent": "absent",
        "negative for pulmonary edema": "absent",
        "edema absent": "absent",
        "no edema": "absent",
        "pulmonary edema not seen": "absent",
        "no evidence of pulmonary edema": "absent",

        # present / confirmed
        "present": "present",
        "pulmonary edema": "present",
        "pulmonary edema present": "present",
        "edema present": "present",
        "positive for pulmonary edema": "present",
        "findings consistent with pulmonary edema": "present",

        # suspected / possible
        "suspected": "suspected",
        "suspected pulmonary edema": "suspected",
        "possible pulmonary edema": "suspected",
        "probable pulmonary edema": "suspected",
        "questionable pulmonary edema": "suspected",
        "cannot exclude pulmonary edema": "suspected",
        "concern for pulmonary edema": "suspected",

        # severity
        "trace": "trace",
        "trace pulmonary edema": "trace",
        "mild": "mild",
        "mild pulmonary edema": "mild",
        "minimal pulmonary edema": "mild",
        "moderate": "moderate",
        "moderate pulmonary edema": "moderate",
        "severe": "severe",
        "severe pulmonary edema": "severe",
        "mild/moderate": "mild to moderate",
        "mild to moderate": "mild to moderate",
        "mild to moderate pulmonary edema": "mild to moderate",
        "moderate/severe": "moderate to severe",
        "moderate to severe": "moderate to severe",
        "moderate to severe pulmonary edema": "moderate to severe",
    }

    if norm in mapping:
        return mapping[norm]

    return norm


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "radiology_Diagnosis_pulmonary_edema_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_pulmonary_edema_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content="Cleaning completed successfully.",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")