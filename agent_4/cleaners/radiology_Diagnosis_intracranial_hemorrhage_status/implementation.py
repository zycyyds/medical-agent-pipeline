import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_intracranial_hemorrhage_status(value: str) -> str:
    if value is None:
        return value

    original = value
    text = str(value).strip()
    if text == "":
        return text

    text = re.sub(r"\s+", " ", text)
    lowered = text.lower()
    lowered = lowered.replace("-", " ").replace("/", " ").replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered).strip()

    if lowered in {
        "confirmed",
        "present",
        "positive",
        "yes",
        "true",
        "intracranial hemorrhage",
        "hemorrhage present",
        "haemorrhage present",
        "ich present",
        "acute intracranial hemorrhage",
        "acute intracranial haemorrhage",
    }:
        return "confirmed"

    if lowered in {
        "ruled out",
        "rule out",
        "ruledout",
        "negative",
        "absent",
        "not present",
        "no",
        "false",
        "no intracranial hemorrhage",
        "no intracranial haemorrhage",
        "without intracranial hemorrhage",
        "without intracranial haemorrhage",
        "intracranial hemorrhage absent",
        "intracranial haemorrhage absent",
        "no evidence of intracranial hemorrhage",
        "no evidence of intracranial haemorrhage",
    }:
        return "ruled_out"

    if lowered in {
        "suspected",
        "possible",
        "probable",
        "query",
        "question of",
        "cannot exclude",
        "not excluded",
        "concerning for",
    }:
        return "suspected"

    if lowered in {
        "indeterminate",
        "equivocal",
        "uncertain",
        "inconclusive",
    }:
        return "indeterminate"

    if lowered in {
        "unknown",
        "unk",
        "undetermined",
        "unable to determine",
        "not specified",
    }:
        return "unknown"

    canonical_forms = {
        "confirmed": "confirmed",
        "ruled out": "ruled_out",
        "suspected": "suspected",
        "indeterminate": "indeterminate",
        "unknown": "unknown",
    }
    if lowered in canonical_forms:
        return canonical_forms[lowered]

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "radiology_Diagnosis_intracranial_hemorrhage_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_intracranial_hemorrhage_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        if output_path == input_path:
            base, ext = os.path.splitext(input_path)
            output_path = base + "_cleaned" + ext

        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned CSV saved to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")