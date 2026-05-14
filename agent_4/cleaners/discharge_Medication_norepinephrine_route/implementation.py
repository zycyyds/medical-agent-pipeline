import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_route_value(value: str) -> str:
    if value is None:
        return value

    original = value
    v = str(value).strip()

    if v == "":
        return v

    # Normalize whitespace and separators lightly
    v = re.sub(r"\s+", " ", v)
    v_std = v.lower().strip()
    v_std = re.sub(r"[\-_]+", " ", v_std)
    v_std = re.sub(r"\s+", " ", v_std)

    # Safe standardization for clearly equivalent IV route expressions
    iv_patterns = [
        r"^iv$",
        r"^i\.?v\.?$",
        r"^intravenous$",
        r"^intravenous route$",
        r"^iv drip$",
        r"^iv gtt$",
        r"^iv infusion$",
        r"^intravenous drip$",
        r"^intravenous gtt$",
        r"^intravenous infusion$",
    ]
    for pattern in iv_patterns:
        if re.fullmatch(pattern, v_std):
            return "IV"

    # Conservative normalization for common route labels without inventing categories
    exact_map = {
        "im": "IM",
        "i.m.": "IM",
        "intramuscular": "IM",
        "sc": "SC",
        "s.c.": "SC",
        "subcutaneous": "SC",
        "sq": "SC",
        "po": "PO",
        "p.o.": "PO",
        "oral": "PO",
    }
    if v_std in exact_map:
        return exact_map[v_std]

    # If no safe mapping, return trimmed/lightly normalized original text
    return v


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        target_col = "discharge_Medication_norepinephrine_route"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_route_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column: {target_col}" if target_col in df.columns else f"Column not found: {target_col}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")