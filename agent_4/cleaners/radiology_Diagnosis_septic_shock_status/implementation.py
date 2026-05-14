import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return value

    text = str(value)
    original = text

    if text == "":
        return text

    text = text.strip()
    if text == "":
        return text

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    # 轻量标点规范化
    text = text.replace("_", " ")
    text = re.sub(r"[;|]+", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")

    # 常见同义表达标准化：仅在语义明确时替换
    replacements = [
        (r"\bseptic[- ]shock\b", "septic shock"),
        (r"\bshock[, ]*septic\b", "septic shock"),
        (r"\bsepsis with septic shock\b", "sepsis with septic shock"),
        (r"\bsevere sepsis with septic shock\b", "severe sepsis with septic shock"),
        (r"\bhx of septic shock\b", "history of septic shock"),
        (r"\bh/o septic shock\b", "history of septic shock"),
        (r"\bhistory septic shock\b", "history of septic shock"),
        (r"\bno septic shock\b", "no septic shock"),
        (r"\bwithout septic shock\b", "without septic shock"),
        (r"\bnegative for septic shock\b", "negative for septic shock"),
        (r"\br/o septic shock\b", "rule out septic shock"),
        (r"\bquery septic shock\b", "query septic shock"),
        (r"\bpossible septic shock\b", "possible septic shock"),
        (r"\bsuspected septic shock\b", "suspected septic shock"),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # 若整格仅为极常见单一状态表达，则进一步统一
    exact_map = {
        "septic shock": "septic shock",
        "history of septic shock": "history of septic shock",
        "hx septic shock": "history of septic shock",
        "h/o septic shock": "history of septic shock",
        "no septic shock": "no septic shock",
        "without septic shock": "without septic shock",
        "negative for septic shock": "negative for septic shock",
        "rule out septic shock": "rule out septic shock",
        "query septic shock": "query septic shock",
        "possible septic shock": "possible septic shock",
        "suspected septic shock": "suspected septic shock",
        "sepsis with septic shock": "sepsis with septic shock",
        "severe sepsis with septic shock": "severe sepsis with septic shock",
    }
    if text in exact_map:
        return exact_map[text]

    # 对包含性文本仅做轻量规范化，保留原有语义
    return text if text != "" else original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        target_col = "radiology_Diagnosis_septic_shock_status"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_text)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content="Cleaning completed successfully.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")