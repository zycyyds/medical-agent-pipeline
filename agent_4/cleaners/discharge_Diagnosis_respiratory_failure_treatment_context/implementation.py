import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "discharge_Diagnosis_respiratory_failure_treatment_context"


def _normalize_text(value):
    if value is None:
        return value

    text = str(value)

    if text == "":
        return text

    # 基础空白规范化
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 轻量标点规范化
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("：", ":").replace("；", ";")
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 常见呼吸/通气相关缩写大小写规范化
    replacements = [
        (r"\bhfnc\b", "HFNC"),
        (r"\bnc\b", "NC"),
        (r"\bniv\b", "NIV"),
        (r"\bnippv\b", "NIPPV"),
        (r"\bbipap\b", "BiPAP"),
        (r"\bcpap\b", "CPAP"),
        (r"\bimv\b", "IMV"),
        (r"\bsimv\b", "SIMV"),
        (r"\bpsv\b", "PSV"),
        (r"\bpeep\b", "PEEP"),
        (r"\becmo\b", "ECMO"),
        (r"\bicu\b", "ICU"),
        (r"\bards\b", "ARDS"),
        (r"\bcopd\b", "COPD"),
        (r"\bspo2\b", "SpO2"),
        (r"\bpao2\b", "PaO2"),
        (r"\bpaco2\b", "PaCO2"),
        (r"\bfio2\b", "FiO2"),
        (r"\bo2\b", "O2"),
        (r"\bph\b", "pH"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 轻量规范常见罗马数字分型写法
    text = re.sub(r"\btype\s*i\b", "Type I", text, flags=re.IGNORECASE)
    text = re.sub(r"\btype\s*ii\b", "Type II", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\s*型呼吸衰竭\b", "I型呼吸衰竭", text, flags=re.IGNORECASE)
    text = re.sub(r"\bii\s*型呼吸衰竭\b", "II型呼吸衰竭", text, flags=re.IGNORECASE)

    # 去除规范化后可能产生的多余空白
    text = re.sub(r"\s+", " ", text).strip()

    return text


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(_normalize_text)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column: {TARGET_COLUMN}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")