import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


# 仅针对常见 ICU 护理单元做轻量、保守映射
STANDARD_UNIT_MAP = {
    "ICU": "ICU",
    "INTENSIVE CARE UNIT": "ICU",
    "CCU": "CCU",
    "CORONARY CARE UNIT": "CCU",
    "CARDIAC CARE UNIT": "CCU",
    "CSRU": "CSRU",
    "CARDIAC SURGERY RECOVERY UNIT": "CSRU",
    "CVICU": "CVICU",
    "CARDIOVASCULAR ICU": "CVICU",
    "CARDIOVASCULAR INTENSIVE CARE UNIT": "CVICU",
    "MICU": "MICU",
    "MEDICAL ICU": "MICU",
    "MEDICAL INTENSIVE CARE UNIT": "MICU",
    "SICU": "SICU",
    "SURGICAL ICU": "SICU",
    "SURGICAL INTENSIVE CARE UNIT": "SICU",
    "TSICU": "TSICU",
    "TRAUMA SICU": "TSICU",
    "TRAUMA SURGICAL ICU": "TSICU",
    "TRAUMA SURGICAL INTENSIVE CARE UNIT": "TSICU",
    "NICU": "NICU",
    "NEONATAL ICU": "NICU",
    "NEONATAL INTENSIVE CARE UNIT": "NICU",
    "PICU": "PICU",
    "PEDIATRIC ICU": "PICU",
    "PEDIATRIC INTENSIVE CARE UNIT": "PICU",
    "NEURO ICU": "Neuro ICU",
    "NEUROICU": "Neuro ICU",
    "NEURO INTENSIVE CARE UNIT": "Neuro ICU",
    "NSICU": "Neuro ICU",
    "NEUROSURGICAL ICU": "Neuro ICU",
    "BURN ICU": "Burn ICU",
    "BURN INTENSIVE CARE UNIT": "Burn ICU",
    "CTICU": "CTICU",
    "CARDIOTHORACIC ICU": "CTICU",
    "CARDIOTHORACIC INTENSIVE CARE UNIT": "CTICU",
}


def _normalize_delimiters(text: str) -> str:
    text = re.sub(r"[，、；;|/]+", ",", text)
    text = re.sub(r"\s*,\s*", ", ", text.strip())
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _canonical_key(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("-", " ")
    text = text.upper()
    return text


def _format_unknown_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"\s+", " ", token)
    # 对明显的 ICU 类缩写做大写规范，其余保留原貌
    upper_token = token.upper()
    if re.fullmatch(r"[A-Z]{2,10}", upper_token):
        return upper_token
    return token


def _standardize_token(token: str) -> str:
    original = token.strip()
    if original == "":
        return ""

    compact = re.sub(r"\s+", " ", original)
    key = _canonical_key(compact)

    if key in STANDARD_UNIT_MAP:
        return STANDARD_UNIT_MAP[key]

    # 处理形如 "Medical ICU (MICU)" / "MICU (Medical ICU)"，保留括号缩写
    m = re.fullmatch(r"(.+?)\s*\(([^()]+)\)", compact)
    if m:
        left = re.sub(r"\s+", " ", m.group(1).strip())
        inner = re.sub(r"\s+", " ", m.group(2).strip())

        left_key = _canonical_key(left)
        inner_key = _canonical_key(inner)

        left_std = STANDARD_UNIT_MAP.get(left_key)
        inner_std = STANDARD_UNIT_MAP.get(inner_key)

        if left_std and inner_std:
            if left_std == inner_std:
                return f"{left_std} ({inner.strip()})"
            return f"{left_std} ({inner.strip()})"

        if left_std:
            return f"{left_std} ({inner.strip()})"

        if inner_std:
            return f"{left.strip()} ({inner_std})"

        return _format_unknown_token(compact)

    return _format_unknown_token(compact)


def _clean_icu_careunit_cell(value: str) -> str:
    if value is None:
        return value
    if not isinstance(value, str):
        value = str(value)

    raw = value.strip()
    if raw == "":
        return value

    normalized = _normalize_delimiters(raw)
    parts = [p.strip() for p in normalized.split(",")]

    cleaned_parts = []
    for part in parts:
        if part == "":
            continue
        cleaned = _standardize_token(part)
        if cleaned != "":
            cleaned_parts.append(cleaned)

    if not cleaned_parts:
        return raw

    return ", ".join(cleaned_parts)


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        if "icu_careunit" in df.columns:
            df["icu_careunit"] = df["icu_careunit"].apply(_clean_icu_careunit_cell)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Successfully cleaned `icu_careunit` and saved to {os.path.basename(output_path)}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")