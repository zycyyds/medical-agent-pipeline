import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "discharge_Diagnosis_coronary_artery_disease_status"


def _normalize_text(value: str) -> str:
    if value is None:
        return value
    s = str(value)

    # 保留缺失
    if s == "":
        return s

    original = s

    # 轻量标准化：去首尾空格、统一空白和常见全角标点
    s = s.strip()
    if s == "":
        return ""

    s = s.replace("\u3000", " ")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("：", ":").replace("，", ",").replace("；", ";")
    s = re.sub(r"\s+", " ", s)

    low = s.lower().strip()

    # 常见未知/不详表达
    if low in {
        "unknown", "unk", "unclear", "not clear", "n/a", "na", "none stated",
        "not sure", "unspecified", "undetermined"
    } or s in {"未知", "不详", "未说明", "不明确"}:
        return "unknown"

    # 否定表达优先判断，避免被“冠心病/cad/chd”误归类
    negative_patterns = [
        r"^(no|none|negative)\b.*\b(cad|chd|coronary artery disease|coronary heart disease|ischemic heart disease)\b",
        r"^(without|deny|denies|denied)\b.*\b(cad|chd|coronary artery disease|coronary heart disease|ischemic heart disease)\b",
        r"\bno evidence of\b.*\b(cad|chd|coronary artery disease|coronary heart disease|ischemic heart disease)\b",
        r"\bnon[- ]?cad\b",
    ]
    for p in negative_patterns:
        if re.search(p, low):
            return "no coronary artery disease"

    chinese_negative_patterns = [
        r"无冠心病",
        r"否认冠心病",
        r"未见冠心病",
        r"排除冠心病",
        r"非冠心病",
        r"冠心病否定",
    ]
    for p in chinese_negative_patterns:
        if re.search(p, s):
            return "no coronary artery disease"

    # 可疑/疑似表达
    suspected_patterns = [
        r"^(suspected|possible|probable|query)\b.*\b(cad|chd|coronary artery disease|coronary heart disease|ischemic heart disease)\b",
        r"^\?+\s*(cad|chd|coronary artery disease|coronary heart disease|ischemic heart disease)\b",
    ]
    for p in suspected_patterns:
        if re.search(p, low):
            return "suspected coronary artery disease"

    chinese_suspected_patterns = [
        r"疑似冠心病",
        r"可疑冠心病",
        r"考虑冠心病",
        r"待排冠心病",
    ]
    for p in chinese_suspected_patterns:
        if re.search(p, s):
            return "suspected coronary artery disease"

    # 明确冠心病/同义表达
    positive_patterns = [
        r"^(cad|chd|coronary artery disease|coronary heart disease|ischemic heart disease)$",
        r"\b(cad|chd|coronary artery disease|coronary heart disease|ischemic heart disease)\b",
    ]
    for p in positive_patterns:
        if re.search(p, low):
            return "coronary artery disease"

    chinese_positive_patterns = [
        r"^冠心病$",
        r"冠状动脉粥样硬化性心脏病",
        r"冠状动脉性心脏病",
    ]
    for p in chinese_positive_patterns:
        if re.search(p, s):
            return "coronary artery disease"

    # 仅做轻量清理后的原样保留
    return s if s != "" else original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_normalize_text)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COL}` and saved cleaned file to `{output_path}`.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")