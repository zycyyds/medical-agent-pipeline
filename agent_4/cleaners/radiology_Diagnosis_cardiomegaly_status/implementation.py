import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_cardiomegaly_status(value: str) -> str:
    if value is None:
        return value

    original = value
    text = str(value).strip()

    if text == "":
        return ""

    # 轻量规范化：压缩空白并标准化匹配键
    text = re.sub(r"\s+", " ", text)
    key = text.lower().strip()
    key = key.replace("-", " ")
    key = key.replace("_", " ")
    key = re.sub(r"\s+", " ", key)

    # 缺失类：保持为空
    missing_tokens = {
        "na", "n/a", "nan", "none", "null", "unknown", "not available", "missing"
    }
    if key in missing_tokens:
        return ""

    # 标准类别映射
    mapping = {
        # present
        "present": "present",
        "positive": "present",
        "yes": "present",
        "y": "present",
        "cardiomegaly": "present",
        "enlarged heart": "present",
        "heart enlarged": "present",

        # absent
        "absent": "absent",
        "negative": "absent",
        "no": "absent",
        "n": "absent",
        "none detected": "absent",
        "no cardiomegaly": "absent",
        "without cardiomegaly": "absent",

        # uncertain
        "uncertain": "uncertain",
        "equivocal": "uncertain",
        "indeterminate": "uncertain",
        "possible": "uncertain",
        "cannot exclude": "uncertain",
        "questionable": "uncertain",
        "suspected": "uncertain",
    }

    if key in mapping:
        return mapping[key]

    # 对常见短语做轻量规则匹配；不确定则保留原值
    if re.fullmatch(r"(no|negative|absent)( cardiomegaly)?", key):
        return "absent"

    if re.fullmatch(r"(yes|positive|present)( cardiomegaly)?", key):
        return "present"

    if "no cardiomegaly" in key or "without cardiomegaly" in key:
        return "absent"

    if "cardiomegaly" in key:
        if any(token in key for token in ["possible", "questionable", "cannot exclude", "equivocal", "indeterminate", "suspected"]):
            return "uncertain"
        return "present"

    if any(token in key for token in ["possible", "questionable", "cannot exclude", "equivocal", "indeterminate", "suspected"]):
        return "uncertain"

    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "radiology_Diagnosis_cardiomegaly_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_cardiomegaly_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        if output_path == input_path:
            output_path = input_path + "_cleaned.csv"

        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed: {os.path.basename(output_path)}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")