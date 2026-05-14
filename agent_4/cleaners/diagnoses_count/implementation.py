import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


_FULLWIDTH_TRANS = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "＋": "+", "－": "-", "．": ".", "，": ",",
    "　": " ", "％": "%"
})


def _clean_diagnoses_count(value: str) -> str:
    if value is None:
        return value

    original = value
    s = str(value)

    if s == "":
        return s

    # 轻量去除首尾空白并统一全角字符
    cleaned = s.strip().translate(_FULLWIDTH_TRANS)

    # 去除成对包裹引号
    if len(cleaned) >= 2 and (
        (cleaned[0] == '"' and cleaned[-1] == '"') or
        (cleaned[0] == "'" and cleaned[-1] == "'")
    ):
        inner = cleaned[1:-1].strip()
        cleaned = inner.translate(_FULLWIDTH_TRANS)

    # 对明确的千分位数字去掉逗号，如 1,234 / -1,234.5
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", cleaned):
        cleaned = cleaned.replace(",", "")
        return cleaned

    # 对明确数值样式，仅返回规范化后的轻量结果（主要是去空白、全角转半角）
    if re.fullmatch(r"[+-]?(?:\d+(\.\d+)?|\.\d+)", cleaned):
        return cleaned

    # 不确定内容时保留原值
    return original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if "diagnoses_count" in df.columns:
            df["diagnoses_count"] = df["diagnoses_count"].apply(_clean_diagnoses_count)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned file saved to {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")