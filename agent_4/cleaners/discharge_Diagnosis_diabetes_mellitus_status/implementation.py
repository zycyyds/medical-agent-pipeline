import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_diabetes_status(value: str) -> str:
    if value is None:
        return value

    text = str(value)
    stripped = text.strip()
    if stripped == "":
        return ""

    # 轻量空白规范化
    normalized_space = re.sub(r"\s+", " ", stripped)
    lower = normalized_space.lower()

    # 常见缺失/未知表达
    if lower in {
        "unknown", "unk", "unclear", "not clear", "n/a", "na", "none",
        "not documented", "not mentioned", "unspecified"
    }:
        return "unknown"

    # 统一常见糖尿病状态表达
    mapping_patterns = [
        (r"^(dm|diabetes|diabetes mellitus)$", "diabetes mellitus"),
        (r"^(type[\s\-]*1|t1dm|dm1|iddm|insulin[\s\-]*dependent diabetes mellitus)$", "type 1 diabetes mellitus"),
        (r"^(type[\s\-]*2|t2dm|dm2|niddm|non[\s\-]*insulin[\s\-]*dependent diabetes mellitus)$", "type 2 diabetes mellitus"),
        (r"^(gestational diabetes|gdm)$", "gestational diabetes mellitus"),
        (r"^(prediabetes|pre[\s\-]*diabetes|impaired glucose tolerance|igt)$", "prediabetes"),
        (r"^(new[\s\-]*onset diabetes|newly diagnosed diabetes)$", "new onset diabetes mellitus"),
        (r"^(diabetes mellitus uncontrolled|uncontrolled diabetes|poorly controlled diabetes)$", "uncontrolled diabetes mellitus"),
        (r"^(diabetes mellitus controlled|controlled diabetes|well controlled diabetes)$", "controlled diabetes mellitus"),
        (r"^(diabetes with complications|complicated diabetes)$", "diabetes mellitus with complications"),
        (r"^(diabetes without complications|uncomplicated diabetes)$", "diabetes mellitus without complications"),
        (r"^(history of diabetes|hx of diabetes|h/o diabetes)$", "history of diabetes mellitus"),
        (r"^(no diabetes|non diabetic|non-diabetic|without diabetes|diabetes negative)$", "no diabetes"),
    ]

    for pattern, replacement in mapping_patterns:
        if re.fullmatch(pattern, lower):
            return replacement

    # 对包含型但较明确的表达做轻量规范
    if "type 1" in lower or "t1dm" in lower or "iddm" in lower:
        return "type 1 diabetes mellitus"
    if "type 2" in lower or "t2dm" in lower or "niddm" in lower:
        return "type 2 diabetes mellitus"
    if "gestational" in lower or lower == "gdm":
        return "gestational diabetes mellitus"
    if "prediabetes" in lower or "pre-diabetes" in lower:
        return "prediabetes"

    # 若原值全为大小写差异，返回小写规范短语；否则保留原值的基本写法
    if lower in {"diabetes mellitus", "diabetes"}:
        return "diabetes mellitus"

    return normalized_space


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        if not os.path.exists(input_path):
            return ToolResponse(content="Input file does not exist.")

        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        target_col = "discharge_Diagnosis_diabetes_mellitus_status"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_diabetes_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content="Cleaning completed successfully.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")