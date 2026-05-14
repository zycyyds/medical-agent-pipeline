import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _clean_intubation_result(value: str) -> str:
    if value is None:
        return value

    original = value
    s = str(value).strip()
    if s == "":
        return s

    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()

    lower = s.lower()

    # 轻量标准化常见缩写/表达
    lower = re.sub(r"\bet\s*tube\b", "ett", lower)
    lower = re.sub(r"\bendotracheal tube\b", "ett", lower)
    lower = re.sub(r"\bendotracheal intubation\b", "intubation", lower)
    lower = re.sub(r"\bintubated successfully\b", "successful", lower)
    lower = re.sub(r"\bsucces+s?ful+\b", "successful", lower)
    lower = re.sub(r"\bfailed\b", "failure", lower)
    lower = re.sub(r"\bunsuccessful\b", "failure", lower)
    lower = re.sub(r"\bdifficult(y)?\b", "difficult", lower)
    lower = re.sub(r"\bunable to intubate\b", "failure", lower)
    lower = re.sub(r"\bnot attempted\b", "not attempted", lower)
    lower = re.sub(r"\bno attempt\b", "not attempted", lower)
    lower = re.sub(r"\bself extubat(ed|ion)\b", "self extubation", lower)
    lower = re.sub(r"\baccidental extubat(ed|ion)\b", "accidental extubation", lower)
    lower = re.sub(r"[;|]+", ",", lower)
    lower = re.sub(r"\s+", " ", lower).strip(" ,;:/-")

    # 缺失样式统一为空
    if lower in {"na", "n/a", "none", "null", "nil", "unknown", "unk"}:
        return ""

    # 明确的未尝试/未插管
    if lower in {"not attempted", "no intubation", "not intubated", "extubated"}:
        mapping = {
            "not attempted": "未尝试",
            "no intubation": "未插管",
            "not intubated": "未插管",
            "extubated": "已拔管",
        }
        return mapping.get(lower, original)

    # 明确单值
    if lower in {"successful", "success"}:
        return "成功"
    if lower in {"failure", "fail"}:
        return "失败"
    if lower in {"difficult"}:
        return "困难"

    # 带原因的失败/困难优先保留临床信息
    if "failure" in lower:
        if "difficult airway" in lower:
            return "失败-困难气道"
        if "multiple attempt" in lower or "multiple attempts" in lower:
            return "失败-多次尝试"
        if "desaturation" in lower or "hypoxia" in lower:
            return "失败-氧合下降"
        if "swelling" in lower or "edema" in lower:
            return "失败-水肿"
        if "secretions" in lower or "blood" in lower or "vomit" in lower:
            return "失败-分泌物/血液影响"
        return "失败"

    if "difficult" in lower:
        if "but successful" in lower or "successful" in lower or "success" in lower:
            return "困难但成功"
        if "airway" in lower:
            return "困难气道"
        if "multiple attempt" in lower or "multiple attempts" in lower:
            return "困难-多次尝试"
        return "困难"

    if "successful" in lower or "success" in lower:
        if "after" in lower and ("attempt" in lower or "tries" in lower):
            return "成功-经多次尝试"
        if "first pass" in lower or "1st pass" in lower or "first attempt" in lower:
            return "成功-首次通过"
        return "成功"

    if "not attempted" in lower:
        return "未尝试"

    if "self extubation" in lower:
        return "自行拔管"

    if "accidental extubation" in lower:
        return "意外拔管"

    # 中文轻量归并
    zh = re.sub(r"\s+", "", s)
    if zh in {"成功", "插管成功", "气管插管成功"}:
        return "成功"
    if zh in {"失败", "插管失败", "气管插管失败"}:
        return "失败"
    if zh in {"困难", "插管困难", "气管插管困难"}:
        return "困难"
    if zh in {"未尝试", "未插管", "未行插管"}:
        return "未尝试"
    if zh in {"已拔管"}:
        return "已拔管"
    if zh in {"自行拔管"}:
        return "自行拔管"
    if zh in {"意外拔管"}:
        return "意外拔管"

    if "困难" in zh and "成功" in zh:
        return "困难但成功"
    if "失败" in zh and "困难气道" in zh:
        return "失败-困难气道"
    if "困难气道" in zh:
        return "困难气道"
    if "多次" in zh and "成功" in zh:
        return "成功-经多次尝试"
    if "首次" in zh and "成功" in zh:
        return "成功-首次通过"

    # 不确定则保留原值，仅做基础去噪
    cleaned = re.sub(r"\s+", " ", s).strip()
    return cleaned


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Procedure_intubation_result"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_intubation_result)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed for {target_col if target_col in df.columns else 'no target column found'}. Output saved to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")