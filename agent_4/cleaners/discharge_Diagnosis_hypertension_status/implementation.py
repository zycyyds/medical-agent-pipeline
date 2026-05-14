import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_hypertension_status(value: str) -> str:
    if value is None:
        return ""

    text = str(value)
    if text == "":
        return ""

    original = text

    # 基础清洗：去首尾空格、压缩中间空白、统一常见分隔符
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("：", ":").replace("；", ";").replace("，", ",")
    text = re.sub(r"\s*[-_/]+\s*", "-", text)
    text = text.strip(" ,;")

    if text == "":
        return ""

    lower_text = text.lower()
    compact = re.sub(r"[\s\-_/:;,.()]+", "", lower_text)

    # 缺失/未知类：仅对明确缺失表达置空
    missing_patterns = [
        r"^(na|n/a|none|null|nil|unknown|unk|not available|not documented)$",
        r"^(未记录|未填写|未提及|未说明|不详|未知|空白)$",
    ]
    for pat in missing_patterns:
        if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
            return ""

    # 明确否定：无高血压/未见高血压
    negative_patterns = [
        r"^(无|否认|未见|未提示)\s*高血压$",
        r"^non[-\s]*hypertension$",
        r"^no[-\s]*hypertension$",
        r"^without[-\s]*hypertension$",
        r"^hypertension[-\s]*negative$",
        r"^not[-\s]*hypertension$",
    ]
    for pat in negative_patterns:
        if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
            return "无高血压"

    if compact in {
        "无高血压", "否认高血压", "未见高血压", "未提示高血压",
        "nonhypertension", "nohypertension", "withouthypertension",
        "hypertensionnegative", "nothypertension"
    }:
        return "无高血压"

    # 正常/血压正常
    normal_patterns = [
        r"^(正常|血压正常|normotension|normotensive)$",
    ]
    for pat in normal_patterns:
        if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
            return "血压正常"

    if compact in {"正常", "血压正常", "normotension", "normotensive"}:
        return "血压正常"

    # 高血压前期
    pre_patterns = [
        r"^(高血压前期|血压偏高|pre[-\s]*hypertension|prehypertension|elevated blood pressure)$",
    ]
    for pat in pre_patterns:
        if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
            return "高血压前期"

    if compact in {"高血压前期", "血压偏高", "prehypertension", "elevatedbloodpressure"}:
        return "高血压前期"

    # 妊娠高血压相关：优先于一般高血压
    pregnancy_map = {
        "妊娠高血压": [
            r"^(妊娠高血压|孕期高血压|gestational hypertension)$",
        ],
        "妊娠期高血压疾病": [
            r"^(妊娠期高血压疾病|pregnancy induced hypertension|pih)$",
        ],
        "子痫前期": [
            r"^(子痫前期|先兆子痫|pre[-\s]*eclampsia|preeclampsia)$",
        ],
        "子痫": [
            r"^(子痫|eclampsia)$",
        ],
    }
    for standard, patterns in pregnancy_map.items():
        for pat in patterns:
            if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
                return standard

    pregnancy_compact_map = {
        "妊娠高血压": "妊娠高血压",
        "孕期高血压": "妊娠高血压",
        "gestationalhypertension": "妊娠高血压",
        "妊娠期高血压疾病": "妊娠期高血压疾病",
        "pregnancyinducedhypertension": "妊娠期高血压疾病",
        "pih": "妊娠期高血压疾病",
        "子痫前期": "子痫前期",
        "先兆子痫": "子痫前期",
        "preeclampsia": "子痫前期",
        "preeclampsia": "子痫前期",
        "子痫": "子痫",
        "eclampsia": "子痫",
    }
    if compact in pregnancy_compact_map:
        return pregnancy_compact_map[compact]

    # 高血压分级
    grade_map = {
        "高血压1级": [
            r"^(高血压\s*[i1一Ⅰ]\s*级|1\s*级高血压|grade\s*1\s*hypertension|hypertension\s*grade\s*1|stage\s*1\s*hypertension)$",
        ],
        "高血压2级": [
            r"^(高血压\s*[ii2二Ⅱ]\s*级|2\s*级高血压|grade\s*2\s*hypertension|hypertension\s*grade\s*2|stage\s*2\s*hypertension)$",
        ],
        "高血压3级": [
            r"^(高血压\s*[iii3三Ⅲ]\s*级|3\s*级高血压|grade\s*3\s*hypertension|hypertension\s*grade\s*3|stage\s*3\s*hypertension)$",
        ],
    }
    for standard, patterns in grade_map.items():
        for pat in patterns:
            if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
                return standard

    grade_compact_map = {
        "高血压1级": "高血压1级",
        "1级高血压": "高血压1级",
        "grade1hypertension": "高血压1级",
        "hypertensiongrade1": "高血压1级",
        "stage1hypertension": "高血压1级",
        "高血压i级": "高血压1级",
        "高血压一级": "高血压1级",
        "高血压Ⅰ级": "高血压1级",

        "高血压2级": "高血压2级",
        "2级高血压": "高血压2级",
        "grade2hypertension": "高血压2级",
        "hypertensiongrade2": "高血压2级",
        "stage2hypertension": "高血压2级",
        "高血压ii级": "高血压2级",
        "高血压二级": "高血压2级",
        "高血压Ⅱ级": "高血压2级",

        "高血压3级": "高血压3级",
        "3级高血压": "高血压3级",
        "grade3hypertension": "高血压3级",
        "hypertensiongrade3": "高血压3级",
        "stage3hypertension": "高血压3级",
        "高血压iii级": "高血压3级",
        "高血压三级": "高血压3级",
        "高血压Ⅲ级": "高血压3级",
    }
    if compact in grade_compact_map:
        return grade_compact_map[compact]

    # 原发/继发
    subtype_map = {
        "原发性高血压": [
            r"^(原发性高血压|高血压病|essential hypertension|primary hypertension)$",
        ],
        "继发性高血压": [
            r"^(继发性高血压|secondary hypertension)$",
        ],
        "高血压危象": [
            r"^(高血压危象|hypertensive crisis)$",
        ],
        "高血压急症": [
            r"^(高血压急症|hypertensive emergency)$",
        ],
        "高血压亚急症": [
            r"^(高血压亚急症|高血压次急症|hypertensive urgency)$",
        ],
    }
    for standard, patterns in subtype_map.items():
        for pat in patterns:
            if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
                return standard

    subtype_compact_map = {
        "原发性高血压": "原发性高血压",
        "高血压病": "原发性高血压",
        "essentialhypertension": "原发性高血压",
        "primaryhypertension": "原发性高血压",
        "继发性高血压": "继发性高血压",
        "secondaryhypertension": "继发性高血压",
        "高血压危象": "高血压危象",
        "hypertensivecrisis": "高血压危象",
        "高血压急症": "高血压急症",
        "hypertensiveemergency": "高血压急症",
        "高血压亚急症": "高血压亚急症",
        "高血压次急症": "高血压亚急症",
        "hypertensiveurgency": "高血压亚急症",
    }
    if compact in subtype_compact_map:
        return subtype_compact_map[compact]

    # 一般高血压
    general_patterns = [
        r"^(高血压|hypertension|htn)$",
    ]
    for pat in general_patterns:
        if re.fullmatch(pat, lower_text) or re.fullmatch(pat, text):
            return "高血压"

    if compact in {"高血压", "hypertension", "htn"}:
        return "高血压"

    # 对带有“高血压”但无法稳妥细分的文本，尽量保留原词，仅做轻量格式清理
    return text if text != "" else original


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_hypertension_status"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_hypertension_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Successfully cleaned column `{target_col}` and saved cleaned file.",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")