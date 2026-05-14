import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COLUMN = "discharge_Diagnosis_atrial_fibrillation_treatment_context"


def _is_null_like(val):
    if pd.isna(val):
        return True
    text = str(val).strip()
    return text == ""


def _basic_normalize(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_component(comp: str) -> str:
    original = _basic_normalize(comp)
    if original == "":
        return ""

    s = original

    # 去掉常见前缀动作词，尽量不影响核心治疗语义
    s = re.sub(r"^(给予|予以|予|应用|使用|服用|口服|静滴|静注|行|接受)\s*", "", s, flags=re.IGNORECASE)

    # 常见术语/药名标准化
    replacements = [
        (r"\bwarfarin\b|华法林钠?", "华法林"),
        (r"\brivaroxaban\b|利伐沙班", "利伐沙班"),
        (r"\bdabigatran(?:\s+etexilate)?\b|达比加群(?:酯)?", "达比加群"),
        (r"\bapixaban\b|阿哌沙班", "阿哌沙班"),
        (r"\bedoxaban\b|艾多沙班", "艾多沙班"),
        (r"\bdoac\b|\bnoac\b|新型口服抗凝药|直接口服抗凝药", "DOAC"),
        (r"\blmwh\b|低分子肝素", "低分子肝素"),
        (r"\baspirin\b|阿司匹林", "阿司匹林"),
        (r"\bclopidogrel\b|氯吡格雷", "氯吡格雷"),

        (r"\bamiodarone\b|胺碘酮", "胺碘酮"),
        (r"\bpropafenone\b|普罗帕酮", "普罗帕酮"),
        (r"\bsotalol\b|索他洛尔", "索他洛尔"),
        (r"\bdronedarone\b|决奈达隆", "决奈达隆"),

        (r"\bmetoprolol(?:\s+(?:succinate|tartrate))?\b|倍他乐克|美托洛尔", "美托洛尔"),
        (r"\bbisoprolol\b|比索洛尔", "比索洛尔"),
        (r"\bdiltiazem\b|地尔硫卓", "地尔硫卓"),
        (r"\bverapamil\b|维拉帕米", "维拉帕米"),
        (r"\bdigoxin\b|地高辛", "地高辛"),

        (r"\brfca\b|radiofrequency ablation|catheter ablation|导管消融|射频消融(?:术)?|房颤消融(?:术)?", "射频消融"),
        (r"\bcardioversion\b|电复律", "电复律"),
        (r"\bpvi\b", "PVI"),
        (r"\bcti\b", "CTI"),
        (r"\blaac\b|\blaao\b|left atrial appendage (?:closure|occlusion)|左心耳封堵", "左心耳封堵"),
        (r"\bpacemaker\b|起搏器", "起搏器"),
        (r"\bmaze\b", "Maze手术"),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # 常见治疗状态轻量规范化
    s = re.sub(r"未予特殊处理|未予治疗|未治疗|未用药", "未予特殊治疗", s)
    s = re.sub(r"保守治疗", "保守治疗", s)
    s = re.sub(r"未抗凝", "未抗凝", s)
    s = re.sub(r"拒绝抗凝|不愿抗凝", "拒绝抗凝", s)

    # 消融/复律状态
    if re.search(r"射频消融", s):
        if re.search(r"拟行|待行|计划行", s):
            return "拟行射频消融"
        if re.search(r"术后|后", s):
            return "射频消融术后"

    if re.search(r"电复律", s):
        if re.search(r"拟行|待行|计划行", s):
            return "拟行电复律"
        if re.search(r"术后|后", s):
            return "电复律后"

    if re.search(r"左心耳封堵", s):
        if re.search(r"术后|后", s):
            return "左心耳封堵术后"

    # 轻量收敛常见后缀
    s = re.sub(r"(华法林|利伐沙班|达比加群|阿哌沙班|艾多沙班)\s*抗凝治疗", r"\1抗凝", s)
    s = re.sub(r"(华法林|利伐沙班|达比加群|阿哌沙班|艾多沙班)\s*抗凝", r"\1抗凝", s)
    s = re.sub(r"\s+", " ", s).strip(" ,，、;；+/|")
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else original


def _split_and_normalize(text: str) -> str:
    text = _basic_normalize(text)
    if text == "":
        return text

    # 先做整句级别的轻量标准化，便于识别英文药名/术语
    normalized = _normalize_component(text)

    # 若已经是明确状态短语，直接返回
    protected_patterns = [
        r"^射频消融术后$",
        r"^拟行射频消融$",
        r"^电复律后$",
        r"^拟行电复律$",
        r"^左心耳封堵术后$",
        r"^未予特殊治疗$",
        r"^保守治疗$",
        r"^未抗凝$",
        r"^拒绝抗凝$",
    ]
    for p in protected_patterns:
        if re.fullmatch(p, normalized):
            return normalized

    # 统一分隔符，仅在单列内做联合治疗标准化
    split_pattern = r"\s*(?:\+|＋|,|，|、|;|；|\||/|联合|合用|并用)\s*"
    parts = re.split(split_pattern, normalized)

    cleaned_parts = []
    seen = set()
    for part in parts:
        part = _normalize_component(part)
        part = re.sub(r"\s+", " ", part).strip(" ,，、;；+/|")
        if not part:
            continue
        if part not in seen:
            seen.add(part)
            cleaned_parts.append(part)

    if not cleaned_parts:
        return normalized

    if len(cleaned_parts) == 1:
        return cleaned_parts[0]

    return " + ".join(cleaned_parts)


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        if not os.path.exists(input_path):
            return ToolResponse(content=f"输入文件不存在: {input_path}")

        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COLUMN in df.columns:
            def clean_value(val):
                if _is_null_like(val):
                    return val
                raw = str(val)
                cleaned = _split_and_normalize(raw)
                return cleaned if cleaned != "" else raw

            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(clean_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index, encoding="utf-8-sig")

        return ToolResponse(
            content=f"清洗完成，输出文件已保存至: {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"清洗失败: {str(e)}")