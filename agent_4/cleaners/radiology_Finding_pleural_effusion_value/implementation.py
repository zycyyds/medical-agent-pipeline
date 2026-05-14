import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "radiology_Finding_pleural_effusion_value"


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if text == "":
        return ""
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("：", ":").replace("；", ";").replace("，", ",").replace("。", ".")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_presence(norm_lower: str):
    generic_absent = {"none", "no", "nil", "无", "未见", "阴性"}
    if norm_lower.strip() in generic_absent:
        return "absent"

    neg_patterns = [
        r"无明显?胸腔积液",
        r"未见明显?胸腔积液",
        r"未见胸腔积液",
        r"胸腔积液未见",
        r"否认胸腔积液",
        r"胸水未见",
        r"无胸水",
        r"未见胸水",
        r"\bno pleural effusion\b",
        r"\bwithout pleural effusion\b",
        r"\bnegative for pleural effusion\b",
        r"\bno effusion\b",
    ]
    for p in neg_patterns:
        if re.search(p, norm_lower):
            return "absent"

    pos_patterns = [
        r"胸腔积液",
        r"胸水",
        r"\bpleural effusions?\b",
        r"\beffusions?\b",
    ]
    for p in pos_patterns:
        if re.search(p, norm_lower):
            return "present"

    return None


def _detect_degree(norm_lower: str):
    degree_patterns = [
        ("trace", [r"微量", r"极少量", r"\btrace\b", r"\bminimal\b"]),
        ("small", [r"少量", r"小量", r"\bsmall\b", r"\bmild\b"]),
        ("moderate", [r"中量", r"中等量", r"\bmoderate\b"]),
        ("large", [r"大量", r"多量", r"\blarge\b", r"\bmassive\b"]),
    ]
    for label, patterns in degree_patterns:
        for p in patterns:
            if re.search(p, norm_lower):
                return label
    return None


def _detect_laterality(norm_lower: str):
    bilateral_patterns = [
        r"双侧",
        r"两侧",
        r"bilateral",
        r"both sides",
    ]
    for p in bilateral_patterns:
        if re.search(p, norm_lower):
            return "bilateral"

    left_patterns = [r"左侧", r"\bleft\b"]
    right_patterns = [r"右侧", r"\bright\b"]

    has_left = any(re.search(p, norm_lower) for p in left_patterns)
    has_right = any(re.search(p, norm_lower) for p in right_patterns)

    if has_left and has_right:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def _detect_change(norm_lower: str):
    change_patterns = [
        ("increased", [r"增多", r"增加", r"加重", r"进展", r"\bincreas(ed|ing)?\b", r"\bworsen(ed|ing)?\b"]),
        ("decreased", [r"减少", r"减轻", r"吸收", r"好转", r"\bdecreas(ed|ing)?\b", r"\bimprov(ed|ing|ement)?\b", r"\bresolv(ed|ing)?\b"]),
        ("stable", [r"稳定", r"无变化", r"未见明显变化", r"相仿", r"\bstable\b", r"\bunchanged\b", r"\bno significant change\b"]),
        ("new", [r"新发", r"新增", r"\bnew\b", r"\bnewly\b"]),
        ("resolved", [r"消失", r"已吸收", r"\bresolved\b", r"\bresolution\b"]),
    ]
    for label, patterns in change_patterns:
        for p in patterns:
            if re.search(p, norm_lower):
                return label
    return None


def _clean_pleural_effusion_value(value: str) -> str:
    original = _normalize_text(value)
    if original == "":
        return original

    norm_lower = original.lower()

    presence = _detect_presence(norm_lower)
    degree = _detect_degree(norm_lower)
    laterality = _detect_laterality(norm_lower)
    change = _detect_change(norm_lower)

    if presence is None and degree is None and laterality is None and change is None:
        return original

    parts = []
    if presence is not None:
        parts.append(f"presence={presence}")
    if degree is not None:
        parts.append(f"degree={degree}")
    if laterality is not None:
        parts.append(f"laterality={laterality}")
    if change is not None:
        parts.append(f"change={change}")
    parts.append(f"raw={original}")

    return "; ".join(parts)


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        if not os.path.exists(input_path):
            return ToolResponse(content=f"Input file not found: {input_path}")

        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_clean_pleural_effusion_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned file saved to: {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")