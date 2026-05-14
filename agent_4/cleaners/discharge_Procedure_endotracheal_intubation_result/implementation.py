import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""

    # normalize spaces and punctuation
    text = text.replace("\u3000", " ")
    text = text.replace("，", ",").replace("；", ";").replace("：", ":")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("＋", "+").replace("／", "/")
    text = re.sub(r"\s+", " ", text).strip()

    lower = text.lower()

    # common exact missing-like markers: keep empty only for strong null markers
    if lower in {"nan", "none", "null", "n/a", "na", "未记录", "未注明", "不详"}:
        return ""

    # common abbreviation normalization
    replacements = [
        (r"\bett\b", "endotracheal tube"),
        (r"\bett\s*#?\s*", "endotracheal tube "),
        (r"\bet\s*tube\b", "endotracheal tube"),
        (r"\bintubated\b", "intubation"),
        (r"\bintubation successful\b", "successful intubation"),
        (r"\bfob\b", "fiberoptic bronchoscope"),
        (r"\bglidescope\b", "video laryngoscope"),
        (r"\bc-mac\b", "video laryngoscope"),
        (r"\bvl\b", "video laryngoscope"),
        (r"\bdl\b", "direct laryngoscope"),
        (r"\bors?\b", "operating room"),
        (r"\bed\b", "emergency department"),
        (r"\bic[u|u]\b", "icu"),
        (r"\bcpr\b", "cardiopulmonary resuscitation"),
        (r"\brsi\b", "rapid sequence induction"),
        (r"\boa\b", "orotracheal"),
        (r"\bnt\b", "nasotracheal"),
        (r"\beasy\b", "easy"),
        (r"\bdifficult\b", "difficult"),
    ]
    for pattern, repl in replacements:
        lower = re.sub(pattern, repl, lower, flags=re.IGNORECASE)

    # Chinese keyword normalization
    zh_map = [
        (r"经口气管插管|口插|经口插管", "orotracheal intubation"),
        (r"经鼻气管插管|鼻插|经鼻插管", "nasotracheal intubation"),
        (r"气管插管成功|插管成功|已插管成功", "successful intubation"),
        (r"插管失败|气管插管失败", "failed intubation"),
        (r"困难气道|插管困难", "difficult airway"),
        (r"顺利插管|容易插管", "easy intubation"),
        (r"急诊", "emergency"),
        (r"手术室|术中", "operating room"),
        (r"icu", "icu"),
        (r"病房", "ward"),
        (r"纤支镜|纤维支气管镜", "fiberoptic bronchoscope"),
        (r"可视喉镜|视频喉镜", "video laryngoscope"),
        (r"直接喉镜", "direct laryngoscope"),
        (r"呼吸衰竭", "respiratory failure"),
        (r"呼吸窘迫", "respiratory distress"),
        (r"心肺复苏", "cardiopulmonary resuscitation"),
        (r"气道保护", "airway protection"),
        (r"误吸", "aspiration"),
        (r"全麻", "general anesthesia"),
    ]
    for pattern, repl in zh_map:
        lower = re.sub(pattern, repl, lower, flags=re.IGNORECASE)

    lower = re.sub(r"\s+", " ", lower).strip()
    return lower


def _extract_size(text: str) -> str:
    patterns = [
        r"(?:size|tube size|ett)\s*[:#]?\s*(\d(?:\.\d)?)\s*(?:mm)?",
        r"(\d(?:\.\d)?)\s*(?:mm)?\s*(?:ett|endotracheal tube)",
        r"(\d(?:\.\d)?)\s*#",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1) + " mm"
    return ""


def _extract_route(text: str) -> str:
    if "orotracheal" in text:
        return "orotracheal"
    if "nasotracheal" in text:
        return "nasotracheal"
    return ""


def _extract_result(text: str) -> str:
    if "failed intubation" in text:
        return "failed"
    if "successful intubation" in text or re.search(r"\bintubation\b", text):
        return "successful"
    return ""


def _extract_difficulty(text: str) -> str:
    if "difficult airway" in text or "difficult intubation" in text:
        return "difficult"
    if "easy intubation" in text or re.search(r"\beasy\b", text):
        return "easy"
    return ""


def _extract_device(text: str) -> str:
    if "fiberoptic bronchoscope" in text:
        return "fiberoptic bronchoscope"
    if "video laryngoscope" in text:
        return "video laryngoscope"
    if "direct laryngoscope" in text:
        return "direct laryngoscope"
    return ""


def _extract_setting(text: str) -> str:
    if "emergency department" in text or re.search(r"\bemergency\b", text):
        return "emergency"
    if "operating room" in text:
        return "operating room"
    if re.search(r"\bicu\b", text):
        return "icu"
    if "ward" in text:
        return "ward"
    return ""


def _extract_reason(text: str) -> str:
    reason_terms = [
        "respiratory failure",
        "respiratory distress",
        "airway protection",
        "cardiopulmonary resuscitation",
        "aspiration",
        "general anesthesia",
    ]
    found = []
    for term in reason_terms:
        if term in text:
            found.append(term)
    return "; ".join(found)


def _rebuild_value(original: str) -> str:
    norm = _normalize_text(original)
    if norm == "":
        return ""

    route = _extract_route(norm)
    result = _extract_result(norm)
    difficulty = _extract_difficulty(norm)
    device = _extract_device(norm)
    size = _extract_size(norm)
    setting = _extract_setting(norm)
    reason = _extract_reason(norm)

    parts = []
    if result:
        parts.append(result)
    if route:
        parts.append(route)
    if difficulty:
        parts.append(difficulty)
    if device:
        parts.append("device=" + device)
    if size:
        parts.append("size=" + size)
    if setting:
        parts.append("setting=" + setting)
    if reason:
        parts.append("reason=" + reason)

    # If structured extraction is too weak, keep normalized original text
    if len(parts) == 0:
        return norm

    # Preserve meaningful unmatched qualifiers lightly if present
    remainder = norm
    removable_terms = [
        "successful intubation", "failed intubation", "intubation",
        "orotracheal intubation", "nasotracheal intubation",
        "orotracheal", "nasotracheal", "difficult airway", "easy intubation",
        "fiberoptic bronchoscope", "video laryngoscope", "direct laryngoscope",
        "emergency department", "emergency", "operating room", "icu", "ward",
        "respiratory failure", "respiratory distress", "airway protection",
        "cardiopulmonary resuscitation", "aspiration", "general anesthesia"
    ]
    for term in removable_terms:
        remainder = re.sub(re.escape(term), " ", remainder, flags=re.IGNORECASE)

    remainder = re.sub(r"(?:size|tube size|ett)\s*[:#]?\s*\d(?:\.\d)?\s*(?:mm)?", " ", remainder, flags=re.IGNORECASE)
    remainder = re.sub(r"\b\d(?:\.\d)?\s*(?:mm)?\s*(?:ett|endotracheal tube)\b", " ", remainder, flags=re.IGNORECASE)
    remainder = re.sub(r"[,:;()/]+", " ", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip()

    if remainder:
        parts.append("note=" + remainder)

    return "; ".join(parts)


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Procedure_endotracheal_intubation_result"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_rebuild_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed: {os.path.basename(output_path)}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")