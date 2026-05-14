import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


TARGET_COL = "cxr_sampled_with_reports_Diagnosis_pulmonary_edema_status"


def _normalize_token(token: str) -> str:
    if token is None:
        return ""
    t = str(token).strip().lower()
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\s+", " ", t)

    direct_map = {
        "confirmed": "confirmed",
        "confirm": "confirmed",
        "positive": "confirmed",
        "present": "confirmed",
        "yes": "confirmed",
        "true": "confirmed",
        "suspected": "suspected",
        "suspicious": "suspected",
        "possible": "suspected",
        "probable": "suspected",
        "likely": "suspected",
        "questionable": "suspected",
        "equivocal": "suspected",
        "ruled out": "ruled out",
        "rule out": "ruled out",
        "ruledout": "ruled out",
        "negative": "ruled out",
        "absent": "ruled out",
        "no": "ruled out",
        "false": "ruled out",
        "none": "ruled out",
        "not present": "ruled out",
        "without": "ruled out",
    }
    if t in direct_map:
        return direct_map[t]

    if re.fullmatch(r"r/?o", t):
        return "ruled out"

    if "rule" in t and "out" in t:
        return "ruled out"
    if "confirm" in t:
        return "confirmed"
    if any(word in t for word in ["suspect", "possible", "probable", "likely", "equivocal", "questionable"]):
        return "suspected"
    if any(word in t for word in ["negative", "absent", "without", "not present"]):
        return "ruled out"
    if t in {"1"}:
        return "confirmed"
    if t in {"0"}:
        return "ruled out"

    return token.strip()


def _clean_status_value(value: str) -> str:
    if value is None:
        return value

    raw = str(value)
    if raw == "":
        return raw

    s = raw.strip()
    if s == "":
        return s

    s_norm = s.lower().strip()
    s_norm = s_norm.replace(";", "|")
    s_norm = s_norm.replace("/", "|")
    s_norm = s_norm.replace("\\", "|")
    s_norm = re.sub(r"\s*\+\s*", "|", s_norm)
    s_norm = re.sub(r"\s*&\s*", "|", s_norm)
    s_norm = re.sub(r"\s*,\s*", "|", s_norm)
    s_norm = re.sub(r"\s*\|\s*", "|", s_norm)
    s_norm = re.sub(r"\s+", " ", s_norm)

    tokens = [tok.strip() for tok in s_norm.split("|") if tok.strip() != ""]
    if not tokens:
        return raw

    normalized = []
    for tok in tokens:
        normalized.append(_normalize_token(tok))

    allowed = {"confirmed", "suspected", "ruled out"}
    if all(item in allowed for item in normalized):
        order = ["confirmed", "suspected", "ruled out"]
        dedup = []
        seen = set()
        for item in order:
            if item in normalized and item not in seen:
                dedup.append(item)
                seen.add(item)
        return "|".join(dedup)

    # 对一些未被分隔但包含多状态关键词的情况做轻量解析
    joined_text = s_norm
    found = []
    if "confirm" in joined_text or re.search(r"\bpositive\b|\bpresent\b|\byes\b|\btrue\b", joined_text):
        found.append("confirmed")
    if any(word in joined_text for word in ["suspect", "possible", "probable", "likely", "equivocal", "questionable"]):
        found.append("suspected")
    if ("rule out" in joined_text or "ruled out" in joined_text or
            re.search(r"\bnegative\b|\babsent\b|\bwithout\b|\bnot present\b|\bno\b|\bfalse\b", joined_text)):
        found.append("ruled out")

    found = [x for x in ["confirmed", "suspected", "ruled out"] if x in found]
    if found:
        return "|".join(found)

    return raw


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].apply(_clean_status_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        if output_path == input_path:
            output_path = input_path + "_cleaned.csv"

        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned column `{TARGET_COL}` and saved cleaned file.",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")