import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _clean_whitespace_and_punct(text: str) -> str:
    if text is None:
        return text

    text = str(text)

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "；": ";",
        "，": ",",
        "：": ":",
        "（": "(",
        "）": ")",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = text.strip()
    text = re.sub(r"[ \t\r\n]+", " ", text)
    text = re.sub(r"\s*([,;:/|])\s*", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r";\s*;", "; ", text)
    text = re.sub(r",\s*,", ", ", text)
    return text.strip(" ;,")


def _extract_tags(text: str):
    s = text.lower()
    tags = []

    def add_tag(tag: str):
        if tag not in tags:
            tags.append(tag)

    # Orientation / A&O variants
    m = re.search(
        r"\b(?:a\s*&\s*o|a\s*/\s*o|ao|aox|axo|aaox|axox|aaox|alert\s*(?:and|&)\s*oriented)\s*x?\s*([1-4])\b",
        s,
        flags=re.I,
    )
    if m:
        add_tag(f"Alert and oriented x{m.group(1)}")
    else:
        m2 = re.search(r"\balert\s*(?:and|&)\s*oriented\s*x?\s*([1-4])\b", s, flags=re.I)
        if m2:
            add_tag(f"Alert and oriented x{m2.group(1)}")
        else:
            m3 = re.search(r"\boriented\s*x?\s*([1-4])\b", s, flags=re.I)
            if m3:
                add_tag(f"Oriented x{m3.group(1)}")

    if re.search(r"\b(?:a\s*&\s*o|a\s*/\s*o|alert\s*(?:and|&)\s*oriented)\b", s, flags=re.I):
        if not any(tag.startswith("Alert and oriented x") for tag in tags):
            add_tag("Alert and oriented")

    # Alertness / consciousness
    if re.search(r"\balert\b", s, flags=re.I):
        if not any(tag.startswith("Alert and oriented") for tag in tags):
            add_tag("Alert")
    if re.search(r"\bawake\b", s, flags=re.I):
        add_tag("Awake")
    if re.search(r"\blethargic\b", s, flags=re.I):
        add_tag("Lethargic")
    if re.search(r"\bdrowsy\b", s, flags=re.I):
        add_tag("Drowsy")
    if re.search(r"\bsomnolent\b", s, flags=re.I):
        add_tag("Somnolent")
    if re.search(r"\bobtunded\b", s, flags=re.I):
        add_tag("Obtunded")
    if re.search(r"\bstupor(?:ous)?\b", s, flags=re.I):
        add_tag("Stuporous")
    if re.search(r"\bunresponsive\b", s, flags=re.I):
        add_tag("Unresponsive")

    # Cognition / behavior
    if re.search(r"\bconfused\b", s, flags=re.I):
        add_tag("Confused")
    if re.search(r"\bdisoriented\b", s, flags=re.I):
        add_tag("Disoriented")
    if re.search(r"\bagitated\b", s, flags=re.I):
        add_tag("Agitated")
    if re.search(r"\bcalm\b", s, flags=re.I):
        add_tag("Calm")
    if re.search(r"\bcooperative\b", s, flags=re.I):
        add_tag("Cooperative")
    if re.search(r"\bappropriate\b", s, flags=re.I):
        add_tag("Appropriate")
    if re.search(r"\bnon[- ]?verbal\b", s, flags=re.I):
        add_tag("Nonverbal")

    # Responsiveness
    if re.search(r"\bresponsive to verbal(?: stimuli)?\b|\bresponds? to verbal\b", s, flags=re.I):
        add_tag("Responsive to verbal stimuli")
    if re.search(r"\bresponsive to pain(?:ful stimuli)?\b|\bresponds? to pain\b", s, flags=re.I):
        add_tag("Responsive to painful stimuli")

    return tags


def _is_status_only_text(text: str, tags) -> bool:
    if not tags:
        return False

    s = text.lower()

    patterns = [
        r"\b(?:a\s*&\s*o|a\s*/\s*o|ao|aox|axo|aaox|axox|aaox)\s*x?\s*[1-4]\b",
        r"\balert\s*(?:and|&)\s*oriented\s*x?\s*[1-4]\b",
        r"\boriented\s*x?\s*[1-4]\b",
        r"\balert\s*(?:and|&)\s*oriented\b",
        r"\balert\b",
        r"\bawake\b",
        r"\blethargic\b",
        r"\bdrowsy\b",
        r"\bsomnolent\b",
        r"\bobtunded\b",
        r"\bstupor(?:ous)?\b",
        r"\bunresponsive\b",
        r"\bconfused\b",
        r"\bdisoriented\b",
        r"\bagitated\b",
        r"\bcalm\b",
        r"\bcooperative\b",
        r"\bappropriate\b",
        r"\bnon[- ]?verbal\b",
        r"\bresponsive to verbal(?: stimuli)?\b",
        r"\bresponds? to verbal\b",
        r"\bresponsive to pain(?:ful stimuli)?\b",
        r"\bresponds? to pain\b",
    ]

    for p in patterns:
        s = re.sub(p, " ", s, flags=re.I)

    s = re.sub(r"[\s,;:/|&().\-]+", "", s)
    return s == ""


def _standardize_known_abbreviations(text: str) -> str:
    t = text

    t = re.sub(
        r"\b(?:a\s*&\s*o|a\s*/\s*o|ao|aox|axo|aaox|axox|aaox)\s*x?\s*([1-4])\b",
        r"Alert and oriented x\1",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\balert\s*(?:and|&)\s*oriented\s*x?\s*([1-4])\b",
        r"Alert and oriented x\1",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\balert\s*(?:and|&)\s*oriented\b",
        "Alert and oriented",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\boriented\s*x?\s*([1-4])\b",
        r"Oriented x\1",
        t,
        flags=re.I,
    )

    # Normalize obvious delimiter runs after substitution
    t = re.sub(r"\s*[,;/|]\s*", "; ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(?:;\s*){2,}", "; ", t)
    return t.strip(" ;,")


def _normalize_mental_status(value: str) -> str:
    if value is None:
        return value

    original = str(value)
    if original == "":
        return original

    cleaned = _clean_whitespace_and_punct(original)
    if cleaned == "":
        return cleaned

    standardized = _standardize_known_abbreviations(cleaned)
    tags = _extract_tags(standardized)

    if _is_status_only_text(standardized, tags):
        return "; ".join(tags)

    return standardized


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_values=[""]
        )

        target_col = "discharge_Finding_mental_status_value"

        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_normalize_mental_status)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Data cleaning completed. Output saved to: {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")