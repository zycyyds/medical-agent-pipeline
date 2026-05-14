import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _format_number(num_str: str) -> str:
    try:
        num = float(num_str)
        if num.is_integer():
            return str(int(num))
        return str(num).rstrip("0").rstrip(".")
    except Exception:
        return num_str.strip()


def _extract_saturation(text: str):
    # 1) Prefer explicit percentages
    pct_matches = re.findall(r"\b(100|[5-9]?\d(?:\.\d+)?)\s*%", text, flags=re.I)
    if pct_matches:
        return _format_number(pct_matches[0]) + "%"

    # 2) Labeled saturation patterns
    labeled_patterns = [
        r"(?:spo2|spo₂|o2\s*sat(?:s|uration)?|oxygen\s*sat(?:s|uration)?|sat(?:s|uration)?)\s*[:=]?\s*(100|[5-9]?\d(?:\.\d+)?)\b",
        r"\b(100|[5-9]?\d(?:\.\d+)?)\b\s*(?:spo2|spo₂|o2\s*sat(?:s|uration)?|oxygen\s*sat(?:s|uration)?|sat(?:s|uration)?)\b",
    ]
    for pat in labeled_patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return _format_number(m.group(1)) + "%"

    # 3) Fallback: standalone plausible saturation number not tied to oxygen flow units
    fallback_matches = list(re.finditer(r"\b(100|[5-9]?\d(?:\.\d+)?)\b", text, flags=re.I))
    for m in fallback_matches:
        start, end = m.span(1)
        after = text[end:end + 12].lower()
        before = text[max(0, start - 12):start].lower()

        # Exclude likely oxygen flow / FiO2 / respiratory settings
        if re.match(r"\s*(?:l|lpm|l/min|lit(?:er|re)?s?\b|%)", after):
            continue
        if re.search(r"(?:fio2|fio₂)\s*$", before):
            continue

        val = m.group(1)
        return _format_number(val) + "%"

    return None


def _extract_room_air(text: str) -> bool:
    return bool(
        re.search(r"\b(?:room air|ra)\b", text, flags=re.I)
    )


def _standardize_device(device: str) -> str:
    d = _normalize_whitespace(device).lower()
    replacements = {
        "nc": "NC",
        "n/c": "NC",
        "nasal cannula": "nasal cannula",
        "face mask": "face mask",
        "facemask": "face mask",
        "mask": "mask",
        "venti mask": "Venturi mask",
        "venturi mask": "Venturi mask",
        "nrb": "NRB",
        "non rebreather": "non-rebreather",
        "non-rebreather": "non-rebreather",
        "nonrebreather": "non-rebreather",
        "bipap": "BiPAP",
        "cpap": "CPAP",
        "hfnc": "HFNC",
        "high flow nasal cannula": "high flow nasal cannula",
        "high-flow nasal cannula": "high flow nasal cannula",
        "trach collar": "trach collar",
    }
    return replacements.get(d, device.strip())


def _extract_oxygen_context(text: str):
    contexts = []

    flow_device_pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(?:l|lpm|l/min|lit(?:er|re)?s?)\b"
        r"(?:\s*(?:o2|oxygen))?"
        r"(?:\s*(?:via|by))?"
        r"(?:\s*(nc|n/c|nasal cannula|face ?mask|mask|venti ?mask|venturi mask|nrb|non[- ]?rebreather|bipap|cpap|hfnc|high[- ]?flow nasal cannula|trach collar))?",
        flags=re.I,
    )

    for m in flow_device_pattern.finditer(text):
        flow = _format_number(m.group(1)) + " L"
        device = m.group(2)
        if device:
            contexts.append(flow + " " + _standardize_device(device))
        else:
            contexts.append(flow)

    device_only_pattern = re.compile(
        r"\b(nc|n/c|nasal cannula|face ?mask|venti ?mask|venturi mask|nrb|non[- ]?rebreather|bipap|cpap|hfnc|high[- ]?flow nasal cannula|trach collar)\b",
        flags=re.I,
    )

    # Add device-only context only if no flow+device captured for that device
    existing_text = " | ".join(contexts).lower()
    for m in device_only_pattern.finditer(text):
        device_std = _standardize_device(m.group(1))
        if device_std.lower() not in existing_text:
            contexts.append(device_std)

    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for c in contexts:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return deduped


def _clean_discharge_oxygen_value(value: str) -> str:
    if value is None:
        return value

    original = value
    text = _normalize_whitespace(str(value))
    if text == "":
        return original

    saturation = _extract_saturation(text)
    room_air = _extract_room_air(text)
    oxygen_contexts = _extract_oxygen_context(text)

    # If nothing reliable extracted, keep original lightly normalized only
    if not saturation and not room_air and not oxygen_contexts:
        return text

    parts = []
    if saturation:
        parts.append(saturation)
    if room_air:
        parts.append("room air")
    for ctx in oxygen_contexts:
        if ctx.lower() != "room air":
            parts.append(ctx)

    if not parts:
        return text

    # If we only found context but no saturation, preserve original to avoid over-cleaning
    if not saturation:
        return text

    return " | ".join(parts)


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Finding_oxygen_saturation_value"
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_clean_discharge_oxygen_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaning completed. Output saved to: {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Cleaning failed: {str(e)}")