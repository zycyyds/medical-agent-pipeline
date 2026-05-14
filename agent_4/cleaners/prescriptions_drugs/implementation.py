import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


_ALIAS_PATTERNS = [
    (r'^\s*apap\b', 'acetaminophen'),
    (r'^\s*tylenol\b', 'acetaminophen'),
    (r'^\s*paracetamol\b', 'acetaminophen'),
    (r'^\s*asa\b', 'aspirin'),
    (r'^\s*ntg\b', 'nitroglycerin'),
    (r'^\s*hctz\b', 'hydrochlorothiazide'),
    (r'^\s*hcq\b', 'hydroxychloroquine'),
    (r'^\s*pcn\b', 'penicillin'),
]

_FLUID_ONLY_PATTERNS = [
    r'^(?:normal\s+saline|ns)$',
    r'^(?:normal\s+saline|ns)\s*(?:flush|flushed)?$',
    r'^saline\s*(?:flush)?$',
    r'^sodium\s+chloride\s*0?\.?9%\s*(?:flush)?$',
    r'^sterile\s+water(?:\s+for\s+injection)?$',
    r'^water(?:\s+for\s+injection)?$',
    r'^d5w$',
    r'^d10w$',
    r'^d50w$',
    r'^dextrose\s*(?:5|10|50)\s*%?(?:\s+in\s+water)?$',
    r'^lactated\s+ringer[\'’]?[s]?$',
    r'^ringer[\'’]?\s*lactate$',
]

_SPLIT_PATTERN = re.compile(r'\s*(?:,|，|;|；|\||/|\n|\r\n?)\s*')


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace('\u3000', ' ')
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*-\s*', '-', s)
    s = re.sub(r'\s*/\s*', '/', s)
    s = re.sub(r'\s*,\s*', ', ', s)
    s = s.strip(" \t;|")
    return s


def _standardize_alias(token: str) -> str:
    original = token
    for pattern, replacement in _ALIAS_PATTERNS:
        if re.search(pattern, token, flags=re.IGNORECASE):
            token = re.sub(pattern, replacement, token, flags=re.IGNORECASE)
            break

    token = re.sub(r'^\s+', '', token)
    if token:
        token = token[0].lower() + token[1:]
    return token if token else original


def _is_fluid_only(token: str) -> bool:
    t = token.strip().lower()
    t = re.sub(r'[().]+', '', t)
    t = re.sub(r'\s+', ' ', t)
    for pattern in _FLUID_ONLY_PATTERNS:
        if re.fullmatch(pattern, t, flags=re.IGNORECASE):
            return True
    return False


def _dedupe_key(token: str) -> str:
    t = token.lower().strip()
    t = re.sub(r'[^\w%./+-]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _clean_prescriptions_drugs_cell(value: str) -> str:
    if value is None:
        return value
    if not isinstance(value, str):
        value = str(value)

    raw = value.strip()
    if raw == "":
        return value

    parts = _SPLIT_PATTERN.split(raw)
    cleaned_parts = []
    seen = set()

    for part in parts:
        token = _normalize_text(part)
        if token == "":
            continue

        token = _standardize_alias(token)
        token = _normalize_text(token)

        if token == "":
            continue

        if _is_fluid_only(token):
            continue

        key = _dedupe_key(token)
        if key == "" or key in seen:
            continue

        seen.add(key)
        cleaned_parts.append(token)

    if not cleaned_parts:
        return ""

    return ", ".join(cleaned_parts)


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if 'prescriptions_drugs' in df.columns:
            df['prescriptions_drugs'] = df['prescriptions_drugs'].apply(_clean_prescriptions_drugs_cell)

        output_path = input_path.replace('.csv', '_cleaned.csv')
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned file saved to: {output_path}",
            metadata={"output_file": output_path}
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")