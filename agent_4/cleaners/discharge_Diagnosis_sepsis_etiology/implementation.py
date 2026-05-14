import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _normalize_text(value: str) -> str:
    if value is None:
        return value

    text = str(value)

    # Preserve obvious missing-like values as original after trim normalization
    stripped = text.strip()
    if stripped == "":
        return text

    # Normalize spaces and separators lightly
    s = stripped
    s = s.replace("\u3000", " ")
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()

    lower = s.lower()

    # Common no-information placeholders: keep standardized but do not infer
    null_like = {
        "unknown": "Unknown",
        "unk": "Unknown",
        "unclear": "Unknown",
        "not clear": "Unknown",
        "n/a": "N/A",
        "na": "N/A",
        "none": "None",
        "nil": "None",
        "-": "-",
        "--": "--",
        "?": "?",
    }
    if lower in null_like:
        return null_like[lower]

    # Light spelling/abbreviation normalization
    replacements = [
        (r"\betio\b", "etiology"),
        (r"\baetiology\b", "etiology"),
        (r"\betiology\b", "etiology"),
        (r"\burosepsis\b", "urinary tract infection-related sepsis"),
        (r"\buti\b", "urinary tract infection"),
        (r"\bpna\b", "pneumonia"),
        (r"\bcap\b", "community-acquired pneumonia"),
        (r"\bhcap\b", "healthcare-associated pneumonia"),
        (r"\bhap\b", "hospital-acquired pneumonia"),
        (r"\bvap\b", "ventilator-associated pneumonia"),
        (r"\bcns\b", "central nervous system"),
        (r"\bgi\b", "gastrointestinal"),
        (r"\bgu\b", "genitourinary"),
        (r"\bbsi\b", "bloodstream infection"),
        (r"\bclabsi\b", "central line-associated bloodstream infection"),
        (r"\bssi\b", "surgical site infection"),
        (r"\bmrsa\b", "MRSA"),
        (r"\bmssa\b", "MSSA"),
        (r"\besbl\b", "ESBL"),
        (r"\bmdro\b", "multidrug-resistant organism"),
        (r"\be\.?\s*coli\b", "E. coli"),
        (r"\bkleb\b", "Klebsiella"),
        (r"\bstaph\.?\s*aureus\b", "Staphylococcus aureus"),
        (r"\bstrep\.?\b", "Streptococcus"),
        (r"\bentero\b", "Enterococcus"),
        (r"\bpseudomonas aeroginosa\b", "Pseudomonas aeruginosa"),
        (r"\bpseudomonas aeruginosa\b", "Pseudomonas aeruginosa"),
        (r"\bacinetobacter baumani[i]?\b", "Acinetobacter baumannii"),
        (r"\baspiration pna\b", "aspiration pneumonia"),
        (r"\bcholangits\b", "cholangitis"),
        (r"\bpyelo(nephritis)?\b", "pyelonephritis"),
        (r"\bintraabdominal\b", "intra-abdominal"),
        (r"\bintra abdominal\b", "intra-abdominal"),
        (r"\bcatheter assoc(iated)?\b", "catheter-associated"),
        (r"\bline assoc(iated)?\b", "line-associated"),
        (r"\bpost op\b", "postoperative"),
        (r"\bpost-op\b", "postoperative"),
    ]
    for pattern, repl in replacements:
        lower = re.sub(pattern, repl, lower, flags=re.IGNORECASE)

    # Synonym consolidation for common etiologies; only when clearly matched
    synonym_patterns = [
        (r"\blung infection\b", "pneumonia"),
        (r"\bchest infection\b", "pneumonia"),
        (r"\bresp(iratory)? source\b", "respiratory source"),
        (r"\burinary source\b", "urinary source"),
        (r"\burologic source\b", "urinary source"),
        (r"\babdominal source\b", "intra-abdominal source"),
        (r"\bintra-abdominal infection\b", "intra-abdominal source"),
        (r"\bbiliary infection\b", "biliary source"),
    ]
    for pattern, repl in synonym_patterns:
        lower = re.sub(pattern, repl, lower, flags=re.IGNORECASE)

    # Normalize common bacteremia wording without over-merging
    lower = re.sub(r"\bbacter[a-z]*emia\b", "bacteremia", lower, flags=re.IGNORECASE)
    lower = re.sub(r"\bsepticemia\b", "bacteremia", lower, flags=re.IGNORECASE)

    # Cleanup repeated punctuation/spaces
    lower = re.sub(r"\s+", " ", lower).strip()
    lower = re.sub(r"\s*,\s*", ", ", lower)
    lower = re.sub(r"\s*;\s*", "; ", lower)

    # Title/case handling:
    # keep well-known uppercase abbreviations, otherwise sentence-like normalization
    tokens = []
    for tok in re.split(r"(\s+|/|-|,|;|\(|\))", lower):
        if tok is None or tok == "":
            continue
        if re.fullmatch(r"\s+|/|-|,|;|\(|\)", tok):
            tokens.append(tok)
            continue
        if tok.upper() in {"MRSA", "MSSA", "ESBL", "CNS"}:
            tokens.append(tok.upper())
        elif tok in {"e.", "coli"}:
            tokens.append(tok)
        else:
            tokens.append(tok)

    result = "".join(tokens)

    # Fix specific organism capitalization
    organism_fixes = [
        (r"\be\. coli\b", "E. coli"),
        (r"\bmrsa\b", "MRSA"),
        (r"\bmssa\b", "MSSA"),
        (r"\besbl\b", "ESBL"),
        (r"\bklebsiella\b", "Klebsiella"),
        (r"\bpseudomonas aeruginosa\b", "Pseudomonas aeruginosa"),
        (r"\bacinetobacter baumannii\b", "Acinetobacter baumannii"),
        (r"\bstaphylococcus aureus\b", "Staphylococcus aureus"),
        (r"\bstreptococcus\b", "Streptococcus"),
        (r"\benterococcus\b", "Enterococcus"),
    ]
    for pattern, repl in organism_fixes:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)

    # Gentle sentence-style capitalization for leading character only
    if result:
        result = result[0].upper() + result[1:]

    return result


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        target_col = "discharge_Diagnosis_sepsis_etiology"
        if target_col in df.columns:
            original_series = df[target_col]
            df[target_col] = original_series.apply(_normalize_text)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Successfully cleaned column `{target_col}` and saved cleaned file.",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {e}")