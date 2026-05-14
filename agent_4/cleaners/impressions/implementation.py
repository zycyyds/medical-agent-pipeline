import os
import re
import pandas as pd
from agentscope.tool import ToolResponse


def _clean_impressions_value(value: str) -> str:
    if value is None:
        return value

    text = str(value)
    if text == "":
        return text

    original = text

    # Remove common de-identification placeholders, e.g. [** Name **]
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)

    # Normalize common full-width punctuation and special dashes/quotes
    replacements = {
        "\u3000": " ",
        "\r\n": "\n",
        "\r": "\n",
        "；": ";",
        "：": ":",
        "，": ",",
        "。": ".",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "‒": "-",
        "―": "-",
        "•": "|",
        "·": ".",
        "\t": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Convert repeated line-based separators / bullets into a unified pipe delimiter
    text = re.sub(r"\n+", " | ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)

    # Convert semicolon-separated long summaries into pipe-separated segments
    text = re.sub(r"\s*;\s*", " | ", text)

    # Collapse excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.:;])", r"\1", text)

    # Normalize repeated delimiters/punctuation
    text = re.sub(r"(?:\s*\|\s*){2,}", " | ", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{2,}", "-", text)

    # Trim leading/trailing separators and whitespace
    text = re.sub(r"^\s*\|\s*", "", text)
    text = re.sub(r"\s*\|\s*$", "", text)
    text = text.strip()

    return text if text != "" else original.strip()


def data_cleaning(input_path: str, index=False) -> ToolResponse:
    try:
        if not os.path.exists(input_path):
            return ToolResponse(content=f"Input file not found: {input_path}")

        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

        if "impressions" in df.columns:
            df["impressions"] = df["impressions"].apply(_clean_impressions_value)

        output_path = input_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=index)

        return ToolResponse(
            content=f"Cleaned file saved to {output_path}",
            metadata={"output_file": output_path},
        )
    except Exception as e:
        return ToolResponse(content=f"Data cleaning failed: {str(e)}")