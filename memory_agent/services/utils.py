from __future__ import annotations

import hashlib
import io
import json
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("output") or block))
            else:
                text = getattr(block, "text", None)
                parts.append(str(text if text is not None else block))
        return "\n".join(part for part in parts if part.strip()).strip()
    return str(content or "").strip()


def compact_json_payload(data: Any, max_chars: int = 2000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


@contextmanager
def suppress_external_output():
    try:
        from loguru import logger

        logger.disable("reme")
        logger.disable("reme_ai")
        logger.disable("flowllm")
    except Exception:
        pass
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield
