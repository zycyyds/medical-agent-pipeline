from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agentscope.formatter import OllamaChatFormatter, OpenAIChatFormatter
    from agentscope.model import OllamaChatModel, OpenAIChatModel
except Exception:  # pragma: no cover - minimal test environments
    OllamaChatFormatter = None
    OpenAIChatFormatter = None
    OllamaChatModel = None
    OpenAIChatModel = None

from memory_agent.config import config
from memory_agent.services.memory_service import get_default_memory_service


def _memory_generate_kwargs() -> dict[str, Any]:
    return {
        "temperature": config.LLM_TEMPERATURE,
        "seed": config.LLM_SEED,
    }


def _create_memory_model_and_formatter():
    api_key = os.environ.get("OPENAI_API_KEY") or config.LLM_API_KEY or None
    base_url = os.environ.get("OPENAI_API_BASE") or config.LLM_BASE_URL
    if api_key or base_url:
        if OpenAIChatModel is None or OpenAIChatFormatter is None:
            raise RuntimeError("OpenAI AgentScope classes are unavailable.")
        model = OpenAIChatModel(
            model_name=config.get_llm_model(),
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url or "https://api.openai.com/v1"},
            generate_kwargs=_memory_generate_kwargs(),
        )
        return model, OpenAIChatFormatter()

    if OllamaChatModel is None or OllamaChatFormatter is None:
        raise RuntimeError("Ollama AgentScope classes are unavailable.")
    model = OllamaChatModel(
        model_name=config.get_llm_model(),
        options={
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
        },
    )
    return model, OllamaChatFormatter()


def _create_memory_agent():
    """Compatibility shim for older tests.

    The main pipeline no longer lets a Memory ReAct agent freely orchestrate a
    reflection loop. It uses MemoryService below. This function only preserves
    the old model-selection contract.
    """

    model, formatter = _create_memory_model_and_formatter()
    return SimpleNamespace(name="MemoryServiceAdapter", model=model, formatter=formatter)


async def get_pipeline_context(query_text: str = "", task_spec: Any | None = None) -> tuple[str, list[str]]:
    _ = task_spec
    return await get_default_memory_service().get_rule_context(query_text=query_text)


async def get_step_memory_context(
    step_name: str,
    query_text: str = "",
    task_spec: Any | None = None,
) -> tuple[str, list[str]]:
    return await get_default_memory_service().get_step_context(
        step_name=step_name,
        query_text=query_text,
        task_spec=task_spec,
    )


async def report_pipeline_result(
    trace_payload: dict[str, Any] | str,
    used_bullet_ids: list[str] | None = None,
    used_memory_ids: list[str] | None = None,
) -> str:
    ids = used_memory_ids if used_memory_ids is not None else used_bullet_ids
    return await get_default_memory_service().report_result(trace_payload, used_memory_ids=ids)
