import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

import main_orchestrator as orchestrator_module
import memory_supervisor as memory_supervisor_module


def test_resolve_model_name_uses_model_name_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gpt-5.4")

    assert orchestrator_module.resolve_model_name("main_orchestrator", "gpt-4.1-mini") == "gpt-5.4"


def test_resolve_model_name_falls_back_to_agent_config_when_env_missing(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)

    assert orchestrator_module.resolve_model_name("main_orchestrator", "gpt-4.1-mini") == orchestrator_module.AGENT_CFG["model_name"]


def test_create_memory_agent_uses_configured_openai_compatible_model(monkeypatch):
    captured = {}

    class FakeOpenAIChatModel:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

    class FakeFormatter:
        pass

    monkeypatch.setattr(memory_supervisor_module, "OpenAIChatModel", FakeOpenAIChatModel, raising=False)
    monkeypatch.setattr(memory_supervisor_module, "OpenAIChatFormatter", FakeFormatter, raising=False)

    agent = memory_supervisor_module._create_memory_agent()

    assert captured["model_kwargs"]["model_name"] == memory_supervisor_module.config.LLM_MODEL
    assert captured["model_kwargs"]["api_key"] == memory_supervisor_module.config.LLM_API_KEY
    assert captured["model_kwargs"]["client_kwargs"]["base_url"] == memory_supervisor_module.config.LLM_BASE_URL
    assert captured["model_kwargs"]["generate_kwargs"] == {
        "temperature": memory_supervisor_module.config.LLM_TEMPERATURE,
        "seed": memory_supervisor_module.config.LLM_SEED,
    }
    assert isinstance(agent.formatter, FakeFormatter)


def test_create_memory_agent_can_fallback_to_ollama_model(monkeypatch):
    captured = {}

    class FakeOllamaChatModel:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

    class FakeFormatter:
        pass

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(memory_supervisor_module.config, "LLM_API_KEY", "")
    monkeypatch.setattr(memory_supervisor_module.config, "LLM_BASE_URL", "")
    monkeypatch.setattr(memory_supervisor_module, "OllamaChatModel", FakeOllamaChatModel, raising=False)
    monkeypatch.setattr(memory_supervisor_module, "OllamaChatFormatter", FakeFormatter, raising=False)

    agent = memory_supervisor_module._create_memory_agent()

    assert captured["model_kwargs"]["model_name"] == memory_supervisor_module.config.LLM_MODEL
    assert set(captured["model_kwargs"]) == {"model_name", "options"}
    assert isinstance(agent.formatter, FakeFormatter)


def test_memory_agent_model_config_is_available():
    assert memory_supervisor_module.config.LLM_MODEL
    assert memory_supervisor_module.config.LLM_BASE_URL
