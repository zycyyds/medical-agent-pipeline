import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import memory_agent.main_memory as main_memory_module


def test_memory_model_factory_builds_openai_model_from_env(monkeypatch):
    captured = {}

    class FakeOpenAIChatModel:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setenv("MODEL_NAME", "gpt-5.4")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:8317/v1")
    monkeypatch.setattr(main_memory_module, "OpenAIChatModel", FakeOpenAIChatModel, raising=False)

    main_memory_module._create_memory_model_and_formatter()

    assert captured["kwargs"]["model_name"] == "gpt-5.4"
    assert captured["kwargs"]["api_key"] == "test-key"
    assert captured["kwargs"]["client_kwargs"]["base_url"] == "http://127.0.0.1:8317/v1"


def test_memory_agent_model_falls_back_to_config_when_env_missing(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)

    assert main_memory_module.config.get_llm_model() == main_memory_module.config.LLM_MODEL


def test_main_memory_uses_openai_model(monkeypatch):
    captured = {}

    class FakeOpenAIChatModel:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    class FakeFormatter:
        pass

    class StopAfterInit(Exception):
        pass

    monkeypatch.setenv("MODEL_NAME", "gpt-5.4")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:8317/v1")
    monkeypatch.setattr(main_memory_module, "OpenAIChatModel", FakeOpenAIChatModel, raising=False)
    monkeypatch.setattr(main_memory_module, "OpenAIChatFormatter", FakeFormatter, raising=False)
    monkeypatch.setattr(main_memory_module, "UserAgent", lambda name: (_ for _ in ()).throw(StopAfterInit()))

    try:
        import asyncio
        asyncio.run(main_memory_module.main())
    except StopAfterInit:
        pass

    assert captured["kwargs"]["model_name"] == "gpt-5.4"
    assert captured["kwargs"]["api_key"] == "test-key"
    assert captured["kwargs"]["client_kwargs"]["base_url"] == "http://127.0.0.1:8317/v1"
