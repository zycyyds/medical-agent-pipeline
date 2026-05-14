import importlib


MODULE_NAME = "agent_4.agents.cleaner_designer_agent"


class DummyOllamaModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyOpenAIModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyFormatter:
    pass


def _reload_module(monkeypatch):
    import configs.loader as loader
    loader.model_config = {
        "agents": {
            "agent_5": {
                "backend": "ollama",
                "model_name": "legacy-fallback",
                "temperature": 0.0,
                "seed": 666,
                "ollama": {
                    "model_name": "yaml-ollama",
                    "temperature": 0.1,
                    "seed": 123,
                },
                "openai": {
                    "model_name": "yaml-openai",
                    "api_base": "http://openai.local/v1",
                    "temperature": 0.2,
                    "timeout": 45,
                },
            },
        }
    }
    importlib.invalidate_caches()
    return importlib.reload(importlib.import_module(MODULE_NAME))


def test_uses_ollama_backend_from_yaml(monkeypatch):
    monkeypatch.delenv("STEP4_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("STEP5_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("STEP4_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("STEP5_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    module = _reload_module(monkeypatch)
    monkeypatch.setattr(module, "OllamaChatModel", DummyOllamaModel)
    monkeypatch.setattr(module, "OllamaChatFormatter", DummyFormatter)

    model, formatter = module._get_model_and_formatter()

    assert isinstance(model, DummyOllamaModel)
    assert model.kwargs["model_name"] == "yaml-ollama"
    assert model.kwargs["options"]["temperature"] == 0.1
    assert model.kwargs["options"]["seed"] == 123
    assert isinstance(formatter, DummyFormatter)


def test_uses_openai_backend_from_yaml(monkeypatch):
    monkeypatch.delenv("STEP4_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("STEP5_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("STEP4_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("STEP5_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("STEP4_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STEP5_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import configs.loader as loader
    loader.model_config = {
        "agents": {
            "agent_5": {
                "backend": "openai",
                "openai": {
                    "model_name": "yaml-openai",
                    "api_base": "http://openai.local/v1",
                    "temperature": 0.2,
                    "timeout": 45,
                },
            }
        }
    }
    module = importlib.reload(importlib.import_module(MODULE_NAME))
    monkeypatch.setattr(module, "OpenAIChatModel", DummyOpenAIModel)
    monkeypatch.setattr(module, "OpenAIChatFormatter", DummyFormatter)

    model, formatter = module._get_model_and_formatter()

    assert isinstance(model, DummyOpenAIModel)
    assert model.kwargs["model_name"] == "yaml-openai"
    assert model.kwargs["api_key"] == "test-key"
    assert model.kwargs["client_kwargs"]["base_url"] == "http://openai.local/v1"
    assert model.kwargs["client_kwargs"]["timeout"] == 45
    assert model.kwargs["generate_kwargs"]["temperature"] == 0.2
    assert isinstance(formatter, DummyFormatter)


def test_step5_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("STEP5_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("STEP5_OLLAMA_MODEL", "env-ollama")
    monkeypatch.setenv("STEP5_OLLAMA_TEMPERATURE", "0.6")
    monkeypatch.setenv("STEP5_OLLAMA_SEED", "999")

    module = _reload_module(monkeypatch)
    monkeypatch.setattr(module, "OllamaChatModel", DummyOllamaModel)
    monkeypatch.setattr(module, "OllamaChatFormatter", DummyFormatter)

    model, _ = module._get_model_and_formatter()

    assert model.kwargs["model_name"] == "env-ollama"
    assert model.kwargs["options"]["temperature"] == 0.6
    assert model.kwargs["options"]["seed"] == 999


def test_generic_model_name_does_not_override_step5(monkeypatch):
    monkeypatch.delenv("STEP4_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("STEP5_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("STEP4_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("STEP5_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-5.4")

    module = _reload_module(monkeypatch)
    monkeypatch.setattr(module, "OllamaChatModel", DummyOllamaModel)
    monkeypatch.setattr(module, "OllamaChatFormatter", DummyFormatter)

    model, _ = module._get_model_and_formatter()

    assert model.kwargs["model_name"] == "yaml-ollama"
