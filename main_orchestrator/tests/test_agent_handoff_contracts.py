import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
STEP1_DIR = PROJECT_ROOT / "step-1"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, STEP1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import orchestrator_agent as orchestrator_agent_module
import step1_agent as step1_agent_module


def test_create_step1_agent_registers_step1_tools(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_register(toolkit):
        captured["registered_step1_tools"] = True
        return toolkit

    monkeypatch.setattr(step1_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(step1_agent_module, "register_step1_tools", fake_register)
    monkeypatch.setattr(step1_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    step1_agent_module.create_step1_agent()

    assert captured["name"] == "Step1ReActAgent"
    assert captured["model"] == "model"
    assert captured["formatter"] == "formatter"
    assert captured["registered_step1_tools"] is True
    assert captured["parallel_tool_calls"] is False


def test_create_orchestrator_agent_registers_handoff_tool(monkeypatch):
    captured = {"tools": []}

    class FakeToolkit:
        def register_tool_function(self, func):
            captured["tools"].append(func.__name__)

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(orchestrator_agent_module, "Toolkit", FakeToolkit)
    monkeypatch.setattr(orchestrator_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(orchestrator_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    orchestrator_agent_module.create_orchestrator_agent()

    assert captured["name"] == "OrchestratorAgent"
    assert captured["tools"] == ["handoff_to_step1_agent"]
    assert captured["parallel_tool_calls"] is False
