import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
STEP1_DIR = PROJECT_ROOT / "step-1"
STEP4_DIR = PROJECT_ROOT / "step-4"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, STEP1_DIR, STEP4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import orchestrator_agent as orchestrator_agent_module
import step1_agent as step1_agent_module
import step4_agent as step4_agent_module
from contracts import TaskSpec, TaskType, WorkerResult


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


def test_create_step4_agent_registers_step4_tools(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_register(toolkit):
        captured["registered_step4_tools"] = True
        return toolkit

    monkeypatch.setattr(step4_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(step4_agent_module, "register_step4_tools", fake_register)
    monkeypatch.setattr(step4_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    step4_agent_module.create_step4_agent()

    assert captured["name"] == "Step4ReActAgent"
    assert captured["model"] == "model"
    assert captured["formatter"] == "formatter"
    assert captured["registered_step4_tools"] is True
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

    spec = TaskSpec(task_type=TaskType.STEP1_ONLY, entry_artifacts={"input_path": str(PROJECT_ROOT)})
    orchestrator_agent_module.create_orchestrator_agent(spec)

    assert captured["name"] == "OrchestratorAgent"
    assert captured["tools"] == ["handoff_step1_from_task_spec_tool"]
    assert captured["parallel_tool_calls"] is False


def test_create_orchestrator_agent_registers_step4_handoff_tool(monkeypatch, tmp_path):
    captured = {"tools": []}

    class FakeToolkit:
        def register_tool_function(self, func):
            captured["tools"].append(func.__name__)

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator_agent_module, "Toolkit", FakeToolkit)
    monkeypatch.setattr(orchestrator_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(orchestrator_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    spec = TaskSpec(task_type=TaskType.RESUME_FROM_STEP2_3, entry_artifacts={"input_csv": str(input_csv)})
    orchestrator_agent_module.create_orchestrator_agent(spec)

    assert captured["name"] == "OrchestratorAgent"
    assert captured["tools"] == ["handoff_step4_from_task_spec_tool"]
    assert captured["parallel_tool_calls"] is False


def test_locked_step1_handoff_tool_preserves_task_spec_input_path(monkeypatch, tmp_path):
    calls = []
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return orchestrator_agent_module.json_tool_response(
            WorkerResult.success("ok", artifacts={"output_file_count": 1}).to_dict()
        )

    monkeypatch.setattr(orchestrator_agent_module, "handoff_to_step1_agent", fake_handoff)
    spec = TaskSpec(task_type=TaskType.STEP1_ONLY, entry_artifacts={"input_path": str(sample)})

    tool = orchestrator_agent_module._make_locked_step1_handoff_tool(spec)
    asyncio.run(tool())

    assert calls[0]["task_type"] == "step1_only"
    assert calls[0]["input_path"] == str(sample)


def test_locked_step4_handoff_tool_preserves_task_spec_input_csv(monkeypatch, tmp_path):
    calls = []
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return orchestrator_agent_module.json_tool_response(
            WorkerResult.success("ok", artifacts={"filtered_csv_path": str(input_csv)}).to_dict()
        )

    monkeypatch.setattr(orchestrator_agent_module, "handoff_to_step4_agent", fake_handoff)
    spec = TaskSpec(task_type=TaskType.RESUME_FROM_STEP2_3, entry_artifacts={"input_csv": str(input_csv)})

    tool = orchestrator_agent_module._make_locked_step4_handoff_tool(spec, memory_context="memory ctx")
    asyncio.run(tool())

    assert calls[0]["task_type"] == "resume_from_step2_3"
    assert calls[0]["input_path"] == str(input_csv)
    assert calls[0]["memory_context"] == "memory ctx"
