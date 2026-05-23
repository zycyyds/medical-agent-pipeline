import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
STEP1_DIR = PROJECT_ROOT / "step-1"
STEP23_DIR = PROJECT_ROOT / "step-2-3"
STEP4_DIR = PROJECT_ROOT / "step-4"
STEP5_DIR = PROJECT_ROOT / "step-5"
STEP6_DIR = PROJECT_ROOT / "step-6"
STEP7_DIR = PROJECT_ROOT / "step-7"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, STEP1_DIR, STEP23_DIR, STEP4_DIR, STEP5_DIR, STEP6_DIR, STEP7_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import orchestrator_agent as orchestrator_agent_module
import step1_agent as step1_agent_module
import step23_agent as step23_agent_module
import step4_agent as step4_agent_module
import step5_agent as step5_agent_module
import step6_agent as step6_agent_module
import step7_agent as step7_agent_module
from contracts import TaskSpec, TaskType, WorkerResult, WorkerStatus


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


def test_create_step23_agent_registers_step23_tools(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_register(toolkit):
        captured["registered_step23_tools"] = True
        return toolkit

    monkeypatch.setattr(step23_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(step23_agent_module, "register_step23_tools", fake_register)
    monkeypatch.setattr(step23_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    step23_agent_module.create_step23_agent()

    assert captured["name"] == "Step2_3ReActAgent"
    assert captured["model"] == "model"
    assert captured["formatter"] == "formatter"
    assert captured["registered_step23_tools"] is True
    assert captured["parallel_tool_calls"] is False


def test_create_step5_agent_registers_step5_tools(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_register(toolkit):
        captured["registered_step5_tools"] = True
        return toolkit

    monkeypatch.setattr(step5_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(step5_agent_module, "register_step5_tools", fake_register)
    monkeypatch.setattr(step5_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    step5_agent_module.create_step5_agent()

    assert captured["name"] == "Step5ReActAgent"
    assert captured["model"] == "model"
    assert captured["formatter"] == "formatter"
    assert captured["registered_step5_tools"] is True
    assert captured["parallel_tool_calls"] is False


def test_create_step6_agent_registers_step6_tools(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_register(toolkit):
        captured["registered_step6_tools"] = True
        return toolkit

    monkeypatch.setattr(step6_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(step6_agent_module, "register_step6_tools", fake_register)
    monkeypatch.setattr(step6_agent_module.runtime, "create_model", lambda: "model")

    step6_agent_module.create_step6_agent()

    assert captured["name"] == "ConsistencyAgent"
    assert captured["model"] == "model"
    assert captured["registered_step6_tools"] is True
    assert captured["parallel_tool_calls"] is False


def test_create_step7_agent_registers_step7_tools(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_register(toolkit):
        captured["registered_step7_tools"] = True
        return toolkit

    monkeypatch.setattr(step7_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(step7_agent_module, "register_step7_tools", fake_register)
    monkeypatch.setattr(step7_agent_module.runtime, "create_model", lambda: "model")

    step7_agent_module.create_step7_agent()

    assert captured["name"] == "DataPrepAgent"
    assert captured["model"] == "model"
    assert captured["registered_step7_tools"] is True
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
    assert captured["tools"] == orchestrator_agent_module.MEMORY_TOOL_NAMES + ["handoff_step1_tool"]
    assert captured["parallel_tool_calls"] is False


def test_create_orchestrator_agent_registers_step23_handoff_tool(monkeypatch, tmp_path):
    captured = {"tools": []}

    class FakeToolkit:
        def register_tool_function(self, func):
            captured["tools"].append(func.__name__)

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    input_dir = tmp_path / "step1_results"
    input_dir.mkdir()
    monkeypatch.setattr(orchestrator_agent_module, "Toolkit", FakeToolkit)
    monkeypatch.setattr(orchestrator_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(orchestrator_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    spec = TaskSpec(task_type=TaskType.STEP2_3_ONLY, entry_artifacts={"input_path": str(input_dir)})
    orchestrator_agent_module.create_orchestrator_agent(spec)

    assert captured["name"] == "OrchestratorAgent"
    assert captured["tools"] == orchestrator_agent_module.MEMORY_TOOL_NAMES + ["handoff_step2_3_tool"]
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

    spec = TaskSpec(
        task_type=TaskType.RESUME_FROM_STEP2_3,
        execution_scope="from_step_to_end",
        start_step="step4",
        allowed_steps=["step4", "step5", "step6", "step7"],
        entry_artifacts={"input_csv": str(input_csv), "task_text": "死亡预测和离院去向预测"},
    )
    orchestrator_agent_module.create_orchestrator_agent(spec)

    assert captured["name"] == "OrchestratorAgent"
    assert captured["tools"] == orchestrator_agent_module.MEMORY_TOOL_NAMES + [
        "handoff_step4_tool",
        "handoff_step5_tool",
        "handoff_step6_tool",
        "handoff_step7_tool",
    ]
    assert captured["parallel_tool_calls"] is False


def test_create_orchestrator_agent_registers_step5_handoff_tool(monkeypatch, tmp_path):
    captured = {"tools": []}

    class FakeToolkit:
        def register_tool_function(self, func):
            captured["tools"].append(func.__name__)

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator_agent_module, "Toolkit", FakeToolkit)
    monkeypatch.setattr(orchestrator_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(orchestrator_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    spec = TaskSpec(task_type=TaskType.STEP5_ONLY, entry_artifacts={"filtered_csv": str(input_csv)})
    orchestrator_agent_module.create_orchestrator_agent(spec)

    assert captured["name"] == "OrchestratorAgent"
    assert captured["tools"] == orchestrator_agent_module.MEMORY_TOOL_NAMES + ["handoff_step5_tool"]
    assert captured["parallel_tool_calls"] is False


def test_create_orchestrator_agent_uses_disabled_memory_prompt(monkeypatch, tmp_path):
    captured = {"tools": []}

    class FakeToolkit:
        def register_tool_function(self, func):
            captured["tools"].append(func.__name__)

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator_agent_module, "Toolkit", FakeToolkit)
    monkeypatch.setattr(orchestrator_agent_module, "ReActAgent", FakeAgent)
    monkeypatch.setattr(orchestrator_agent_module, "create_openai_model_and_formatter", lambda *_args: ("model", "formatter"))

    spec = TaskSpec(task_type=TaskType.STEP5_ONLY, entry_artifacts={"filtered_csv": str(input_csv)})
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=False)
    orchestrator_agent_module.create_orchestrator_agent(spec, planner_runtime=runtime)

    assert captured["tools"] == ["handoff_step5_tool"]
    assert "本轮未注册 Memory 工具" in captured["sys_prompt"]
    assert "首次 handoff 前必须先显式做一次 Memory 决策" not in captured["sys_prompt"]


def test_planner_step_tool_rejects_disallowed_step():
    spec = TaskSpec(
        task_type=TaskType.STEP5_ONLY,
        execution_scope="only",
        start_step="step5",
        allowed_steps=["step5"],
        entry_artifacts={"filtered_csv": "/tmp/filtered.csv"},
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=False)

    response = asyncio.run(runtime.handoff_step6_tool())
    payload = orchestrator_agent_module.parse_tool_response(response)

    assert payload["status"] == "NEEDS_REPAIR"
    assert "不允许调用" in payload["summary"]
    assert payload["artifacts"]["allowed_steps"] == ["step5"]


def test_planner_does_not_propagate_step1_output_root_to_later_steps(tmp_path):
    spec = TaskSpec(
        task_type=TaskType.FULL_PIPELINE,
        execution_scope="full_pipeline",
        start_step="step1",
        allowed_steps=["step1", "step2_3", "step4", "step5", "step6", "step7"],
        entry_artifacts={"input_path": str(tmp_path / "mimic-mini"), "task_text": "死亡预测"},
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=False)
    step1_output = tmp_path / "program" / "output" / "step1_results"

    runtime._record("step1", WorkerResult.success("step1 ok", artifacts={"output_root": str(step1_output)}))
    payload = orchestrator_agent_module._step23_call_payload(runtime._runtime_task_spec())

    assert runtime.artifacts["step1_output_root"] == str(step1_output)
    assert runtime.artifacts["input_path"] == str(step1_output)
    assert "output_root" not in runtime.artifacts
    assert payload["input_path"] == str(step1_output)
    assert payload["output_root"].endswith("program/output/step2_3_results")


def test_planner_blocks_later_step_after_previous_failure():
    spec = TaskSpec(
        task_type=TaskType.RESUME_FROM_STEP4,
        execution_scope="from_step_to_end",
        start_step="step5",
        allowed_steps=["step5", "step6", "step7"],
        entry_artifacts={"filtered_csv": "/tmp/filtered.csv", "task_text": "死亡预测"},
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=False)
    runtime._record(
        "step5",
        WorkerResult.failure(
            "Step5 产物标准化失败。",
            issues=["same file"],
            status=WorkerStatus.NEEDS_REPAIR,
            artifacts={"cleaned_csv_path": "/tmp/cleaned.csv"},
        ),
    )

    response = asyncio.run(runtime.handoff_step6_tool())
    payload = orchestrator_agent_module.parse_tool_response(response)

    assert payload["status"] == "NEEDS_REPAIR"
    assert "前置步骤 step5 未成功" in payload["summary"]
    assert payload["artifacts"]["failed_step"] == "step5"


def test_planner_rejects_out_of_order_step_call_in_full_pipeline(tmp_path):
    spec = TaskSpec(
        task_type=TaskType.FULL_PIPELINE,
        execution_scope="full_pipeline",
        start_step="step1",
        allowed_steps=["step1", "step2_3", "step4", "step5", "step6", "step7"],
        entry_artifacts={"input_path": str(tmp_path / "mimic-mini"), "task_text": "死亡预测"},
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=False)

    response = asyncio.run(runtime.handoff_step6_tool())
    payload = orchestrator_agent_module.parse_tool_response(response)

    assert payload["status"] == "NEEDS_REPAIR"
    assert "当前应先执行 step1" in payload["summary"]
    assert payload["artifacts"]["requested_step"] == "step6"
    assert payload["artifacts"]["expected_next_step"] == "step1"


def test_planner_memory_step_search_updates_runtime_context(monkeypatch):
    spec = TaskSpec(
        task_type=TaskType.STEP5_ONLY,
        execution_scope="only",
        start_step="step5",
        allowed_steps=["step5"],
        entry_artifacts={"filtered_csv": "/tmp/filtered.csv"},
        intent_summary="清洗数据",
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=True)

    async def fake_get_step_memory_context(step_name, query_text="", task_spec=None):
        assert step_name == "step5"
        assert query_text == "清洗数据"
        assert task_spec.allowed_steps == ["step5"]
        return "step5 memory context", ["mem-step5"]

    monkeypatch.setattr(orchestrator_agent_module, "get_step_memory_context", fake_get_step_memory_context)

    response = asyncio.run(runtime.memory_search_step_tool("step5"))
    payload = orchestrator_agent_module.parse_tool_response(response)

    assert payload["status"] == "SUCCESS"
    assert payload["used_memory_ids"] == ["mem-step5"]
    assert runtime.context == "step5 memory context"
    assert runtime.memory_used_ids == ["mem-step5"]
    assert runtime.memory_events[-1]["state"] == "MEMORY_REFLECTION"


def test_planner_memory_step_search_rejects_disallowed_step():
    spec = TaskSpec(
        task_type=TaskType.STEP5_ONLY,
        execution_scope="only",
        start_step="step5",
        allowed_steps=["step5"],
        entry_artifacts={"filtered_csv": "/tmp/filtered.csv"},
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=True)

    response = asyncio.run(runtime.memory_search_step_tool("step6"))
    payload = orchestrator_agent_module.parse_tool_response(response)

    assert payload["status"] == "NEEDS_REPAIR"
    assert payload["requested_step"] == "step6"
    assert payload["allowed_steps"] == ["step5"]


def test_planner_requires_explicit_memory_decision_before_handoff():
    spec = TaskSpec(
        task_type=TaskType.STEP5_ONLY,
        execution_scope="only",
        start_step="step5",
        allowed_steps=["step5"],
        entry_artifacts={"filtered_csv": "/tmp/filtered.csv"},
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=True)

    response = asyncio.run(runtime.handoff_step5_tool())
    payload = orchestrator_agent_module.parse_tool_response(response)

    assert payload["status"] == "NEEDS_REPAIR"
    assert "必须先显式决定是否使用 Memory" in payload["summary"]
    assert payload["artifacts"]["requested_step"] == "step5"
    assert "memory_skip_tool" in payload["artifacts"]["available_memory_tools"]
    assert runtime.results == []
    assert runtime.memory_events[-1]["state"] == "MEMORY_REFLECTION"


def test_planner_memory_skip_allows_handoff(monkeypatch):
    calls = []

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return orchestrator_agent_module.json_tool_response(
            WorkerResult.success("ok", artifacts={"cleaned_csv_path": "/tmp/cleaned.csv"}).to_dict()
        )

    monkeypatch.setattr(orchestrator_agent_module, "handoff_to_step5_agent", fake_handoff)
    spec = TaskSpec(
        task_type=TaskType.STEP5_ONLY,
        execution_scope="only",
        start_step="step5",
        allowed_steps=["step5"],
        entry_artifacts={"filtered_csv": "/tmp/filtered.csv"},
    )
    runtime = orchestrator_agent_module.HandoffPlannerRuntime(spec, context="", enable_memory_agent=True)

    skip_response = asyncio.run(runtime.memory_skip_tool("输入和顺序已经明确。"))
    skip_payload = orchestrator_agent_module.parse_tool_response(skip_response)
    response = asyncio.run(runtime.handoff_step5_tool())
    payload = orchestrator_agent_module.parse_tool_response(response)

    assert skip_payload["status"] == "SUCCESS"
    assert runtime.memory_decision_made is True
    assert runtime.memory_decision_action == "skip"
    assert calls
    assert payload["status"] == "SUCCESS"


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


def test_locked_step23_handoff_tool_preserves_task_spec_input_path(monkeypatch, tmp_path):
    calls = []
    input_dir = tmp_path / "step1_results"
    input_dir.mkdir()

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return orchestrator_agent_module.json_tool_response(
            WorkerResult.success("ok", artifacts={"next_input_csv": str(input_dir / "input.csv")}).to_dict()
        )

    monkeypatch.setattr(orchestrator_agent_module, "handoff_to_step23_agent", fake_handoff)
    spec = TaskSpec(task_type=TaskType.STEP2_3_ONLY, entry_artifacts={"input_path": str(input_dir), "step23_mode": "directory"})

    tool = orchestrator_agent_module._make_locked_step23_handoff_tool(spec)
    asyncio.run(tool())

    assert calls[0]["task_type"] == "step2_3_only"
    assert calls[0]["input_path"] == str(input_dir)
    assert calls[0]["mode"] == "directory"


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
    spec = TaskSpec(
        task_type=TaskType.RESUME_FROM_STEP2_3,
        entry_artifacts={"input_csv": str(input_csv), "task_text": "死亡预测和离院去向预测"},
    )

    tool = orchestrator_agent_module._make_locked_step4_handoff_tool(spec, memory_context="memory ctx")
    asyncio.run(tool())

    assert calls[0]["task_type"] == "resume_from_step2_3"
    assert calls[0]["input_path"] == str(input_csv)
    assert calls[0]["task_text"] == "死亡预测和离院去向预测"
    assert calls[0]["memory_context"] == "memory ctx"


def test_locked_step5_handoff_tool_preserves_task_spec_filtered_csv(monkeypatch, tmp_path):
    calls = []
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return orchestrator_agent_module.json_tool_response(
            WorkerResult.success("ok", artifacts={"next_input_csv": str(input_csv)}).to_dict()
        )

    monkeypatch.setattr(orchestrator_agent_module, "handoff_to_step5_agent", fake_handoff)
    spec = TaskSpec(task_type=TaskType.RESUME_FROM_STEP4, entry_artifacts={"filtered_csv": str(input_csv)})

    tool = orchestrator_agent_module._make_locked_step5_handoff_tool(spec)
    asyncio.run(tool())

    assert calls[0]["task_type"] == "resume_from_step4"
    assert calls[0]["input_path"] == str(input_csv)
