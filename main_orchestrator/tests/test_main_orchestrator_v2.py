import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

import main_orchestrator as orchestrator_module
import orchestrator_agent as orchestrator_agent_module
import router_agent as router_agent_module
import supervisors as supervisors_module
from contracts import PipelineRunResult, TaskSpec, TaskType, WorkerResult, WorkerStatus
from state_machine import StateMachineOrchestrator


def test_parse_args_supports_disable_memory_agent():
    args = orchestrator_module.parse_args(["--disable-memory-agent"])
    assert args.disable_memory_agent is True


def test_no_input_defaults_to_interactive_mode():
    args = orchestrator_module.parse_args([])

    assert orchestrator_module._should_enter_interactive(args) is True


def test_input_argument_runs_one_shot_mode():
    args = orchestrator_module.parse_args(["处理 mimic-10，只跑 step1"])

    assert orchestrator_module._should_enter_interactive(args) is False


def test_interactive_continue_uses_previous_records_path():
    previous = PipelineRunResult(
        status=WorkerStatus.SUCCESS,
        task_spec=TaskSpec(task_type=TaskType.STEP1_ONLY),
        worker_results=[
            {
                "state": "STEP1_RECORDING",
                "result": {
                    "artifacts": {
                        "records_path": "/tmp/reorganized_output/_meta/records.json",
                    }
                },
            }
        ],
    )

    assert (
        orchestrator_module._resolve_interactive_input("继续", previous)
        == "从 /tmp/reorganized_output/_meta/records.json 继续跑"
    )


def test_route_records_json_to_resume(tmp_path):
    records_path = tmp_path / "reorganized_output" / "_meta" / "records.json"
    records_path.parent.mkdir(parents=True)
    records_path.write_text("[]", encoding="utf-8")

    spec = orchestrator_module.route_task(str(records_path), project_root=tmp_path)

    assert spec.task_type == TaskType.RESUME_FROM_RECORDS
    assert spec.entry_artifacts["records_path"] == str(records_path.resolve())


def test_route_natural_language_step1_input(tmp_path):
    sample = tmp_path / "mimic-10"
    sample.mkdir()

    spec = orchestrator_module.route_task("处理 mimic-10，只跑 step1", project_root=tmp_path)

    assert spec.task_type == TaskType.STEP1_ONLY
    assert spec.entry_artifacts["input_path"] == str(sample.resolve())


def test_route_extracts_absolute_path_from_chinese_sentence(tmp_path):
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    spec = orchestrator_module.route_task(f"请测试{sample}这个数据集", project_root=PROJECT_ROOT)

    assert spec.task_type == TaskType.FULL_PIPELINE
    assert spec.entry_artifacts["input_path"] == str(sample.resolve())


def test_task_router_agent_falls_back_without_model_credentials(tmp_path, monkeypatch):
    sample = tmp_path / "mimic-10"
    sample.mkdir()
    monkeypatch.setattr(router_agent_module, "has_model_credentials", lambda _agent_key: False)

    spec = asyncio.run(router_agent_module.route_task_with_agent("处理 mimic-10，只跑 step1", project_root=tmp_path))

    assert spec.task_type == TaskType.STEP1_ONLY
    assert spec.entry_artifacts["input_path"] == str(sample.resolve())
    assert "兜底" in spec.intent_summary


def test_task_router_agent_completes_missing_path_from_deterministic_route(tmp_path):
    sample = tmp_path / "mimic-mini"
    sample.mkdir()
    incomplete = TaskSpec(task_type=TaskType.STEP1_ONLY)

    spec = router_agent_module._complete_with_deterministic_route(
        incomplete,
        f"请测试{sample}这个数据集，只跑 step1",
        tmp_path,
    )

    assert spec.entry_artifacts["input_path"] == str(sample.resolve())


def test_state_machine_step1_only_does_not_run_later_steps():
    calls = []

    class FakeStep1:
        async def run_from_input(self, input_path):
            calls.append(("step1", input_path))
            return WorkerResult.success("step1 ok")

    class FakeLegacy:
        async def run_step2_3(self, input_data):
            calls.append(("step2_3", input_data))
            return WorkerResult.success("step2_3 ok")

    spec = TaskSpec(
        task_type=TaskType.STEP1_ONLY,
        entry_artifacts={"input_path": "/tmp/rawdata"},
        intent_summary="只跑 Step1",
    )
    result = asyncio.run(
        StateMachineOrchestrator(
            spec,
            step1_supervisor=FakeStep1(),
            legacy_supervisor=FakeLegacy(),
        ).run()
    )

    assert result.status.value == "SUCCESS"
    assert calls == [("step1", "/tmp/rawdata")]


def test_orchestrator_direct_handoff_preserves_input_path(monkeypatch, tmp_path):
    calls = []
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return WorkerResult.success("step1 ok", artifacts={"records_count": 1, "output_file_count": 1})

    monkeypatch.setattr(orchestrator_agent_module, "run_step1_handoff", fake_handoff)
    spec = TaskSpec(
        task_type=TaskType.STEP1_ONLY,
        entry_artifacts={"input_path": str(sample)},
        resume_from_step="step1",
    )

    result = asyncio.run(orchestrator_agent_module.run_with_orchestrator_agent(spec, enable_memory_agent=False))

    assert result.status == WorkerStatus.SUCCESS
    assert calls[0]["task_type"] == "step1_only"
    assert calls[0]["input_path"] == str(sample)


def test_step1_supervisor_delegates_to_handoff(monkeypatch, tmp_path):
    calls = []

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return WorkerResult.failure(
            summary="fake handoff stopped before validation",
            status=supervisors_module.WorkerStatus.NEEDS_ESCALATION,
            artifacts={"records_path": kwargs["records_path"]},
        )

    monkeypatch.setattr(supervisors_module, "run_step1_handoff", fake_handoff)
    supervisor = supervisors_module.Step1Supervisor(
        records_path=tmp_path / "records.json",
        output_root=tmp_path / "output",
        generated_script_path=tmp_path / "generated.py",
    )

    result = asyncio.run(supervisor.run_from_input(tmp_path / "raw"))

    assert result.status == supervisors_module.WorkerStatus.NEEDS_ESCALATION
    assert calls == [
        {
            "task_type": "step1_only",
            "input_path": str(tmp_path / "raw"),
            "records_path": str(tmp_path / "records.json"),
            "output_root": str(tmp_path / "output"),
            "generated_script_path": str(tmp_path / "generated.py"),
        }
    ]
