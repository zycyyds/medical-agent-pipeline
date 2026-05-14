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
from contracts import TaskSpec, TaskType, WorkerResult
from state_machine import StateMachineOrchestrator


def test_parse_args_supports_disable_memory_agent():
    args = orchestrator_module.parse_args(["--disable-memory-agent"])
    assert args.disable_memory_agent is True


def test_route_records_json_to_resume(tmp_path):
    records_path = tmp_path / "reorganized_output" / "_meta" / "records.json"
    records_path.parent.mkdir(parents=True)
    records_path.write_text("[]", encoding="utf-8")

    spec = orchestrator_module.route_task(str(records_path), project_root=tmp_path)

    assert spec.task_type == TaskType.RESUME_FROM_RECORDS
    assert spec.entry_artifacts["records_path"] == str(records_path.resolve())


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
