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


def test_parse_args_supports_disable_memory_agent():
    args = orchestrator_module.parse_args(["--disable-memory-agent"])
    assert args.disable_memory_agent is True


def test_no_input_defaults_to_interactive_mode():
    args = orchestrator_module.parse_args([])

    assert orchestrator_module._should_enter_interactive(args) is True


def test_input_argument_runs_one_shot_mode():
    args = orchestrator_module.parse_args(["处理 mimic-10，只跑 step1"])

    assert orchestrator_module._should_enter_interactive(args) is False


def test_memory_ids_from_result_collects_orchestrator_memory_events():
    result = PipelineRunResult(
        status=WorkerStatus.SUCCESS,
        task_spec=TaskSpec(task_type=TaskType.STEP5_ONLY),
        worker_results=[
            {
                "state": "MEMORY_REFLECTION",
                "result": {
                    "artifacts": {
                        "used_memory_ids": ["mem-a", "mem-b", "mem-a"],
                    }
                },
            }
        ],
    )

    assert orchestrator_module._memory_ids_from_result(result) == ["mem-a", "mem-b"]


def test_memory_recorded_by_orchestrator_detects_record_tool_event():
    result = PipelineRunResult(
        status=WorkerStatus.SUCCESS,
        task_spec=TaskSpec(task_type=TaskType.STEP5_ONLY),
        worker_results=[
            {
                "state": "MEMORY_REFLECTION",
                "result": {
                    "artifacts": {
                        "memory_recorded": True,
                    }
                },
            }
        ],
    )

    assert orchestrator_module._memory_recorded_by_orchestrator(result) is True


def test_task_type_override_recomputes_execution_boundary(tmp_path):
    sample = tmp_path / "filtered.csv"
    sample.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    spec = orchestrator_module._preview_task_spec_for_task_text(str(sample), task_type="step5_only")

    assert spec.task_type == TaskType.STEP5_ONLY
    assert spec.execution_scope == "only"
    assert spec.start_step == "step5"
    assert spec.allowed_steps == ["step5"]


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


def test_route_natural_language_step1_hyphen_input(tmp_path):
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    spec = orchestrator_module.route_task(
        f"输入{sample}这个数据集，且只运行step-1",
        project_root=tmp_path,
    )

    assert spec.task_type == TaskType.STEP1_ONLY
    assert spec.entry_artifacts["input_path"] == str(sample.resolve())


def test_route_natural_language_step4_input_csv(tmp_path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    spec = orchestrator_module.route_task(f"请运行 step4 处理 {input_csv}", project_root=tmp_path)

    assert spec.task_type == TaskType.STEP4_ONLY
    assert spec.execution_scope == "only"
    assert spec.start_step == "step4"
    assert spec.allowed_steps == ["step4"]
    assert spec.entry_artifacts["input_csv"] == str(input_csv.resolve())


def test_route_natural_language_step5_input_csv(tmp_path):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    spec = orchestrator_module.route_task(f"请只运行 step5 数据清洗，处理 {input_csv}", project_root=tmp_path)

    assert spec.task_type == TaskType.STEP5_ONLY
    assert spec.execution_scope == "only"
    assert spec.start_step == "step5"
    assert spec.allowed_steps == ["step5"]
    assert spec.entry_artifacts["filtered_csv"] == str(input_csv.resolve())


def test_route_from_step5_to_end_sets_allowed_steps(tmp_path):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    spec = orchestrator_module.route_task(f"请从 step5 继续到最后，处理 {input_csv}", project_root=tmp_path)

    assert spec.task_type == TaskType.RESUME_FROM_STEP5
    assert spec.execution_scope == "from_step_to_end"
    assert spec.start_step == "step5"
    assert spec.allowed_steps == ["step5", "step6", "step7"]
    assert spec.entry_artifacts["filtered_csv"] == str(input_csv.resolve())


def test_route_step5_quality_repair_is_not_generic_repair(tmp_path):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    spec = orchestrator_module.route_task(f"请运行 step5 数据质量修复，处理 {input_csv}", project_root=tmp_path)

    assert spec.task_type == TaskType.STEP5_ONLY
    assert spec.execution_scope == "only"
    assert spec.allowed_steps == ["step5"]
    assert spec.resume_from_step == "step5"


def test_route_natural_language_step6_input_csv(tmp_path):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    spec = orchestrator_module.route_task(f"请只运行 step6 一致性验证，处理 {input_csv}", project_root=tmp_path)

    assert spec.task_type == TaskType.STEP6_ONLY
    assert spec.execution_scope == "only"
    assert spec.start_step == "step6"
    assert spec.allowed_steps == ["step6"]
    assert spec.entry_artifacts["filtered_csv"] == str(input_csv.resolve())
    assert spec.resume_from_step == "step6"


def test_route_natural_language_step7_passed_json(tmp_path):
    passed_json = tmp_path / "passed_patients_20260519_000000.json"
    passed_json.write_text('{"passed_patient_ids": ["1"]}', encoding="utf-8")

    spec = orchestrator_module.route_task(f"请只运行 step7，处理 {passed_json}", project_root=tmp_path)

    assert spec.task_type == TaskType.STEP7_ONLY
    assert spec.execution_scope == "only"
    assert spec.start_step == "step7"
    assert spec.allowed_steps == ["step7"]
    assert spec.entry_artifacts["passed_patients_json"] == str(passed_json.resolve())
    assert spec.resume_from_step == "step7"


def test_route_natural_language_step7_collects_three_explicit_inputs(tmp_path):
    filtered_csv = tmp_path / "filtered.csv"
    selection_report = tmp_path / "selection_report.json"
    passed_json = tmp_path / "passed_patients_20260519_000000.json"
    filtered_csv.write_text("id,value\n1,2\n", encoding="utf-8")
    selection_report.write_text('{"task": "死亡预测"}', encoding="utf-8")
    passed_json.write_text('{"passed_patient_ids": ["1"]}', encoding="utf-8")

    spec = orchestrator_module.route_task(
        f"请只运行 step7，输入 {selection_report} {passed_json} {filtered_csv}，本次任务裁剪的目标是死亡预测",
        project_root=tmp_path,
    )

    assert spec.task_type == TaskType.STEP7_ONLY
    assert spec.allowed_steps == ["step7"]
    assert spec.entry_artifacts["filtered_csv"] == str(filtered_csv.resolve())
    assert spec.entry_artifacts["selection_report"] == str(selection_report.resolve())
    assert spec.entry_artifacts["passed_patients_json"] == str(passed_json.resolve())
    assert spec.entry_artifacts["task_text"] == "死亡预测"


def test_route_natural_language_step23_input_dir(tmp_path):
    step1_output = tmp_path / "step1_results"
    step1_output.mkdir()

    spec = orchestrator_module.route_task(f"请只运行 step2-3，处理 {step1_output}", project_root=tmp_path)

    assert spec.task_type == TaskType.STEP2_3_ONLY
    assert spec.execution_scope == "only"
    assert spec.start_step == "step2_3"
    assert spec.allowed_steps == ["step2_3"]
    assert spec.entry_artifacts["input_path"] == str(step1_output.resolve())


def test_route_extracts_absolute_path_from_chinese_sentence(tmp_path):
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    spec = orchestrator_module.route_task(f"请测试{sample}这个数据集", project_root=PROJECT_ROOT)

    assert spec.task_type == TaskType.FULL_PIPELINE
    assert spec.entry_artifacts["input_path"] == str(sample.resolve())


def test_route_full_pipeline_extracts_interactive_task_text(tmp_path):
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    spec = orchestrator_module.route_task(
        f"请测试{sample}这个数据集，本次任务裁剪的目标是死亡预测和离院去向预测",
        project_root=tmp_path,
    )

    assert spec.task_type == TaskType.FULL_PIPELINE
    assert spec.execution_scope == "full_pipeline"
    assert spec.start_step == "step1"
    assert spec.allowed_steps == ["step1", "step2_3", "step4", "step5", "step6", "step7"]
    assert spec.entry_artifacts["input_path"] == str(sample.resolve())
    assert spec.entry_artifacts["task_text"] == "死亡预测和离院去向预测"


def test_interactive_task_text_prompt_needed_for_full_pipeline(tmp_path):
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    assert orchestrator_module._needs_interactive_task_text(f"请测试{sample}这个数据集") is True
    assert (
        orchestrator_module._needs_interactive_task_text(
            f"请测试{sample}这个数据集，本次任务裁剪的目标是死亡预测"
        )
        is False
    )


def test_interactive_task_text_prompt_needed_for_step7(tmp_path):
    filtered_csv = tmp_path / "filtered.csv"
    filtered_csv.write_text("id,value\n1,2\n", encoding="utf-8")

    assert orchestrator_module._needs_interactive_task_text(f"请只运行 step7，处理 {filtered_csv}") is True
    assert (
        orchestrator_module._needs_interactive_task_text(
            f"请只运行 step7，处理 {filtered_csv}，本次任务裁剪的目标是死亡预测"
        )
        is False
    )


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


def test_orchestrator_agent_uses_tool_result_instead_of_direct_handoff(monkeypatch, tmp_path):
    created = []
    sample = tmp_path / "mimic-mini"
    sample.mkdir()

    class FakeAgent:
        async def __call__(self, _msg):
            class FakeResponse:
                def get_text_content(self):
                    return ""

            return FakeResponse()

    def fake_create_agent(task_spec, context=""):
        assert context == ""
        created.append(task_spec)
        return FakeAgent()

    async def fake_collect_tool_results(_agent):
        return {
            "handoff_step1_tool": [
                WorkerResult.success("step1 ok", artifacts={"records_count": 1, "output_file_count": 1}).to_dict()
            ]
        }

    monkeypatch.setattr(orchestrator_agent_module, "has_model_credentials", lambda _agent_key: True)
    monkeypatch.setattr(orchestrator_agent_module, "create_orchestrator_agent", fake_create_agent)
    monkeypatch.setattr(orchestrator_agent_module, "collect_tool_results", fake_collect_tool_results)
    spec = TaskSpec(
        task_type=TaskType.STEP1_ONLY,
        entry_artifacts={"input_path": str(sample)},
        resume_from_step="step1",
    )

    result = asyncio.run(orchestrator_agent_module.run_with_orchestrator_agent(spec, enable_memory_agent=False))

    assert result.status == WorkerStatus.SUCCESS
    assert created == [spec]


def test_orchestrator_step1_missing_input_needs_repair():
    spec = TaskSpec(task_type=TaskType.STEP1_ONLY, entry_artifacts={}, resume_from_step="step1")

    result = asyncio.run(orchestrator_agent_module.run_with_orchestrator_agent(spec, enable_memory_agent=False))

    assert result.status == WorkerStatus.NEEDS_REPAIR
    assert "input_path" in " ".join(result.repair_ticket.validator_errors)


def test_orchestrator_agent_uses_step4_tool_result(monkeypatch, tmp_path):
    created = []
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    class FakeAgent:
        async def __call__(self, _msg):
            class FakeResponse:
                def get_text_content(self):
                    return ""

            return FakeResponse()

    def fake_create_agent(task_spec, context=""):
        assert context == ""
        created.append(task_spec)
        return FakeAgent()

    async def fake_collect_tool_results(_agent):
        return {
            "handoff_step4_tool": [
                WorkerResult.success(
                    "step4 ok",
                    artifacts={"filtered_csv_path": str(input_csv), "selection_report_path": str(tmp_path / "report.json")},
                ).to_dict()
            ]
        }

    monkeypatch.setattr(orchestrator_agent_module, "has_model_credentials", lambda _agent_key: True)
    monkeypatch.setattr(orchestrator_agent_module, "create_orchestrator_agent", fake_create_agent)
    monkeypatch.setattr(orchestrator_agent_module, "collect_tool_results", fake_collect_tool_results)
    spec = TaskSpec(
        task_type=TaskType.STEP4_ONLY,
        entry_artifacts={"input_csv": str(input_csv), "task_text": "demo task"},
        resume_from_step="step4",
    )

    result = asyncio.run(orchestrator_agent_module.run_with_orchestrator_agent(spec, enable_memory_agent=False))

    assert result.status == WorkerStatus.SUCCESS
    assert created == [spec]
    assert result.states_visited[-1].value == "DONE"


def test_orchestrator_agent_uses_step23_tool_result(monkeypatch, tmp_path):
    created = []
    input_dir = tmp_path / "step1_results"
    next_input_dir = tmp_path / "next_input"
    input_dir.mkdir()
    next_input_dir.mkdir()
    next_input = next_input_dir / "input.csv"
    next_input.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    class FakeAgent:
        async def __call__(self, _msg):
            class FakeResponse:
                def get_text_content(self):
                    return ""

            return FakeResponse()

    def fake_create_agent(task_spec, context=""):
        assert context == ""
        created.append(task_spec)
        return FakeAgent()

    async def fake_collect_tool_results(_agent):
        return {
            "handoff_step2_3_tool": [
                WorkerResult.success(
                    "step2-3 ok",
                    artifacts={"next_input_csv": str(next_input), "output_row_count": 1, "output_column_count": 2},
                ).to_dict()
            ]
        }

    monkeypatch.setattr(orchestrator_agent_module, "has_model_credentials", lambda _agent_key: True)
    monkeypatch.setattr(orchestrator_agent_module, "create_orchestrator_agent", fake_create_agent)
    monkeypatch.setattr(orchestrator_agent_module, "collect_tool_results", fake_collect_tool_results)
    spec = TaskSpec(
        task_type=TaskType.STEP2_3_ONLY,
        entry_artifacts={"input_path": str(input_dir)},
        resume_from_step="step2_3",
    )

    result = asyncio.run(orchestrator_agent_module.run_with_orchestrator_agent(spec, enable_memory_agent=False))

    assert result.status == WorkerStatus.SUCCESS
    assert created == [spec]
    assert result.states_visited[-1].value == "DONE"


def test_orchestrator_agent_uses_step5_tool_result(monkeypatch, tmp_path):
    created = []
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    class FakeAgent:
        async def __call__(self, _msg):
            class FakeResponse:
                def get_text_content(self):
                    return ""

            return FakeResponse()

    def fake_create_agent(task_spec, context=""):
        assert context == ""
        created.append(task_spec)
        return FakeAgent()

    async def fake_collect_tool_results(_agent):
        return {
            "handoff_step5_tool": [
                WorkerResult.success(
                    "step5 ok",
                    artifacts={"next_input_csv": str(input_csv), "cleaned_csv_path": str(input_csv)},
                ).to_dict()
            ]
        }

    monkeypatch.setattr(orchestrator_agent_module, "has_model_credentials", lambda _agent_key: True)
    monkeypatch.setattr(orchestrator_agent_module, "create_orchestrator_agent", fake_create_agent)
    monkeypatch.setattr(orchestrator_agent_module, "collect_tool_results", fake_collect_tool_results)
    spec = TaskSpec(
        task_type=TaskType.STEP5_ONLY,
        entry_artifacts={"filtered_csv": str(input_csv)},
        resume_from_step="step5",
    )

    result = asyncio.run(orchestrator_agent_module.run_with_orchestrator_agent(spec, enable_memory_agent=False))

    assert result.status == WorkerStatus.SUCCESS
    assert created == [spec]
    assert result.states_visited[-1].value == "DONE"


def test_orchestrator_planner_fails_when_required_steps_are_missing():
    spec = TaskSpec(
        task_type=TaskType.RESUME_FROM_STEP5,
        execution_scope="from_step_to_end",
        start_step="step6",
        allowed_steps=["step6", "step7"],
        entry_artifacts={
            "filtered_csv": "/tmp/filtered.csv",
            "selection_report": "/tmp/selection_report.json",
            "task_text": "死亡预测",
        },
    )

    result = orchestrator_agent_module._pipeline_from_planner_worker_results(
        spec,
        [
            (
                "step6",
                WorkerResult.success(
                    "step6 ok",
                    artifacts={"passed_patients_json": "/tmp/passed_patients.json"},
                ),
            )
        ],
    )

    assert result.status == WorkerStatus.NEEDS_REPAIR
    assert result.repair_ticket.repair_type == "llm_orchestration_missing_steps"
    assert result.repair_ticket.input_artifacts["missing_steps"] == ["step7"]


def test_orchestrator_planner_fails_when_required_steps_are_out_of_order():
    spec = TaskSpec(
        task_type=TaskType.RESUME_FROM_STEP5,
        execution_scope="from_step_to_end",
        start_step="step6",
        allowed_steps=["step6", "step7"],
        entry_artifacts={
            "filtered_csv": "/tmp/filtered.csv",
            "selection_report": "/tmp/selection_report.json",
            "task_text": "死亡预测",
        },
    )

    result = orchestrator_agent_module._pipeline_from_planner_worker_results(
        spec,
        [
            ("step7", WorkerResult.success("step7 ok")),
            ("step6", WorkerResult.success("step6 ok")),
        ],
    )

    assert result.status == WorkerStatus.NEEDS_REPAIR
    assert "required_steps" in " ".join(result.repair_ticket.validator_errors)


def test_orchestrator_planner_succeeds_when_all_required_steps_run_in_order():
    spec = TaskSpec(
        task_type=TaskType.RESUME_FROM_STEP5,
        execution_scope="from_step_to_end",
        start_step="step6",
        allowed_steps=["step6", "step7"],
        entry_artifacts={
            "filtered_csv": "/tmp/filtered.csv",
            "selection_report": "/tmp/selection_report.json",
            "task_text": "死亡预测",
        },
    )

    result = orchestrator_agent_module._pipeline_from_planner_worker_results(
        spec,
        [
            ("step6", WorkerResult.success("step6 ok")),
            ("step7", WorkerResult.success("step7 ok")),
        ],
    )

    assert result.status == WorkerStatus.SUCCESS
    assert [state.value for state in result.states_visited] == ["INIT", "STEP6_RUNNING", "STEP7_RUNNING", "MEMORY_REFLECTION", "DONE"]


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


def test_step4_supervisor_delegates_to_handoff(monkeypatch, tmp_path):
    calls = []
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return WorkerResult.failure(
            summary="fake handoff stopped before validation",
            status=supervisors_module.WorkerStatus.NEEDS_ESCALATION,
            artifacts={"input_path": kwargs["input_path"]},
        )

    monkeypatch.setattr(supervisors_module, "run_step4_handoff", fake_handoff)
    supervisor = supervisors_module.Step4Supervisor(output_root=tmp_path / "step4")

    result = asyncio.run(supervisor.run_from_input(input_csv, task_text="demo task"))

    assert result.status == supervisors_module.WorkerStatus.NEEDS_ESCALATION
    assert calls == [
        {
            "task_type": "resume_from_step2_3",
            "input_path": str(input_csv),
            "output_root": str(tmp_path / "step4"),
            "task_text": "demo task",
            "memory_context": "",
        }
    ]


def test_step23_supervisor_delegates_to_handoff(monkeypatch, tmp_path):
    calls = []
    input_dir = tmp_path / "step1_results"
    input_dir.mkdir()

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return WorkerResult.failure(
            summary="fake handoff stopped before validation",
            status=supervisors_module.WorkerStatus.NEEDS_ESCALATION,
            artifacts={"input_path": kwargs["input_path"]},
        )

    monkeypatch.setattr(supervisors_module, "run_step23_handoff", fake_handoff)
    supervisor = supervisors_module.Step2_3Supervisor(output_root=tmp_path / "step2_3")

    result = asyncio.run(supervisor.run_from_input(input_dir))

    assert result.status == supervisors_module.WorkerStatus.NEEDS_ESCALATION
    assert calls == [
        {
            "task_type": "step2_3_only",
            "input_path": str(input_dir),
            "output_root": str(tmp_path / "step2_3"),
            "mode": "auto",
            "user_hint": "",
        }
    ]


def test_step5_supervisor_delegates_to_handoff(monkeypatch, tmp_path):
    calls = []
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("subject_id,value\n1,2\n", encoding="utf-8")

    async def fake_handoff(**kwargs):
        calls.append(kwargs)
        return WorkerResult.failure(
            summary="fake handoff stopped before validation",
            status=supervisors_module.WorkerStatus.NEEDS_ESCALATION,
            artifacts={"input_path": kwargs["input_path"]},
        )

    monkeypatch.setattr(supervisors_module, "run_step5_handoff", fake_handoff)
    supervisor = supervisors_module.Step5Supervisor(output_root=tmp_path / "step5")

    result = asyncio.run(supervisor.run_from_input(input_csv))

    assert result.status == supervisors_module.WorkerStatus.NEEDS_ESCALATION
    assert calls == [
        {
            "task_type": "resume_from_step4",
            "input_path": str(input_csv),
            "output_root": str(tmp_path / "step5"),
        }
    ]


def test_step6_supervisor_delegates_to_agent6_runtime(monkeypatch, tmp_path):
    calls = []
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
    report_txt = tmp_path / "step6" / "consistency_report.txt"
    report_json = tmp_path / "step6" / "consistency_report.json"
    passed_json = tmp_path / "step6" / "passed_patients.json"
    report_txt.parent.mkdir()
    report_txt.write_text("ok", encoding="utf-8")
    report_json.write_text("{}", encoding="utf-8")
    passed_json.write_text('{"passed_count": 2, "passed_patient_ids": ["1", "2"]}', encoding="utf-8")

    async def fake_run_step6_handoff(**kwargs):
        calls.append(kwargs)
        return WorkerResult.success(
            "ok",
            artifacts={
            "input_csv": str(input_csv),
            "raw_csv": str(input_csv),
            "output_step6": str(report_txt.parent),
            "step6_report_txt": str(report_txt),
            "step6_report_json": str(report_json),
            "passed_patients_json": str(passed_json),
            "passed_patient_count": 2,
            "passed_patient_ids_sample": ["1", "2"],
            },
        )

    monkeypatch.setattr(supervisors_module, "run_step6_handoff", fake_run_step6_handoff)
    supervisor = supervisors_module.Step6Supervisor(output_root=tmp_path / "step6")

    result = asyncio.run(supervisor.run_from_input(input_csv, memory_context="memory ctx"))

    assert result.status == WorkerStatus.SUCCESS
    assert calls[0]["input_path"] == str(input_csv)
    assert calls[0]["output_root"] == str(tmp_path / "step6")
    assert calls[0]["memory_context"] == "memory ctx"
    assert calls[0]["task_type"] == "resume_from_step5"
    assert result.artifacts["passed_patient_count"] == 2
    assert result.artifacts["passed_patient_ids_sample"] == ["1", "2"]


def test_step7_supervisor_delegates_to_agent7_runtime(monkeypatch, tmp_path):
    calls = []
    input_csv = tmp_path / "filtered.csv"
    selection_report = tmp_path / "selection_report.json"
    passed_json = tmp_path / "passed_patients.json"
    output_dir = tmp_path / "step7"
    input_csv.write_text("id,value,label\n1,a,A\n2,b,B\n", encoding="utf-8")
    selection_report.write_text("{}", encoding="utf-8")
    passed_json.write_text('{"passed_count": 2, "passed_patient_ids": ["1", "2"]}', encoding="utf-8")
    output_dir.mkdir()
    dataset_csv = output_dir / "ml_dataset_mock.csv"
    dataset_jsonl = output_dir / "ml_dataset_mock.jsonl"
    dataset_json = output_dir / "ml_dataset_mock.json"
    model_config = output_dir / "model_config_mock.json"
    for path in (dataset_csv, dataset_jsonl, dataset_json, model_config):
        path.write_text("{}", encoding="utf-8")

    async def fake_run_step7_handoff(**kwargs):
        calls.append(kwargs)
        return WorkerResult.success(
            "ok",
            artifacts={
            "input_csv": str(input_csv),
            "selection_report": str(selection_report),
            "passed_patients_json": str(passed_json),
            "passed_patient_count": 2,
            "output_step7": str(output_dir),
            "step7_dataset_csvs": [str(dataset_csv)],
            "step7_dataset_jsonls": [str(dataset_jsonl)],
            "step7_dataset_jsons": [str(dataset_json)],
            "model_config_jsons": [str(model_config)],
            },
        )

    monkeypatch.setattr(supervisors_module, "run_step7_handoff", fake_run_step7_handoff)
    supervisor = supervisors_module.Step7Supervisor(output_root=output_dir)

    result = asyncio.run(
        supervisor.run_from_input(
            input_csv,
            selection_report=selection_report,
            passed_patients_json=passed_json,
            memory_context="memory ctx",
        )
    )

    assert result.status == WorkerStatus.SUCCESS
    assert calls[0]["input_path"] == str(input_csv)
    assert calls[0]["selection_report"] == str(selection_report)
    assert calls[0]["passed_patients_json"] == str(passed_json)
    assert calls[0]["output_root"] == str(output_dir)
    assert calls[0]["task_type"] == "resume_from_step6"
    assert result.artifacts["step7_dataset_count"] == 1
    assert result.artifacts["memory_context_used"] is True
