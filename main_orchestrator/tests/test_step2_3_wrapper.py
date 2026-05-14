import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from agentscope.tool import ToolResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
STEP_WRAPPERS = ORCHESTRATOR_DIR / "step_wrappers.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

import step_wrappers


class _TraceCollectorStub:
    def __init__(self):
        self.calls = []

    def record_step(self, **kwargs):
        self.calls.append(kwargs)


class _FakeReactMedicalAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def reply(self, _msg):
        return SimpleNamespace(
            metadata={
                "success": True,
                "data_type": "csv",
                "statistics": {"row_count": 2},
                "output_dir": self.kwargs["output_dir"],
                "output_csv": str(Path(self.kwargs["output_dir"]) / "cleaned.csv"),
            },
            content="处理完成",
        )


class _ExplodingReactMedicalAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def reply(self, _msg):
        raise RuntimeError("boom")


class _MergedCsvReactMedicalAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def reply(self, _msg):
        return SimpleNamespace(
            metadata={
                "success": True,
                "data_type": "directory",
                "statistics": {"total_patients": 2},
                "output_dir": self.kwargs["output_dir"],
                "output_merged_csv": str(Path(self.kwargs["output_dir"]) / "merged.csv"),
                "output_patients_csv": str(Path(self.kwargs["output_dir"]) / "patients.csv"),
            },
            content="处理完成",
        )


class _AdmissionWideReactMedicalAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def reply(self, _msg):
        return SimpleNamespace(
            metadata={
                "success": True,
                "data_type": "directory",
                "statistics": {"total_patients": 2},
                "output_dir": self.kwargs["output_dir"],
                "output_admission_wide_csv": str(Path(self.kwargs["output_dir"]) / "admission_wide.csv"),
                "output_merged_csv": str(Path(self.kwargs["output_dir"]) / "merged.csv"),
                "output_patients_csv": str(Path(self.kwargs["output_dir"]) / "patients.csv"),
            },
            content="澶勭悊瀹屾垚",
        )


class _PatientsCsvReactMedicalAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def reply(self, _msg):
        return SimpleNamespace(
            metadata={
                "success": True,
                "data_type": "directory",
                "statistics": {"total_patients": 2},
                "output_dir": self.kwargs["output_dir"],
                "output_patients_csv": str(Path(self.kwargs["output_dir"]) / "patients.csv"),
            },
            content="处理完成",
        )


class _NoCsvReactMedicalAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def reply(self, _msg):
        return SimpleNamespace(
            metadata={
                "success": True,
                "data_type": "csv",
                "statistics": {"row_count": 1},
                "output_dir": self.kwargs["output_dir"],
            },
            content="处理完成",
        )


def test_step2_3_wrapper_module_compiles():
    compile(STEP_WRAPPERS.read_text(encoding="utf-8"), str(STEP_WRAPPERS), "exec")


def test_get_default_step2_3_input_path_prefers_step1_original_input(monkeypatch, tmp_path):
    step1_results = tmp_path / "program" / "output" / "step1_results"
    step1_results.mkdir(parents=True)
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()

    monkeypatch.setattr(step_wrappers, "_get_step1_results_dir", lambda: str(step1_results))

    resolved = step_wrappers._get_default_step2_3_input_path(str(rawdata_dir))

    assert resolved == str(rawdata_dir)


def test_get_default_step2_3_input_path_prefers_populated_step1_results(monkeypatch, tmp_path):
    step1_results = tmp_path / "program" / "output" / "step1_results"
    patient_table = step1_results / "10000032" / "table"
    patient_table.mkdir(parents=True)
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()
    trace_stub = SimpleNamespace(
        steps=[{"step_name": "Step1 data recognition", "input_data": str(rawdata_dir)}]
    )

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "_get_step1_results_dir", lambda: str(step1_results))

    resolved = step_wrappers._get_default_step2_3_input_path(str(rawdata_dir))

    assert resolved == str(step1_results)


def test_get_default_step2_3_input_path_falls_back_to_step1_trace(monkeypatch, tmp_path):
    step1_results = tmp_path / "program" / "output" / "step1_results"
    step1_results.mkdir(parents=True)
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()
    trace_stub = SimpleNamespace(
        steps=[{"step_name": "Step1 data recognition", "input_data": str(rawdata_dir)}]
    )

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "_get_step1_results_dir", lambda: str(step1_results))

    resolved = step_wrappers._get_default_step2_3_input_path("Step1 summary text")

    assert resolved == str(rawdata_dir)


def test_get_default_step2_3_input_path_populated_candidate_overrides_trace(monkeypatch, tmp_path):
    step1_results = tmp_path / "program" / "output" / "step1_results"
    (step1_results / "10000032" / "table").mkdir(parents=True)
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()
    trace_stub = SimpleNamespace(
        steps=[{"step_name": "Step1 data recognition", "input_data": str(rawdata_dir)}]
    )

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "_get_step1_results_dir", lambda: str(tmp_path / "empty_step1_results"))

    resolved = step_wrappers._get_default_step2_3_input_path(str(step1_results))

    assert resolved == str(step1_results)


def test_run_step2_3_publishes_next_input_from_output_csv(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    memory_calls = []
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)
    output_csv = results_dir / "cleaned.csv"
    output_csv.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")

    async def fake_report_last_step_reflection(**kwargs):
        memory_calls.append(kwargs)
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "step1_results"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(results_dir / "next_input"))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _FakeReactMedicalAgent)

    result = asyncio.run(step_wrappers.run_step2_3_medical_data_cleaner("/tmp/rawdata", context="playbook"))

    next_input_csv = results_dir / "next_input" / "input.csv"
    assert isinstance(result, ToolResponse)
    assert next_input_csv.exists()
    assert next_input_csv.read_text(encoding="utf-8") == output_csv.read_text(encoding="utf-8")
    assert "传递给Step4(next_input):" in str(result.content)
    assert len(trace_stub.calls) == 1
    trace_call = trace_stub.calls[0]
    assert trace_call["step_name"] == "Step2_3 医学数据清洗与标准化"
    assert trace_call["context"] == "playbook"
    assert any(event["tool_name"] == "step2_3_publish_next_input" and event["status"] == "succeeded" for event in trace_call["tool_events"])
    assert len(memory_calls) == 1
    assert memory_calls[0]["input_text"] == str(tmp_path / "step1_results")


def test_run_step2_3_records_failure_trace_when_agent_reply_raises(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    memory_calls = []
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)

    async def fake_report_last_step_reflection(**kwargs):
        memory_calls.append(kwargs)
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "step1_results"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _ExplodingReactMedicalAgent)

    result = asyncio.run(step_wrappers.run_step2_3_medical_data_cleaner("/tmp/rawdata", context="ctx"))

    assert isinstance(result, ToolResponse)
    assert "医学数据清洗与标准化失败" in str(result.content)
    assert len(trace_stub.calls) == 1
    trace_call = trace_stub.calls[0]
    assert trace_call["step_name"] == "Step2_3 医学数据清洗与标准化"
    assert any(event["tool_name"] == "step2_3_error" for event in trace_call["tool_events"])
    assert len(memory_calls) == 1



def test_run_step2_3_prefers_admission_wide_csv_for_next_input(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)
    admission_wide_csv = results_dir / "admission_wide.csv"
    admission_wide_csv.write_text("subject_id,hadm_id\n1,10\n", encoding="utf-8")
    merged_csv = results_dir / "merged.csv"
    merged_csv.write_text("patient_id,feature\n1,x\n", encoding="utf-8")
    patients_csv = results_dir / "patients.csv"
    patients_csv.write_text("patient_id,count\n1,2\n", encoding="utf-8")

    async def fake_report_last_step_reflection(**kwargs):
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "rawdata"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(results_dir / "next_input"))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _AdmissionWideReactMedicalAgent)

    result = asyncio.run(step_wrappers.run_step2_3_medical_data_cleaner("/tmp/rawdata", context="ctx"))

    next_input_csv = results_dir / "next_input" / "input.csv"
    assert isinstance(result, ToolResponse)
    assert next_input_csv.exists()
    assert next_input_csv.read_text(encoding="utf-8") == admission_wide_csv.read_text(encoding="utf-8")
    assert "Admission wide CSV: " + str(admission_wide_csv) in str(result.content)
    assert any(event["tool_name"] == "step2_3_publish_next_input" and event["status"] == "succeeded" for event in trace_stub.calls[0]["tool_events"])


def test_run_step2_3_prefers_output_merged_csv_for_next_input(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)
    merged_csv = results_dir / "merged.csv"
    merged_csv.write_text("patient_id,feature\n1,x\n", encoding="utf-8")
    patients_csv = results_dir / "patients.csv"
    patients_csv.write_text("patient_id,count\n1,2\n", encoding="utf-8")

    async def fake_report_last_step_reflection(**kwargs):
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "step1_results"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(results_dir / "next_input"))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _MergedCsvReactMedicalAgent)

    result = asyncio.run(step_wrappers.run_step2_3_medical_data_cleaner("/tmp/rawdata", context="ctx"))

    next_input_csv = results_dir / "next_input" / "input.csv"
    assert isinstance(result, ToolResponse)
    assert next_input_csv.exists()
    assert next_input_csv.read_text(encoding="utf-8") == merged_csv.read_text(encoding="utf-8")
    assert "主输出CSV: " + str(merged_csv) in str(result.content)
    assert any(event["tool_name"] == "step2_3_publish_next_input" and event["status"] == "succeeded" for event in trace_stub.calls[0]["tool_events"])



def test_run_step2_3_falls_back_to_output_patients_csv(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)
    patients_csv = results_dir / "patients.csv"
    patients_csv.write_text("patient_id,count\n1,2\n", encoding="utf-8")

    async def fake_report_last_step_reflection(**kwargs):
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "step1_results"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(results_dir / "next_input"))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _PatientsCsvReactMedicalAgent)

    result = asyncio.run(step_wrappers.run_step2_3_medical_data_cleaner("/tmp/rawdata", context="ctx"))

    next_input_csv = results_dir / "next_input" / "input.csv"
    assert isinstance(result, ToolResponse)
    assert next_input_csv.exists()
    assert next_input_csv.read_text(encoding="utf-8") == patients_csv.read_text(encoding="utf-8")
    assert "传递给Step4(next_input):" in str(result.content)
    assert any(event["tool_name"] == "step2_3_publish_next_input" and event["status"] == "succeeded" for event in trace_stub.calls[0]["tool_events"])



def test_run_step2_3_fails_when_no_csv_is_available(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)

    async def fake_report_last_step_reflection(**kwargs):
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "step1_results"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(results_dir / "next_input"))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _NoCsvReactMedicalAgent)
    monkeypatch.setattr(step_wrappers, "_resolve_step2_3_output_csv_for_next_input", lambda output_dir, started_at: "")

    result = asyncio.run(step_wrappers.run_step2_3_medical_data_cleaner("/tmp/rawdata", context="ctx"))

    next_input_csv = results_dir / "next_input" / "input.csv"
    assert isinstance(result, ToolResponse)
    assert "医学数据清洗与标准化失败" in str(result.content)
    assert not next_input_csv.exists()
    assert any(event["tool_name"] == "step2_3_publish_next_input" and event["status"] == "failed" for event in trace_stub.calls[0]["tool_events"])



def test_run_step2_3_falls_back_to_scanned_output_csv(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)
    scanned_csv = results_dir / "nested" / "all_patients.csv"
    scanned_csv.parent.mkdir(parents=True)
    scanned_csv.write_text("patient_id,count\n1,2\n", encoding="utf-8")

    async def fake_report_last_step_reflection(**kwargs):
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "step1_results"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(results_dir / "next_input"))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _NoCsvReactMedicalAgent)

    result = asyncio.run(step_wrappers.run_step2_3_medical_data_cleaner("/tmp/rawdata", context="ctx"))

    next_input_csv = results_dir / "next_input" / "input.csv"
    assert isinstance(result, ToolResponse)
    assert next_input_csv.exists()
    assert next_input_csv.read_text(encoding="utf-8") == scanned_csv.read_text(encoding="utf-8")
    assert "主输出CSV: " + str(scanned_csv) in str(result.content)
    assert any(event["tool_name"] == "step2_3_select_next_input" and event["status"] == "succeeded" for event in trace_stub.calls[0]["tool_events"])



def test_step4_prefers_step2_3_next_input_csv(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    memory_calls = []
    project_root = PROJECT_ROOT
    step4_input_dir = tmp_path / "step1_results"
    step4_input_dir.mkdir(parents=True)
    step2_3_next_input_dir = tmp_path / "program" / "output" / "step2_3_results" / "next_input"
    step2_3_next_input_dir.mkdir(parents=True)
    next_input_csv = step2_3_next_input_dir / "input.csv"
    next_input_csv.write_text("id,value\n1,a\n", encoding="utf-8")
    workspace_dir = tmp_path / "program" / "output" / "step4_results"
    final_output_csv = workspace_dir / "final.csv"
    final_output_csv.parent.mkdir(parents=True)
    final_output_csv.write_text("id,value\n1,a\n", encoding="utf-8")
    validation_report = workspace_dir / "validation.json"
    validation_report.write_text("{}", encoding="utf-8")
    detailed_report = workspace_dir / "detail.json"
    detailed_report.write_text("{}", encoding="utf-8")
    failed_columns = workspace_dir / "failed.json"
    failed_columns.write_text("[]", encoding="utf-8")
    captured = {}

    async def fake_report_last_step_reflection(**kwargs):
        memory_calls.append(kwargs)
        return None

    async def fake_run_data_quality_repair(input_dir, workspace_dir, validation_config_path, verbose):
        captured["input_dir"] = input_dir
        return {
            "success": True,
            "input_dir": str(step4_input_dir),
            "input_csv": input_dir,
            "workspace_dir": workspace_dir,
            "final_output_csv": str(final_output_csv),
            "validation_report_path": str(validation_report),
            "detailed_report_path": str(detailed_report),
            "failed_columns_path": str(failed_columns),
            "quality_score": 0.95,
            "generated_cleaners": 1,
            "generation_failures": {},
            "runtime_failures": {},
        }

    fake_agent4_module = types.SimpleNamespace(run_data_quality_repair=fake_run_data_quality_repair)

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step4_input_path", lambda input_data: str(step4_input_dir))
    monkeypatch.setattr(step_wrappers, "_get_project_root", lambda: str(project_root))
    monkeypatch.setattr(step_wrappers, "_get_step4_results_dir", lambda: str(workspace_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(step2_3_next_input_dir))
    monkeypatch.setattr(step_wrappers, "_get_step4_next_input_dir", lambda: str(workspace_dir / "next_input"))
    monkeypatch.setattr(step_wrappers.importlib, "import_module", lambda name: fake_agent4_module if name == "agent_4.main" else __import__(name))

    result = asyncio.run(step_wrappers.run_step4_data_quality_repair(str(step4_input_dir), context="ctx"))

    assert isinstance(result, ToolResponse)
    assert captured["input_dir"] == str(next_input_csv)
    assert (workspace_dir / "next_input" / "input.csv").exists()
    assert (workspace_dir / "next_input" / "input.csv").read_text(encoding="utf-8") == final_output_csv.read_text(encoding="utf-8")
    assert "预处理输入CSV: " + str(next_input_csv) in str(result.content)
    assert any(event["tool_name"] == "step4_resolve_next_input_csv" and event["status"] == "succeeded" for event in trace_stub.calls[0]["tool_events"])
    assert len(memory_calls) == 1


def test_run_step2_3_skips_memory_when_disabled(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    memory_calls = []
    results_dir = tmp_path / "program" / "output" / "step2_3_results"
    results_dir.mkdir(parents=True)
    output_csv = results_dir / "cleaned.csv"
    output_csv.write_text("id,value\n1,a\n", encoding="utf-8")

    async def fake_report_last_step_reflection(**kwargs):
        memory_calls.append(kwargs)
        return "should-not-run"

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step2_3_input_path", lambda input_data: str(tmp_path / "step1_results"))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_results_dir", lambda: str(results_dir))
    monkeypatch.setattr(step_wrappers, "_get_step2_3_next_input_dir", lambda: str(results_dir / "next_input"))
    monkeypatch.setattr(step_wrappers, "_load_step2_3_react_agent_class", lambda: _FakeReactMedicalAgent)

    result = asyncio.run(
        step_wrappers.run_step2_3_medical_data_cleaner(
            "/tmp/rawdata",
            context="playbook",
            enable_memory_agent=False,
        )
    )

    assert isinstance(result, ToolResponse)
    assert len(trace_stub.calls) == 1
    assert memory_calls == []


def test_step4_missing_input_skips_memory_when_disabled(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    memory_calls = []
    missing_dir = tmp_path / "missing-step4-input"

    async def fake_report_last_step_reflection(**kwargs):
        memory_calls.append(kwargs)
        return "should-not-run"

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_default_step4_input_path", lambda input_data: str(missing_dir))

    result = asyncio.run(
        step_wrappers.run_step4_data_quality_repair(
            str(missing_dir),
            context="ctx",
            enable_memory_agent=False,
        )
    )

    assert isinstance(result, ToolResponse)
    assert "输入目录不存在" in str(result.content)
    assert len(trace_stub.calls) == 1
    assert memory_calls == []
