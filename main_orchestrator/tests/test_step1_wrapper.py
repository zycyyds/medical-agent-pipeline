import asyncio
import sys
from pathlib import Path

from agentscope.tool import ToolResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
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


def test_run_step1_modal_recognition_uses_new_codegen_entry_and_reports_memory(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    memory_calls = []

    async def fake_run_step1_codegen(input_path: str, output_root: Path | None = None, emit=print, script_path=None):
        assert input_path == "/tmp/rawdata"
        assert output_root == Path(step_wrappers._get_step1_results_dir())
        return {
            "status": "SUCCESS",
            "summary_text": "Step1 完成",
            "records_path": str(tmp_path / "reorganized_output" / "_meta" / "records.json"),
            "generated_script_path": str(tmp_path / "generated_reorganizer.py"),
            "step1_output_root": str(tmp_path / "step1_results"),
            "raw_messages": [{"role": "assistant", "content_blocks": [{"type": "text", "text": "done"}]}],
            "tool_events": [{"tool_name": "record_source_file", "status": "succeeded", "tool_input": {}, "tool_output": "ok", "error": ""}],
            "runtime_result": {"status": "SUCCESS"},
        }

    async def fake_report_last_step_reflection(**kwargs):
        memory_calls.append(kwargs)
        return "memory ok"

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_step1_results_dir", lambda: str(tmp_path / "step1_results"))
    monkeypatch.setattr("agent_1.codegen_agent.run_step1_codegen", fake_run_step1_codegen)

    result = asyncio.run(step_wrappers.run_step1_modal_recognition("/tmp/rawdata", context="playbook"))

    assert isinstance(result, ToolResponse)
    assert "Step1: 数据感知与模态识别完成" in str(result.content)
    assert len(trace_stub.calls) == 1
    trace_call = trace_stub.calls[0]
    assert trace_call["step_name"] == "数据感知与模态识别"
    assert trace_call["context"] == "playbook"
    assert trace_call["tool_events"][0]["tool_name"] == "record_source_file"
    assert len(memory_calls) == 1
    assert memory_calls[0]["input_text"] == "/tmp/rawdata"
    assert "Step1: 数据感知与模态识别完成" in memory_calls[0]["orchestrator_summary"]


def test_run_step1_modal_recognition_renders_paths_from_step1_contract(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()

    async def fake_run_step1_codegen(input_path: str, output_root: Path | None = None, emit=print, script_path=None):
        return {
            "status": "SUCCESS",
            "summary_text": "Step1 完成",
            "records_path": str(tmp_path / "reorganized_output" / "_meta" / "records.json"),
            "generated_script_path": str(tmp_path / "generated_reorganizer.py"),
            "step1_output_root": str(tmp_path / "step1_results"),
            "raw_messages": [{"role": "assistant", "content_blocks": [{"type": "text", "text": "done"}]}],
            "tool_events": [{"tool_name": "run_react_codegen_cycle", "status": "succeeded", "tool_input": {}, "tool_output": "ok", "error": ""}],
            "runtime_result": {"status": "SUCCESS", "issues": []},
        }

    async def fake_report_last_step_reflection(**kwargs):
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_step1_results_dir", lambda: str(tmp_path / "step1_results"))
    monkeypatch.setattr("agent_1.codegen_agent.run_step1_codegen", fake_run_step1_codegen)

    result = asyncio.run(step_wrappers.run_step1_modal_recognition("/tmp/rawdata", context="ctx"))

    text = str(result.content)
    assert "输出目录:" in text
    assert str(tmp_path / "step1_results") in text
    assert str(tmp_path / "generated_reorganizer.py") in text
    assert str(tmp_path / "reorganized_output" / "_meta" / "records.json") in text



def test_run_step1_modal_recognition_returns_failure_summary_on_codegen_error(monkeypatch, tmp_path):
    trace_stub = _TraceCollectorStub()
    memory_calls = []

    async def fake_run_step1_codegen(input_path: str, output_root: Path | None = None, emit=print, script_path=None):
        raise RuntimeError("boom")

    async def fake_report_last_step_reflection(**kwargs):
        memory_calls.append(kwargs)
        return None

    monkeypatch.setattr(step_wrappers, "_trace_collector", trace_stub)
    monkeypatch.setattr(step_wrappers, "report_last_step_reflection", fake_report_last_step_reflection)
    monkeypatch.setattr(step_wrappers, "_get_step1_results_dir", lambda: str(tmp_path / "step1_results"))
    monkeypatch.setattr("agent_1.codegen_agent.run_step1_codegen", fake_run_step1_codegen)

    result = asyncio.run(step_wrappers.run_step1_modal_recognition("/tmp/rawdata", context="ctx"))

    assert isinstance(result, ToolResponse)
    assert "执行失败" in str(result.content)
    assert len(trace_stub.calls) == 1
    assert trace_stub.calls[0]["tool_events"][0]["status"] == "failed"
    assert memory_calls == []
