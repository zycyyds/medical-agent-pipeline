import asyncio
import json

from memory_agent.services.experience_memory_service import ExperienceMemoryService
from memory_agent.services.memory_service import MemoryService
from memory_agent.services.rule_memory_service import RuleMemoryService


class FakeRuleMemory:
    async def search_rules(self, query_text, max_results=5):
        assert query_text == "run step1"
        return "rule text", ["rule:one"], []

    async def summarize_trace(self, trace_payload):
        assert trace_payload["success"] is True
        return "rule written", []

    def status(self):
        return {"backend": "fake_rule"}


class FakeExperienceMemory:
    async def retrieve(self, query_text, task_spec=None):
        assert query_text == "run step1"
        return "task text", "tool text", ["exp:one"], []

    async def record_from_trace(self, trace_payload):
        assert trace_payload["success"] is True
        return "experience written", []

    def status(self):
        return {"backend": "fake_experience"}


def test_memory_service_formats_context_and_records_result():
    service = MemoryService(rule_service=FakeRuleMemory(), experience_service=FakeExperienceMemory())

    context, used_ids = asyncio.run(service.get_context("run step1"))

    assert "# Memory Context" in context
    assert "## Rule Memory\nrule text" in context
    assert "## Similar Step Experience\ntask text" in context
    assert "## Tool Memory\ntool text" in context
    assert used_ids == ["rule:one", "exp:one"]

    summary = asyncio.run(service.report_result({"success": True}))

    assert "rule written" in summary
    assert "experience written" in summary


def test_rule_memory_ensure_store_creates_memory_file(tmp_path):
    from memory_agent.config import Config

    cfg = Config()
    cfg.REME_LIGHT_ROOT = str(tmp_path / "playbook_light")

    service = RuleMemoryService(cfg)
    memory_file = tmp_path / "playbook_light" / "MEMORY.md"

    status = service.status()

    assert status["backend"] == "reme_light"
    assert memory_file.exists()
    assert "MemoryAgent Rule Memory" in memory_file.read_text(encoding="utf-8")


def test_experience_memory_records_step1_without_full_script_body(tmp_path, monkeypatch):
    from memory_agent.config import Config

    calls = {"task": [], "tool": []}

    class FakeTaskMemory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
            return None

        async def record_to_memory(self, thinking, content, **kwargs):
            calls["task"].append({"thinking": thinking, "content": content, "kwargs": kwargs})
            return type("Resp", (), {"content": "task ok"})()

    class FakeToolMemory(FakeTaskMemory):
        async def record_to_memory(self, thinking, content, **kwargs):
            calls["tool"].append({"thinking": thinking, "content": content, "kwargs": kwargs})
            return type("Resp", (), {"content": "tool ok"})()

    script = tmp_path / "generated.py"
    script.write_text("print('secret script body')\n", encoding="utf-8")

    cfg = Config()
    cfg.REME_VECTOR_ROOT = str(tmp_path / "vector")
    cfg.LLM_API_KEY = "llm-key"
    cfg.LLM_BASE_URL = "http://example.test/v1"
    cfg.EMBEDDING_API_KEY = "embedding-key"
    cfg.EMBEDDING_BASE_URL = "http://embedding.test/v1"
    cfg.EMBEDDING_MODEL = "text-embedding-3-small"

    service = ExperienceMemoryService(cfg, task_memory_cls=FakeTaskMemory, tool_memory_cls=FakeToolMemory)
    summary, warnings = asyncio.run(
        service.record_from_trace(
            {
                "success": True,
                "steps": [
                    {
                        "state": "STEP1_RECORDING",
                        "result": {
                            "status": "SUCCESS",
                            "summary": "ok",
                            "artifacts": {
                                "input_path": str(tmp_path),
                                "records_count": 3,
                                "written_files": 4,
                                "generated_script_path": str(script),
                            },
                        },
                    }
                ],
            }
        )
    )

    serialized = json.dumps(calls, ensure_ascii=False)
    assert "task ok" in summary
    assert "tool ok" in summary
    assert warnings == []
    assert "secret script body" not in serialized
    assert str(script) in serialized
    assert calls["task"][0]["kwargs"]["score"] == 1.0


def test_experience_memory_records_step4_selection_report_without_raw_responses(tmp_path):
    from memory_agent.config import Config

    calls = {"task": [], "tool": []}

    class FakeTaskMemory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
            return None

        async def record_to_memory(self, thinking, content, **kwargs):
            calls["task"].append({"thinking": thinking, "content": content, "kwargs": kwargs})
            return type("Resp", (), {"content": "task ok"})()

    class FakeToolMemory(FakeTaskMemory):
        async def record_to_memory(self, thinking, content, **kwargs):
            calls["tool"].append({"thinking": thinking, "content": content, "kwargs": kwargs})
            return type("Resp", (), {"content": "tool ok"})()

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("patient_id,visit_date,diagnosis_note\n1,2026-01-01,A\n", encoding="utf-8")
    filtered = tmp_path / "filtered.csv"
    filtered.write_text("patient_id,visit_date\n1,2026-01-01\n", encoding="utf-8")
    report = tmp_path / "selection_report.json"
    report.write_text(
        json.dumps(
            {
                "task_text": "筛选诊断相关字段",
                "task_spec": {"task_type": "prediction", "need_modalities": ["Diagnosis"]},
                "term_profile": {"column_terms": ["diagnosis"], "negative_terms": []},
                "column_counts": {"original_column_count": 3, "final_column_count": 2},
                "agent_execution": {
                    "term_planner_mode": "llm",
                    "term_planner_raw_response": "raw secret response",
                },
                "agent_decisions": {
                    "structured_keep_overrides": [
                        {"column": "patient_id", "action": "forced_keep_structured_field"}
                    ]
                },
                "final_columns": ["patient_id", "visit_date"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = Config()
    cfg.REME_VECTOR_ROOT = str(tmp_path / "vector")
    cfg.LLM_API_KEY = "llm-key"
    cfg.LLM_BASE_URL = "http://example.test/v1"
    cfg.EMBEDDING_API_KEY = "embedding-key"
    cfg.EMBEDDING_BASE_URL = "http://embedding.test/v1"
    cfg.EMBEDDING_MODEL = "text-embedding-3-small"

    service = ExperienceMemoryService(cfg, task_memory_cls=FakeTaskMemory, tool_memory_cls=FakeToolMemory)
    summary, warnings = asyncio.run(
        service.record_from_trace(
            {
                "success": True,
                "steps": [
                    {
                        "state": "STEP4_RUNNING",
                        "result": {
                            "status": "SUCCESS",
                            "summary": "step4 ok",
                            "artifacts": {
                                "input_csv_path": str(input_csv),
                                "filtered_csv_path": str(filtered),
                                "selection_report_path": str(report),
                            },
                        },
                    }
                ],
            }
        )
    )

    serialized = json.dumps(calls, ensure_ascii=False)
    assert "task ok" in summary
    assert "tool ok" in summary
    assert warnings == []
    assert "step4_experience" in serialized
    assert "structured_keep_overrides" in serialized
    assert "raw secret response" not in serialized
    assert calls["task"][0]["kwargs"]["score"] == 1.0


def test_experience_memory_retrieves_step4_tool_names(tmp_path):
    from memory_agent.config import Config

    captured = {"tool_names": []}

    class FakeTaskSpec:
        def to_dict(self):
            return {"task_type": "resume_from_step2_3", "entry_artifacts": {"input_csv": "input.csv"}}

    class FakeTaskMemory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
            return None

        async def retrieve(self, msg, limit=3):
            return "task memory"

    class FakeToolMemory(FakeTaskMemory):
        async def retrieve_from_memory(self, tool_names, limit=3):
            captured["tool_names"] = list(tool_names)
            return type("Resp", (), {"content": "tool memory"})()

    cfg = Config()
    cfg.REME_VECTOR_ROOT = str(tmp_path / "vector")
    cfg.LLM_API_KEY = "llm-key"
    cfg.LLM_BASE_URL = "http://example.test/v1"
    cfg.EMBEDDING_API_KEY = "embedding-key"
    cfg.EMBEDDING_BASE_URL = "http://embedding.test/v1"
    cfg.EMBEDDING_MODEL = "text-embedding-3-small"

    service = ExperienceMemoryService(cfg, task_memory_cls=FakeTaskMemory, tool_memory_cls=FakeToolMemory)
    _task_text, _tool_text, _ids, warnings = asyncio.run(
        service.retrieve(
            "运行 step4",
            task_spec=FakeTaskSpec(),
        )
    )

    assert warnings == []
    assert captured["tool_names"] == ExperienceMemoryService.STEP4_TOOL_NAMES
