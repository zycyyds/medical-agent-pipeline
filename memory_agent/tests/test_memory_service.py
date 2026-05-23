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


def test_memory_service_separates_rule_and_step_context():
    calls = {"rule": 0, "experience": 0}

    class RuleOnlyMemory:
        async def search_rules(self, query_text, max_results=5):
            calls["rule"] += 1
            return "rule only", ["rule:one"], []

        async def summarize_trace(self, trace_payload):
            return "rule written", []

        def status(self):
            return {"backend": "rule_only"}

    class StepOnlyMemory:
        async def retrieve(self, query_text, task_spec=None, step_name=None):
            calls["experience"] += 1
            assert step_name == "step4"
            return "task step4", "tool step4", ["exp:step4"], []

        async def record_from_trace(self, trace_payload):
            return "experience written", []

        def status(self):
            return {"backend": "step_only"}

    service = MemoryService(rule_service=RuleOnlyMemory(), experience_service=StepOnlyMemory())

    rule_context, rule_ids = asyncio.run(service.get_rule_context("run pipeline"))
    assert "rule only" in rule_context
    assert rule_ids == ["rule:one"]
    assert calls == {"rule": 1, "experience": 0}

    step_context, step_ids = asyncio.run(service.get_step_context("step4", "run step4"))
    assert "task step4" in step_context
    assert "tool step4" in step_context
    assert step_ids == ["exp:step4"]
    assert calls == {"rule": 1, "experience": 1}


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
    tool_content = "\n".join(calls["tool"][0]["content"])
    assert "schema_summary" not in tool_content
    assert "sample_files" not in tool_content
    assert all(len(item) <= service.TOOL_EVENT_MAX_CHARS for item in calls["tool"][0]["content"])


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
    tool_content = "\n".join(calls["tool"][0]["content"])
    assert "schema_summary" not in tool_content
    assert "sample_files" not in tool_content
    assert "term_profile" not in tool_content
    assert "final_columns" not in tool_content
    assert "raw secret response" not in tool_content
    assert all(len(item) <= service.TOOL_EVENT_MAX_CHARS for item in calls["tool"][0]["content"])


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


def test_experience_memory_retrieves_tool_names_from_explicit_step(tmp_path):
    from memory_agent.config import Config

    captured = {"tool_names": []}

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
            "泛化描述，不包含 step5 关键词",
            step_name="STEP5_RUNNING",
        )
    )

    assert warnings == []
    assert captured["tool_names"] == ExperienceMemoryService.STEP5_TOOL_NAMES


def test_experience_memory_retrieves_step6_tool_names(tmp_path):
    from memory_agent.config import Config

    captured = {"tool_names": []}

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
            "运行一致性验证",
            step_name="STEP6_RUNNING",
        )
    )

    assert warnings == []
    assert captured["tool_names"] == ExperienceMemoryService.STEP6_TOOL_NAMES


def test_experience_memory_retrieves_step7_tool_names(tmp_path):
    from memory_agent.config import Config

    captured = {"tool_names": []}

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
            "生成机器学习训练数据集",
            step_name="STEP7_RUNNING",
        )
    )

    assert warnings == []
    assert captured["tool_names"] == ExperienceMemoryService.STEP7_TOOL_NAMES


def test_experience_memory_records_step6_without_full_patient_ids(tmp_path):
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

    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
    report_txt = tmp_path / "consistency_report.txt"
    report_json = tmp_path / "consistency_report.json"
    passed_json = tmp_path / "passed_patients.json"
    report_txt.write_text("report", encoding="utf-8")
    report_json.write_text("{}", encoding="utf-8")
    passed_json.write_text(
        json.dumps({"passed_count": 1000, "passed_patient_ids": [str(i) for i in range(1000)]}),
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
                        "state": "STEP6_RUNNING",
                        "result": {
                            "status": "SUCCESS",
                            "summary": "step6 ok",
                            "artifacts": {
                                "input_csv": str(input_csv),
                                "output_step6": str(tmp_path),
                                "step6_report_txt": str(report_txt),
                                "step6_report_json": str(report_json),
                                "passed_patients_json": str(passed_json),
                                "passed_patient_count": 1000,
                                "passed_patient_ids_sample": ["0", "1", "2"],
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
    assert "step6_experience" in serialized
    assert "passed_patient_count" in serialized
    assert '"999"' not in serialized
    assert calls["task"][0]["kwargs"]["score"] == 1.0
    assert all(len(item) <= service.TOOL_EVENT_MAX_CHARS for item in calls["tool"][0]["content"])


def test_experience_memory_records_step7_compact_outputs(tmp_path):
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

    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,value,label\n1,a,A\n2,b,B\n", encoding="utf-8")
    selection_report = tmp_path / "selection_report.json"
    passed_json = tmp_path / "passed_patients.json"
    dataset_csv = tmp_path / "ml_dataset_mock.csv"
    dataset_json = tmp_path / "ml_dataset_mock.json"
    model_config = tmp_path / "model_config_mock.json"
    for path in (selection_report, passed_json, dataset_csv, dataset_json, model_config):
        path.write_text("{}", encoding="utf-8")

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
                        "state": "STEP7_RUNNING",
                        "result": {
                            "status": "SUCCESS",
                            "summary": "step7 ok",
                            "artifacts": {
                                "input_csv": str(input_csv),
                                "selection_report": str(selection_report),
                                "passed_patients_json": str(passed_json),
                                "passed_patient_count": 1000,
                                "output_step7": str(tmp_path),
                                "step7_dataset_count": 1,
                                "step7_dataset_csvs": [str(dataset_csv)],
                                "step7_dataset_jsons": [str(dataset_json)],
                                "model_config_jsons": [str(model_config)],
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
    assert "step7_experience" in serialized
    assert "step7_dataset_count" in serialized
    assert calls["task"][0]["kwargs"]["score"] == 1.0
    assert all(len(item) <= service.TOOL_EVENT_MAX_CHARS for item in calls["tool"][0]["content"])
