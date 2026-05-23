import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("run_step6_ml_pipeline.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("agent6_step6_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_standalone_defaults_use_step6_data_input_and_program_output():
    module = _load_module()
    project_root = MODULE_PATH.parents[1]

    assert module.get_default_input_dir() == str(project_root / "step-6" / "data_input")
    assert module.get_default_output_step6_dir() == str(project_root / "program" / "output" / "step6_results")


def test_read_data_maps_common_id_columns_to_patient_id(tmp_path):
    for column_name in ("id", "subject_id", "record_index", "hadm_id"):
        csv_path = tmp_path / f"{column_name}.csv"
        csv_path.write_text(f"{column_name},value\n1,a\n2,b\n", encoding="utf-8")

        module = _load_module()
        module.configure_runtime({"input_csv": str(csv_path), "patient_id_col": ""})
        df = module._read_data(str(csv_path))

        assert "patient_id" in df.columns
        assert column_name not in df.columns or column_name == "patient_id"
        assert df["patient_id"].astype(str).tolist() == ["1", "2"]


def test_parse_passed_ids_handles_empty_list():
    module = _load_module()

    assert module._parse_passed_ids("PASSED_PATIENTS: []") == []


def test_parse_passed_ids_requires_marker():
    module = _load_module()

    assert module._parse_passed_ids("Step 6 finished without marker") is None


def test_save_step6_report_uses_tool_cache_for_compact_marker(tmp_path):
    input_csv = tmp_path / "filtered.csv"
    output_dir = tmp_path / "step6_results"
    input_csv.write_text("id,gender\n1,F\n2,M\n", encoding="utf-8")

    module = _load_module()
    module.configure_runtime({"input_csv": str(input_csv), "output_step6_dir": str(output_dir), "patient_id_col": ""})
    module._LAST_ANALYSIS_PASSED_IDS = ["1", "2"]

    module.save_step6_report("Step 6 report\nPASSED_PATIENTS: <stored_in_step6_tool_cache>")

    passed_file = next(output_dir.glob("passed_patients_*.json"))
    assert '"passed_count": 2' in passed_file.read_text(encoding="utf-8")


def test_run_step6_only_mock_agent_writes_outputs(monkeypatch, tmp_path):
    module = _load_module()
    input_csv = tmp_path / "filtered.csv"
    output_dir = tmp_path / "step6_results"
    input_csv.write_text("id,gender,age\n1,F,30\n1,F,30\n2,M,40\n", encoding="utf-8")

    class FakeAgent:
        async def __call__(self, _msg):
            text = "Step 6 mock report\nPASSED_PATIENTS: ['1', '2']"
            return SimpleNamespace(content=text)

    monkeypatch.setattr(module, "create_consistency_agent", lambda: FakeAgent())

    result = module.asyncio.run(
        module.run_step6_only(
            {
                "input_csv": str(input_csv),
                "output_step6_dir": str(output_dir),
                "patient_id_col": "",
            }
        )
    )

    assert result["success"] is True
    assert result["error"] == ""
    assert result["input_csv"] == str(input_csv.resolve())
    assert result["output_step6"] == str(output_dir.resolve())
    assert result["passed_patient_ids"] == ["1", "2"]
    assert Path(result["step6_report_txt"]).is_file()
    assert Path(result["step6_report_json"]).is_file()
    assert Path(result["passed_patients_json"]).is_file()


def test_run_step6_only_accepts_report_saved_by_tool_without_final_marker(monkeypatch, tmp_path):
    module = _load_module()
    input_csv = tmp_path / "filtered.csv"
    output_dir = tmp_path / "step6_results"
    input_csv.write_text("id,gender\n1,F\n2,M\n", encoding="utf-8")

    class FakeAgent:
        async def __call__(self, _msg):
            module._LAST_ANALYSIS_PASSED_IDS = ["1", "2"]
            module.save_step6_report("Saved report\nPASSED_PATIENTS: <stored_in_step6_tool_cache>")
            return SimpleNamespace(content="Step 6 saved successfully.")

    monkeypatch.setattr(module, "create_consistency_agent", lambda: FakeAgent())

    result = module.asyncio.run(
        module.run_step6_only(
            {
                "input_csv": str(input_csv),
                "output_step6_dir": str(output_dir),
                "patient_id_col": "",
            }
        )
    )

    assert result["success"] is True
    assert result["passed_patient_ids"] == ["1", "2"]
    assert Path(result["passed_patients_json"]).is_file()


def test_run_step6_only_fails_without_marker_or_saved_artifact(monkeypatch, tmp_path):
    module = _load_module()
    input_csv = tmp_path / "filtered.csv"
    output_dir = tmp_path / "step6_results"
    input_csv.write_text("id,gender\n1,F\n", encoding="utf-8")

    class FakeAgent:
        async def __call__(self, _msg):
            return SimpleNamespace(content="Step 6 finished without passed marker.")

    monkeypatch.setattr(module, "create_consistency_agent", lambda: FakeAgent())

    result = module.asyncio.run(
        module.run_step6_only(
            {
                "input_csv": str(input_csv),
                "output_step6_dir": str(output_dir),
                "patient_id_col": "",
            }
        )
    )

    assert result["success"] is False
    assert "PASSED_PATIENTS" in result["error"]
