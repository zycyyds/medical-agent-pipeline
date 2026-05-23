import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("run_step7_ml_pipeline.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("agent7_step7_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_standalone_defaults_use_main_pipeline_artifacts():
    module = _load_module()
    project_root = MODULE_PATH.parents[1]

    assert module.get_default_input_dir() == str(project_root / "step-7" / "data_input")
    assert module.get_default_input_csv() == str(project_root / "program" / "output" / "step5_results" / "next_input" / "filtered.csv")
    assert module.get_default_selection_report() == str(project_root / "program" / "output" / "step4_results" / "next_input" / "selection_report.json")
    assert module.get_default_output_step6_dir() == str(project_root / "program" / "output" / "step6_results")
    assert module.get_default_output_step7_dir() == str(project_root / "program" / "output" / "step7_results")


def test_read_data_maps_common_id_columns_to_patient_id(tmp_path):
    for column_name in ("id", "subject_id", "record_index", "hadm_id"):
        csv_path = tmp_path / f"{column_name}.csv"
        csv_path.write_text(f"{column_name},value\n1,a\n2,b\n", encoding="utf-8")

        module = _load_module()
        module.configure_runtime({"input_csv": str(csv_path), "passed_patients_json": str(tmp_path / "missing.json")})
        df = module._read_data(str(csv_path))

        assert "patient_id" in df.columns
        assert column_name not in df.columns or column_name == "patient_id"
        assert df["patient_id"].astype(str).tolist() == ["1", "2"]


def test_load_passed_patient_ids_reads_explicit_json(tmp_path):
    module = _load_module()
    passed_json = tmp_path / "passed_patients.json"
    _write_json(passed_json, {"passed_patient_ids": ["1", 2], "passed_count": 2})

    module.configure_runtime({"passed_patients_json": str(passed_json)})

    assert module._load_passed_patient_ids() == ["1", "2"]


def test_run_step7_only_fails_without_passed_patient_json(tmp_path):
    module = _load_module()
    input_csv = tmp_path / "filtered.csv"
    selection_report = tmp_path / "selection_report.json"
    input_csv.write_text("id,feature,label\n1,10,A\n", encoding="utf-8")
    _write_json(selection_report, {"final_columns": ["id", "feature", "label"], "task": "mock"})

    result = module.asyncio.run(
        module.run_step7_only(
            {
                "input_csv": str(input_csv),
                "selection_report": str(selection_report),
                "passed_patients_json": str(tmp_path / "missing.json"),
                "output_step7_dir": str(tmp_path / "step7_results"),
            }
        )
    )

    assert result["success"] is False
    assert "passed_patients" in result["error"]


def test_run_step7_only_mock_agent_writes_dataset_outputs(monkeypatch, tmp_path):
    module = _load_module()
    input_csv = tmp_path / "filtered.csv"
    selection_report = tmp_path / "selection_report.json"
    passed_json = tmp_path / "passed_patients.json"
    output_dir = tmp_path / "step7_results"
    input_csv.write_text(
        "id,feature,label\n1,10,A\n2,20,B\n3,30,A\n",
        encoding="utf-8",
    )
    _write_json(selection_report, {"final_columns": ["id", "feature", "label"], "task": "mock"})
    _write_json(passed_json, {"passed_patient_ids": ["1", "2", "3"], "passed_count": 3})
    monkeypatch.setattr(module, "_local_icd10_lookup", lambda _diagnosis: ("", "NOT_FOUND", ""))

    class FakeAgent:
        async def __call__(self, _msg):
            module.format_and_save_ml_dataset(
                label_column="label",
                label_mapping={"A": 0, "B": 1},
                task_name="mock",
            )
            return SimpleNamespace(content="mock step7 completed")

    monkeypatch.setattr(module, "create_dataprep_agent", lambda: FakeAgent())

    result = module.asyncio.run(
        module.run_step7_only(
            {
                "input_csv": str(input_csv),
                "selection_report": str(selection_report),
                "passed_patients_json": str(passed_json),
                "output_step7_dir": str(output_dir),
                "patient_id_col": "",
            }
        )
    )

    assert result["success"] is True
    assert result["error"] == ""
    assert result["input_csv"] == str(input_csv.resolve())
    assert result["selection_report"] == str(selection_report.resolve())
    assert result["passed_patients_json"] == str(passed_json.resolve())
    assert result["passed_patient_count"] == 3
    assert len(result["step7_dataset_csvs"]) == 1
    assert len(result["step7_dataset_jsonls"]) == 1
    assert len(result["step7_dataset_jsons"]) == 1
    assert len(result["model_config_jsons"]) == 1
    for key in ("step7_dataset_csvs", "step7_dataset_jsonls", "step7_dataset_jsons", "model_config_jsons"):
        assert Path(result[key][0]).is_file()
