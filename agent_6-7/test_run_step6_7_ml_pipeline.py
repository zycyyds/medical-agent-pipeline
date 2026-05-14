import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_step6_7_ml_pipeline.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("step6_7_pipeline_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_data_maps_id_to_patient_id(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")

    module = _load_module()
    df = module._read_data(str(csv_path))

    assert "patient_id" in df.columns
    assert "id" not in df.columns
    assert df["patient_id"].astype(str).tolist() == ["1", "2"]


def test_read_data_maps_subject_id_to_patient_id(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("subject_id,hadm_id,value\n100,200,a\n101,201,b\n", encoding="utf-8")

    module = _load_module()
    df = module._read_data(str(csv_path))

    assert "patient_id" in df.columns
    assert "subject_id" not in df.columns
    assert df["patient_id"].astype(str).tolist() == ["100", "101"]


def test_parse_passed_ids_handles_empty_list():
    module = _load_module()

    assert module._parse_passed_ids("PASSED_PATIENTS: []") == []


def test_parse_passed_ids_requires_marker():
    module = _load_module()

    assert module._parse_passed_ids("Step 6 finished without marker") is None


def test_standalone_defaults_use_data_input_and_split_program_output():
    module = _load_module()
    project_root = MODULE_PATH.parents[1]

    assert module.get_default_input_dir() == str(project_root / "agent_6-7" / "data_input")
    assert module.get_default_output_step6_dir() == str(project_root / "program" / "output" / "step6_results")
    assert module.get_default_output_step7_dir() == str(project_root / "program" / "output" / "step7_results")
