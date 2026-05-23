import importlib.util
import asyncio
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "agent_2-3" / "config.py"
RUNTIME_PATH = PROJECT_ROOT / "step-2-3" / "step23_runtime.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_agent23_chat_and_embedding_config_are_separate(monkeypatch):
    module = _load_module(CONFIG_PATH, "agent23_config_test")
    for key in (
        "AGENT_2_3_API_KEY",
        "AGENT_2_3_BASE_URL",
        "AGENT_2_3_MODEL",
        "AGENT_2_3_EMBEDDING_API_KEY",
        "AGENT_2_3_EMBEDDING_BASE_URL",
        "AGENT_2_3_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(
        module,
        "get_agent_config",
        lambda _key: {
            "api_key": "chat-key",
            "base_url": "https://chat.example/v1",
            "model": "chat-model",
            "embedding": {
                "api_key": "embed-key",
                "base_url": "https://embed.example/v1",
                "model": "embed-model",
                "dimensions": 1536,
            },
        },
    )

    chat_cfg = module.get_api_config()
    embedding_cfg = module.get_embedding_config()

    assert chat_cfg["api_key"] == "chat-key"
    assert chat_cfg["api_base"] == "https://chat.example/v1"
    assert chat_cfg["model_name"] == "chat-model"
    assert embedding_cfg["api_key"] == "embed-key"
    assert embedding_cfg["api_base"] == "https://embed.example/v1"
    assert embedding_cfg["model_name"] == "embed-model"
    assert embedding_cfg["dimensions"] == 1536


def test_step23_normalize_and_validate_next_input_contract(tmp_path):
    module = _load_module(RUNTIME_PATH, "step23_runtime_test")
    source = tmp_path / "filled_merged.csv"
    source.write_text("subject_id,value\n1,ok\n", encoding="utf-8")
    output_root = tmp_path / "step2_3_results"

    normalized = module.normalize_step23_outputs(
        output_root=str(output_root),
        filled_merged_csv=str(source),
        summary={"status": "SUCCESS"},
    )
    validation = module.validate_step23_output(
        next_input_csv=normalized["next_input_csv"],
        next_summary_json=normalized["next_summary_json"],
    )

    assert Path(normalized["next_input_csv"]).name == "input.csv"
    assert Path(normalized["next_input_csv"]).is_file()
    assert validation["passed"] is True
    assert validation["details"]["output_row_count"] == 1


def test_step23_auto_selects_directory_for_notes_and_structured(tmp_path):
    module = _load_module(RUNTIME_PATH, "step23_runtime_mode_test")
    table_dir = tmp_path / "100" / "table"
    notes_dir = table_dir / "notes"
    structured_dir = table_dir / "structured"
    notes_dir.mkdir(parents=True)
    structured_dir.mkdir()
    (notes_dir / "radiology_notes.csv").write_text(
        "subject_id,hadm_id,note_id,text\n100,200,n1,long clinical note text\n",
        encoding="utf-8",
    )
    (structured_dir / "patients.csv").write_text("subject_id,gender\n100,F\n", encoding="utf-8")

    scan = module.scan_step23_input(str(tmp_path))
    selected = module.select_step23_pipeline(str(tmp_path), scan_artifacts=scan, requested_mode="auto")

    assert scan["patient_count"] == 1
    assert scan["notes_csv_count"] == 1
    assert selected["resolved_mode"] == "directory"


def test_step23_directory_pipeline_writes_wide_csv_without_llm(tmp_path):
    module = _load_module(RUNTIME_PATH, "step23_runtime_directory_test")
    table_dir = tmp_path / "100" / "table"
    notes_dir = table_dir / "notes"
    structured_dir = table_dir / "structured"
    notes_dir.mkdir(parents=True)
    structured_dir.mkdir()
    (structured_dir / "patients.csv").write_text("subject_id,gender,anchor_age\n100,F,45\n", encoding="utf-8")
    (structured_dir / "admissions.csv").write_text("subject_id,hadm_id,admittime\n100,200,2020\n", encoding="utf-8")
    (structured_dir / "diagnoses_icd.csv").write_text("subject_id,hadm_id,icd_code\n100,200,A1\n", encoding="utf-8")
    (notes_dir / "radiology_notes.csv").write_text(
        "subject_id,hadm_id,note_id,text\n100,200,n1,\"vertigo note text repeated repeated repeated repeated\"\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        module.run_step23_directory_pipeline(
            str(tmp_path),
            output_root=str(tmp_path / "out"),
            use_llm=False,
        )
    )
    wide_csv = Path(result["admission_wide_csv"])
    header = wide_csv.read_text(encoding="utf-8-sig").splitlines()[0]

    assert wide_csv.is_file()
    assert result["resolved_mode"] == "directory"
    assert result["wide_row_count"] == 1
    assert "structured__diagnoses_icd__icd_code" in header
    assert "clinical_text" in header
