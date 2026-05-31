from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

from autonomous_pipeline.steps.step2 import logic, tools


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_step1_output(root: Path) -> Path:
    step1 = root / "step1_results"
    patient = step1 / "10000032"
    _write_csv(
        patient / "table" / "structured" / "patients.csv",
        [{"subject_id": "10000032", "gender": "F", "anchor_age": "52"}],
    )
    _write_csv(
        patient / "table" / "structured" / "admissions.csv",
        [{"subject_id": "10000032", "hadm_id": "2001", "admittime": "2026-01-01"}],
    )
    _write_csv(
        patient / "table" / "structured" / "diagnoses_icd.csv",
        [
            {"subject_id": "10000032", "hadm_id": "2001", "seq_num": "1", "icd_code": "5715", "icd_version": "9"},
            {"subject_id": "10000032", "hadm_id": "2001", "seq_num": "2", "icd_code": "5723", "icd_version": "9"},
        ],
    )
    _write_csv(
        patient / "table" / "notes" / "discharge_notes.csv",
        [
            {
                "subject_id": "10000032",
                "hadm_id": "2001",
                "note_id": "n1",
                "note_type": "discharge",
                "note_seq": "1",
                "text": "Patient has cirrhosis with ascites. Bilirubin 3.2 mg/dL.",
            }
        ],
    )
    figure = patient / "figure" / "images" / "p10000032" / "s1" / "x.jpg"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"fake image")
    return step1


def test_step2_public_tools_accept_reason_argument():
    expected = {
        "resolve_step2_input",
        "scan_step2_input",
        "select_step2_pipeline",
        "collect_directory_payloads",
        "extract_entities",
        "cluster_entity_names",
        "build_admission_wide",
        "run_ocr",
        "extract_ocr_entities",
        "fill_table_from_entities",
        "normalize_step2_outputs",
        "validate_step2_output",
    }
    actual = {func.__name__ for func in tools.TOOL_FUNCTIONS}
    assert expected <= actual
    for func in tools.TOOL_FUNCTIONS:
        signature = inspect.signature(func)
        assert "reason" in signature.parameters, func.__name__
        assert signature.parameters["reason"].default == ""


def test_scan_step2_input_profiles_step1_directory(tmp_path):
    step1 = _make_step1_output(tmp_path)

    result = tools.scan_step2_input(str(step1), str(tmp_path / "out"))

    assert result["status"] == "SUCCESS"
    assert result["artifacts"]["patient_count"] == 1
    assert result["artifacts"]["image_count"] == 1
    assert result["artifacts"]["notes_csv_count"] == 1
    assert result["artifacts"]["structured_csv_count"] == 3
    assert result["artifacts"]["text_csv_count"] == 1


def test_directory_payloads_entities_clustering_and_wide_table(tmp_path, monkeypatch):
    step1 = _make_step1_output(tmp_path)
    output_root = tmp_path / "out"

    payloads = tools.collect_directory_payloads(str(step1), str(output_root))
    assert payloads["status"] == "SUCCESS"
    assert Path(payloads["artifacts"]["patient_payloads_json"]).exists()
    assert Path(payloads["artifacts"]["text_tasks_jsonl"]).exists()
    assert payloads["artifacts"]["text_task_count"] == 1

    async def fake_extract(text: str, max_chars: int = 4000):
        return {
            "success": True,
            "entities": [
                {"category": "Diagnosis", "name": "cirrhosis", "status": "confirmed"},
                {"category": "LabResult", "name": "bilirubin", "value": "3.2", "unit": "mg/dL"},
            ],
            "impression": "cirrhosis with ascites",
            "indication": "",
        }

    monkeypatch.setattr(logic, "extract_from_text", fake_extract)
    extracted = tools.extract_entities(payloads["artifacts"]["text_tasks_jsonl"], str(output_root), source_kind="directory", concurrency=3)
    assert extracted["status"] == "SUCCESS"
    assert extracted["artifacts"]["entity_count"] == 2

    monkeypatch.setattr(
        logic,
        "get_embeddings",
        lambda names: [[0.0, 0.0], [1.0, 0.0]] if len(names) == 2 else [[0.0, 0.0] for _ in names],
    )
    clustered = tools.cluster_entity_names(extracted["artifacts"]["entities_long_csv"], str(output_root))
    assert clustered["status"] == "SUCCESS"
    assert clustered["artifacts"]["entity_column_count_after_clustering"] >= 1

    wide = tools.build_admission_wide(
        payloads["artifacts"]["patient_payloads_json"],
        clustered["artifacts"]["clustered_entities_csv"],
        str(output_root),
    )
    assert wide["status"] == "SUCCESS"
    assert wide["artifacts"]["wide_row_count"] == 1
    cols, rows = logic.read_csv_rows(wide["artifacts"]["admission_wide_csv"])
    assert rows[0]["subject_id"] == "10000032"
    assert rows[0]["hadm_id"] == "2001"
    assert rows[0]["structured__diagnoses_icd__row_count"] == "2"
    assert rows[0]["discharge_Diagnosis_cirrhosis_status"] == "confirmed"
    assert rows[0]["discharge_LabResult_bilirubin_value"] == "3.2"


def test_ocr_cache_extraction_fill_and_normalize(tmp_path, monkeypatch):
    step1 = _make_step1_output(tmp_path)
    output_root = tmp_path / "out"

    monkeypatch.setattr(logic, "ocr_image", lambda path: "Cirrhosis noted on report")
    ocr = tools.run_ocr(str(step1), str(output_root), ocr_workers=2)
    assert ocr["status"] == "SUCCESS"
    assert ocr["artifacts"]["successful_ocr"] == 1

    async def fake_extract(text: str, max_chars: int = 4000):
        return {
            "success": True,
            "entities": [{"category": "LabResult", "name": "风湿因子", "value": "18.5", "unit": "IU/mL"}],
            "impression": "",
            "indication": "",
        }

    monkeypatch.setattr(logic, "extract_from_text", fake_extract)
    extracted = tools.extract_ocr_entities(ocr["artifacts"]["ocr_cache_path"], str(output_root), results_dir=ocr["artifacts"]["results_dir"], concurrency=2)
    assert extracted["status"] == "SUCCESS"

    template = step1 / "10000032" / "table" / "template.csv"
    _write_csv(template, [{"subject_id": "10000032", "existing": "keep", "辅助检查-实验室检查-类风湿因子": ""}])
    filled = tools.fill_table_from_entities(str(step1), extracted["artifacts"]["entities_csv"], str(output_root), use_llm=False)
    assert filled["status"] == "SUCCESS"
    cols, rows = logic.read_csv_rows(filled["artifacts"]["filled_merged_csv"])
    assert rows[0]["existing"] == "keep"
    assert rows[0]["辅助检查-实验室检查-类风湿因子"] == "18.5"
    assert Path(filled["artifacts"]["filled_mappings_json"]).exists()

    normalized = tools.normalize_step2_outputs(str(output_root), filled_merged_csv=filled["artifacts"]["filled_merged_csv"])
    assert normalized["status"] == "SUCCESS"
    valid = tools.validate_step2_output(normalized["artifacts"]["next_input_csv"], normalized["artifacts"]["next_summary_json"])
    assert valid["status"] == "SUCCESS"
    trace = json.loads((output_root / "step2_results" / "_meta" / "tool_trace.json").read_text(encoding="utf-8"))
    assert any(item["tool"] == "run_ocr" for item in trace)
