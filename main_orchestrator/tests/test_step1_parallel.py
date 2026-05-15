import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

from step1_parallel import build_records_parallel, run_reorganize_from_records, scan_source_files
from validators import validate_records, validate_step1_output


def _build_sample(root: Path) -> None:
    image_dir = root / "images" / "p10000032" / "s1"
    image_dir.mkdir(parents=True)
    (image_dir / "xray.jpg").write_bytes(b"fake")
    structured = root / "structured"
    structured.mkdir()
    pd.DataFrame(
        [
            {"subject_id": "10000032", "value": "a"},
            {"subject_id": "10000033", "value": "b"},
        ]
    ).to_csv(structured / "patients.csv", index=False)
    (root / "mini_manifest.json").write_text("{}", encoding="utf-8")


def test_step1_legacy_records_all_files_but_only_processes_supported_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("STEP1_DOCLAYOUT_ENABLED", "0")
    sample = tmp_path / "mimic"
    sample.mkdir()
    _build_sample(sample)
    serial_records = tmp_path / "serial_records.json"
    parallel_records = tmp_path / "parallel_records.json"

    serial = build_records_parallel(sample, serial_records, max_workers=1, parallel_enabled=False)
    parallel = build_records_parallel(sample, parallel_records, max_workers=4, parallel_enabled=True)

    assert serial["source_files"] == 3
    assert serial["processable_files"] == 2
    assert serial["unsupported_files"] == 1
    assert parallel["source_files"] == 3
    assert parallel["processable_files"] == 2
    assert parallel["unsupported_files"] == 1
    assert json.loads(serial_records.read_text(encoding="utf-8")) == json.loads(
        parallel_records.read_text(encoding="utf-8")
    )
    records = json.loads(parallel_records.read_text(encoding="utf-8"))
    manifest_record = next(item for item in records if item["source_name"] == "mini_manifest.json")
    assert manifest_record["observations"]["modality"] == "unsupported"
    assert manifest_record["observations"]["status"] == "unsupported"
    validation = validate_records(parallel_records, expected_count=3)
    assert validation.passed


def test_reorganize_from_records_splits_tables_by_subject_id(tmp_path, monkeypatch):
    monkeypatch.setenv("STEP1_DOCLAYOUT_ENABLED", "0")
    sample = tmp_path / "mimic"
    sample.mkdir()
    _build_sample(sample)
    records_path = tmp_path / "records.json"
    output_root = tmp_path / "step1_results"

    build_records_parallel(sample, records_path, max_workers=4, parallel_enabled=True)
    result = run_reorganize_from_records(records_path, output_root, input_root=sample)

    assert result["status"] == "SUCCESS"
    assert result["records_count"] == 3
    assert result["processed_records"] == 2
    assert result["skipped_records"] == 1
    assert (output_root / "10000032" / "table" / "structured" / "patients.csv").is_file()
    assert (output_root / "10000033" / "table" / "structured" / "patients.csv").is_file()
    assert (output_root / "10000032" / "figure" / "images" / "p10000032" / "s1" / "xray.jpg").is_file()
    assert not (output_root / "unknown" / "unsupported" / "mini_manifest.json").exists()


def test_reorganize_from_records_parallel_matches_serial_output(tmp_path, monkeypatch):
    monkeypatch.setenv("STEP1_DOCLAYOUT_ENABLED", "0")
    sample = tmp_path / "mimic"
    sample.mkdir()
    _build_sample(sample)
    records_path = tmp_path / "records.json"
    serial_output = tmp_path / "serial_step1_results"
    parallel_output = tmp_path / "parallel_step1_results"

    build_records_parallel(sample, records_path, max_workers=4, parallel_enabled=True)
    serial = run_reorganize_from_records(
        records_path,
        serial_output,
        input_root=sample,
        parallel_enabled=False,
    )
    parallel = run_reorganize_from_records(
        records_path,
        parallel_output,
        input_root=sample,
        max_workers=2,
        parallel_enabled=True,
    )

    assert serial["status"] == "SUCCESS"
    assert parallel["status"] == "SUCCESS"
    assert parallel["parallel_enabled"] is True
    assert parallel["max_workers"] == 2
    assert parallel["written_files"] == serial["written_files"]

    serial_files = sorted(path.relative_to(serial_output) for path in serial_output.rglob("*") if path.is_file())
    parallel_files = sorted(path.relative_to(parallel_output) for path in parallel_output.rglob("*") if path.is_file())
    assert parallel_files == serial_files


def test_step1_output_validator_reports_full_modality_counts(tmp_path):
    output_root = tmp_path / "step1_results"
    for idx in range(25):
        path = output_root / f"p{idx:05d}" / "table" / "structured" / f"{idx}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("subject_id,value\n1,a\n", encoding="utf-8")
    for idx in range(3):
        path = output_root / f"p{idx:05d}" / "figure" / "images" / f"{idx}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    result = validate_step1_output(output_root, expected_min_files=28)

    assert result.passed
    assert result.details["modality_counts"] == {"figure": 3, "table": 25}
    assert sum(result.details["sample_modality_counts"].values()) == 20


def test_scan_mimic10_fixture_if_present():
    mimic10 = PROJECT_ROOT / "mimic-10"
    if not mimic10.is_dir():
        return
    tasks = scan_source_files(mimic10)
    suffix_counts = {}
    for task in tasks:
        suffix_counts[task.suffix] = suffix_counts.get(task.suffix, 0) + 1

    assert len(tasks) >= 31
    assert suffix_counts[".csv"] == 9
    assert suffix_counts[".jpg"] == 22
