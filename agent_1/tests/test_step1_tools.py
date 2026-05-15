import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP1_DIR = PROJECT_ROOT / "step-1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))

from step1_tools_core import (
    build_records_parallel_tool,
    run_reorganize_from_records_tool,
    scan_source_files_tool,
    validate_records_tool,
    validate_step1_output_tool,
)


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


def test_scan_source_files_tool_returns_serialized_tasks(tmp_path):
    sample = tmp_path / "mimic"
    sample.mkdir()
    _build_sample(sample)

    payload = json.loads(str(scan_source_files_tool(str(sample)).content))

    assert payload["status"] == "SUCCESS"
    assert payload["artifacts"]["source_files"] == 3
    assert len(payload["artifacts"]["tasks"]) == 3


def test_step1_tools_build_validate_and_reorganize(tmp_path, monkeypatch):
    monkeypatch.setenv("STEP1_DOCLAYOUT_ENABLED", "0")
    sample = tmp_path / "mimic"
    sample.mkdir()
    _build_sample(sample)
    records_path = tmp_path / "records.json"
    output_root = tmp_path / "step1_results"

    build_payload = json.loads(
        str(build_records_parallel_tool(str(sample), str(records_path), max_workers=2, parallel_enabled=True).content)
    )
    validate_payload = json.loads(str(validate_records_tool(str(records_path), expected_count=3).content))
    run_payload = json.loads(
        str(run_reorganize_from_records_tool(str(records_path), str(output_root), input_root=str(sample)).content)
    )
    output_payload = json.loads(str(validate_step1_output_tool(str(output_root), expected_min_files=2).content))

    assert build_payload["status"] == "SUCCESS"
    assert build_payload["artifacts"]["records_count"] == 3
    assert validate_payload["status"] == "SUCCESS"
    assert run_payload["status"] == "SUCCESS"
    assert run_payload["artifacts"]["processed_records"] == 2
    assert output_payload["status"] == "SUCCESS"
    assert (output_root / "10000032" / "figure" / "images" / "p10000032" / "s1" / "xray.jpg").is_file()
