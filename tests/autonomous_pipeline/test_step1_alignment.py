from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from autonomous_pipeline.steps.step1 import logic, tools


def _tool(name: str):
    assert hasattr(tools, name), f"missing Step1 tool: {name}"
    return getattr(tools, name)


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_scan_filters_hidden_files_and_extracts_mimic_patient_id(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / ".DS_Store").write_text("junk", encoding="utf-8")
    (raw / "images" / ".DS_Store").parent.mkdir(parents=True)
    (raw / "images" / ".DS_Store").write_text("junk", encoding="utf-8")
    jpg = raw / "images" / "p10000032" / "s1" / "x.jpg"
    jpg.parent.mkdir(parents=True)
    jpg.write_bytes(b"fake image")

    result = logic.scan_source_files(str(raw))

    assert result["file_count"] == 1
    assert result["tasks"][0]["relative_path"] == "images/p10000032/s1/x.jpg"
    assert result["tasks"][0]["patient_id"] == "10000032"


def test_step1_records_schema_doclayout_table_split_and_reorganize(tmp_path):
    raw = tmp_path / "raw"
    image = raw / "images" / "p10000032" / "s1" / "x.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake image")
    _write_csv(
        raw / "structured" / "diagnoses_icd.csv",
        "subject_id,icd_code\n10000032,5715\n10000033,I10\n",
    )

    out = tmp_path / "out"
    scan = _tool("scan_source_files")(str(raw), str(out))
    base = _tool("build_base_records")(scan["artifacts"]["tasks_path"], str(out))
    visual = _tool("infer_visual_modality")(base["artifacts"]["base_records_path"], str(out))
    table = _tool("infer_table_split_strategy")(base["artifacts"]["base_records_path"], str(out))
    reduced = _tool("reduce_records")(
        base["artifacts"]["base_records_path"],
        visual["artifacts"]["visual_patches_path"],
        table["artifacts"]["table_split_patches_path"],
        str(out),
    )
    records_path = reduced["artifacts"]["records_path"]
    records_payload = json.loads(Path(records_path).read_text(encoding="utf-8"))

    assert isinstance(records_payload, list)
    assert {r["source_name"] for r in records_payload} == {"x.jpg", "diagnoses_icd.csv"}
    image_record = next(r for r in records_payload if r["source_name"] == "x.jpg")
    table_record = next(r for r in records_payload if r["source_name"] == "diagnoses_icd.csv")
    assert image_record["patient_id"] == "10000032"
    assert image_record["observations"]["modality"] in {"figure", "ocr"}
    assert image_record["worker_evidence"]["modality"] == "doclayout-yolo"
    assert table_record["observations"]["modality"] == "table"
    assert table_record["observations"]["should_split"] is True
    assert table_record["observations"]["id_column"] == "subject_id"

    valid_records = _tool("validate_records")(records_path, 2)
    assert valid_records["status"] == "SUCCESS"
    reorganized = _tool("reorganize_from_records")(records_path, str(out), str(raw))
    assert reorganized["status"] == "SUCCESS"
    output_root = Path(reorganized["artifacts"]["output_root"])
    assert (output_root / "10000032" / "figure" / "images" / "p10000032" / "s1" / "x.jpg").exists()
    assert (output_root / "10000032" / "table" / "structured" / "diagnoses_icd.csv").exists()
    assert (output_root / "10000033" / "table" / "structured" / "diagnoses_icd.csv").exists()

    valid_output = _tool("validate_step1_output")(str(out), 3)
    assert valid_output["status"] == "SUCCESS"
    trace = json.loads((out / "step1_results" / "_meta" / "tool_trace.json").read_text(encoding="utf-8"))
    assert any(item["tool"] == "infer_visual_modality" for item in trace)


def test_validate_records_rejects_missing_modality(tmp_path):
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps([
            {
                "source_path": str(tmp_path / "x.jpg"),
                "source_name": "x.jpg",
                "patient_id": "10000032",
                "relative_path": "images/p10000032/x.jpg",
                "file_info": {"suffix": ".jpg", "size_bytes": 1, "processable": True},
                "observations": {"should_split": False, "id_column": None, "status": "ready"},
                "worker_evidence": {},
            }
        ]),
        encoding="utf-8",
    )

    result = _tool("validate_records")(str(records_path), 1)

    assert result["status"] == "NEEDS_REPAIR"
    assert any("modality" in issue for issue in result["issues"])


def test_doclayout_ocr_figure_is_normalized_to_figure(monkeypatch, tmp_path):
    image_path = tmp_path / "p10000032" / "image.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"0" * 2048)

    class FakeResult:
        def tojson(self):
            return json.dumps(
                [
                    {"class": 3, "box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
                    {"class": 1, "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
                ]
            )

    class FakeModel:
        def predict(self, paths, imgsz, conf, device, verbose):
            return [FakeResult() for _ in paths]

    from autonomous_pipeline.steps.step1 import doclayout

    monkeypatch.setattr(doclayout, "_get_model", lambda model_path=None: (FakeModel(), "/fake/model.pt"))
    monkeypatch.setattr(doclayout, "_figure_area_ratio", lambda detections, width, height: 0.6)
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imread=lambda path: types.SimpleNamespace(shape=(100, 100, 3))))

    result = doclayout.classify_visual_records(
        [
            {
                "source_path": str(image_path),
                "relative_path": "images/p10000032/image.jpg",
                "file_info": {"suffix": ".jpg"},
            }
        ]
    )

    patch = result["patches"][0]
    assert patch["modality"] == "figure"
    assert patch["layout_analysis"]["raw_modality"] == "ocr+figure"


def test_doclayout_can_be_disabled_with_legacy_env(monkeypatch, tmp_path):
    from autonomous_pipeline.steps.step1 import doclayout

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake image bytes larger than tiny")
    monkeypatch.setenv("STEP1_DOCLAYOUT_ENABLED", "0")

    result = doclayout.classify_visual_records(
        [
            {
                "source_path": str(image_path),
                "relative_path": "images/p10000032/image.jpg",
                "file_info": {"suffix": ".jpg"},
            }
        ]
    )

    patch = result["patches"][0]
    assert result["doclayout_used"] is False
    assert patch["evidence"] == "doclayout disabled"
    assert patch["modality"] == "figure"


def test_reorganize_from_records_reports_parallel_progress(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("STEP1_WORKERS", raising=False)
    monkeypatch.setenv("STEP1_MAX_WORKERS", "3")
    monkeypatch.setenv("STEP1_REORGANIZE_MAX_WORKERS", "2")
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    for index in range(3):
        source = raw / f"p1000003{index}" / f"note{index}.txt"
        source.parent.mkdir(parents=True)
        source.write_text(f"note {index}", encoding="utf-8")
        records.append(
            {
                "source_path": str(source),
                "source_name": source.name,
                "patient_id": f"1000003{index}",
                "relative_path": f"p1000003{index}/{source.name}",
                "file_info": {"suffix": ".txt", "processable": True},
                "observations": {"modality": "text", "status": "ready"},
                "worker_evidence": {"modality": "suffix"},
            }
        )
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")

    result = logic.reorganize_from_records(str(records_path), str(tmp_path / "out"), input_root=str(raw))

    err = capsys.readouterr().err
    assert result["status"] == "SUCCESS"
    assert result["workers"] == 2
    assert "ReorganizeWorker 并行执行: workers=2" in err
    assert "ReorganizeWorker records:" in err


def test_step1_worker_count_uses_legacy_env_names(monkeypatch):
    monkeypatch.delenv("STEP1_WORKERS", raising=False)
    monkeypatch.setenv("STEP1_MAX_WORKERS", "3")
    monkeypatch.delenv("STEP1_REORGANIZE_MAX_WORKERS", raising=False)

    assert logic.get_step1_max_workers() == 3
    assert logic.get_step1_reorganize_max_workers() == 3

    monkeypatch.setenv("STEP1_REORGANIZE_MAX_WORKERS", "2")
    assert logic.get_step1_reorganize_max_workers() == 2
