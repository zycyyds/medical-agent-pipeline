from __future__ import annotations

import json
import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cli_lists_new_step_names_only():
    result = subprocess.run(
        [sys.executable, "-m", "autonomous_pipeline.cli", "list-steps"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "step1" in result.stdout
    assert "step6" in result.stdout
    assert "step2_3" not in result.stdout


def test_cli_dry_run_uses_step2_name(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "autonomous_pipeline.cli", "run-step", "--step", "step2", "--input", str(tmp_path), "--task", "肝病诊断", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "任务目标：肝病诊断" in result.stdout
    assert "required_tool_sequence" not in result.stdout


def test_step1_prompt_is_importable_and_task_is_optional(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "autonomous_pipeline.cli", "run-step", "--step", "step1", "--input", str(tmp_path), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "输入路径" in result.stdout
    assert "Step1 不使用疾病任务目标" in result.stdout


def test_step1_doclayout_default_device_is_mps(monkeypatch, tmp_path):
    image_path = tmp_path / "p10000032" / "image.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"0" * 2048)

    captured = {}

    class FakeResult:
        def tojson(self):
            return "[]"

    class FakeModel:
        def predict(self, paths, imgsz, conf, device, verbose):
            captured["device"] = device
            return [FakeResult() for _ in paths]

    from autonomous_pipeline.steps.step1 import doclayout

    monkeypatch.delenv("STEP1_DOCLAYOUT_DEVICE", raising=False)
    monkeypatch.setattr(doclayout, "_get_model", lambda model_path=None: (FakeModel(), "/fake/model.pt"))

    result = doclayout.classify_visual_records(
        [
            {
                "source_path": str(image_path),
                "relative_path": "images/p10000032/image.jpg",
                "file_info": {"suffix": ".jpg"},
            }
        ]
    )

    assert result["doclayout_used"] is True
    assert result["device"] == "mps"
    assert captured["device"] == "mps"


def test_tool_wrapper_prints_agent_decision_reason(capsys):
    from autonomous_pipeline.agent_factory import wrap_tool_function

    def sample_tool(reason: str = ""):
        """sample tool"""
        return {"status": "SUCCESS", "summary": "ok", "artifacts": {"records_count": 1}, "issues": [], "next_recommendation": "none"}

    wrapped = wrap_tool_function(sample_tool)
    wrapped(reason="需要先扫描输入目录，才能构建后续 records")

    err = capsys.readouterr().err
    assert "[Agent 决策] sample_tool: 需要先扫描输入目录" in err
    assert "[Agent 编排] 调用工具 sample_tool" in err


def test_step1_public_tools_accept_reason_argument():
    from autonomous_pipeline.steps.step1.tools import TOOL_FUNCTIONS

    for func in TOOL_FUNCTIONS:
        signature = inspect.signature(func)
        assert "reason" in signature.parameters, func.__name__
        assert signature.parameters["reason"].default == ""


def test_step1_tools_return_uniform_json(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    (data / "note.txt").write_text("patient has liver disease", encoding="utf-8")
    from autonomous_pipeline.steps.step1.tools import build_records, scan_source_files

    scan = scan_source_files(str(data), str(tmp_path / "out"))
    assert scan["status"] == "SUCCESS"
    assert scan["artifacts"]["file_count"] == 1
    assert "tasks_path" in scan["artifacts"]
    records = build_records(str(data), str(tmp_path / "out"))
    assert records["status"] == "SUCCESS"
    assert Path(records["artifacts"]["records_path"]).exists()


def test_step6_excludes_icd_label_source_from_features(tmp_path):
    rows = [
        {"patient_id": "p1", "structured__diagnoses_icd__icd_code": "K7290|I10", "age": "70", "albumin": "3.1"},
        {"patient_id": "p2", "structured__diagnoses_icd__icd_code": "I10", "age": "65", "albumin": "4.0"},
    ]
    csv_path = tmp_path / "input.csv"
    import csv
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    from autonomous_pipeline.steps.step6.tools import build_ml_dataset, create_icd_label, select_feature_columns, validate_dataset

    labeled = create_icd_label(str(csv_path), "structured__diagnoses_icd__icd_code", str(tmp_path / "out"))
    assert labeled["status"] == "SUCCESS"
    assert labeled["artifacts"]["positive_count"] == 1
    features = select_feature_columns(labeled["artifacts"]["labeled_csv"], str(tmp_path / "out"))
    feature_payload = json.loads(Path(features["artifacts"]["feature_columns_json"]).read_text(encoding="utf-8"))
    assert "structured__diagnoses_icd__icd_code" not in feature_payload["feature_columns"]
    dataset = build_ml_dataset(labeled["artifacts"]["labeled_csv"], features["artifacts"]["feature_columns_json"], str(tmp_path / "out"), "liver_test")
    validation = validate_dataset(dataset["artifacts"]["dataset_csv"], dataset["artifacts"]["model_config_json"])
    assert validation["status"] == "SUCCESS"
