from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_gold(root: Path) -> Path:
    visible = root / "gold" / "visible"
    visible.mkdir(parents=True)
    (visible / "cases.jsonl").write_text(
        json.dumps({"病历": {"性别": "F"}, "诊断列表": ["肝硬化"], "实验室检验": [{"AST": 88}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return root / "gold"


def test_gold_loader_requires_visible_cases(tmp_path):
    from autonomous_pipeline.gold import validate_gold_standard

    gold_dir = _write_gold(tmp_path)
    summary = validate_gold_standard(gold_dir, output_root=tmp_path / "run", include_hidden=True)
    assert summary["valid"] is True
    assert summary["visible_count"] == 1
    assert summary["hidden_count"] == 0
    assert Path(summary["summary_path"]).is_file()


def test_task_analysis_builds_task_spec_from_visible_gold_and_raw_sample(tmp_path):
    from autonomous_pipeline.task_analysis import logic

    gold_dir = _write_gold(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "diagnoses_icd.csv").write_text("subject_id,icd_code\n10000032,5715\n", encoding="utf-8")
    result = logic.build_task_spec(raw, gold_dir, tmp_path / "run", task_text="肝病诊断")
    spec = json.loads(Path(result["task_spec_path"]).read_text(encoding="utf-8"))
    assert spec["task_goal"] == "肝病诊断"
    assert "诊断列表" in spec["content_expectations"]
    assert "diagnoses_icd" in spec["source_priority"]


def test_evaluator_checks_patient_cases_and_ml_dataset(tmp_path):
    from autonomous_pipeline.evaluator import logic

    gold_dir = _write_gold(tmp_path)
    round_root = tmp_path / "round_01"
    step4_next = round_root / "step4_results" / "next_input"
    step4_next.mkdir(parents=True)
    (step4_next / "patient_cases.jsonl").write_text(
        json.dumps({"patient_id": "10000032", "sections": {"诊断列表": [{"value": "肝硬化"}], "实验室检验": [{"value": "AST 88"}]}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    step6_next = round_root / "step6_results" / "next_input"
    step6_next.mkdir(parents=True)
    dataset = round_root / "step6_results" / "ml_dataset.csv"
    dataset.write_text("patient_id,feature,y,label_name\n10000032,1,1,liver\n10000764,0,0,no_liver\n", encoding="utf-8")
    model_config = round_root / "step6_results" / "model_config.json"
    model_config.write_text(json.dumps({"feature_names": ["feature"]}), encoding="utf-8")
    (step6_next / "latest_dataset.json").write_text(json.dumps({"dataset_csv": str(dataset), "model_config_json": str(model_config)}), encoding="utf-8")

    result = logic.evaluate_round_outputs(round_root, gold_dir, tmp_path / "eval", previous_score=0.0)
    assert result["score"] > 0.5
    assert Path(result["private_report_path"]).is_file()
    assert Path(result["public_feedback_path"]).is_file()


def test_planner_tools_resolve_and_inspect_round_artifacts(tmp_path):
    from autonomous_pipeline.planner import logic

    raw = tmp_path / "raw"
    raw.mkdir()
    round_root = tmp_path / "round"
    assert logic.resolve_step_input(round_root, "step1", raw)["input_path"] == str(raw.resolve())
    assert logic.resolve_step_input(round_root, "step3", raw)["input_path"].endswith("step2_results/next_input/input.csv")
    artifacts = logic.inspect_round_artifacts(round_root)
    assert artifacts["complete"] is False
    assert "step1" in artifacts["missing"]


def test_cli_run_all_dry_run_uses_planner_prompt(tmp_path):
    gold_dir = _write_gold(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autonomous_pipeline.cli",
            "run-all",
            "--input",
            str(raw),
            "--task",
            "肝病诊断",
            "--gold-dir",
            str(gold_dir),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "task_analysis" in result.stdout
    assert "planner round 01" in result.stdout
    assert "required_tool_sequence" not in result.stdout
