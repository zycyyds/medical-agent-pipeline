import json
from pathlib import Path

import pandas as pd

import step4_runtime


def test_step4_runtime_normalizes_and_validates_outputs(tmp_path, monkeypatch):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"subject_id": ["1"], "value": ["2"], "drop_me": ["x"]}).to_csv(input_csv, index=False)
    output_root = tmp_path / "step4_results"

    class FakeSelector:
        def run(self, input_csv_path, task_text, output_dir, memory_context=None):
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            filtered = output / "input_filtered_20260518_120000.csv"
            report = output / "input_selection_report_20260518_120000.json"
            pd.DataFrame({"subject_id": ["1"], "value": ["2"]}).to_csv(filtered, index=False)
            report.write_text(
                json.dumps(
                    {
                        "task_text": task_text,
                        "column_counts": {
                            "original_column_count": 3,
                            "prefiltered_column_count": 3,
                            "selector_candidate_column_count": 2,
                            "final_column_count": 2,
                        },
                        "final_columns": ["subject_id", "value"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {"filtered_csv_path": str(filtered), "selection_report_path": str(report)}

    monkeypatch.setattr(step4_runtime, "TaskDrivenColumnSelector", FakeSelector)

    result = step4_runtime.run_step4_pipeline(input_csv, output_root=output_root, task_text="demo")

    assert result["status"] == "SUCCESS"
    assert Path(result["next_input_csv"]).is_file()
    assert Path(result["next_selection_report"]).is_file()
    assert Path(result["next_input_csv"]).name == "filtered.csv"
    assert result["final_column_count"] == 2


def test_step4_validation_rejects_mismatched_columns(tmp_path):
    filtered = tmp_path / "filtered.csv"
    report = tmp_path / "selection_report.json"
    pd.DataFrame({"subject_id": ["1"], "wrong": ["2"]}).to_csv(filtered, index=False)
    report.write_text(json.dumps({"final_columns": ["subject_id", "value"]}), encoding="utf-8")

    result = step4_runtime.validate_step4_output(filtered, report)

    assert result.passed is False
    assert any("列顺序" in issue for issue in result.issues)
