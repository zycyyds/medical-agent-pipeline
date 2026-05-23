from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP5_DIR = PROJECT_ROOT / "step-5"
if str(STEP5_DIR) not in sys.path:
    sys.path.insert(0, str(STEP5_DIR))

import agent_5.main as agent5_main
import step5_runtime


def test_agent5_standalone_defaults_use_cleaning_paths():
    project_root = Path(agent5_main.get_project_root())

    assert agent5_main.get_default_input_dir() == str(project_root / "agent_5" / "data_input")
    assert agent5_main.get_default_workspace_dir() == str(project_root / "program" / "output" / "step5_results")


def test_agent5_runtime_keeps_non_low_columns_original_when_llm_disabled(tmp_path):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text(
        "subject_id,age,note\n1,\" 1,234 \",\"patient has vertigo, dizziness, and long clinical comments.\"\n",
        encoding="utf-8",
    )

    result = agent5_main.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(tmp_path / "step5"),
            "--disable-llm",
        ]
    )

    assert result["status"] == "SUCCESS"
    cleaned = pd.read_csv(result["next_input_csv"], dtype=object, keep_default_na=False)
    assert list(cleaned.columns) == ["subject_id", "age", "note"]
    assert len(cleaned) == 1
    assert cleaned.loc[0, "age"] == " 1,234 "
    assert cleaned.loc[0, "note"] == "patient has vertigo, dizziness, and long clinical comments."
    assert set(result["llm_noop_columns"]) == {"age", "note"}
    assert result["changed_cell_count"] == 0
    assert Path(result["next_data_quality_report"]).is_file()
    assert Path(result["next_column_risk_report"]).is_file()


def test_agent5_high_risk_llm_cleaner_updates_only_target_column(tmp_path, monkeypatch):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,note\n1,foo note\n2,keep\n", encoding="utf-8")

    def fake_llm(prompt: str) -> str:
        if "结构化 CSV 数据分析师" in prompt:
            return "note 是文本列，只清理明确噪声，不能修改其他列。"
        return _cleaner_package('df["note"] = df["note"].str.replace("foo", "bar", regex=False)')

    monkeypatch.setattr(step5_runtime, "_call_high_risk_llm", fake_llm)
    result = agent5_main.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(tmp_path / "step5"),
            "--llm-workers",
            "1",
        ]
    )

    assert result["status"] == "SUCCESS"
    assert result["high_risk_generated_cleaners"] == ["note"]
    cleaned = pd.read_csv(result["next_input_csv"], dtype=object, keep_default_na=False)
    assert cleaned["id"].tolist() == ["1", "2"]
    assert cleaned["note"].tolist() == ["bar note", "keep"]
    assert result["changed_cell_count"] == 1
    assert list(Path(result["generated_scripts_dir"]).glob("note_*/implementation.py"))


def test_step5_medium_risk_uses_llm_cleaner(tmp_path, monkeypatch):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,note\n1,foo note\n2,keep\n", encoding="utf-8")
    risk_report = tmp_path / "column_risk_report.json"
    risk_report.write_text(
        """
{
  "columns": [
    {"column_name": "id", "risk_level": "low"},
    {"column_name": "note", "risk_level": "medium"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    def fake_llm(prompt: str) -> str:
        if "结构化 CSV 数据分析师" in prompt:
            return "note 是中风险文本列，只清理明确噪声，不能修改其他列。"
        return _cleaner_package('df["note"] = df["note"].str.replace("foo", "bar", regex=False)')

    monkeypatch.setattr(step5_runtime, "_call_high_risk_llm", fake_llm)
    result = step5_runtime.run_step5_data_cleaning(
        input_csv_path=input_csv,
        column_risk_report_path=risk_report,
        output_root=tmp_path / "step5",
        llm_workers=1,
        enable_llm=True,
    )

    cleaned = pd.read_csv(result["cleaned_csv_path"], dtype=object, keep_default_na=False)
    assert cleaned["id"].tolist() == ["1", "2"]
    assert cleaned["note"].tolist() == ["bar note", "keep"]
    assert result["llm_generated_cleaners"] == ["note"]
    assert list(Path(result["generated_scripts_dir"]).glob("note_*/implementation.py"))
    changes = Path(result["cleaning_changes_path"]).read_text(encoding="utf-8")
    assert '"risk_level": "medium"' in changes
    assert '"method": "llm_generated_cleaner"' in changes


def test_agent5_high_risk_generation_failure_keeps_original(tmp_path, monkeypatch):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,note\n1,foo note\n", encoding="utf-8")

    def fake_llm(prompt: str) -> str:
        if "结构化 CSV 数据分析师" in prompt:
            return "note 是文本列，只清理明确噪声。"
        return """
===CLEANER_MD===
---
name: note
---
bad
===END_CLEANER_MD===
===IMPLEMENTATION_PY===
def data_cleaning(:
===END_IMPLEMENTATION_PY===
"""

    monkeypatch.setattr(step5_runtime, "_call_high_risk_llm", fake_llm)
    result = agent5_main.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(tmp_path / "step5"),
            "--llm-workers",
            "1",
        ]
    )

    cleaned = pd.read_csv(result["next_input_csv"], dtype=object, keep_default_na=False)
    assert cleaned.loc[0, "note"] == "foo note"
    assert result["high_risk_noop_columns"] == ["note"]
    assert "note" in result["high_risk_failed_columns"]
    assert any("generation failed" in warning for warning in result["warnings"])


def test_agent5_high_risk_probe_blocks_other_column_changes(tmp_path, monkeypatch):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,note\n1,foo note\n", encoding="utf-8")

    def fake_llm(prompt: str) -> str:
        if "结构化 CSV 数据分析师" in prompt:
            return "note 是文本列，只清理明确噪声。"
        return _cleaner_package('df["id"] = "999"')

    monkeypatch.setattr(step5_runtime, "_call_high_risk_llm", fake_llm)
    result = agent5_main.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(tmp_path / "step5"),
            "--llm-workers",
            "1",
        ]
    )

    cleaned = pd.read_csv(result["next_input_csv"], dtype=object, keep_default_na=False)
    assert str(cleaned.loc[0, "id"]) == "1"
    assert cleaned.loc[0, "note"] == "foo note"
    errors = result["high_risk_failed_columns"]["note"]["generation_errors"]
    assert any("非目标列" in error for error in errors)


def test_agent5_high_risk_runtime_repair_is_used(tmp_path, monkeypatch):
    input_csv = tmp_path / "filtered.csv"
    input_csv.write_text("id,note\n1,foo note\n", encoding="utf-8")
    calls = {"repair": 0}

    def fake_llm(prompt: str) -> str:
        if "结构化 CSV 数据分析师" in prompt:
            return "note 是文本列，只清理明确噪声。"
        if "运行失败" in prompt or "修正要求" in prompt:
            calls["repair"] += 1
            return _cleaner_package('df["note"] = df["note"].str.replace("foo", "fixed", regex=False)')
        return _cleaner_package(
            'if "run_" in input_path:\n'
            '        raise RuntimeError("run failure")\n'
            '    df["note"] = df["note"].str.replace("foo", "bar", regex=False)'
        )

    monkeypatch.setattr(step5_runtime, "_call_high_risk_llm", fake_llm)
    result = agent5_main.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(tmp_path / "step5"),
            "--llm-workers",
            "1",
        ]
    )

    cleaned = pd.read_csv(result["next_input_csv"], dtype=object, keep_default_na=False)
    assert cleaned.loc[0, "note"] == "fixed note"
    assert calls["repair"] >= 1
    assert result["high_risk_generated_cleaners"] == ["note"]


def _cleaner_package(body: str) -> str:
    return f"""
===CLEANER_MD===
---
name: note
description: 仅清洗 note 列
---
1. clean note
===END_CLEANER_MD===

===IMPLEMENTATION_PY===
import pandas as pd
from agentscope.tool import ToolResponse

def data_cleaning(input_path: str, index=False) -> ToolResponse:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])
    if "note" in df.columns:
        {body}
    output_path = input_path.replace(".csv", "_cleaned.csv")
    df.to_csv(output_path, index=index)
    return ToolResponse(content="ok", metadata={{"output_file": output_path}})
===END_IMPLEMENTATION_PY===
"""
