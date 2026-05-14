import asyncio
import json
from pathlib import Path

import pandas as pd

import agent_4.data_quality_repair as step4


def test_identifier_and_code_columns_are_report_only():
    df = pd.DataFrame(
        {
            "subject_id": [" 001 ", "002"],
            "icd_code": [" I10 ", "E11"],
        }
    )

    for column in df.columns:
        values = step4.sample_values(df[column])
        profile = step4.infer_column_profile(column, values)
        decision = step4.classify_column_risk(column, df[column], profile)

        assert decision["risk_level"] == "low"
        assert decision["cleaning_mode"] == "report_only"

    cleaned = step4.apply_builtin_safe_cleaning(df.copy(), {})

    pd.testing.assert_frame_equal(cleaned, df)


def test_builtin_safe_cleaning_normalizes_numeric_values():
    df = pd.DataFrame({"lab_value": [" 1,234.5 ", "nan", "-2.0"]})
    decisions = {
        "lab_value": {
            "risk_level": "medium",
            "cleaning_mode": "builtin_safe_clean",
            "profile": {"column_type": "numeric"},
        }
    }

    cleaned = step4.apply_builtin_safe_cleaning(df, decisions)

    assert cleaned["lab_value"].tolist() == ["1234.5", "", "-2.0"]


def test_builtin_safe_cleaning_normalizes_only_unambiguous_dates():
    df = pd.DataFrame({"chart_date": ["2024/01/03", "03/01/2024", "2024-01-03 08:30:00"]})
    decisions = {
        "chart_date": {
            "risk_level": "medium",
            "cleaning_mode": "builtin_safe_clean",
            "profile": {"column_type": "datetime"},
        }
    }

    cleaned = step4.apply_builtin_safe_cleaning(df, decisions)

    assert cleaned["chart_date"].tolist() == ["2024-01-03", "03/01/2024", "2024-01-03 08:30:00"]


def test_free_text_column_is_high_risk_llm_cleaner():
    series = pd.Series(
        [
            "Patient has cirrhosis with ascites and hepatic failure.",
            "Long discharge summary with inconsistent narrative findings.",
        ]
    )
    profile = step4.infer_column_profile("diagnosis_note", step4.sample_values(series))

    decision = step4.classify_column_risk("diagnosis_note", series, profile)

    assert decision["risk_level"] == "high"
    assert decision["cleaning_mode"] == "llm_cleaner"


def test_task_specific_terms_do_not_force_high_risk_without_shape_signal():
    df = pd.DataFrame(
        {
            "deathtime": ["2024-01-01 08:30:00", "2024-01-02 09:45:00"],
            "length_of_stay": ["3.5", "8.0"],
            "liver_score": ["1.2", "2.4"],
            "hepatic_flag": ["0", "1"],
        }
    )

    for column in df.columns:
        values = step4.sample_values(df[column])
        profile = step4.infer_column_profile(column, values)
        decision = step4.classify_column_risk(column, df[column], profile)

        assert decision["cleaning_mode"] != "llm_cleaner"


def test_run_data_quality_repair_keeps_all_columns_and_only_calls_llm_for_high_risk(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    source_csv = input_dir / "input.csv"
    pd.DataFrame(
        {
            "subject_id": [" 001 ", "002"],
            "lab_value": [" 1,234.5 ", "7.0"],
            "chart_date": ["2024/01/03", "03/01/2024"],
            "diagnosis_note": [
                "Patient has cirrhosis with ascites and hepatic failure.",
                "Long discharge summary with mortality risk notes.",
            ],
        }
    ).to_csv(source_csv, index=False)

    called_columns = []

    async def fake_build_cleaner(column, values, profile, human, derived_path, cleaner_dir, csv_path):
        called_columns.append(column)
        cleaner_dir = Path(cleaner_dir)
        implementation = (
            "import pandas as pd\n"
            "from agentscope.tool import ToolResponse\n\n"
            "def data_cleaning(input_path: str, index=False) -> ToolResponse:\n"
            "    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[''])\n"
            f"    if '{column}' in df.columns:\n"
            f"        df['{column}'] = df['{column}'].astype(str).str.strip()\n"
            "    output_path = input_path.replace('.csv', '_cleaned.csv')\n"
            "    df.to_csv(output_path, index=False)\n"
            "    return ToolResponse(content='ok', metadata={'output_file': output_path})\n"
        )
        with open(cleaner_dir / "implementation.py", "w", encoding="utf-8") as f:
            f.write(implementation)
        with open(cleaner_dir / "CLEANER.md", "w", encoding="utf-8") as f:
            f.write("---\nname: fake\n---\n")
        return {"cleaner_name": step4.safe_name(column), "summary": "fake", "profile": profile}, []

    monkeypatch.setattr(step4, "build_cleaner", fake_build_cleaner)

    result = asyncio.run(
        step4.run_data_quality_repair(
            input_dir=str(input_dir),
            workspace_dir=str(workspace_dir),
            validation_config_path=step4.get_default_validation_config_path(),
            verbose=False,
        )
    )

    assert result["success"] is True
    assert called_columns == ["diagnosis_note"]
    assert result["report_only_columns"] == 1
    assert result["builtin_cleaned_columns"] == 2
    assert result["llm_cleaner_columns"] == 1

    output_df = pd.read_csv(result["final_output_csv"], dtype=str, keep_default_na=False)
    assert output_df.columns.tolist() == ["subject_id", "lab_value", "chart_date", "diagnosis_note"]
    assert output_df["subject_id"].tolist() == [" 001 ", "002"]
    assert output_df["lab_value"].tolist() == ["1234.5", "7.0"]
    assert output_df["chart_date"].tolist() == ["2024-01-03", "03/01/2024"]

    with open(result["column_risk_report_path"], "r", encoding="utf-8") as f:
        risk_report = json.load(f)
    assert risk_report["columns"]["subject_id"]["cleaning_mode"] == "report_only"
    assert risk_report["columns"]["lab_value"]["cleaning_mode"] == "builtin_safe_clean"
    assert risk_report["columns"]["diagnosis_note"]["cleaning_mode"] == "llm_cleaner"


def test_standalone_defaults_use_data_input_and_program_output():
    project_root = step4.PROJECT_ROOT

    assert step4.get_default_input_dir() == str(project_root / "agent_4" / "data_input")
    assert step4.get_default_workspace_dir() == str(project_root / "program" / "output" / "step5_results")
