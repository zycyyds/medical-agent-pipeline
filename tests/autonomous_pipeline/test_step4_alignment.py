from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd


def _write_step3_filtered(path: Path) -> Path:
    rows = [
        {
            "subject_id": "10000032",
            "hadm_id": "22595853",
            "gender": "F",
            "anchor_age": " 52 ",
            "dischtime": "2180/05/10 12:30",
            "discharge_Diagnosis_hcv_cirrhosis_status": "foo confirmed",
            "discharge_LabResult_ast_value": " 1,234 ",
        },
        {
            "subject_id": "10000764",
            "hadm_id": "27897940",
            "gender": "M",
            "anchor_age": "61",
            "dischtime": "2181-01-01 01:02:03",
            "discharge_Diagnosis_hcv_cirrhosis_status": "keep",
            "discharge_LabResult_ast_value": "88",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    (path.parent / "selection_report.json").write_text(json.dumps({"final_columns": list(rows[0])}, ensure_ascii=False), encoding="utf-8")
    return path


def _cleaner_package(body: str) -> str:
    return f"""
===CLEANER_MD===
---
name: fake_cleaner
description: fake test cleaner
---
1. clean target column only
===END_CLEANER_MD===

===IMPLEMENTATION_PY===
import pandas as pd
from agentscope.tool import ToolResponse

def data_cleaning(input_path: str, index=False) -> ToolResponse:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])
    {body}
    output_path = input_path.replace(".csv", "_cleaned.csv")
    df.to_csv(output_path, index=index)
    return ToolResponse(content="ok", metadata={{"output_file": output_path}})
===END_IMPLEMENTATION_PY===
"""


def test_step4_public_tools_accept_reason_argument():
    from autonomous_pipeline.steps.step4.tools import TOOL_FUNCTIONS

    for func in TOOL_FUNCTIONS:
        signature = inspect.signature(func)
        assert "reason" in signature.parameters, func.__name__
        assert signature.parameters["reason"].default == ""


def test_step4_old_step5_style_cleaning_pipeline_without_llm(tmp_path):
    from autonomous_pipeline.steps.step4 import tools

    input_csv = _write_step3_filtered(tmp_path / "step3_results" / "next_input" / "filtered.csv")
    output_root = str(tmp_path / "program" / "output")

    resolved = tools.resolve_step4_input(str(input_csv), output_root, reason="定位 Step3 输出")
    assert resolved["status"] == "SUCCESS"
    assert resolved["artifacts"]["selection_report_path"].endswith("selection_report.json")

    profile = tools.profile_step4_table(resolved["artifacts"]["input_csv_path"], output_root, workers=2, reason="并行 profile")
    assert profile["status"] == "SUCCESS"
    assert profile["artifacts"]["row_count"] == 2
    assert profile["artifacts"]["column_count"] == 7

    risks = tools.classify_step4_column_risks(
        resolved["artifacts"]["input_csv_path"],
        profile["artifacts"]["profile_report_path"],
        output_root,
        workers=2,
        reason="旧规则风险分类",
    )
    assert risks["status"] == "SUCCESS"
    assert risks["artifacts"]["risk_counts"]["high"] >= 1

    cleaned = tools.run_step4_data_cleaning(
        resolved["artifacts"]["input_csv_path"],
        risks["artifacts"]["column_risk_report_path"],
        output_root,
        workers=2,
        llm_workers=1,
        enable_llm=False,
        reason="禁用真实 LLM 验证旧版 noop 回退",
    )
    assert cleaned["status"] == "SUCCESS"
    assert Path(cleaned["artifacts"]["cleaned_csv_path"]).is_file()
    assert cleaned["artifacts"]["llm_noop_columns"]

    normalized = tools.normalize_step4_outputs(
        cleaned["artifacts"]["cleaned_csv_path"],
        cleaned["artifacts"]["data_quality_report_path"],
        cleaned["artifacts"]["column_risk_report_path"],
        output_root,
        selection_report_path=resolved["artifacts"]["selection_report_path"],
        reason="生成 next_input",
    )
    assert normalized["status"] == "SUCCESS"
    assert Path(normalized["artifacts"]["next_input_csv"]).is_file()
    assert Path(normalized["artifacts"]["next_selection_report"]).is_file()
    assert Path(normalized["artifacts"]["next_patient_cases_jsonl"]).is_file()
    assert normalized["artifacts"]["patient_case_count"] == 2

    validated = tools.validate_step4_output(
        resolved["artifacts"]["input_csv_path"],
        cleaned["artifacts"]["cleaned_csv_path"],
        cleaned["artifacts"]["data_quality_report_path"],
        cleaned["artifacts"]["column_risk_report_path"],
        normalized["artifacts"]["next_input_csv"],
        normalized["artifacts"]["next_data_quality_report"],
        normalized["artifacts"]["next_column_risk_report"],
        reason="校验 Step4 输出",
    )
    assert validated["status"] == "SUCCESS"
    trace = Path(output_root) / "step4_results" / "_meta" / "tool_trace.json"
    assert trace.is_file()


def test_step4_llm_cleaner_updates_only_target_column(tmp_path, monkeypatch):
    from autonomous_pipeline.steps.step4 import logic

    input_csv = _write_step3_filtered(tmp_path / "input.csv")
    output_root = tmp_path / "out"

    def fake_llm(prompt: str) -> str:
        return _cleaner_package('df["discharge_Diagnosis_hcv_cirrhosis_status"] = df["discharge_Diagnosis_hcv_cirrhosis_status"].str.replace("foo", "bar", regex=False)')

    monkeypatch.setattr(logic, "_call_high_risk_llm", fake_llm)

    profile = logic.profile_step4_table(input_csv, output_root=output_root, workers=1)
    risks = logic.classify_step4_column_risks(input_csv, profile["profile_report_path"], output_root=output_root, workers=1)
    result = logic.run_step4_data_cleaning(input_csv, risks["column_risk_report_path"], output_root=output_root, workers=1, llm_workers=1, enable_llm=True)

    cleaned = pd.read_csv(result["cleaned_csv_path"], dtype=object, keep_default_na=False)
    assert cleaned["subject_id"].tolist() == ["10000032", "10000764"]
    assert cleaned["discharge_Diagnosis_hcv_cirrhosis_status"].tolist() == ["bar confirmed", "keep"]
    assert "discharge_Diagnosis_hcv_cirrhosis_status" in result["high_risk_generated_cleaners"]


def test_step4_patient_cases_are_content_oriented(tmp_path):
    from autonomous_pipeline.steps.step4 import logic

    input_csv = _write_step3_filtered(tmp_path / "input.csv")
    artifacts = logic.build_patient_cases(input_csv, output_root=tmp_path / "out", task_text="肝病诊断")
    path = Path(artifacts["patient_cases_jsonl"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert artifacts["patient_case_count"] == 2
    assert rows[0]["patient_id"] == "10000032"
    assert "诊断列表" in rows[0]["sections"]
    assert "实验室检验" in rows[0]["sections"]
