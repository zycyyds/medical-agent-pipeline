from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd


def _write_step4_filtered(path: Path) -> Path:
    rows = [
        {
            "subject_id": "10000032",
            "hadm_id": "22595853",
            "gender": "F",
            "anchor_age": "52",
            "race": "WHITE",
            "discharge_Diagnosis_hcv_cirrhosis_status": "confirmed",
            "discharge_LabResult_ast_value": "88",
        },
        {
            "subject_id": "10000032",
            "hadm_id": "22595853",
            "gender": "F",
            "anchor_age": "52",
            "race": "WHITE",
            "discharge_Diagnosis_hcv_cirrhosis_status": "confirmed",
            "discharge_LabResult_ast_value": "88",
        },
        {
            "subject_id": "10000764",
            "hadm_id": "27897940",
            "gender": "M",
            "anchor_age": "61",
            "race": "BLACK",
            "discharge_Diagnosis_hcv_cirrhosis_status": "suspected",
            "discharge_LabResult_ast_value": "41",
        },
        {
            "subject_id": "10000764",
            "hadm_id": "27897940",
            "gender": "F",
            "anchor_age": "61",
            "race": "BLACK",
            "discharge_Diagnosis_hcv_cirrhosis_status": "confirmed",
            "discharge_LabResult_ast_value": "43",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_step5_public_tools_accept_reason_argument():
    from autonomous_pipeline.steps.step5.tools import TOOL_FUNCTIONS

    for func in TOOL_FUNCTIONS:
        signature = inspect.signature(func)
        assert "reason" in signature.parameters, func.__name__
        assert signature.parameters["reason"].default == ""


def test_step5_old_step6_style_consistency_pipeline(tmp_path):
    from autonomous_pipeline.steps.step5 import tools

    input_csv = _write_step4_filtered(tmp_path / "step4_results" / "next_input" / "filtered.csv")
    output_root = str(tmp_path / "program" / "output")

    resolved = tools.resolve_step5_input(str(input_csv), output_root, reason="定位 Step4 输出")
    assert resolved["status"] == "SUCCESS"
    assert resolved["artifacts"]["input_csv"].endswith("filtered.csv")

    overview = tools.load_data_overview(resolved["artifacts"]["input_csv"], output_root, reason="读取数据概况")
    assert overview["status"] == "SUCCESS"
    assert overview["artifacts"]["patient_id_column"] == "subject_id"
    assert overview["artifacts"]["patient_count"] == 2

    analysis = tools.analyze_all_patients_consistency(
        resolved["artifacts"]["input_csv"],
        output_root,
        threshold=0.70,
        reason="按旧 Step6 规则批量打分",
    )
    assert analysis["status"] == "SUCCESS"
    assert analysis["artifacts"]["total_patients"] == 2
    assert analysis["artifacts"]["passed_patients"] == 1
    assert Path(analysis["artifacts"]["analysis_json"]).is_file()

    saved = tools.save_step5_report(
        resolved["artifacts"]["input_csv"],
        analysis["artifacts"]["analysis_json"],
        output_root,
        report_content="Step5 LLM report\nPASSED_PATIENTS: <stored_in_step5_report_json>",
        reason="保存报告和通过患者名单",
    )
    assert saved["status"] == "SUCCESS"
    assert Path(saved["artifacts"]["next_input_csv"]).is_file()
    assert Path(saved["artifacts"]["next_passed_patients_json"]).is_file()

    passed_payload = json.loads(Path(saved["artifacts"]["next_passed_patients_json"]).read_text(encoding="utf-8"))
    assert passed_payload["passed_patient_ids"] == ["10000032"]
    report_payload = json.loads(Path(saved["artifacts"]["json_report"]).read_text(encoding="utf-8"))
    assert report_payload["report_author"] == "agent_llm"

    validated = tools.validate_step5_output(
        saved["artifacts"]["json_report"],
        saved["artifacts"]["passed_patients_json"],
        saved["artifacts"]["next_input_dir"],
        reason="校验 Step5 输出",
    )
    assert validated["status"] == "SUCCESS"

    trace = Path(output_root) / "step5_results" / "_meta" / "tool_trace.json"
    assert trace.is_file()
