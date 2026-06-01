from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd


def _write_sample_step2_output(path: Path) -> Path:
    rows = [
        {
            "subject_id": "10000032",
            "hadm_id": "22595853",
            "gender": "F",
            "anchor_age": 52,
            "structured__diagnoses_icd__icd_code": "5715|I10",
            "structured__diagnoses_icd__row_count": 8,
            "radiology__Diagnosis__cirrhosis__status": "confirmed",
            "discharge__LabResult__bilirubin__value": 3.2,
            "clinical_text": "The patient has cirrhosis and ascites with elevated bilirubin.",
            "constant_col": "same",
            "all_missing": None,
        },
        {
            "subject_id": "10000764",
            "hadm_id": "27897940",
            "gender": "M",
            "anchor_age": 61,
            "structured__diagnoses_icd__icd_code": "I10",
            "structured__diagnoses_icd__row_count": 3,
            "radiology__Diagnosis__cirrhosis__status": "",
            "discharge__LabResult__bilirubin__value": 0.8,
            "clinical_text": "No liver diagnosis is stated in this short summary.",
            "constant_col": "same",
            "all_missing": None,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_step3_public_tools_accept_reason_argument():
    from autonomous_pipeline.steps.step3.tools import TOOL_FUNCTIONS

    for func in TOOL_FUNCTIONS:
        signature = inspect.signature(func)
        assert "reason" in signature.parameters, func.__name__
        assert signature.parameters["reason"].default == ""


def test_step3_old_step4_style_column_selection_pipeline(tmp_path, monkeypatch):
    from autonomous_pipeline.steps.step3 import logic, tools

    llm_stages: list[str] = []

    def fake_json_llm(prompt: str, temperature: float = 0.0):
        if "Analyze this medical data task" in prompt:
            llm_stages.append("task_spec")
            return {
                "task_type": "cohort_selection",
                "need_modalities": ["Identifiers/Keys", "Demographics", "Diagnosis", "Labs", "ClinicalNotes"],
                "index_event": "肝病诊断",
                "time_window": "",
                "uncertainty": 0.2,
            }
        if "Generate task-specific medical recall terms" in prompt:
            llm_stages.append("term_profile")
            return {
                "task_domain": "肝病诊断",
                "trigger_terms": ["liver", "hepatic", "cirrhosis", "bilirubin", "icd"],
                "column_terms": ["liver", "cirrhosis", "bilirubin", "icd"],
                "direct_diagnosis_terms": ["cirrhosis", "icd"],
                "supportive_evidence_terms": ["bilirubin"],
                "background_disease_terms": [],
                "imaging_finding_terms": [],
                "lab_marker_terms": ["bilirubin"],
                "treatment_context_terms": [],
                "short_terms": ["icd"],
                "negative_terms": [],
                "reason": "fake llm term profile",
            }
        if "You are selecting columns" in prompt:
            llm_stages.append("column_selector")
            return {
                "must_have_columns": ["subject_id", "hadm_id", "structured__diagnoses_icd__icd_code"],
                "useful_columns": ["radiology__Diagnosis__cirrhosis__status", "discharge__LabResult__bilirubin__value"],
                "maybe_columns": ["clinical_text", "constant_col"],
                "drop_columns": [],
                "reason_by_column": {
                    "structured__diagnoses_icd__icd_code": "ICD diagnosis source",
                    "discharge__LabResult__bilirubin__value": "liver evidence lab",
                },
                "warnings": [],
            }
        raise AssertionError(f"unexpected fake LLM prompt: {prompt[:120]}")

    monkeypatch.setattr(logic, "_call_json_llm", fake_json_llm)

    input_csv = _write_sample_step2_output(tmp_path / "step2_results" / "next_input" / "input.csv")
    output_root = str(tmp_path / "program" / "output")

    resolved = tools.resolve_step3_input(str(input_csv), output_root, reason="定位 Step2 交接宽表")
    assert resolved["status"] == "SUCCESS"
    assert resolved["artifacts"]["input_csv_path"] == str(input_csv)

    task = tools.resolve_step3_task_text("肝病诊断", output_root, reason="确认任务目标")
    assert task["status"] == "SUCCESS"

    profile = tools.profile_schema(str(input_csv), output_root, max_rows=1000, reason="生成 schema profile")
    assert profile["status"] == "SUCCESS"
    schema_path = profile["artifacts"]["schema_profile_path"]
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    assert schema["dataset_info"]["total_rows"] == 2
    assert schema["columns"]["structured__diagnoses_icd__icd_code"]["inferred_modality"] == "Diagnosis"
    assert schema["columns"]["discharge__LabResult__bilirubin__value"]["inferred_modality"] == "Labs"

    task_spec = tools.plan_task_spec("肝病诊断", output_root, reason="调用 LLM 规划任务")
    assert task_spec["status"] == "SUCCESS"
    assert "Diagnosis" in task_spec["artifacts"]["task_spec"]["need_modalities"]

    terms = tools.plan_medical_terms("肝病诊断", output_root, reason="调用 LLM 规划术语")
    assert terms["status"] == "SUCCESS"
    term_profile = terms["artifacts"]["term_profile"]
    assert "liver" in term_profile["trigger_terms"]
    assert "bilirubin" in term_profile["supportive_evidence_terms"]

    candidates = tools.build_candidate_columns(
        schema_path,
        task_spec["artifacts"]["task_spec_path"],
        terms["artifacts"]["term_profile_path"],
        output_root,
        reason="结合 profile 和术语构建候选列",
    )
    assert candidates["status"] == "SUCCESS"
    candidate_cols = candidates["artifacts"]["candidate_columns"]
    assert "subject_id" in candidate_cols
    assert "hadm_id" in candidate_cols
    assert "structured__diagnoses_icd__icd_code" in candidate_cols
    assert "discharge__LabResult__bilirubin__value" in candidate_cols
    assert "all_missing" not in candidate_cols

    selected = tools.select_columns(
        schema_path,
        task_spec["artifacts"]["task_spec_path"],
        terms["artifacts"]["term_profile_path"],
        candidates["artifacts"]["candidate_columns_path"],
        output_root,
        reason="调用 LLM 选择列",
    )
    assert selected["status"] == "SUCCESS"

    gated = tools.apply_hard_gates(
        selected["artifacts"]["raw_selection_path"],
        schema_path,
        output_root,
        reason="移除无效列并生成最终报告",
    )
    assert gated["status"] == "SUCCESS"
    final_columns = gated["artifacts"]["final_columns"]
    assert "subject_id" in final_columns
    assert "hadm_id" in final_columns
    assert "structured__diagnoses_icd__icd_code" in final_columns
    assert "discharge__LabResult__bilirubin__value" in final_columns
    assert "constant_col" not in final_columns
    assert "all_missing" not in final_columns

    exported = tools.export_filtered_table(str(input_csv), gated["artifacts"]["selection_report_path"], output_root, reason="导出 Step3 标准交接产物")
    assert exported["status"] == "SUCCESS"
    assert Path(exported["artifacts"]["next_input_csv"]).is_file()
    assert Path(exported["artifacts"]["next_selection_report"]).is_file()

    validated = tools.validate_step3_output(
        exported["artifacts"]["filtered_csv"],
        exported["artifacts"]["selection_report_path"],
        exported["artifacts"]["next_input_csv"],
        exported["artifacts"]["next_selection_report"],
        reason="校验 Step3 输出",
    )
    assert validated["status"] == "SUCCESS"
    trace_path = Path(output_root) / "step3_results" / "_meta" / "tool_trace.json"
    assert trace_path.is_file()
    trace_tools = [item["tool"] for item in json.loads(trace_path.read_text(encoding="utf-8"))]
    assert "plan_task_spec" in trace_tools
    assert "apply_hard_gates" in trace_tools
    assert llm_stages == ["task_spec", "term_profile", "column_selector"]


def test_step3_llm_is_required_for_task_cropping(tmp_path, monkeypatch):
    from autonomous_pipeline.steps.step3 import logic, tools

    monkeypatch.setattr(logic, "_call_json_llm", lambda prompt, temperature=0.0: None)
    result = tools.plan_task_spec("肝病诊断", str(tmp_path / "out"), reason="任务裁剪必须调用 LLM")
    assert result["status"] == "NEEDS_REPAIR"
    assert "必须调用 LLM" in result["issues"][0]
