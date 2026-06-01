from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd


def _write_step5_next_input(root: Path) -> tuple[Path, Path]:
    next_dir = root / "step5_results" / "next_input"
    next_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"subject_id": "10000032", "hadm_id": "1", "gender": "F", "anchor_age": 52, "structured__diagnoses_icd__icd_code": "5715 | 07070", "structured__diagnoses_icd__icd_version": "9", "discharge_LabResult_ast_value": 88, "admittime": "2180-01-01"},
        {"subject_id": "10000764", "hadm_id": "2", "gender": "M", "anchor_age": 61, "structured__diagnoses_icd__icd_code": "I10", "structured__diagnoses_icd__icd_version": "10", "discharge_LabResult_ast_value": 41, "admittime": "2180-01-02"},
        {"subject_id": "10000898", "hadm_id": "3", "gender": "F", "anchor_age": 70, "structured__diagnoses_icd__icd_code": "5723", "structured__diagnoses_icd__icd_version": "9", "discharge_LabResult_ast_value": 100, "admittime": "2180-01-03"},
    ]
    csv_path = next_dir / "filtered.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    passed_path = next_dir / "passed_patients.json"
    passed_path.write_text(json.dumps({"passed_patient_ids": ["10000032", "10000764", "10000898"], "passed_count": 3}, ensure_ascii=False), encoding="utf-8")
    return csv_path, passed_path


def test_step6_public_tools_accept_reason_argument():
    from autonomous_pipeline.steps.step6.tools import TOOL_FUNCTIONS

    for func in TOOL_FUNCTIONS:
        signature = inspect.signature(func)
        assert "reason" in signature.parameters, func.__name__
        assert signature.parameters["reason"].default == ""


def test_step6_icd_binary_dataset_pipeline(tmp_path):
    from autonomous_pipeline.steps.step6 import tools

    csv_path, passed_path = _write_step5_next_input(tmp_path)
    output_root = str(tmp_path / "program" / "output")

    resolved = tools.resolve_step6_inputs(str(csv_path.parent), output_root, reason="解析 Step5 next_input")
    assert resolved["status"] == "SUCCESS"
    assert resolved["artifacts"]["input_csv"] == str(csv_path.resolve())
    assert resolved["artifacts"]["passed_patients_json"] == str(passed_path.resolve())

    context = tools.load_task_context(
        resolved["artifacts"]["input_csv"],
        output_root,
        task_text="肝病诊断",
        passed_patients_json=resolved["artifacts"]["passed_patients_json"],
        reason="加载任务上下文",
    )
    assert context["status"] == "SUCCESS"
    assert "structured__diagnoses_icd__icd_code" in context["artifacts"]["icd_code_columns"]

    label = tools.create_icd_binary_label(
        resolved["artifacts"]["input_csv"],
        "structured__diagnoses_icd__icd_code",
        output_root,
        passed_patients_json=resolved["artifacts"]["passed_patients_json"],
        reason="创建肝病 ICD 二分类标签",
    )
    assert label["status"] == "SUCCESS"
    assert label["artifacts"]["positive_count"] == 2
    assert label["artifacts"]["negative_count"] == 1

    features = tools.select_feature_columns(
        label["artifacts"]["labeled_csv"],
        output_root,
        label["artifacts"]["label_column"],
        label_source_column=label["artifacts"]["label_source_column"],
        reason="自动选择特征并排除泄漏源",
    )
    assert features["status"] == "SUCCESS"
    excluded = features["artifacts"]["excluded_columns"]
    assert excluded["structured__diagnoses_icd__icd_code"] == "label_source_column"
    assert excluded["admittime"] == "time_or_date_leakage"

    dataset = tools.format_and_save_ml_dataset(
        label["artifacts"]["labeled_csv"],
        features["artifacts"]["feature_columns_json"],
        output_root,
        label["artifacts"]["label_column"],
        label["artifacts"]["label_mapping"],
        task_name="liver_disease_icd_binary",
        icd10_mapping={"liver_disease": "K76.9", "no_liver_disease": "NOT_FOUND"},
        reason="导出 ML 数据集",
    )
    assert dataset["status"] == "SUCCESS"

    validated = tools.validate_dataset(dataset["artifacts"]["dataset_csv"], dataset["artifacts"]["model_config_json"], reason="校验数据集")
    assert validated["status"] == "SUCCESS"

    df = pd.read_csv(dataset["artifacts"]["dataset_csv"])
    assert "structured__diagnoses_icd__icd_code" not in df.columns
    assert "admittime" not in df.columns
    assert "label_ICD10" in df.columns
    assert "K76.9" in set(df["label_ICD10"].dropna().astype(str))
    assert set(df["y"].astype(int)) == {0, 1}

    trace = Path(output_root) / "step6_results" / "_meta" / "tool_trace.json"
    assert trace.is_file()


def test_step6_diagnosis_field_mapping_tools(tmp_path, monkeypatch):
    from autonomous_pipeline.steps.step6 import logic, tools

    input_csv = tmp_path / "filtered.csv"
    pd.DataFrame(
        [
            {"patient_id": "p1", "discharge_Diagnosis_liver_status": "肝硬化", "feature_value": 1},
            {"patient_id": "p2", "discharge_Diagnosis_liver_status": "脂肪肝", "feature_value": 2},
        ]
    ).to_csv(input_csv, index=False, encoding="utf-8-sig")
    output_root = str(tmp_path / "program" / "output")
    monkeypatch.setattr(logic, "ICD10_USE_LLM", False)
    monkeypatch.setattr(logic, "_llm_extract_disease_content", lambda field, value: str(value).strip())
    monkeypatch.setattr(logic, "_local_icd10_lookup", lambda name: ("FAKE", "K74.600" if name == "肝硬化" else "K76.000", name))

    fields = tools.discover_diagnosis_fields(str(input_csv), output_root, reason="识别诊断字段")
    assert fields["status"] == "SUCCESS"
    assert "discharge_Diagnosis_liver_status" in fields["artifacts"]["diagnosis_fields"]

    mapped = tools.extract_and_map_all_diagnosis_fields(
        str(input_csv),
        fields["artifacts"]["diagnosis_fields"],
        output_root,
        reason="映射 ICD10",
    )
    assert mapped["status"] == "SUCCESS"
    assert mapped["artifacts"]["icd10_mapping"]["肝硬化"] == "K74.600"
