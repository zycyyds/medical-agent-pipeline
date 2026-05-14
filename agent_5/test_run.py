#!/usr/bin/env python
"""
Test script to run the Medical Column Selector
"""
import pandas as pd
import os
import tempfile
import json
from medical_column_selector import TaskDrivenColumnSelector
from medical_column_selector.components.schema_profiler import SchemaProfiler
from medical_column_selector.components.hard_gates import HardGates
from medical_column_selector.agents.column_selector import ColumnSelectorAgent

def create_sample_data():
    """Create sample medical data for testing."""
    sample_data = {
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "filename": ["case_001.csv", "case_002.csv", "case_003.csv", "case_004.csv", "case_005.csv"],
        "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],  # PHI
        "last_name": ["Doe", "Smith", "Johnson", "Brown", "Wilson"],  # PHI
        "dob": ["1980-01-15", "1990-05-20", "1975-11-10", "1985-03-25", "1992-07-30"],  # PHI
        "gender": ["M", "F", "M", "F", "M"],
        "admission_date": ["2023-01-10", "2023-01-15", "2023-01-20", "2023-01-25", "2023-02-01"],
        "discharge_date": ["2023-01-15", "2023-01-20", "2023-01-25", "2023-01-30", "2023-02-05"],
        "diagnosis_code": ["J45.909", "E11.9", "I10", "J44.1", "E10.9"],
        "diagnosis_description": ["Asthma", "Type 2 DM", "Hypertension", "COPD", "Type 1 DM"],
        "medication": ["Albuterol", "Metformin", "Lisinopril", "Tiotropium", "Insulin"],
        "lab_glucose": [180, 220, 160, 190, 250],
        "lab_hba1c": [7.2, 8.1, 6.8, 7.5, 9.0],
        "vital_bp_sys": [140, 135, 150, 145, 155],
        "vital_bp_dia": [90, 85, 95, 90, 100],
        "procedure_count": [2, 1, 3, 1, 2],
        "outcome_days": [5, 5, 5, 5, 4],
        "free_text_notes": ["Patient responded well", "Required follow-up", "Stable condition", "Improved significantly", "Monitoring needed"]
    }
    return sample_data

def test_identifier_handling_component_level():
    """Regression check for task-supporting identifiers."""
    print("Running identifier retention regression check...")
    sample_df = pd.DataFrame(create_sample_data())

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_csv:
        sample_df.to_csv(temp_csv.name, index=False)
        temp_csv_path = temp_csv.name

    try:
        profiler = SchemaProfiler(max_rows=100)
        schema_profile = profiler.analyze_csv(temp_csv_path)

        assert schema_profile["columns"]["patient_id"]["is_potential_phi"] is False
        assert schema_profile["columns"]["filename"]["is_potential_phi"] is False

        agent = ColumnSelectorAgent(model_name="dashscope/qwen-max")
        task_spec = {
            "task_type": "cohort_selection",
            "need_modalities": ["Demographics", "Labs"],
            "privacy_mode": "strict",
        }
        selection = json.loads(agent._create_mock_response(task_spec, {"columns": schema_profile["columns"]}))

        assert "patient_id" in selection["useful_columns"]
        assert "filename" in selection["drop_columns"]
        print("Identifier retention regression check passed.")
    finally:
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)

def test_final_column_order_preserved():
    """Final exported columns should follow the original CSV order."""
    print("Running final column order regression check...")
    schema_profile = {
        "columns": {
            "col_a": {"is_potential_phi": False, "is_constant": False, "missing_rate": 0.0, "dtype_guess": "float64", "inferred_modality": "Labs"},
            "col_b": {"is_potential_phi": False, "is_constant": False, "missing_rate": 0.0, "dtype_guess": "object", "inferred_modality": "Identifiers/Keys"},
            "col_c": {"is_potential_phi": False, "is_constant": False, "missing_rate": 0.0, "dtype_guess": "float64", "inferred_modality": "Labs"},
            "col_d": {"is_potential_phi": False, "is_constant": False, "missing_rate": 0.0, "dtype_guess": "float64", "inferred_modality": "Labs"},
        }
    }
    selection_result = {
        "must_have_columns": ["col_d", "col_a"],
        "useful_columns": ["col_b"],
        "maybe_columns": ["col_c"],
        "drop_columns": [],
        "reason_by_column": {},
        "warnings": [],
        "transform_suggestions": {},
        "task_spec": {"task_type": "cohort_selection", "need_modalities": ["Labs"]},
    }

    result = HardGates(privacy_mode="strict", keep_free_text=True, max_final_columns=10).apply_gates(
        selection_result=selection_result,
        schema_profile=schema_profile,
    )

    assert result["final_columns"] == ["col_a", "col_b", "col_c", "col_d"]
    print("Final column order regression check passed.")

def test_column_selector():
    """Test the column selector functionality."""
    print("Creating sample data...")
    sample_data = create_sample_data()
    sample_df = pd.DataFrame(sample_data)

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_csv:
        sample_df.to_csv(temp_csv.name, index=False)
        temp_csv_path = temp_csv.name

    try:
        print(f"Created temporary CSV: {temp_csv_path}")

        # Define the task
        task_text = "Identify patients with Type 2 diabetes and analyze glucose control factors"
        print(f"Task: {task_text}")

        # Initialize selector with configuration
        config = {
            "privacy_mode": "strict",  # Should remove PHI columns
            "keep_free_text": False,   # Should handle free text appropriately
            "max_rows_profile": 100,
            "max_final_columns": 50,
            "require_llm": False,
        }

        print("Initializing TaskDrivenColumnSelector...")
        selector = TaskDrivenColumnSelector(config=config)

        print("Running column selection...")
        # Run the selection process
        result_paths = selector.run(
            input_csv_path=temp_csv_path,
            task_text=task_text
        )

        print("\nSUCCESS! Results:")
        print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
        print(f"Selection Report: {result_paths['selection_report_path']}")

        # Verify the filtered CSV was created and has expected columns
        filtered_df = pd.read_csv(result_paths['filtered_csv_path'])
        print(f"\nOriginal columns: {len(sample_df.columns)}")
        print(f"Filtered columns: {len(filtered_df.columns)}")
        print(f"Columns kept: {list(filtered_df.columns)}")

        # Show first few rows of filtered data
        print(f"\nFirst few rows of filtered data:")
        print(filtered_df.head())

        # Load and display selection report
        with open(result_paths['selection_report_path'], 'r') as f:
            report = json.load(f)
        print(f"\nSelection Report Summary:")
        print(f"- Final columns: {len(report['final_columns'])}")
        print(f"- Task type: {report['task_spec']['task_type']}")
        print(f"- Needed modalities: {report['task_spec']['need_modalities']}")
        print(f"- Uncertainty: {report['task_spec']['uncertainty']}")
        if report['warnings']:
            print(f"- Warnings: {len(report['warnings'])}")
            for warning in report['warnings']:
                print(f"  * {warning}")

        return result_paths

    finally:
        # Clean up temporary file
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)
        print(f"\nCleaned up temporary file: {temp_csv_path}")

def test_column_selector_normalizes_llm_column_names():
    """LLM 返回列名有轻微格式差异时，仍应能映射回 schema。"""
    print("Running column name normalization regression check...")
    agent = ColumnSelectorAgent(model_name="dashscope/qwen-max")
    schema_profile = {
        "columns": {
            "住院天数": {},
            "年龄_入院时": {},
            "死亡时间": {},
        }
    }
    llm_json = json.dumps({
        "must_have_columns": ["住院 天数", "年龄-入院时"],
        "useful_columns": ["死亡时间"],
        "maybe_columns": [],
        "drop_columns": [],
        "reason_by_column": {
            "住院 天数": "length of stay outcome",
            "年龄-入院时": "baseline demographic factor",
        },
        "warnings": [],
        "transform_suggestions": {}
    }, ensure_ascii=False)

    parsed = agent._parse_selection_result(llm_json, schema_profile)

    assert parsed["must_have_columns"] == ["住院天数", "年龄_入院时"]
    assert parsed["useful_columns"] == ["死亡时间"]
    assert parsed["reason_by_column"]["住院天数"] == "length of stay outcome"
    assert parsed["reason_by_column"]["年龄_入院时"] == "baseline demographic factor"
    print("Column name normalization regression check passed.")

if __name__ == "__main__":
    print("Testing Medical Column Selector...")
    try:
        test_identifier_handling_component_level()
        test_final_column_order_preserved()
        test_column_selector_normalizes_llm_column_names()
        test_column_selector()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
