# Medical Task-Driven Column Selector

A sophisticated medical data preprocessing module that uses AgentScope framework to intelligently select relevant columns from medical datasets based on task descriptions. The module implements a multi-agent workflow to analyze medical research tasks and filter columns appropriately while considering privacy requirements.

## Features

- **Multi-Agent Workflow**: Uses TaskPlannerAgent and ColumnSelectorAgent with AgentScope framework
- **Privacy Protection**: Implements strict hard gates for PHI (Protected Health Information) protection
- **Smart Column Selection**: Analyzes task requirements and data schema to select relevant columns
- **LLM-Driven Selection**: By default, the column selector LLM sees the full eligible schema instead of a heuristic subset
- **Configurable Privacy Modes**: Supports both "strict" and "research" privacy modes
- **Detailed Reporting**: Generates comprehensive audit reports of selection decisions
- **Data Quality Checks**: Removes constant and all-missing columns automatically

## Architecture

The module implements a 5-step workflow:

1. **Schema Profiler**: Analyzes CSV structure and generates column profiles with data types, missing rates, examples, and statistical measures
2. **Task Planner Agent**: Interprets the medical task text and generates structured specifications with required data modalities
3. **Column Selector Agent**: Matches task requirements with data schema to categorize columns into must-have/useful/maybe/drop
4. **Hard Gates**: Applies privacy and quality rules to enforce column removal regardless of agent decisions
5. **Export & Report**: Generates filtered CSV and detailed selection report

## Modalities Supported

The system recognizes and categorizes columns into these medical data modalities:
- Identifiers/Keys: Patient IDs, MRNs, etc.
- Time: Dates, timestamps, durations
- Demographics: Age, gender, race, etc.
- Encounter: Visits, admissions, providers
- Diagnosis: ICD codes, conditions
- Medication: Drugs, dosages, orders
- Labs: Test results, values
- Vitals: Blood pressure, heart rate, etc.
- Procedures: Interventions, surgeries
- Outcomes: Mortality, readmissions
- ClinicalNotes: Text notes, summaries

## Usage

### Basic Usage

```python
from medical_column_selector import TaskDrivenColumnSelector

# Initialize with configuration
config = {
    "privacy_mode": "strict",        # "strict" or "research"
    "keep_free_text": False,         # Whether to preserve free text columns
    "max_rows_profile": 1000,        # Max rows to profile
    "max_final_columns": 100,        # Max columns in output
    "require_llm": True              # Fail fast if the LLM client cannot be initialized or called
}

selector = TaskDrivenColumnSelector(config=config)

# Run the selection process
result = selector.run(
    input_csv_path="your_medical_data.csv",
    task_text="Identify patients with Type 2 diabetes and analyze factors associated with glucose control."
)

# Access results
filtered_csv_path = result["filtered_csv_path"]
selection_report_path = result["selection_report_path"]
```

### Configuration Options

- `privacy_mode`:
  - `"strict"`: Removes all identified PHI columns regardless of relevance
  - `"research"`: Less aggressive PHI removal for research contexts
  - Task-supporting dataset keys such as `patient_id`, `subject_id`, `visit_id`, and `encounter_id` are treated as operational identifiers by default so they can be retained for joins and traceability
- `keep_free_text`: Preserve free text columns (may contain PHI)
- `max_rows_profile`: Limit rows used for schema profiling
- `max_final_columns`: Maximum number of columns in output
- `sampling_method`: How to sample data ("random", "head", "tail")
- `random_seed`: Seed for reproducible random sampling
- `require_llm`:
  - `True` (default): the workflow must successfully initialize and call the configured LLM
  - `False`: allow fallback template logic for debugging only
- `selector_input_mode`:
  - `"full_schema"` (default): pass the full eligible schema to the column-selection LLM so task-based screening is decided by the model
  - `"heuristic_prefilter"`: keep the older heuristic candidate reduction before the selector LLM

## Installation

1. Install AgentScope framework:
```bash
pip install agentscope
```

2. Install other dependencies:
```bash
pip install pandas numpy
```

3. Set up your preferred LLM provider configuration in config.json

## Example

See `examples/example_usage.py` for a complete working example with sample medical data.

## Security & Privacy

The module implements multiple layers of privacy protection:

1. **Automatic PHI Detection**: Identifies columns that likely contain PHI based on naming patterns
   - Operational file metadata and common dataset linkage keys are not treated as PHI by default unless they look like direct identifiers such as MRN, SSN, or insurance/account numbers
2. **Hard Gates**: Enforces removal of PHI columns in strict mode regardless of agent decisions
3. **Data Anonymization**: Redacts example values in schema profiles for PHI columns
4. **Audit Trail**: Complete logging of all column selection decisions and privacy overrides

## Output Files

The system generates two main output files:

1. **Filtered CSV**: Contains only the selected columns relevant to the task
2. **Selection Report (JSON)**: Comprehensive audit log including:
   - Original task text
   - Generated task specification
   - Schema profile of original data
   - Agent selection decisions
   - Hard gate overrides
   - Final column list
   - Warnings and considerations

## Customization

To extend functionality:

1. Modify the agent prompts in the agent classes to change selection behavior
2. Adjust the hard gates logic in the HardGates component
3. Enhance the schema profiler to recognize additional data patterns
4. Add new modalities to expand the range of recognizable medical data types

## Error Handling

The system includes robust error handling:

- JSON validation for agent outputs
- Re-try mechanisms for failed agent calls (via AgentScope)
- Graceful handling of missing or malformed data
- Detailed error reporting in the selection report
