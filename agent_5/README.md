# Agent 5

Agent 5 is reserved for Step5 data cleaning and data quality repair.

Step4 task-oriented column selection lives in:

- `agent_4/medical_column_selector/`
- `step-4/`
- `agent_4/main.py`

Do not put task column selection code back under `agent_5`.

Current status:

- The standalone Agent5 entrypoint is `agent_5/main.py`.
- The data cleaning runtime is intentionally separate from Step4 column selection.
- If `agent_5/data_quality_repair.py` is not present yet, running Agent5 will fail with a clear boundary message instead of importing Step4 code.
