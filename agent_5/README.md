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
- Step5 uses low / medium / high column risk routing:
  - low-risk: report only.
  - medium-risk: LLM-generated column-level cleaning script, with output shape validation.
  - high-risk: LLM-generated column-level cleaning script, with output shape validation.
- Standalone example:
  ```bash
  python agent_5/main.py --input program/output/step4_results/next_input/filtered.csv
  ```
  Use `--disable-llm` to keep medium/high-risk columns unchanged without calling LLM.
