# Step6 Consistency Validation

Step6 runs patient-level consistency validation as a standalone module.

## Run

```bash
python step-6/run_step6_ml_pipeline.py \
  --input program/output/step5_results/next_input/filtered.csv \
  --output program/output/step6_results
```

Use `--reuse-existing` to reuse the latest existing Step6 report in the output directory.

## Outputs

`program/output/step6_results/`:

- `consistency_report_<timestamp>.txt`
- `consistency_report_<timestamp>.json`
- `passed_patients_<timestamp>.json`
