# Step7 ML Dataset Generation

独立 Step7 负责基于 Step5 的 `filtered.csv`、Step4 的 `selection_report.json`、Step6 的 `passed_patients_*.json` 生成 ML 训练数据集。

## 默认输入

- filtered CSV: `program/output/step5_results/next_input/filtered.csv`
- selection report: `program/output/step4_results/next_input/selection_report.json`
- passed patients: `program/output/step6_results/passed_patients_20260519_183132.json`
- task text: 来自主链路交互输入；独立运行时用 `--task-text` 显式传入，或从 `selection_report.json` 的 `task_text` / `task` 字段读取
- output: `program/output/step7_results`

## 运行

```bash
/opt/anaconda3/envs/py310/bin/python step-7/run_step7_ml_pipeline.py \
  --input /Users/mkbk/PycharmProjects/new/program/output/step5_results/next_input/filtered.csv \
  --selection-report /Users/mkbk/PycharmProjects/new/program/output/step4_results/next_input/selection_report.json \
  --passed-patients-json /Users/mkbk/PycharmProjects/new/program/output/step6_results/passed_patients_20260519_183132.json \
  --task-text "死亡预测" \
  --output /Users/mkbk/PycharmProjects/new/program/output/step7_results
```

找不到或无法解析 `passed_patients_*.json`、缺少 `task_text` 时，Step7 会失败，不会回退到全量患者或生成无关任务。
