---
name: discharge_Diagnosis_altered_mental_status_status
description: 仅清洗 `discharge_Diagnosis_altered_mental_status_status` 列
---
1. 仅在目标列存在时清洗，统一大小写、首尾空格和常见分隔符
2. 轻量标准化常见诊断状态表达为 `confirmed`、`suspected`、`ruled_out`，空值保持为空，不确定值尽量保留原文