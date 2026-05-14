---
name: discharge_Diagnosis_congestive_heart_failure_status
description: 仅清洗 `discharge_Diagnosis_congestive_heart_failure_status` 列
---

1. 仅在目标列存在时清洗该列，统一大小写、空白、常见分隔符与拼写变体。
2. 将值标准化为 `confirmed`、`suspected`、`history_of` 等稳定枚举；不确定时保留原值，不影响其他列。