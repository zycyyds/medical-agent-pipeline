---
name: discharge_Diagnosis_sepsis_status
description: 仅清洗 `discharge_Diagnosis_sepsis_status` 列
---
1. 仅在目标列存在时处理，统一大小写、首尾空格和多余空格，轻量规范化常见空值表示。
2. 将明确可判定的同义写法标准化为 `confirmed`、`suspected`、`history of`；不确定的原值保留不变。