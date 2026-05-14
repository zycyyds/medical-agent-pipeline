---
name: radiology_Diagnosis_congestive_heart_failure_status
description: 仅清洗 `radiology_Diagnosis_congestive_heart_failure_status` 列
---
1. 仅对 `radiology_Diagnosis_congestive_heart_failure_status` 做轻量文本规范化：去首尾空格、压缩多余空白、统一常见大小写/拼写变体。
2. 将明确同义表达标准化为有限枚举：`confirmed`、`suspected`、`absent`、`indeterminate`；空值/缺失保持为空字符串；无法确定的原值尽量保留不改。