---
name: radiology_Diagnosis_pneumonia_status
description: 仅清洗 `radiology_Diagnosis_pneumonia_status` 列
---

1. 仅在目标列存在时处理该列，统一首尾空格、连续空格与常见分隔符的轻量格式。
2. 将明确同义值规范为有限类别（如 `confirmed`、`suspected`、`ruled_out`、`indeterminate`、`unknown`），不确定的值仅做轻量去空格后保留原意。