---
name: discharge_Diagnosis_obesity_status
description: 仅清洗 `discharge_Diagnosis_obesity_status` 列

---
1. 仅在列存在时清洗 `discharge_Diagnosis_obesity_status`，保留其他列、行数与索引不变。
2. 对该列做轻量文本规范化：去除首尾空格、压缩多余空格、统一大小写，并仅对明确同义表达做小范围标准化；不确定时保留原词义。