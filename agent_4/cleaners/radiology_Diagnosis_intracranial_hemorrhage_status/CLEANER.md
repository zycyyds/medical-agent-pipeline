---
name: radiology_Diagnosis_intracranial_hemorrhage_status
description: 仅清洗 `radiology_Diagnosis_intracranial_hemorrhage_status` 列
---

1. 仅对 `radiology_Diagnosis_intracranial_hemorrhage_status` 做轻量标准化：去首尾空格、压缩多余空白、统一大小写与常见分隔符。
2. 规范常见同义词到稳定分类，如 `confirmed`、`ruled_out`、`suspected`、`unknown`；不确定映射时保留原值，不处理其他列。