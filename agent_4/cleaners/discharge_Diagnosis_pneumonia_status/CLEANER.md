---
name: discharge_Diagnosis_pneumonia_status
description: 仅清洗 `discharge_Diagnosis_pneumonia_status` 列
---
1. 仅针对 `discharge_Diagnosis_pneumonia_status` 做轻量清洗：去除首尾空格、压缩多余空白、统一大小写匹配。
2. 规范常见同义表达为稳定枚举值；对无法确定或未覆盖的原值保持不变，不推断缺失/未知。