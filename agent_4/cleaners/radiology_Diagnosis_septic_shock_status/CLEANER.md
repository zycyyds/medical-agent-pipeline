---
name: radiology_Diagnosis_septic_shock_status
description: 仅清洗 `radiology_Diagnosis_septic_shock_status` 列
---
1. 仅在目标列存在时清洗：统一大小写、去除首尾空格、压缩多余空白并做轻量标点规范化。
2. 标准化与脓毒性休克相关的常见同义表达；对不确定、否定、既往史等表述保留语义，不做随意二值化。