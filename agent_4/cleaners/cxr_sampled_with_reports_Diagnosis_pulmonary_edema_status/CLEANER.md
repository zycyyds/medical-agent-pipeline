---
name: cxr_sampled_with_reports_Diagnosis_pulmonary_edema_status
description: 仅清洗 `cxr_sampled_with_reports_Diagnosis_pulmonary_edema_status` 列
---
1. 仅在目标列存在时进行轻量标准化，统一大小写、空白、常见分隔符与同义表达。
2. 标准化为 `confirmed`、`suspected`、`ruled out` 及其复合多值形式，保留无法确定的原值，不修改其他列。