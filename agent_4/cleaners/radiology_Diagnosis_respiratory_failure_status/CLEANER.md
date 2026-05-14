---
name: radiology_Diagnosis_respiratory_failure_status
description: 仅清洗 `radiology_Diagnosis_respiratory_failure_status` 列
---

1. 仅在目标列存在时处理；统一首尾空格、连续空格与常见分隔符格式。
2. 对明确同义表述做轻量标准化，映射为有限状态标签；无法确定的值仅做轻量规范化后原样保留。