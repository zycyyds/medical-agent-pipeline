---
name: discharge_Diagnosis_atrial_fibrillation_status
description: 仅清洗 `discharge_Diagnosis_atrial_fibrillation_status` 列
---

1. 仅在目标列存在时处理该列；先做去首尾空格、合并多余空白、统一常见大小写与连字符写法。
2. 将房颤状态的明显同义写法标准化为有限类别；对含义不明确或无法可靠判断的值保留原值，不修改其他列。