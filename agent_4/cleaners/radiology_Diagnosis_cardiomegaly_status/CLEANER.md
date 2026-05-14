---
name: radiology_Diagnosis_cardiomegaly_status
description: 仅清洗 `radiology_Diagnosis_cardiomegaly_status` 列
---
1. 仅在目标列存在时清洗：去除首尾空格、压缩多余空白、统一大小写用于匹配。
2. 将常见同义表达规范为标准类别：`present`、`absent`、`uncertain`；空值保持为空，无法确定的原值保留。