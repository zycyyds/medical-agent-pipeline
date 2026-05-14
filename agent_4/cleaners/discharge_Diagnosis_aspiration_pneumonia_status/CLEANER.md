---
name: discharge_Diagnosis_aspiration_pneumonia_status
description: 仅清洗 `discharge_Diagnosis_aspiration_pneumonia_status` 列
---
1. 仅在目标列存在时处理，统一空格、去除首尾空白，并做轻量大小写规范化。
2. 将常见同义表述映射为标准状态标签：`present`、`absent`、`suspected`、`ruled_out`、`history_of`、`unknown`；不确定的原值尽量保留。