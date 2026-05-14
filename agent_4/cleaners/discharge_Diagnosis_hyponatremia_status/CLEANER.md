---
name: discharge_Diagnosis_hyponatremia_status
description: 仅清洗 `discharge_Diagnosis_hyponatremia_status` 列
---
1. 仅在目标列存在时清洗：去除首尾空格、压缩多余空白，并做轻量文本规范化。
2. 将常见同义表达标准化为统一状态标签（如 present / absent / history_of / resolved / unknown），对不确定值尽量保留原始语义，不处理其他列。