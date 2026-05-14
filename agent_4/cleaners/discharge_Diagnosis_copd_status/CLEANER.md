---
name: discharge_Diagnosis_copd_status
description: 仅清洗 `discharge_Diagnosis_copd_status` 列

---
1. 仅在存在 `discharge_Diagnosis_copd_status` 列时处理该列，保留其他列、行数与索引不变。
2. 对该列做轻量文本规范化：去除首尾空格、压缩多余空白、统一常见 COPD 状态同义表达；缺失值保持缺失，不确定的原值仅做轻量空白清理后保留。