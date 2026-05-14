---
name: discharge_Diagnosis_esrd_status
description: 仅清洗 `discharge_Diagnosis_esrd_status` 列

---
1. 仅在目标列存在时处理：去除首尾空白、合并多余空白，保留空值为空字符串。
2. 对明确的 ESRD 状态文本做轻量标准化映射（如 `yes/esrd`→`ESRD`，`no/non-esrd`→`No ESRD`，`unknown/unk`→`Unknown`）；无法确定的原值保留不变。