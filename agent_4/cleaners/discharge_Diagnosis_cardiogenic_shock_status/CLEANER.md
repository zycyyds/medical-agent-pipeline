---
name: discharge_Diagnosis_cardiogenic_shock_status
description: 仅清洗 `discharge_Diagnosis_cardiogenic_shock_status` 列

---
1. 仅在目标列存在时处理，统一前后空格、压缩多余空白、规范常见分隔符与大小写
2. 仅对明确同义表达做轻量映射，如 yes/no、present/absent、positive/negative、unknown 等；不确定值保留原样