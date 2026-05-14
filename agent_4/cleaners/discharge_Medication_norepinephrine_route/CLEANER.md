---
name: discharge_Medication_norepinephrine_route
description: 仅清洗 `discharge_Medication_norepinephrine_route` 列

---
1. 仅在目标列存在时进行轻量标准化：去首尾空白、压缩多余空格、统一常见缩写大小写与分隔符写法。
2. 对明确等价的静脉给药表达做安全映射，如 `IV`、`IV drip`、`IV gtt` 统一为 `IV`；其余不确定值保留原义。