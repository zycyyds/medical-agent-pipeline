---
name: radiology_Diagnosis_pulmonary_edema_status
description: 仅清洗 `radiology_Diagnosis_pulmonary_edema_status` 列

---
1. 仅处理 `radiology_Diagnosis_pulmonary_edema_status`：统一首尾空格、大小写、分隔符与常见表达格式，保留缺失。
2. 仅对高置信同义值做轻量规范化，如 `no pulmonary edema`→`absent`、`pulmonary edema present`→`present`；不合并 suspected 与 confirmed，不确定值保留原义。