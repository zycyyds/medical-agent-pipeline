---
name: radiology_Diagnosis_pulmonary_hypertension_status
description: 仅清洗 `radiology_Diagnosis_pulmonary_hypertension_status` 列
---
1. 统一大小写、首尾空格与多余空白，清理常见分隔符格式。
2. 规范肺动脉高压相关同义表达，保留 suspected、possible、history of、no evidence of 等不确定或否定状态。
3. 无法确定时保留原始语义，不做二值化、不删除非确诊状态信息。