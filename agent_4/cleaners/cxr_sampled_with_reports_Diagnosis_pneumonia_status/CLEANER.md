---
name: cxr_sampled_with_reports_Diagnosis_pneumonia_status
description: 仅清洗 `cxr_sampled_with_reports_Diagnosis_pneumonia_status` 列
---
1. 统一 `cxr_sampled_with_reports_Diagnosis_pneumonia_status` 的分隔符，按多标签拆分并去除首尾空白。
2. 轻量标准化常见肺炎状态同义表达，保留原有标签顺序与共现信息；无法确定的值尽量保留原样。