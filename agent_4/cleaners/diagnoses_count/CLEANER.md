---
name: diagnoses_count
description: 仅清洗 `diagnoses_count` 列
---

1. 仅在 `diagnoses_count` 列存在时处理，保留其他列、行数与索引不变
2. 对 `diagnoses_count` 做轻量数值格式标准化：去除首尾空白、统一全角数字/符号、对明确的千分位数值去掉分隔逗号；不确定内容保留原值