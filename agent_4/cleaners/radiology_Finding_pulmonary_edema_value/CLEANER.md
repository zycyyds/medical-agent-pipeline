---
name: radiology_Finding_pulmonary_edema_value
description: 仅清洗 `radiology_Finding_pulmonary_edema_value` 列
---

1. 仅对 `radiology_Finding_pulmonary_edema_value` 做轻量文本规范化：统一空白、常见分隔符、大小写与少量同义词，不改动其他列。
2. 从原始描述中保守提取 `status`（present/absent/uncertain）、`severity`（mild/moderate/severe）、`trend`（improved/worsened/stable）标签；无法确定时保留原意并标记为 `unknown`。
3. 输出为单列标准化字符串，保留原文：`status=...; severity=...; trend=...; text=...`；空值仅做基础清理，不强行填充或数值化。