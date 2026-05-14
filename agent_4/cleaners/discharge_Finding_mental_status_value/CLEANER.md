---
name: discharge_Finding_mental_status_value
description: 仅清洗 `discharge_Finding_mental_status_value` 列
---
1. 轻量规范化文本：去除首尾空白、统一常见标点与分隔符、压缩多余空格。
2. 标准化常见精神状态缩写/同义表达：如 `A&O x3`、`AOx4`、`alert and oriented` 等映射为统一写法；对明显仅由状态标签组成的短文本提取为标准标签串，无法确定时保留原始临床细节。