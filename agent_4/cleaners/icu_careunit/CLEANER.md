---
name: icu_careunit
description: 仅清洗 `icu_careunit` 列
---

1. 仅在 `icu_careunit` 存在时处理；按逗号类分隔符拆分多值，统一空格、分隔符与大小写。
2. 对常见 ICU/CCU/NICU/SICU/MICU/CVICU/PICU 等护理单元做谨慎标准化；不确定映射时保留原值，不删除括号缩写。
3. 输出仍写回 `icu_careunit` 单列，其他列、行数、索引保持不变。