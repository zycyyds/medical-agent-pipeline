---
name: radiology_Finding_cardiomegaly_value
description: 仅清洗 `radiology_Finding_cardiomegaly_value` 列
---
1. 仅在目标列存在时处理；统一大小写、空白和常见标点/分隔符，做轻量规范化。
2. 识别心影增大主状态并归一为 `absent` / `present` / `mild` / `moderate` / `severe`。
3. 尽量保留变化修饰信息，如 `stable` / `unchanged` / `improved` / `worsened` / `new`，按“主状态 + 修饰”输出；无法确定时保留原值。