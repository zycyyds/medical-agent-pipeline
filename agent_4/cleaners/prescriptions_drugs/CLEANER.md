---
name: prescriptions_drugs
description: 仅清洗 `prescriptions_drugs` 列

---
1. 按逗号、分号、换行等清单分隔符拆分 `prescriptions_drugs`，清理首尾空白、统一常见分隔与大小写格式，不对普通句子做分词。
2. 对少量高置信常见药名缩写/别名做轻量规范化，保留剂型/浓度/剂量等附加信息；删除明显仅为稀释液/容器液体的条目，并在规范化后按药项去重。