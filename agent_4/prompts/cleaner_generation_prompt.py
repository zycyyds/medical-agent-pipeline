def cleaner_generation_prompt(column_name: str, auto_desc: str, cleaner_name: str, column_profile: dict) -> str:
    return f"""你是一个结构化 CSV 数据清洗专家，请为字段 `{column_name}` 生成一个 AgentScope cleaner。

只生成一个单列 cleaner，只能处理 `{column_name}`，不能修改其他列。

列画像:
{column_profile}

分析摘要:
{auto_desc}

严格按下面格式输出，不要加解释：

===CLEANER_MD===
这里放 YAML+Markdown 描述
===END_CLEANER_MD===

===IMPLEMENTATION_PY===
这里放完整 Python 代码
===END_IMPLEMENTATION_PY===

cleaner_md 模板:
---
name: {cleaner_name}
description: 仅清洗 `{column_name}` 列

---
1. 简短步骤
2. 简短步骤

implementation_py 要求:
- 导入只能使用 `pandas`、`os`、`agentscope.tool`、`re`
- 必须定义 `def data_cleaning(input_path: str, index=False) -> ToolResponse:`
- 读取 CSV 时使用 `pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])`
- 只在列存在时处理 `{column_name}`
- 保持其他列、行数、索引不变
- 优先做轻量规范化，不确定就保留原值
- identifier/code 列不要删重、不要填缺失、不要重编号、不要强制转 int
- 保存到 `input_path.replace('.csv', '_cleaned.csv')`
- 成功时只能返回 `ToolResponse(content="...", metadata={{"output_file": output_path}})`
- 失败时只能返回 `ToolResponse(content="...")`
- 不要使用 `ToolResponse(success=...)`、`ToolResponse(status=...)`、`message=` 这类参数
"""
