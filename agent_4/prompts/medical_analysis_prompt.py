# prompts.py

def medical_analysis_prompt(col: str, sample_vals: list, column_profile: dict, human_advice: str = ""):
    base_prompt = f"""
你是一位结构化 CSV 数据分析师。

列名: {col}
列画像: {column_profile}
样本: {sample_vals}

请输出一段简短分析，内容只包括：
1. 这列大致是什么类型
2. 应采取什么清洗策略
3. 哪些操作应该避免
尽量控制在 120 字以内。
"""

    if human_advice:
        base_prompt += f"""
人类知识:
{human_advice}
请优先参考人类知识。
"""
    else:
        base_prompt += "\n当前没有人工知识，请只基于列名、列画像和样本给出简短分析。"

    return base_prompt
