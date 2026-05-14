# utils/generate_knowledge_template.py
import pandas as pd
import json
import os

def generate_knowledge_template(csv_path: str, output_path: str):
    """
    读取CSV文件，获取列名，并生成一个知识库模板JSON文件。
    模板中的建议部分留空，供人类专家填写。
    如果输出文件已存在，会先将其删除（清空）。

    Args:
        csv_path (str): 输入CSV文件的路径。
        output_path (str): 输出JSON模板文件的路径。
    """
    # 读取CSV文件
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到输入文件 {csv_path}")
        return

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    
    # 获取所有列名
    columns = df.columns.tolist()
    
    # 清空或创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 检查并删除已存在的输出文件
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🗑️ 已删除旧的知识库模板: {output_path}")

    # 创建一个字典，键为列名，值为**空字符串**
    template_dict = {}
    for col in columns:
        template_dict[col] = "" # 值设置为空字符串

    # 将字典写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已成功生成新的知识库模板: {output_path}")
    print(f"📝 请编辑该文件，为每一列在右侧的双引号内填入具体的清洗建议。")


if __name__ == "__main__":
    # 配置路径 - 根据您的结构修改
    CSV_PATH = "./data/input.csv"
    OUTPUT_PATH = "./human_knowledge.json"
    
    generate_knowledge_template(CSV_PATH, OUTPUT_PATH)