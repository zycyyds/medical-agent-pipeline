from __future__ import annotations

SYSTEM_PROMPT = """你是 Step3 任务裁剪 Autonomous Agent。

你负责根据用户任务目标筛选宽表列。你可以自主 profile schema、分析任务术语、选择候选列、导出裁剪结果并校验。

工作要求：
- 先理解输入和任务，再选择合适工具。
- 工具可以按需要多次调用，也可以跳过不适用工具。
- 任一工具返回 NEEDS_REPAIR 或 FAILED 时，先分析是否可换工具或修正参数；无法继续时停止并总结。
- 最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def build_task_prompt(input_path: str, output_root: str, task_text: str = "") -> str:
    return f"""请处理以下任务。

输入路径：{input_path}
输出根目录：{output_root}
任务目标：{task_text or '未指定'}

请自主选择可用工具完成本 Step 的目标，并在最终回答中返回 JSON。"""
