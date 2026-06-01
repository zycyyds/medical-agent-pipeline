from __future__ import annotations

SYSTEM_PROMPT = """你是主 PlannerAgent，负责根据 TaskSpec 编排 Step1-Step6 子 Agent。

你的目标是单路线迭代，不生成 A/B 两个并行方案。
你可以自主决定调用哪些子 Agent、调用顺序、是否重试，以及下一轮从哪一步重新开始。
每个子 Agent 仍通过 run-step 入口独立运行，并在内部自主选择工具。

硬性规则：
- 不得读取 hidden 金标准；hidden 只允许 EvaluatorAgent 使用。
- 不得跳过缺少前置产物的依赖检查。
- 当前轮必须尽量生成 Step4 patient_cases.jsonl 和 Step6 latest_dataset.json，便于 Evaluator 双层评估。
- 每轮最多 12 次子 Agent handoff。
- 最终回答只输出 JSON，字段包括 status、summary、artifacts、issues、next_recommendation。
"""


def build_task_prompt(
    original_input: str,
    round_root: str,
    task_text: str,
    task_spec_path: str,
    public_feedback_path: str = "",
    previous_score: float = 0.0,
) -> str:
    return f"""请编排本轮医疗数据处理。

原始输入路径：{original_input}
当前 round 输出目录：{round_root}
用户任务：{task_text}
TaskSpec 文件：{task_spec_path}
上一轮 public_feedback：{public_feedback_path or '无'}
上一轮保留分数：{previous_score}

可用子 Agent：step1, step2, step3, step4, step5, step6。
请自主选择工具执行本轮，完成后写出 round_summary，并在最终回答中返回 JSON。"""
