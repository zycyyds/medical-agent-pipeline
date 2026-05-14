"""
Memory Agent 独立交互入口

用法：
    cd <项目根目录>
    python memory_agent/main_memory.py
"""

import asyncio

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OllamaChatFormatter
from agentscope.message import Msg
from agentscope.model import OllamaChatModel
from agentscope.tool import Toolkit

from memory_agent.config import config
from memory_agent.memory_tool import (
    MEMORY_BANK_PATH,
    curator_apply_reflection,
    curator_grow_and_refine,
    playbook_add_bullet,
    playbook_get_context,
    playbook_get_statistics,
    playbook_list_bullets,
    reflector_analyze,
    update_strategy_count,
)


async def main():
    agentscope.init(project="MemoryAgent", name="MemoryAgentStandalone")

    model = OllamaChatModel(
        model_name=config.LLM_MODEL,
        enable_thinking=config.LLM_ENABLE_THINKING,
        options={
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
        },
    )
    formatter = OllamaChatFormatter()

    toolkit = Toolkit()
    toolkit.register_tool_function(playbook_get_context)
    toolkit.register_tool_function(playbook_list_bullets)
    toolkit.register_tool_function(playbook_get_statistics)
    toolkit.register_tool_function(update_strategy_count)
    toolkit.register_tool_function(playbook_add_bullet)
    toolkit.register_tool_function(reflector_analyze)
    toolkit.register_tool_function(curator_apply_reflection)
    toolkit.register_tool_function(curator_grow_and_refine)

    sys_prompt = f"""你是一个记忆反思智能体 (Memory Agent)，负责管理 ACE 架构中的策略本 (Playbook)。
并且全程用中文回复用户。

你的职责：
1. 维护一个 ACE 风格的 Playbook，而不是单段总结
2. 分析其他 Agent 的执行轨迹，判断哪些 bullet 有帮助、哪些有害
3. 把 reflection 交给 curator 生成增量 delta，再由系统做确定性合并
4. 通过 grow-and-refine 去重、归并、清理低效策略

你可以使用的工具分为三类：

【Playbook 工具 - 策略本管理】
- playbook_get_context: 获取当前 ACE Playbook 上下文，支持按 query 检索 bullet
- playbook_list_bullets: 查看当前有效 bullets
- playbook_get_statistics: 查看当前统计
- update_strategy_count(bullet_ids, is_helpful): 批量更新策略的有帮助/有害计数
- playbook_add_bullet(content, bullet_type): 仅在明确需要手工补规则时使用

【Reflector 工具 - 反思分析】
- reflector_analyze(trace_json, max_refinement_rounds=3): 基于预结构化 trace 分析执行轨迹

【Curator 工具 - 整理优化】
- curator_apply_reflection(reflection_json, trace_json): 基于 reflection 和结构化 trace 生成 delta 并合并进 playbook
- curator_grow_and_refine(): 清理无效策略，优化策略本

当前记忆库路径：{MEMORY_BANK_PATH}
"""

    agent = ReActAgent(
        name="MemoryAgent",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        max_iters=8,
    )

    user = UserAgent(name="User")

    msg = Msg(name="system", content="请输入指令。", role="system")
    print(f"\n{agent.name}: {msg.content}")

    while True:
        try:
            try:
                msg = await user(msg)
            except KeyboardInterrupt:
                print("\n已退出。")
                break

            user_input = str(msg.content).strip()
            if user_input.lower() in ["exit", "quit", "q", "退出"]:
                print("再见！")
                break

            msg = await agent(msg)
        except EOFError:
            break


if __name__ == "__main__":
    asyncio.run(main())
