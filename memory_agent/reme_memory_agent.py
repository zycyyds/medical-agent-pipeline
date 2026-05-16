"""
RemeMemoryAgent 独立交互入口。

用法：
    cd <项目根目录>
    python memory_agent/reme_memory_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OllamaChatFormatter, OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OllamaChatModel, OpenAIChatModel
from agentscope.tool import Toolkit

from memory_agent.config import config
from memory_agent.reme_memory_tool import (
    STEP1_SCRIPT_CASES_PATH,
    build_step1_input_profile,
    get_reme_memory_status,
    initialize_reme_memory_store,
    list_step1_script_cases,
    record_step1_script_case,
    retrieve_step1_script_case,
)


def _create_reme_model_and_formatter():
    api_key = os.environ.get("OPENAI_API_KEY") or config.LLM_API_KEY or None
    base_url = os.environ.get("OPENAI_API_BASE") or config.LLM_BASE_URL
    if api_key or base_url:
        model = OpenAIChatModel(
            model_name=config.get_llm_model(),
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url or "https://api.openai.com/v1"},
            generate_kwargs={
                "temperature": config.LLM_TEMPERATURE,
                "seed": config.LLM_SEED,
            },
        )
        return model, OpenAIChatFormatter()

    model = OllamaChatModel(
        model_name=config.get_llm_model(),
        options={
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
        },
    )
    return model, OllamaChatFormatter()


async def main():
    agentscope.init(project="RemeMemoryAgent", name="RemeMemoryAgentStandalone")

    model, formatter = _create_reme_model_and_formatter()

    toolkit = Toolkit()
    toolkit.register_tool_function(initialize_reme_memory_store)
    toolkit.register_tool_function(build_step1_input_profile)
    toolkit.register_tool_function(record_step1_script_case)
    toolkit.register_tool_function(retrieve_step1_script_case)
    toolkit.register_tool_function(list_step1_script_cases)
    toolkit.register_tool_function(get_reme_memory_status)

    sys_prompt = f"""你是 RemeMemoryAgent，负责管理可复用产物记忆。
并且全程用中文回复用户。

你的职责：
1. 按用户定义的 schema 保存 Step1 脚本案例元数据
2. 根据输入 profile 检索相似历史案例
3. 列出案例并解释候选案例为什么相似
4. 报告当前记忆后端状态
5. 首次使用时可调用 initialize_reme_memory_store 初始化 JSONL fallback 存储

严格限制：
- 不执行脚本
- 不决定是否跳过 Step1
- 不修改 main_orchestrator
- 不写入 memory_agent/memory_bank.json
- 不把完整脚本正文保存进 JSONL，只保存路径、hash、大小、执行摘要和输入 profile

当前 Step1 案例库路径：{STEP1_SCRIPT_CASES_PATH}
"""

    agent = ReActAgent(
        name="RemeMemoryAgent",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        max_iters=8,
    )

    user = UserAgent(name="User")

    msg = Msg(name="system", content="请输入 RemeMemoryAgent 指令。", role="system")
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
