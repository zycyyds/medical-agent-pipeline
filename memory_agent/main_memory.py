"""
Memory Agent 独立交互入口。

用法：
    python memory_agent/main_memory.py
    python memory_agent/main_memory.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OllamaChatFormatter, OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OllamaChatModel, OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

from memory_agent.config import config
from memory_agent.services.memory_service import get_default_memory_service


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemoryAgent ReMe service entrypoint.")
    parser.add_argument("--status", action="store_true", help="Print MemoryAgent/ReMe backend status and exit.")
    parser.add_argument("--json", action="store_true", help="Print command output as JSON.")
    return parser.parse_args(argv)


def _memory_generate_kwargs() -> dict[str, Any]:
    return {
        "temperature": config.LLM_TEMPERATURE,
        "seed": config.LLM_SEED,
    }


def _create_memory_model_and_formatter():
    api_key = os.environ.get("OPENAI_API_KEY") or config.LLM_API_KEY or None
    base_url = os.environ.get("OPENAI_API_BASE") or config.LLM_BASE_URL
    if api_key or base_url:
        model = OpenAIChatModel(
            model_name=config.get_llm_model(),
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url or "https://api.openai.com/v1"},
            generate_kwargs=_memory_generate_kwargs(),
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


async def memory_status_tool() -> ToolResponse:
    """查看当前 MemoryAgent / ReMe 后端状态。"""

    return _json_response(get_default_memory_service().status())


async def memory_search_tool(query_text: str) -> ToolResponse:
    """按任务 query 检索 ReMeLight 规则记忆和 ReMe Step/Tool 经验记忆。"""

    context, used_ids = await get_default_memory_service().get_context(query_text=query_text)
    return _json_response({"context": context, "used_memory_ids": used_ids})


async def memory_report_tool(trace_json: str) -> ToolResponse:
    """把结构化 trace 写入 ReMeLight 规则记忆和 ReMe Step/Tool 经验记忆。"""

    try:
        payload = json.loads(trace_json)
    except json.JSONDecodeError:
        payload = {"raw_trace_text": trace_json, "success": False}
    summary = await get_default_memory_service().report_result(payload)
    return _json_response({"summary": summary})


def _print_payload(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


async def main(argv: list[str] | None = None):
    args = parse_args(argv or [])
    service = get_default_memory_service()

    if args.status:
        _print_payload(service.status(), args.json)
        return

    agentscope.init(project="MemoryAgent", name="MemoryAgentStandalone")
    model, formatter = _create_memory_model_and_formatter()

    toolkit = Toolkit()
    toolkit.register_tool_function(memory_status_tool)
    toolkit.register_tool_function(memory_search_tool)
    toolkit.register_tool_function(memory_report_tool)

    sys_prompt = f"""你是 MemoryAgent，负责受控管理多智能体流水线的 ReMe 记忆。
你全程用中文回复用户。

你的权限边界：
- 你只能检索、总结、写入记忆。
- 你不能决定主 pipeline 下一步执行哪个 Step。
- 你不能修改 records.json、生成脚本或业务输出。
- 你不能保存完整医疗数据、完整 CSV、完整图片或完整脚本正文。

工具：
- memory_status_tool：查看 ReMeLight 和 ReMe Task/Tool 后端状态。
- memory_search_tool(query_text)：检索规则记忆和相似经验。
- memory_report_tool(trace_json)：把结构化执行 trace 写入记忆。

当前 ReMeLight 目录：{config.resolve_project_path(config.REME_LIGHT_ROOT)}
当前 ReMe Vector 目录：{config.resolve_project_path(config.REME_VECTOR_ROOT)}
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
    msg = Msg(name="system", content="请输入 MemoryAgent 指令。输入 exit 退出。", role="system")
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
    asyncio.run(main(sys.argv[1:]))
