from __future__ import annotations

import argparse
import asyncio
import re
import sys

from agentscope.message import Msg

from autonomous_pipeline.agent_factory import create_react_agent
from autonomous_pipeline.config import DEFAULT_OUTPUT_ROOT

from . import tools
from .prompt import SYSTEM_PROMPT, build_task_prompt

STEP_TITLE = "Step1 数据重组"


def create_agent():
    return create_react_agent("step1", SYSTEM_PROMPT, tools.TOOL_FUNCTIONS, max_iters=14)


def build_natural_language_prompt(user_text: str, output_root: str) -> str:
    return f"""你正在独立运行 Step1 数据重组 Agent。

默认输出根目录：{output_root}

用户自然语言请求：{user_text}

请从用户请求中识别要处理的输入路径。Step1 不需要疾病任务目标；如果用户提到疾病或诊断任务，只作为备注，不影响 Step1 输出结构。请自主选择工具完成 Step1，并在终端工具日志之外，最终只返回 JSON。"""


def _is_connection_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    if name in {"APIConnectionError", "RemoteProtocolError", "ConnectError", "ReadTimeout", "APITimeoutError"}:
        return True
    return "connection error" in text or "server disconnected" in text or "remote protocol" in text


def _message_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _clean_model_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


async def run_natural_language(user_text: str, output_root: str, retries: int = 2):
    agent = create_agent()
    prompt = build_natural_language_prompt(user_text, output_root)
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return await agent(Msg(name="user", role="user", content=prompt))
        except Exception as exc:
            last_error = exc
            if not _is_connection_error(exc) or attempt >= retries:
                raise
            wait_seconds = min(2 ** attempt, 8)
            print(f"[LLM Retry] 模型连接异常，{wait_seconds}s 后重试 ({attempt + 1}/{retries})：{type(exc).__name__}: {exc}", file=sys.stderr)
            await asyncio.sleep(wait_seconds)
    raise RuntimeError(f"Step1 failed before tool execution: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step1 数据重组独立 Autonomous Agent")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录，默认使用新工作区 program/output")
    parser.add_argument("--retries", type=int, default=2, help="模型连接异常重试次数")
    parser.add_argument("--once", default="", help="只执行一条自然语言请求，例如：处理 /path/to/data")
    args = parser.parse_args()

    if args.once:
        response = asyncio.run(run_natural_language(args.once, args.output_root, retries=args.retries))
        print(_clean_model_text(_message_text(response)))
        return

    print("Step1AutonomousAgent 已启动。直接输入自然语言请求，例如：处理 /Users/mkbk/PycharmProjects/new/mimic-mini。输入 exit 退出。")
    while True:
        try:
            text = input("Step1> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"exit", "quit", "q"}:
            return
        if not text:
            continue
        response = asyncio.run(run_natural_language(text, args.output_root, retries=args.retries))
        print(_clean_model_text(_message_text(response)))


__all__ = ["STEP_TITLE", "create_agent", "build_task_prompt", "build_natural_language_prompt", "run_natural_language"]


if __name__ == "__main__":
    main()
