import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentscope.agent import ReActAgent
from agentscope.formatter import OllamaChatFormatter, OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OllamaChatModel, OpenAIChatModel
from agentscope.tool import Toolkit

from memory_agent.config import config
from memory_agent.core import parse_execution_trace_to_model
from memory_agent.memory_tool import (
    MEMORY_BANK_PATH,
    curator_apply_reflection,
    curator_grow_and_refine,
    get_playbook_context_data,
    reflector_analyze,
    set_current_reflection_payload,
    set_current_trace_payload,
    update_strategy_count,
)


def _format_message_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    continue
                elif block_type == "tool_result":
                    parts.append(str(block.get("output", "")))
                elif block_type == "tool_use":
                    parts.append(f"[tool_use] {block.get('name')}")
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _has_visible_answer_content(content) -> bool:
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                if str(block).strip():
                    return True
                continue

            block_type = block.get("type")
            if block_type == "text" and str(block.get("text", "")).strip():
                return True
            if block_type == "tool_result" and str(block.get("output", "")).strip():
                return True
        return False
    return bool(str(content).strip())


def _strip_thinking_from_msg(msg: Msg) -> Msg:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return msg
    content_blocks = [
        block for block in msg.get_content_blocks() if str(block.get("type") or "") != "thinking"
    ]
    sanitized_msg = Msg(
        name=msg.name,
        content=content_blocks,
        role=msg.role,
        metadata=msg.metadata,
        timestamp=msg.timestamp,
        invocation_id=msg.invocation_id,
    )
    sanitized_msg.id = msg.id
    return sanitized_msg


def register_no_thinking_print_hook(agent: ReActAgent) -> ReActAgent:
    def _hook(_agent: ReActAgent, kwargs: dict):
        msg = kwargs.get("msg")
        if isinstance(msg, Msg):
            kwargs["msg"] = _strip_thinking_from_msg(msg)
        return kwargs

    agent.register_instance_hook("pre_print", "strip_thinking_for_console", _hook)
    return agent


def _parse_tool_result_output(output) -> dict | None:
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, list):
        rendered = _format_message_content(output)
    else:
        rendered = str(output or "").strip()
    if not rendered:
        return None
    try:
        parsed = json.loads(rendered)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _collect_tool_results(agent: ReActAgent) -> dict[str, list[dict]]:
    collected: dict[str, list[dict]] = {}
    if not hasattr(agent.memory, "get_memory"):
        return collected

    memory_msgs = await agent.memory.get_memory()
    for msg in memory_msgs:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_name = str(block.get("name") or "").strip()
            parsed = _parse_tool_result_output(block.get("output"))
            if tool_name and parsed is not None:
                collected.setdefault(tool_name, []).append(parsed)
    return collected


def _reflector_used_fallback(reflector_result: dict) -> bool:
    analysis = str(reflector_result.get("analysis") or "")
    error_identification = str(reflector_result.get("error_identification") or "")
    return (
        "已使用确定性安全规则" in analysis
        or "模型未返回可解析 JSON" in error_identification
        or "未通过规则校验" in error_identification
    )


async def _build_deterministic_memory_summary(agent: ReActAgent) -> str | None:
    tool_results = await _collect_tool_results(agent)
    reflector_result = (tool_results.get("reflector_analyze") or [None])[-1]
    curator_result = (tool_results.get("curator_apply_reflection") or [None])[-1]
    refine_result = (tool_results.get("curator_grow_and_refine") or [None])[-1]

    if not all([reflector_result, curator_result, refine_result]):
        return None

    helpful_updates = 0
    harmful_updates = 0
    for update_result in tool_results.get("update_strategy_count", []):
        success_count = int(update_result.get("success_count", 0) or 0)
        if bool(update_result.get("is_helpful")):
            helpful_updates += success_count
        else:
            harmful_updates += success_count

    new_bullets = int(curator_result.get("new_bullets", 0) or 0)
    skipped_duplicates = int(curator_result.get("skipped_duplicates", 0) or 0)
    removed_count = int(refine_result.get("removed_count", 0) or 0)
    merged_count = int(refine_result.get("merged_count", 0) or 0)

    reflection_status = "否（已使用安全 fallback）" if _reflector_used_fallback(reflector_result) else "是"
    summary_lines = [
        "### 记忆闭环完成总结",
        "",
        f"- **反思结构化成功**：{reflection_status}",
        f"- **helpful 更新条数**：{helpful_updates} 条",
        f"- **harmful 更新条数**：{harmful_updates} 条",
        f"- **新增策略条数**：{new_bullets} 条",
        f"- **去重跳过条数**：{skipped_duplicates} 条",
        f"- **清理低效策略**：{removed_count} 条",
        f"- **合并重复策略**：{merged_count} 组",
    ]

    reflection_source = str(curator_result.get("reflection_source") or "").strip()
    reflection_quality = str(curator_result.get("reflection_quality") or "").strip()
    if reflection_source:
        summary_lines.append(f"- **reflection 来源**：`{reflection_source}`")
    if reflection_quality:
        summary_lines.append(f"- **reflection 质量**：`{reflection_quality}`")

    if new_bullets == 0 and skipped_duplicates > 0:
        summary_lines.append("")
        summary_lines.append("本轮未新增策略；Curator 试图应用的规则与现有策略重复，已被去重跳过。")

    return "\n".join(summary_lines)


_REQUIRED_MEMORY_TOOLS = {
    "reflector_analyze",
    "curator_apply_reflection",
    "curator_grow_and_refine",
}


def _get_missing_required_tools(called_tools: set[str]) -> list[str]:
    return sorted(_REQUIRED_MEMORY_TOOLS - called_tools)


def _memory_generate_kwargs() -> dict:
    return {
        "temperature": config.LLM_TEMPERATURE,
        "seed": config.LLM_SEED,
    }


def _create_memory_model_and_formatter():
    api_key = os.environ.get("OPENAI_API_KEY") or config.LLM_API_KEY or None
    base_url = os.environ.get("OPENAI_API_BASE") or config.LLM_BASE_URL
    if api_key or base_url:
        model = OpenAIChatModel(
            model_name=config.LLM_MODEL,
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url or "https://api.openai.com/v1"},
            generate_kwargs=_memory_generate_kwargs(),
        )
        return model, OpenAIChatFormatter()

    model = OllamaChatModel(
        model_name=config.LLM_MODEL,
        options={
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
        },
    )
    return model, OllamaChatFormatter()


async def _run_memory_agent_with_required_first_tool(
    agent: ReActAgent,
    msg: Msg,
) -> Msg:
    await agent.memory.add(msg)
    await agent._retrieve_from_long_term_memory(msg)
    await agent._retrieve_from_knowledge(msg)

    has_used_tool = False
    called_tools: set[str] = set()

    for _ in range(agent.max_iters):
        msg_reasoning = await agent._reasoning()
        tool_calls = msg_reasoning.get_content_blocks("tool_use")

        if tool_calls:
            has_used_tool = True
            called_tools.update(
                str(tool_call.get("name", "")).strip()
                for tool_call in tool_calls
                if str(tool_call.get("name", "")).strip()
            )
            if agent.parallel_tool_calls:
                import asyncio

                await asyncio.gather(*(agent._acting(tool_call) for tool_call in tool_calls))
            else:
                for tool_call in tool_calls:
                    await agent._acting(tool_call)
            continue

        if (
            has_used_tool
            and not _get_missing_required_tools(called_tools)
            and _has_visible_answer_content(msg_reasoning.content)
        ):
            return msg_reasoning

        missing_tools = _get_missing_required_tools(called_tools)
        missing_msg = (
            f"还没完成记忆闭环；至少还缺少这些工具调用：{', '.join(missing_tools)}。"
            if missing_tools
            else "继续完成记忆闭环；如果还没完成，就继续调用你判断所需的工具，不要只输出思考。"
        )
        hint_msg = Msg(
            "user",
            f"<system-hint>{missing_msg}</system-hint>",
            "user",
        )
        await agent._reasoning_hint_msgs.add(hint_msg)
        if agent.print_hint_msg:
            await agent.print(hint_msg)

    reply_msg = await agent._summarizing()
    await agent.memory.add(reply_msg)
    return reply_msg


def _create_memory_agent() -> ReActAgent:
    toolkit = Toolkit()
    toolkit.register_tool_function(reflector_analyze)
    toolkit.register_tool_function(update_strategy_count)
    toolkit.register_tool_function(curator_apply_reflection)
    toolkit.register_tool_function(curator_grow_and_refine)

    model, formatter = _create_memory_model_and_formatter()

    sys_prompt = f"""你是一个全局反思与记忆智能体 (Memory Supervisor)，负责 ACE 架构中的策略本 (Playbook) 维护。
你全程使用中文。

### 核心任务：
当你接收到一段执行痕迹 (Trace) 时，当前结构化 trace 已经在系统外部预加载为 `CURRENT_TRACE`。
你会看到一份简要摘要，必要时再参考原始调试文本。

你必须自主决定工具调用，但推荐闭环顺序为：
1. 先调用 `reflector_analyze(max_refinement_rounds=3)`；如果你确实需要调试覆写，才显式传 `trace_json`。
   成功后，当前结构化 reflection 会自动预加载为 `CURRENT_REFLECTION`。
2. 如果反思结果里有 `helpful_bullet_ids`，调用 `update_strategy_count(bullet_ids, is_helpful=True)`。
3. 如果反思结果里有 `harmful_bullet_ids`，调用 `update_strategy_count(bullet_ids, is_helpful=False)`。
4. 优先直接调用 `curator_apply_reflection()`；默认会自动读取 `CURRENT_REFLECTION`。只有在调试兼容旧调用时，才显式传 `reflection_json`。
5. 最后无论是否新增策略，都必须调用一次 `curator_grow_and_refine()` 完成本轮收尾。

### 工具调用规范：
- 不要自己手工重建、压缩或改写 trace_json；优先直接使用已经预加载的 `CURRENT_TRACE`。
- 不要自己手工拼接 `reflection_json`；优先直接使用已经预加载的 `CURRENT_REFLECTION`。
- `RAW_TRACE_TEXT` 只作补充证据，不能代替结构化 trace。
- 不要自己手写新增 bullet 内容，新增必须通过 `curator_apply_reflection` 完成。
- 不要调用不存在的工具。

### 最终总结：
最终总结只能基于工具返回结果生成，明确说明：
- 反思是否结构化成功
- helpful/harmful 分别更新了多少条
- 新增了多少条策略
- 去重跳过了多少条
- 清理了多少条低效策略
- 合并了多少组重复策略

如果中间某一步失败或拿不到结构化结果，要直接说明失败点，不能伪造“已新增策略”。

当前记忆库路径：{MEMORY_BANK_PATH}
"""

    agent = ReActAgent(
        name="MemorySupervisor",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        max_iters=10,
    )
    return register_no_thinking_print_hook(agent)


async def get_pipeline_context(query_text: str = "") -> tuple[str, list[str]]:
    return get_playbook_context_data(query_text=query_text)


async def report_pipeline_result(trace_payload: dict[str, object] | str, used_bullet_ids: list[str] | None = None) -> str:
    if isinstance(trace_payload, str):
        legacy = parse_execution_trace_to_model(trace_payload)
        payload = {
            "input_text": legacy.input_text,
            "duration_seconds": legacy.duration,
            "orchestrator_summary": legacy.output_text,
            "retrieved_bullet_ids": list(used_bullet_ids or legacy.used_bullet_ids),
            "steps": [],
            "raw_trace_text": trace_payload,
            "success": legacy.success,
            "failure_signals": legacy.failure_signals,
        }
    else:
        payload = dict(trace_payload or {})
        if used_bullet_ids is not None and not payload.get("retrieved_bullet_ids"):
            payload["retrieved_bullet_ids"] = list(used_bullet_ids)

    structured_trace = json.dumps(payload, ensure_ascii=False, indent=2)
    raw_trace_text = str(payload.get("raw_trace_text") or "")
    set_current_reflection_payload(None)
    set_current_trace_payload(payload)

    print(f"\n[Orchestrator] 正在将执行痕迹发送给 Memory Supervisor 进行反思... (Structured Trace 长度: {len(structured_trace)} 字符)")
    agent = _create_memory_agent()

    steps = payload.get("steps") or []
    step_summaries = []
    for idx, step in enumerate(steps[:5], 1):
        if not isinstance(step, dict):
            continue
        tool_names = [
            str(event.get("tool_name") or "")
            for event in (step.get("tool_events") or [])
            if isinstance(event, dict) and str(event.get("tool_name") or "")
        ]
        step_summaries.append(
            f"- Step {idx}: {step.get('step_name', '')} | tools={tool_names or ['无']} | output={str(step.get('final_output') or '')[:200]}"
        )
    summary_text = "\n".join(step_summaries) if step_summaries else "- 无步骤摘要"

    msg = Msg(
        name="orchestrator",
        role="user",
        content=(
            "本次流水线执行完成。当前结构化 trace 已预加载为 CURRENT_TRACE，请优先直接调用工具，不要自己重建 trace_json。\n\n"
            f"输入: {payload.get('input_text', '')}\n"
            f"耗时: {payload.get('duration_seconds', 0.0)}s\n"
            f"上游总结: {payload.get('orchestrator_summary', '')}\n"
            f"retrieved_bullet_ids: {payload.get('retrieved_bullet_ids', [])}\n"
            "步骤摘要:\n"
            f"{summary_text}\n\n"
            "如需调试，以下是原始痕迹摘要（默认不要用它替代 CURRENT_TRACE）：\n"
            f"{raw_trace_text[:3000]}\n\n"
            "请严格按照系统提示完成反思闭环。"
        ),
    )

    try:
        result_msg = await _run_memory_agent_with_required_first_tool(agent, msg)
        deterministic_summary = await _build_deterministic_memory_summary(agent)
        if deterministic_summary:
            return deterministic_summary
        return _format_message_content(result_msg.content)
    except Exception as exc:
        return f"记忆闭环执行失败：{type(exc).__name__}: {exc}"
    finally:
        set_current_reflection_payload(None)
