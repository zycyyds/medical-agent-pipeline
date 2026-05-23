from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, ToolResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
AGENTS_DIR = ORCHESTRATOR_DIR / "agents"
CORE_DIR = ORCHESTRATOR_DIR / "core"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, AGENTS_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runtime import (
    collect_tool_results,
    create_openai_model_and_formatter,
    has_model_credentials,
    json_tool_response,
    make_user_msg,
    parse_agent_text_json,
)
from contracts import TaskSpec, TaskType
from router import route_task


TASK_ROUTER_PROMPT = """你是 TaskRouterAgent，只负责把用户输入分流成 TaskSpec。

你必须调用 route_task_tool，不要自己处理数据，不要调用任何 Step 工具。
返回时只输出 route_task_tool 返回的 JSON，不要添加解释。
"""


def _print_route_decision(spec: TaskSpec, source: str) -> None:
    print("\n" + "=" * 72)
    print(f"[TaskRouter] DECISION source={source}")
    print(f"[TaskRouter]   task_type: {spec.task_type.value}")
    print(f"[TaskRouter]   execution_scope: {spec.execution_scope or '(unset)'}")
    print(f"[TaskRouter]   start_step: {spec.start_step or '(unset)'}")
    print(f"[TaskRouter]   allowed_steps: {spec.allowed_steps or []}")
    if spec.resume_from_step:
        print(f"[TaskRouter]   resume_from_step: {spec.resume_from_step}")
    if spec.intent_summary:
        print(f"[TaskRouter]   reason: {spec.intent_summary}")
    for key in ("input_path", "records_path", "input_csv", "filtered_csv", "selection_report", "passed_patients_json", "task_text"):
        value = spec.entry_artifacts.get(key)
        if value not in (None, "", [], {}):
            print(f"[TaskRouter]   artifact.{key}: {value}")
    print("=" * 72)


def route_task_tool(user_input: str, project_root: str | None = None) -> ToolResponse:
    """Route a natural language request or artifact path to a TaskSpec JSON."""
    spec = route_task(user_input=user_input, project_root=project_root)
    return json_tool_response(spec.to_dict())


def task_spec_from_dict(payload: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_type=TaskType(str(payload.get("task_type") or TaskType.FULL_PIPELINE.value)),
        entry_artifacts=dict(payload.get("entry_artifacts") or {}),
        resume_from_step=str(payload.get("resume_from_step") or ""),
        intent_summary=str(payload.get("intent_summary") or ""),
        confidence=float(payload.get("confidence", 1.0) or 0.0),
        execution_scope=str(payload.get("execution_scope") or ""),
        start_step=str(payload.get("start_step") or ""),
        allowed_steps=[str(item) for item in payload.get("allowed_steps") or []],
    )


def _complete_with_deterministic_route(spec: TaskSpec, user_input: str, project_root: str | Path) -> TaskSpec:
    deterministic = route_task(user_input, project_root=project_root)
    if not spec.entry_artifacts and deterministic.entry_artifacts:
        spec.entry_artifacts = dict(deterministic.entry_artifacts)
    elif deterministic.entry_artifacts:
        for key, value in deterministic.entry_artifacts.items():
            current = spec.entry_artifacts.get(key)
            if not current:
                spec.entry_artifacts[key] = value
    if not spec.resume_from_step and deterministic.resume_from_step:
        spec.resume_from_step = deterministic.resume_from_step
    if not spec.execution_scope and deterministic.execution_scope:
        spec.execution_scope = deterministic.execution_scope
    if not spec.start_step and deterministic.start_step:
        spec.start_step = deterministic.start_step
    if not spec.allowed_steps and deterministic.allowed_steps:
        spec.allowed_steps = list(deterministic.allowed_steps)
    return spec


def create_task_router_agent() -> ReActAgent:
    toolkit = Toolkit()
    toolkit.register_tool_function(route_task_tool)
    model, formatter = create_openai_model_and_formatter("main_orchestrator", "gpt-4.1-mini")
    agent = ReActAgent(
        name="TaskRouterAgent",
        sys_prompt=TASK_ROUTER_PROMPT,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=4,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


async def route_task_with_agent(user_input: str, project_root: str | Path) -> TaskSpec:
    """Route through TaskRouterAgent when model credentials exist, otherwise use deterministic fallback."""
    if not has_model_credentials("main_orchestrator"):
        spec = route_task(user_input, project_root=project_root)
        spec.intent_summary = f"{spec.intent_summary}（TaskRouterAgent 未配置模型密钥，使用确定性 route_task 兜底）"
        spec.confidence = min(spec.confidence, 0.9)
        _print_route_decision(spec, source="deterministic_fallback_no_credentials")
        return spec

    agent = create_task_router_agent()
    payload = {
        "user_input": user_input,
        "project_root": str(Path(project_root).resolve()),
        "required_output": "TaskSpec JSON",
    }
    msg = make_user_msg("user", json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        res = await agent(msg)
        tool_results = await collect_tool_results(agent)
        routed = (tool_results.get("route_task_tool") or [None])[-1]
        if routed:
            spec = _complete_with_deterministic_route(task_spec_from_dict(routed), user_input, project_root)
            _print_route_decision(spec, source="llm_tool_route_task_tool")
            return spec
        parsed = parse_agent_text_json(res.get_text_content() or "")
        if parsed:
            spec = _complete_with_deterministic_route(task_spec_from_dict(parsed), user_input, project_root)
            _print_route_decision(spec, source="llm_final_json")
            return spec
    except Exception as exc:
        spec = route_task(user_input, project_root=project_root)
        spec.intent_summary = f"{spec.intent_summary}（TaskRouterAgent 调用失败，使用确定性 route_task 兜底: {exc}）"
        spec.confidence = min(spec.confidence, 0.7)
        _print_route_decision(spec, source="deterministic_fallback_exception")
        return spec

    spec = route_task(user_input, project_root=project_root)
    spec.intent_summary = f"{spec.intent_summary}（TaskRouterAgent 未返回可解析 TaskSpec，使用确定性 route_task 兜底）"
    spec.confidence = min(spec.confidence, 0.7)
    _print_route_decision(spec, source="deterministic_fallback_unparsed")
    return spec
