from __future__ import annotations

import time
from typing import Any

from autonomous_pipeline.contracts import failed, ok, repair

from . import logic


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    if output_root:
        payload.setdefault("artifacts", {})["tool_trace_path"] = logic.append_tool_trace(output_root, tool, args, payload, started)
    return payload


def resolve_step_input(round_root: str, step: str, original_input: str, reason: str = "") -> dict:
    """根据当前轮目录和目标 Step 推导默认 handoff 输入路径。

    Args:
        round_root: 当前 Planner round 目录。
        step: 目标子 Agent，取值 step1 到 step6。
        original_input: 用户提供的完整原始数据路径，Step1 使用该路径。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.resolve_step_input(round_root, step, original_input)
        return _trace(round_root, "resolve_step_input", locals(), ok("Step 输入路径已推导。", artifacts), started)
    except Exception as exc:
        return _trace(round_root, "resolve_step_input", locals(), failed("Step 输入路径推导失败。", [str(exc)], {"step": step}), started)


def run_child_step(step: str, input_path: str, output_root: str, task_text: str = "", task_spec_path: str = "", retries: int = 2, reason: str = "") -> dict:
    """通过 run-step 终端入口调用一个子 Agent，保留子 Agent 自主工具编排和终端日志。

    Args:
        step: 要执行的子 Agent，取值 step1 到 step6。
        input_path: 当前子 Agent 的输入路径。
        output_root: 当前 round 目录；子 Agent 产物写入该目录下的 stepX_results。
        task_text: 用户任务目标。
        task_spec_path: TaskAnalysisAgent 生成的 task_spec.json 路径。
        retries: 子 Agent transient LLM 连接错误重试次数。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        result = logic.run_child_step(step, input_path, output_root, task_text=task_text, task_spec_path=task_spec_path, retries=retries)
        status = ok(f"{step} 子 Agent 执行完成。", result) if result["returncode"] == 0 and result["expected_output_exists"] else repair(f"{step} 子 Agent 执行后缺少期望产物。", [f"returncode={result['returncode']}", f"expected_output={result['expected_output']}"], result)
        return _trace(output_root, "run_child_step", locals(), status, started)
    except Exception as exc:
        return _trace(output_root, "run_child_step", locals(), failed(f"{step} 子 Agent 执行失败。", [str(exc)], {"step": step, "input_path": input_path}), started)


def inspect_round_artifacts(round_root: str, reason: str = "") -> dict:
    """检查当前 round 是否已生成 Step1-Step6 标准 handoff 产物和 patient_cases.jsonl。

    Args:
        round_root: 当前 Planner round 目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    artifacts = logic.inspect_round_artifacts(round_root)
    payload = ok("当前轮产物完整。", artifacts) if artifacts["complete"] else repair("当前轮产物不完整。", artifacts["missing"], artifacts)
    return _trace(round_root, "inspect_round_artifacts", locals(), payload, started)


def write_round_summary(round_root: str, planner_response: str = "", feedback_path: str = "", reason: str = "") -> dict:
    """写出当前 round summary，供 LoopRunner 和 Evaluator 使用。

    Args:
        round_root: 当前 Planner round 目录。
        planner_response: Planner 对本轮执行的简要总结。
        feedback_path: 上一轮 public_feedback.json 路径；第一轮可为空。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.write_round_summary(round_root, planner_response=planner_response, feedback_path=feedback_path)
        return _trace(round_root, "write_round_summary", locals(), ok("Round summary 已写出。", artifacts), started)
    except Exception as exc:
        return _trace(round_root, "write_round_summary", locals(), failed("Round summary 写出失败。", [str(exc)], {"round_root": round_root}), started)


TOOL_FUNCTIONS = [resolve_step_input, run_child_step, inspect_round_artifacts, write_round_summary]
