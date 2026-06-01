from __future__ import annotations

import time
from typing import Any

from autonomous_pipeline.contracts import ok, repair

from . import logic


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    if output_root:
        payload.setdefault("artifacts", {})["tool_trace_path"] = logic.append_tool_trace(output_root, tool, args, payload, started)
    return payload


def evaluate_round_outputs(round_root: str, gold_dir: str, evaluation_root: str, previous_score: float = 0.0, reason: str = "") -> dict:
    """评估一轮完整链路产物，同时检查患者级病例 JSONL 和 Step6 ML 数据集。

    Args:
        round_root: Planner round 目录，包含 step4_results 和 step6_results。
        gold_dir: 人工金标准目录。hidden 仅在本工具内读取，不返回给 Planner。
        evaluation_root: 本轮评估报告输出目录。
        previous_score: 当前保留方案上一轮分数，用于判断是否提升。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.evaluate_round_outputs(round_root, gold_dir, evaluation_root, previous_score=previous_score)
        status = ok("本轮产物评估完成。", artifacts) if artifacts["score"] >= 0.65 else repair("本轮产物需要继续修复。", artifacts["issues"], artifacts)
        return _trace(evaluation_root, "evaluate_round_outputs", locals(), status, started)
    except Exception as exc:
        payload = repair("本轮产物评估失败。", [str(exc)], {"round_root": round_root, "gold_dir": gold_dir})
        return _trace(evaluation_root, "evaluate_round_outputs", locals(), payload, started)


TOOL_FUNCTIONS = [evaluate_round_outputs]
