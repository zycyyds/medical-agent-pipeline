from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from autonomous_pipeline.contracts import failed, ok, repair
from autonomous_pipeline.gold import validate_gold_standard

from . import logic


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    if output_root:
        payload.setdefault("artifacts", {})["tool_trace_path"] = logic.append_tool_trace(output_root, tool, args, payload, started)
    return payload


def load_gold_examples(gold_dir: str, output_root: str, reason: str = "") -> dict:
    """读取并校验人工金标准目录，确认 visible JSONL 可用于任务分析。

    Args:
        gold_dir: 人工金标准目录，必须包含 visible/*.jsonl；hidden 可选。
        output_root: Planner run 目录，用于写 gold_summary 和 tool_trace。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        gold = validate_gold_standard(gold_dir, output_root=Path(output_root), include_hidden=False)
        payload = ok("人工可见金标准已加载。", {k: v for k, v in gold.items() if k not in {"visible_cases", "hidden_cases", "mappings"}}) if gold["valid"] else repair("人工金标准不可用。", gold["issues"], {"gold_dir": str(gold_dir)})
        return _trace(output_root, "load_gold_examples", locals(), payload, started)
    except Exception as exc:
        return _trace(output_root, "load_gold_examples", locals(), failed("人工金标准读取失败。", [str(exc)], {"gold_dir": gold_dir}), started)


def sample_raw_dataset(input_path: str, output_root: str, max_files: int = 40, reason: str = "") -> dict:
    """只读抽样原始数据目录，统计文件类型、表头和少量文本片段。

    Args:
        input_path: 完整原始数据目录或文件。
        output_root: Planner run 目录，用于写 raw_sample 和 tool_trace。
        max_files: 最多抽样文件数，默认 40。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        sample = logic.sample_raw_dataset(input_path, output_root, max_files=max_files)
        return _trace(output_root, "sample_raw_dataset", locals(), ok("原始目录只读抽样完成。", sample), started)
    except Exception as exc:
        return _trace(output_root, "sample_raw_dataset", locals(), failed("原始目录抽样失败。", [str(exc)], {"input_path": input_path}), started)


def build_task_spec(input_path: str, gold_dir: str, output_root: str, task_text: str = "", reason: str = "") -> dict:
    """根据用户任务、可见人工病例和原始目录抽样生成 TaskSpec。

    Args:
        input_path: 完整原始数据目录或文件。
        gold_dir: 人工金标准目录，必须包含 visible/*.jsonl。
        output_root: Planner run 目录，输出 task_spec.json。
        task_text: 用户任务目标，例如 肝病诊断。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.build_task_spec(input_path, gold_dir, output_root, task_text=task_text)
        return _trace(output_root, "build_task_spec", locals(), ok("TaskSpec 已生成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "build_task_spec", locals(), failed("TaskSpec 生成失败。", [str(exc)], {"input_path": input_path, "gold_dir": gold_dir}), started)


TOOL_FUNCTIONS = [load_gold_examples, sample_raw_dataset, build_task_spec]
