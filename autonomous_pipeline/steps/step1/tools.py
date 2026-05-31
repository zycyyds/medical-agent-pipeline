from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from autonomous_pipeline.contracts import failed, ok, repair
from . import logic


def _maybe_trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    if output_root:
        try:
            trace_path = logic.append_tool_trace(output_root, tool, args, payload, started)
            payload.setdefault("artifacts", {})["tool_trace_path"] = trace_path
        except Exception:
            pass
    return payload


def scan_source_files(input_path: str, output_root: str | None = None, reset_trace: bool = True, reason: str = "") -> dict:
    """扫描原始输入，过滤隐藏/缓存/输出文件，并返回可记录的源文件任务列表。

    Args:
        input_path: 原始医疗数据目录或单文件路径。Agent 应先调用该工具了解输入规模、suffix 分布和 patient_id 解析结果。
        output_root: 新工作区输出根目录；提供后会写入 source_tasks.json 和 Step1 tool trace。
        reset_trace: 是否把 scan 视为新一轮 Step1 开始并清空旧 tool trace，默认 True。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    if output_root and reset_trace:
        trace_path = Path(output_root) / "step1_results" / "_meta" / "tool_trace.json"
        if trace_path.exists():
            trace_path.unlink()
    try:
        result = logic.scan_source_files(input_path)
        artifacts = {
            "input_path": result["input_path"],
            "file_count": result["file_count"],
            "by_suffix": result["by_suffix"],
            "sample_tasks": result["sample_tasks"],
        }
        if output_root:
            artifacts["tasks_path"] = logic.write_source_tasks(result["tasks"], output_root)
        else:
            artifacts["tasks"] = result["tasks"]
        payload = ok("原始输入扫描完成。", artifacts)
    except Exception as exc:
        payload = failed("原始输入扫描失败。", [str(exc)], {"input_path": input_path})
    return _maybe_trace(output_root, "scan_source_files", {"input_path": input_path, "reset_trace": reset_trace}, payload, started)


def build_base_records(tasks_path: str, output_root: str, reason: str = "") -> dict:
    """根据扫描任务文件生成基础 records，包含 source_path、patient_id、relative_path、file_info 和基础 modality。

    Args:
        tasks_path: scan_source_files 返回的 artifacts.tasks_path 路径。不要把 tasks 列表直接作为参数传入。
        output_root: 新工作区输出根目录，工具会写入 step1_results/_meta/base_records.json。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    try:
        tasks = logic.load_source_tasks(tasks_path)
        payload = ok("基础 records 已生成。", logic.build_base_records(tasks, output_root))
    except Exception as exc:
        payload = failed("基础 records 生成失败。", [str(exc)], {"tasks_path": tasks_path, "output_root": output_root})
    return _maybe_trace(output_root, "build_base_records", {"tasks_path": tasks_path}, payload, started)


def infer_visual_modality(base_records_path: str, output_root: str, model_path: str | None = None, reason: str = "") -> dict:
    """使用 DocLayout-YOLO 判断图片/PDF的 modality，并生成 visual modality patch。

    Args:
        base_records_path: build_base_records 返回的 base_records.json 路径。
        output_root: 新工作区输出根目录，工具会写入 visual_modality_patches.json 和 tool trace。
        model_path: 可选的 DocLayout-YOLO 权重路径；不传时按配置候选路径自动查找。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    try:
        payload = ok("DocLayout-YOLO 视觉 modality 推断完成。", logic.infer_visual_modality(base_records_path, output_root, model_path=model_path))
    except Exception as exc:
        payload = failed("DocLayout-YOLO 视觉 modality 推断失败。", [str(exc)], {"base_records_path": base_records_path})
    return _maybe_trace(output_root, "infer_visual_modality", {"base_records_path": base_records_path, "model_path": model_path or "auto"}, payload, started)


def infer_table_split_strategy(base_records_path: str, output_root: str, reason: str = "") -> dict:
    """检测表格是否含 subject_id/patient_id/hadm_id/stay_id/id 等列，并生成拆分策略 patch。

    Args:
        base_records_path: build_base_records 返回的 base_records.json 路径。
        output_root: 新工作区输出根目录，工具会写入 table_split_patches.json 和 tool trace。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    try:
        payload = ok("表格拆分策略推断完成。", logic.infer_table_split_strategy(base_records_path, output_root))
    except Exception as exc:
        payload = failed("表格拆分策略推断失败。", [str(exc)], {"base_records_path": base_records_path})
    return _maybe_trace(output_root, "infer_table_split_strategy", {"base_records_path": base_records_path}, payload, started)


def reduce_records(base_records_path: str, visual_patches_path: str, table_split_patches_path: str, output_root: str, reason: str = "") -> dict:
    """合并基础 records、DocLayout patch 和表格拆分 patch，生成最终 records.json。

    Args:
        base_records_path: build_base_records 返回的 base_records.json 路径。
        visual_patches_path: infer_visual_modality 返回的 visual_modality_patches.json 路径。
        table_split_patches_path: infer_table_split_strategy 返回的 table_split_patches.json 路径。
        output_root: 新工作区输出根目录，工具会写入 step1_results/_meta/records.json。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    try:
        payload = ok("records reducer 合并完成。", logic.reduce_records(base_records_path, visual_patches_path, table_split_patches_path, output_root))
    except Exception as exc:
        payload = failed("records reducer 合并失败。", [str(exc)], {"base_records_path": base_records_path})
    return _maybe_trace(output_root, "reduce_records", {"base_records_path": base_records_path}, payload, started)


def validate_records(records_path: str, expected_count: int | None = None, reason: str = "") -> dict:
    """校验 records.json 是否为旧 Step1 schema 兼容格式，并检查关键字段和 modality。

    Args:
        records_path: reduce_records 返回的最终 records.json 路径。
        expected_count: 可选的期望 records 数；用于对照扫描文件数量。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    result = logic.validate_records(records_path, expected_count=expected_count)
    payload = ok("records 校验通过。", result) if result["passed"] else repair("records 校验未通过。", result["issues"], result)
    output_root = None
    p = Path(records_path)
    if p.parent.name == "_meta" and p.parent.parent.name == "step1_results":
        output_root = str(p.parent.parent.parent)
    return _maybe_trace(output_root, "validate_records", {"records_path": records_path, "expected_count": expected_count}, payload, started)


def reorganize_from_records(records_path: str, output_root: str, input_root: str | None = None, reason: str = "") -> dict:
    """按 records.json 重组文件到 patient_id/modality/source_relative_parent/filename，并按 subject_id 拆表。

    Args:
        records_path: validate_records 通过后的 records.json 路径。
        output_root: 新工作区输出根目录，工具会重建 step1_results 患者级目录。
        input_root: 可选的原始输入根目录；不传时根据 records 推断。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    try:
        result = logic.reorganize_from_records(records_path, output_root, input_root=input_root)
        payload = ok("Step1 患者级文件重组完成。", result) if result["status"] == "SUCCESS" else failed("Step1 患者级文件重组失败。", result.get("errors", []), result)
    except Exception as exc:
        payload = failed("Step1 患者级文件重组失败。", [str(exc)], {"records_path": records_path})
    return _maybe_trace(output_root, "reorganize_from_records", {"records_path": records_path, "input_root": input_root}, payload, started)


def validate_step1_output(output_root: str, expected_min_files: int = 1, reason: str = "") -> dict:
    """校验 Step1 患者级输出目录、records、tool trace 和 modality 层级是否可用。

    Args:
        output_root: 新工作区输出根目录。
        expected_min_files: 期望输出文件数下限，默认至少 1 个。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    result = logic.validate_output(output_root, expected_min_files=expected_min_files)
    payload = ok("Step1 输出校验通过。", result) if result["passed"] else repair("Step1 输出校验未通过。", result["issues"], result)
    return _maybe_trace(output_root, "validate_step1_output", {"output_root": output_root, "expected_min_files": expected_min_files}, payload, started)


def build_records(input_path: str, output_root: str, reason: str = "") -> dict:
    """兼容旧测试的便捷工具：扫描输入并一次性生成最终 records.json。

    Args:
        input_path: 原始医疗数据目录或文件路径。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的简短中文理由；仅用于终端决策日志，不参与业务逻辑。
    """
    started = time.time()
    try:
        payload = ok("records.json 已生成。", logic.build_records_compat(input_path, output_root))
    except Exception as exc:
        payload = failed("records.json 生成失败。", [str(exc)], {"input_path": input_path, "output_root": output_root})
    return _maybe_trace(output_root, "build_records", {"input_path": input_path}, payload, started)


TOOL_FUNCTIONS = [
    scan_source_files,
    build_base_records,
    infer_visual_modality,
    infer_table_split_strategy,
    reduce_records,
    validate_records,
    reorganize_from_records,
    validate_step1_output,
]
