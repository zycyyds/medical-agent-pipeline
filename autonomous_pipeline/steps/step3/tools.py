from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from autonomous_pipeline.contracts import failed, ok, repair

from . import logic


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    root = _pipeline_output_root(output_root)
    if root:
        trace_path = logic.append_tool_trace(root, tool, args, payload, started)
        payload.setdefault("artifacts", {})["tool_trace_path"] = trace_path
    return payload


def _pipeline_output_root(output_root: str | None) -> str | None:
    if not output_root:
        return None
    path = Path(output_root).expanduser()
    if path.name == "step3_results":
        return str(path.parent)
    return str(path)


def _infer_output_root_from_step3_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if "step3_results" not in parts:
        return None
    idx = parts.index("step3_results")
    return str(Path(*parts[:idx])) if idx else None


def resolve_step3_input(input_path: str, output_root: str | None = None, reason: str = "") -> dict:
    """解析 Step3 输入路径，定位可裁剪的宽表 CSV/XLSX。

    Args:
        input_path: Step2 输出的 next_input/input.csv，或包含 input.csv/filtered.csv 的目录。
        output_root: 新工作区输出根目录；提供时会写入 Step3 tool_trace。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.resolve_input_csv(input_path)
        return _trace(output_root, "resolve_step3_input", locals(), ok("Step3 输入路径已解析。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "resolve_step3_input", locals(), repair("Step3 输入路径解析失败。", [str(exc)], {"input_path": input_path}), started)


def resolve_step3_task_text(task_text: str, output_root: str | None = None, reason: str = "") -> dict:
    """解析 Step3 任务目标文本，确保任务裁剪有明确医学目标。

    Args:
        task_text: 用户任务目标，例如 肝病诊断、肝硬化诊断。
        output_root: 新工作区输出根目录；提供时会写入 Step3 tool_trace。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.resolve_task_text(task_text)
        return _trace(output_root, "resolve_step3_task_text", locals(), ok("Step3 任务目标已解析。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "resolve_step3_task_text", locals(), repair("Step3 任务目标解析失败。", [str(exc)], {"task_text": task_text}), started)


def profile_schema(input_csv: str, output_root: str, max_rows: int = 1000, reason: str = "") -> dict:
    """分析输入宽表 schema、缺失率、样例值、PHI 风险和列模态。

    Args:
        input_csv: Step2 输出的患者级或 admission 级宽表 CSV/XLSX。
        output_root: 新工作区输出根目录。
        max_rows: 用于 profile 的最大采样行数，默认 1000。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.profile_schema(input_csv, _pipeline_output_root(output_root) or output_root, max_rows=max_rows)
        return _trace(output_root, "profile_schema", locals(), ok("Schema profile 完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "profile_schema", locals(), failed("Schema profile 失败。", [str(exc)], {"input_csv": input_csv}), started)


def plan_task_spec(task_text: str, output_root: str, reason: str = "") -> dict:
    """将自然语言任务目标规划成任务类型、需要的数据模态和索引事件。

    Args:
        task_text: 用户任务目标，例如 肝病诊断。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.plan_task_spec(task_text, _pipeline_output_root(output_root) or output_root)
        return _trace(output_root, "plan_task_spec", locals(), ok("任务规划完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "plan_task_spec", locals(), repair("任务规划失败。", [str(exc)], {"task_text": task_text}), started)


def plan_medical_terms(task_text: str, output_root: str, reason: str = "") -> dict:
    """生成任务相关医学术语、诊断词、证据词、检查指标词和负向词。

    Args:
        task_text: 用户任务目标，例如 肝病诊断。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.plan_medical_terms(task_text, _pipeline_output_root(output_root) or output_root)
        return _trace(output_root, "plan_medical_terms", locals(), ok("医学术语规划完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "plan_medical_terms", locals(), repair("医学术语规划失败。", [str(exc)], {"task_text": task_text}), started)


def build_candidate_columns(schema_profile_path: str, task_spec_path: str, term_profile_path: str, output_root: str, max_candidates: int | None = None, reason: str = "") -> dict:
    """结合 schema、任务规划和术语规划，构建候选列集合。

    Args:
        schema_profile_path: profile_schema 生成的 schema_profile.json 路径。
        task_spec_path: plan_task_spec 生成的 task_spec.json 路径。
        term_profile_path: plan_medical_terms 生成的 term_profile.json 路径。
        output_root: 新工作区输出根目录。
        max_candidates: 最多保留候选列数；None 表示不限制。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.build_candidate_columns(schema_profile_path, task_spec_path, term_profile_path, _pipeline_output_root(output_root) or output_root, max_candidates=max_candidates)
        return _trace(output_root, "build_candidate_columns", locals(), ok("候选列构建完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "build_candidate_columns", locals(), repair("候选列构建失败。", [str(exc)], {}), started)


def select_columns(schema_profile_path: str, task_spec_path: str, term_profile_path: str, candidate_columns_path: str, output_root: str, reason: str = "") -> dict:
    """从候选列中选择 must/useful/maybe/drop 列，保留任务裁剪理由。

    Args:
        schema_profile_path: profile_schema 生成的 schema_profile.json 路径。
        task_spec_path: plan_task_spec 生成的 task_spec.json 路径。
        term_profile_path: plan_medical_terms 生成的 term_profile.json 路径。
        candidate_columns_path: build_candidate_columns 生成的 candidate_columns.json 路径。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.select_columns(schema_profile_path, task_spec_path, term_profile_path, candidate_columns_path, _pipeline_output_root(output_root) or output_root)
        return _trace(output_root, "select_columns", locals(), ok("任务裁剪列选择完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "select_columns", locals(), repair("任务裁剪列选择失败。", [str(exc)], {}), started)


def apply_hard_gates(raw_selection_path: str, schema_profile_path: str, output_root: str, keep_free_text: bool = False, max_final_columns: int | None = None, reason: str = "") -> dict:
    """应用硬规则门控，移除 PHI、全空列、无效常量列，并生成最终 selection_report。

    Args:
        raw_selection_path: select_columns 生成的 raw_selection_result.json 路径。
        schema_profile_path: profile_schema 生成的 schema_profile.json 路径。
        output_root: 新工作区输出根目录。
        keep_free_text: 是否允许保留大段临床文本列；默认 False。
        max_final_columns: 最终列数上限；None 表示不限制。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.apply_hard_gates(raw_selection_path, schema_profile_path, _pipeline_output_root(output_root) or output_root, keep_free_text=keep_free_text, max_final_columns=max_final_columns)
        return _trace(output_root, "apply_hard_gates", locals(), ok("硬规则门控完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "apply_hard_gates", locals(), repair("硬规则门控失败。", [str(exc)], {}), started)


def export_filtered_table(input_csv: str, selection_report_path: str, output_root: str, reason: str = "") -> dict:
    """导出裁剪后的 filtered.csv、selection_report.json 和 next_input 标准交接目录。

    Args:
        input_csv: Step2 输出的原始宽表 CSV/XLSX。
        selection_report_path: apply_hard_gates 生成的 selection_report.json。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.export_filtered_table(input_csv, selection_report_path, _pipeline_output_root(output_root) or output_root)
        return _trace(output_root, "export_filtered_table", locals(), ok("任务裁剪结果已导出。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "export_filtered_table", locals(), failed("任务裁剪结果导出失败。", [str(exc)], {"input_csv": input_csv}), started)


def validate_step3_output(filtered_csv_path: str, selection_report_path: str, next_input_csv: str | None = None, next_selection_report: str | None = None, reason: str = "") -> dict:
    """校验 Step3 输出是否可交给 Step4 数据清洗。

    Args:
        filtered_csv_path: export_filtered_table 返回的 filtered_csv 或 next_input_csv 路径。
        selection_report_path: export_filtered_table 返回的 selection_report_path 或 next_selection_report 路径。
        next_input_csv: 可选的 next_input/filtered.csv 路径。
        next_selection_report: 可选的 next_input/selection_report.json 路径。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    result = logic.validate_output(filtered_csv_path, selection_report_path, next_input_csv, next_selection_report)
    payload = ok("Step3 输出校验通过。", result["details"]) if result["passed"] else repair("Step3 输出校验未通过。", result["issues"], result["details"])
    output_root = _infer_output_root_from_step3_path(filtered_csv_path) or _infer_output_root_from_step3_path(selection_report_path) or _infer_output_root_from_step3_path(next_input_csv)
    return _trace(output_root, "validate_step3_output", locals(), payload, started) if output_root else payload


TOOL_FUNCTIONS = [
    resolve_step3_input,
    resolve_step3_task_text,
    profile_schema,
    plan_task_spec,
    plan_medical_terms,
    build_candidate_columns,
    select_columns,
    apply_hard_gates,
    export_filtered_table,
    validate_step3_output,
]
