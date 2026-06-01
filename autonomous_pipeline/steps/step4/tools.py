from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from autonomous_pipeline.contracts import failed, ok, repair

from . import logic


def _step_output_root(output_root: str | None) -> str | None:
    if not output_root:
        return None
    path = Path(output_root).expanduser()
    if path.name == "step4_results":
        return str(path)
    return str(path / "step4_results")


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    step_root = _step_output_root(output_root)
    if step_root:
        trace_path = logic.append_tool_trace(step_root, tool, args, payload, started)
        payload.setdefault("artifacts", {})["tool_trace_path"] = trace_path
    return payload


def _infer_output_root_from_step4_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if "step4_results" not in parts:
        return None
    idx = parts.index("step4_results")
    return str(Path(*parts[:idx])) if idx else None


def resolve_step4_input(input_path: str, output_root: str | None = None, reason: str = "") -> dict:
    """解析 Step4 输入路径，定位 Step3 输出的 filtered.csv。

    Args:
        input_path: Step3 输出的 next_input/filtered.csv，或包含 filtered.csv/input.csv 的目录。
        output_root: 新工作区输出根目录；提供时会写入 Step4 tool_trace。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.resolve_step4_input(input_path)
        return _trace(output_root, "resolve_step4_input", locals(), ok("Step4 输入路径已解析。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "resolve_step4_input", locals(), repair("Step4 输入路径解析失败。", [str(exc)], {"input_path": input_path}), started)


def profile_step4_table(input_csv_path: str, output_root: str, workers: int = 4, reason: str = "") -> dict:
    """并行统计待清洗表格的列画像，包含缺失、唯一值、长度、文本比例和安全清洗机会。

    Args:
        input_csv_path: Step3 输出的 filtered.csv。
        output_root: 新工作区输出根目录。
        workers: 并行 profile worker 数，默认沿用旧逻辑 4。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.profile_step4_table(input_csv_path, output_root=_step_output_root(output_root), workers=workers, emit=_emit)
        return _trace(output_root, "profile_step4_table", locals(), ok("Step4 表格 profile 完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "profile_step4_table", locals(), failed("Step4 表格 profile 失败。", [str(exc)], {"input_csv_path": input_csv_path}), started)


def classify_step4_column_risks(input_csv_path: str, profile_report_path: str, output_root: str, workers: int = 4, reason: str = "") -> dict:
    """并行按旧 Step5 规则将列分为 low、medium、high 风险。

    Args:
        input_csv_path: Step3 输出的 filtered.csv。
        profile_report_path: profile_step4_table 生成的 column_profile JSON。
        output_root: 新工作区输出根目录。
        workers: 并行风险分类 worker 数，默认沿用旧逻辑 4。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.classify_step4_column_risks(input_csv_path, profile_report_path, output_root=_step_output_root(output_root), workers=workers, emit=_emit)
        return _trace(output_root, "classify_step4_column_risks", locals(), ok("Step4 列风险分类完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "classify_step4_column_risks", locals(), failed("Step4 列风险分类失败。", [str(exc)], {"profile_report_path": profile_report_path}), started)


def run_step4_data_cleaning(input_csv_path: str, column_risk_report_path: str, output_root: str, workers: int = 4, llm_workers: int = 32, enable_llm: bool = True, reason: str = "") -> dict:
    """执行旧 Step5 数据清洗：low 列报告，medium/high 列生成并验证单列 cleaner，失败则 noop 回退。

    Args:
        input_csv_path: Step3 输出的 filtered.csv。
        column_risk_report_path: classify_step4_column_risks 生成的风险报告。
        output_root: 新工作区输出根目录。
        workers: 普通 worker 数，默认沿用旧逻辑 4。
        llm_workers: LLM cleaner 并发数，默认沿用旧逻辑 32。
        enable_llm: 是否启用 LLM cleaner，默认 True；关闭时 high/medium 列只写 noop 产物。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.run_step4_data_cleaning(
            input_csv_path,
            column_risk_report_path,
            output_root=_step_output_root(output_root),
            workers=workers,
            llm_workers=llm_workers,
            enable_llm=enable_llm,
            emit=_emit,
        )
        return _trace(output_root, "run_step4_data_cleaning", locals(), ok("Step4 数据清洗完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "run_step4_data_cleaning", locals(), repair("Step4 数据清洗失败。", [str(exc)], {"input_csv_path": input_csv_path}), started)


def build_step4_patient_cases(cleaned_csv_path: str, output_root: str, task_text: str = "", reason: str = "") -> dict:
    """从清洗后的宽表生成患者级病例 JSONL，供 Planner Evaluator 与人工 JSONL 做内容语义评估。

    Args:
        cleaned_csv_path: run_step4_data_cleaning 生成的 cleaned CSV。
        output_root: 新工作区输出根目录。
        task_text: 用户任务目标，例如 肝病诊断。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.build_patient_cases(cleaned_csv_path, output_root=_step_output_root(output_root), task_text=task_text)
        return _trace(output_root, "build_step4_patient_cases", locals(), ok("Step4 患者级病例 JSONL 已生成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "build_step4_patient_cases", locals(), repair("Step4 患者级病例 JSONL 生成失败。", [str(exc)], {"cleaned_csv_path": cleaned_csv_path}), started)


def normalize_step4_outputs(cleaned_csv_path: str, data_quality_report_path: str, column_risk_report_path: str, output_root: str, selection_report_path: str | None = None, task_text: str = "", reason: str = "") -> dict:
    """标准化 Step4 产物到 next_input，供 Step5 一致性验证继续使用。

    Args:
        cleaned_csv_path: run_step4_data_cleaning 生成的 cleaned CSV。
        data_quality_report_path: run_step4_data_cleaning 生成的数据质量报告。
        column_risk_report_path: classify_step4_column_risks 生成的风险报告。
        output_root: 新工作区输出根目录。
        selection_report_path: 可选 Step3 selection_report，会复制到 next_input。
        task_text: 用户任务目标；用于写入 patient_cases.jsonl。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.normalize_step4_outputs(
            cleaned_csv_path,
            data_quality_report_path,
            column_risk_report_path,
            output_root=_step_output_root(output_root),
            selection_report_path=selection_report_path,
            task_text=task_text,
        )
        return _trace(output_root, "normalize_step4_outputs", locals(), ok("Step4 next_input 已生成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "normalize_step4_outputs", locals(), repair("Step4 输出标准化失败。", [str(exc)], {"cleaned_csv_path": cleaned_csv_path}), started)


def validate_step4_output(input_csv_path: str, cleaned_csv_path: str, data_quality_report_path: str | None = None, column_risk_report_path: str | None = None, next_input_csv: str | None = None, next_data_quality_report: str | None = None, next_column_risk_report: str | None = None, reason: str = "") -> dict:
    """校验 Step4 清洗没有改变行列结构，并确认关键产物存在。

    Args:
        input_csv_path: Step4 清洗前 CSV。
        cleaned_csv_path: run_step4_data_cleaning 生成的 cleaned CSV。
        data_quality_report_path: 可选数据质量报告路径。
        column_risk_report_path: 可选列风险报告路径。
        next_input_csv: 可选 next_input/filtered.csv 路径。
        next_data_quality_report: 可选 next_input/data_quality_report.json 路径。
        next_column_risk_report: 可选 next_input/column_risk_report.json 路径。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    result = logic.validate_step4_output(
        input_csv_path,
        cleaned_csv_path,
        data_quality_report_path=data_quality_report_path,
        column_risk_report_path=column_risk_report_path,
        next_input_csv=next_input_csv,
        next_data_quality_report=next_data_quality_report,
        next_column_risk_report=next_column_risk_report,
    )
    payload = ok("Step4 输出校验通过。", result.details) if result.passed else repair("Step4 输出校验未通过。", result.issues, result.details)
    output_root = _infer_output_root_from_step4_path(cleaned_csv_path) or _infer_output_root_from_step4_path(next_input_csv)
    return _trace(output_root, "validate_step4_output", locals(), payload, started) if output_root else payload


TOOL_FUNCTIONS = [
    resolve_step4_input,
    profile_step4_table,
    classify_step4_column_risks,
    run_step4_data_cleaning,
    build_step4_patient_cases,
    normalize_step4_outputs,
    validate_step4_output,
]
