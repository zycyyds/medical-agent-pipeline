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
    if path.name == "step5_results":
        return str(path)
    if path.name.startswith("step5"):
        return str(path.parent / "step5_results")
    return str(path / "step5_results")


def _pipeline_output_root(output_root: str | None) -> str | None:
    if not output_root:
        return None
    path = Path(output_root).expanduser()
    return str(path.parent if path.name.startswith("step5") else path)


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    step_root = _step_output_root(output_root)
    if step_root:
        trace_path = logic.append_tool_trace(step_root, tool, args, payload, started)
        payload.setdefault("artifacts", {})["tool_trace_path"] = trace_path
    return payload


def _infer_output_root_from_step5_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if "step5_results" not in parts:
        return None
    idx = parts.index("step5_results")
    return str(Path(*parts[:idx])) if idx else None


def resolve_step5_input(input_path: str, output_root: str | None = None, reason: str = "") -> dict:
    """解析 Step5 输入路径，定位 Step4 输出的 filtered.csv。

    Args:
        input_path: Step4 输出的 next_input/filtered.csv，或包含 filtered.csv/input.csv 的目录。
        output_root: 新工作区输出根目录；提供时会写入 Step5 tool_trace。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.resolve_step5_input(input_path)
        return _trace(output_root, "resolve_step5_input", locals(), ok("Step5 输入路径已解析。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "resolve_step5_input", locals(), repair("Step5 输入路径解析失败。", [str(exc)], {"input_path": input_path}), started)


def load_data_overview(input_csv: str, output_root: str, patient_id_col: str | None = None, reason: str = "") -> dict:
    """读取数据概况，识别患者 ID 列、患者数、记录数分布和需跳过的长文本列。

    Args:
        input_csv: Step4 清洗后的 filtered.csv。
        output_root: 新工作区输出根目录。
        patient_id_col: 可选，显式指定患者 ID 列；为空时自动识别。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.load_data_overview(input_csv, output_root=_pipeline_output_root(output_root) or output_root, patient_id_col=patient_id_col)
        return _trace(output_root, "load_data_overview", locals(), ok("Step5 数据概况加载完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "load_data_overview", locals(), failed("Step5 数据概况加载失败。", [str(exc)], {"input_csv": input_csv}), started)


def analyze_all_patients_consistency(input_csv: str, output_root: str, patient_id_col: str | None = None, threshold: float = 0.70, reason: str = "") -> dict:
    """按旧 Step6 规则批量计算患者级一致性分数和通过名单。

    Args:
        input_csv: Step4 清洗后的 filtered.csv。
        output_root: 新工作区输出根目录。
        patient_id_col: 可选，显式指定患者 ID 列；为空时自动识别并规范为 patient_id。
        threshold: 通过阈值，默认沿用旧 Step6 的 0.70。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.analyze_all_patients_consistency(
            input_csv,
            output_root=_pipeline_output_root(output_root) or output_root,
            patient_id_col=patient_id_col,
            threshold=threshold,
            emit=_emit,
        )
        return _trace(output_root, "analyze_all_patients_consistency", locals(), ok("Step5 患者一致性分析完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "analyze_all_patients_consistency", locals(), failed("Step5 患者一致性分析失败。", [str(exc)], {"input_csv": input_csv}), started)


def save_step5_report(input_csv: str, analysis_json_path: str, output_root: str, report_content: str | None = None, allow_auto_report: bool = False, reason: str = "") -> dict:
    """保存 Step5 一致性报告、通过患者名单，并标准化 next_input 供 Step6 使用。

    Args:
        input_csv: Step4 清洗后的 filtered.csv。
        analysis_json_path: analyze_all_patients_consistency 生成的 patient_consistency JSON。
        output_root: 新工作区输出根目录。
        report_content: Agent 撰写的中文摘要报告，必须包含 PASSED_PATIENTS 标记。
        allow_auto_report: 仅测试/降级时可设为 True；正式 autonomous 流程必须由 Agent 提供 report_content。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.save_step5_report(
            input_csv,
            analysis_json_path,
            output_root=_pipeline_output_root(output_root) or output_root,
            report_content=report_content,
            allow_auto_report=allow_auto_report,
        )
        return _trace(output_root, "save_step5_report", locals(), ok("Step5 一致性报告已保存。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "save_step5_report", locals(), repair("Step5 报告保存失败。", [str(exc)], {"analysis_json_path": analysis_json_path}), started)


def validate_step5_output(json_report_path: str, passed_patients_json: str, next_input_dir: str | None = None, reason: str = "") -> dict:
    """校验 Step5 报告和通过患者名单，并确认 next_input 可交给 Step6。

    Args:
        json_report_path: save_step5_report 生成的 consistency_report JSON。
        passed_patients_json: save_step5_report 生成的 passed_patients JSON。
        next_input_dir: 可选，Step5 next_input 目录；提供时检查 filtered.csv、passed_patients.json、consistency_report.json。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    result = logic.validate_step5_output(json_report_path, passed_patients_json, next_input_dir=next_input_dir)
    payload = ok("Step5 输出校验通过。", result) if result["passed"] else repair("Step5 输出校验未通过。", result["issues"], result)
    output_root = _infer_output_root_from_step5_path(json_report_path) or _infer_output_root_from_step5_path(passed_patients_json) or _infer_output_root_from_step5_path(next_input_dir)
    return _trace(output_root, "validate_step5_output", locals(), payload, started) if output_root else payload


TOOL_FUNCTIONS = [
    resolve_step5_input,
    load_data_overview,
    analyze_all_patients_consistency,
    save_step5_report,
    validate_step5_output,
]
