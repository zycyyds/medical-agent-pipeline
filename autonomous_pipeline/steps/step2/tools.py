from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from autonomous_pipeline.contracts import failed, ok, repair

from . import logic


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    if output_root:
        trace_path = logic.append_tool_trace(output_root, tool, args, payload, started)
        payload.setdefault("artifacts", {})["tool_trace_path"] = trace_path
    return payload


def resolve_step2_input(input_path: str, output_root: str | None = None, reason: str = "") -> dict:
    """解析 Step2 输入路径，确认它是可处理的 Step1 输出目录。

    Args:
        input_path: Step1 输出目录路径，例如 program/output/step1_results。
        output_root: 新工作区输出根目录；提供时会写入 Step2 tool_trace。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.resolve_input(input_path)
        return _trace(output_root, "resolve_step2_input", locals(), ok("Step2 输入路径已解析。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "resolve_step2_input", locals(), repair("Step2 输入路径解析失败。", [str(exc)], {"input_path": input_path}), started)


def scan_step2_input(input_path: str, output_root: str | None = None, reason: str = "") -> dict:
    """扫描 Step2 输入目录，统计患者、图片、notes 表、structured 表和文本列。

    Args:
        input_path: Step1 输出目录路径。
        output_root: 新工作区输出根目录；提供时会写入 Step2 tool_trace。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.scan_input(input_path, output_root)
        return _trace(output_root, "scan_step2_input", locals(), ok("Step2 输入目录扫描完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "scan_step2_input", locals(), failed("Step2 输入目录扫描失败。", [str(exc)], {"input_path": input_path}), started)


def select_step2_pipeline(input_path: str, output_root: str | None = None, requested_mode: str = "auto", user_hint: str = "", scan_artifacts: dict[str, Any] | None = None, reason: str = "") -> dict:
    """根据扫描结果或用户提示选择 directory 或 ocr_fill 处理模式。

    Args:
        input_path: Step1 输出目录路径。
        output_root: 新工作区输出根目录；提供时会写入 Step2 tool_trace。
        requested_mode: auto、directory 或 ocr_fill。
        user_hint: 用户自然语言提示，用于判断是否显式要求 OCR/回填或目录结构化。
        scan_artifacts: scan_step2_input 返回的 artifacts，可避免重复扫描。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.select_pipeline(input_path, scan_artifacts=scan_artifacts, requested_mode=requested_mode, user_hint=user_hint)
        return _trace(output_root, "select_step2_pipeline", locals(), ok("Step2 pipeline 模式已选择。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "select_step2_pipeline", locals(), repair("Step2 pipeline 模式选择失败。", [str(exc)], {"input_path": input_path, "requested_mode": requested_mode}), started)


def collect_directory_payloads(input_path: str, output_root: str, reason: str = "") -> dict:
    """收集 directory 模式所需的患者结构化表和 notes 文本任务。

    Args:
        input_path: Step1 输出目录路径。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.collect_directory_payloads(input_path, output_root)
        return _trace(output_root, "collect_directory_payloads", locals(), ok("Directory payload 收集完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "collect_directory_payloads", locals(), failed("Directory payload 收集失败。", [str(exc)], {"input_path": input_path}), started)


def extract_entities(text_tasks_jsonl: str, output_root: str, source_kind: str = "directory", concurrency: int = 100, results_dir: str | None = None, reason: str = "") -> dict:
    """并发调用 LLM 从文本任务中抽取医学实体。

    Args:
        text_tasks_jsonl: collect_directory_payloads 或 OCR 阶段生成的文本任务 JSONL。
        output_root: 新工作区输出根目录。
        source_kind: directory 或 ocr，用于决定输出文件命名。
        concurrency: LLM 并发数，默认沿用旧逻辑 100。
        results_dir: 可选的结果目录；OCR 模式通常传 run_ocr 返回的 results_dir。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.extract_entities(text_tasks_jsonl, output_root, source_kind=source_kind, concurrency=concurrency, results_dir_path=results_dir)
        return _trace(output_root, "extract_entities", locals(), ok("医学实体抽取完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "extract_entities", locals(), repair("医学实体抽取失败。", [str(exc)], {"text_tasks_jsonl": text_tasks_jsonl}), started)


def cluster_entity_names(entities_csv: str, output_root: str, threshold: float = 0.25, reason: str = "") -> dict:
    """用 embedding 层次聚类对实体名称做语义聚类，并写出 canonical_name。

    Args:
        entities_csv: extract_entities 生成的实体 CSV。
        output_root: 新工作区输出根目录。
        threshold: 层次聚类距离阈值，默认沿用旧逻辑 0.25，约等于余弦相似度 > 0.75。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.cluster_entity_names(entities_csv, output_root, threshold=threshold)
        return _trace(output_root, "cluster_entity_names", locals(), ok("实体名称聚类完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "cluster_entity_names", locals(), repair("实体名称聚类失败。", [str(exc)], {"entities_csv": entities_csv}), started)


def build_admission_wide(patient_payloads_json: str, clustered_entities_csv: str, output_root: str, reason: str = "") -> dict:
    """合并结构化表和聚类实体，生成 admission 级宽表。

    Args:
        patient_payloads_json: collect_directory_payloads 生成的患者 payload JSON。
        clustered_entities_csv: cluster_entity_names 生成的实体 CSV。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.build_admission_wide(patient_payloads_json, clustered_entities_csv, output_root)
        return _trace(output_root, "build_admission_wide", locals(), ok("Admission 宽表生成完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "build_admission_wide", locals(), repair("Admission 宽表生成失败。", [str(exc)], {"patient_payloads_json": patient_payloads_json}), started)


def run_ocr(input_path: str, output_root: str, ocr_workers: int = 64, cache_path: str | None = None, reason: str = "") -> dict:
    """对 Step1 输出目录中的患者图片并行执行 OCR，并写出 ocr_cache.json。

    Args:
        input_path: Step1 输出目录路径。
        output_root: 新工作区输出根目录。
        ocr_workers: OCR 并行线程数，默认沿用旧逻辑 64。
        cache_path: 可选 OCR 缓存路径，存在时复用已有结果。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.run_ocr(input_path, output_root, ocr_workers=ocr_workers, cache_path=cache_path)
        return _trace(output_root, "run_ocr", locals(), ok("OCR 识别完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "run_ocr", locals(), repair("OCR 识别失败。", [str(exc)], {"input_path": input_path}), started)


def extract_ocr_entities(ocr_cache_path: str, output_root: str, results_dir: str | None = None, concurrency: int = 100, reason: str = "") -> dict:
    """从 OCR 缓存文本中并发抽取医学实体。

    Args:
        ocr_cache_path: run_ocr 生成的 ocr_cache.json。
        output_root: 新工作区输出根目录。
        results_dir: 可选结果目录，通常使用 run_ocr 返回的 results_dir。
        concurrency: LLM 并发数，默认沿用旧逻辑 100。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.extract_ocr_entities(ocr_cache_path, output_root, results_dir_path=results_dir, concurrency=concurrency)
        return _trace(output_root, "extract_ocr_entities", locals(), ok("OCR 实体抽取完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "extract_ocr_entities", locals(), repair("OCR 实体抽取失败。", [str(exc)], {"ocr_cache_path": ocr_cache_path}), started)


def fill_table_from_entities(input_path: str, entities_csv: str, output_root: str, use_llm: bool = True, reason: str = "") -> dict:
    """将 OCR 实体回填到患者模板表，只填写空值，不覆盖已有值。

    Args:
        input_path: Step1 输出目录路径。
        entities_csv: extract_ocr_entities 生成的实体 CSV。
        output_root: 新工作区输出根目录。
        use_llm: 是否允许 LLM 辅助列匹配；默认 True，规则匹配后仅对剩余空列做高置信度映射。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.fill_table_from_entities(input_path, entities_csv, output_root, use_llm=use_llm)
        return _trace(output_root, "fill_table_from_entities", locals(), ok("OCR 实体表格回填完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "fill_table_from_entities", locals(), repair("OCR 实体表格回填失败。", [str(exc)], {"input_path": input_path, "entities_csv": entities_csv}), started)


def normalize_step2_outputs(output_root: str, filled_merged_csv: str | None = None, fallback_csv: str | None = None, summary: dict[str, Any] | None = None, reason: str = "") -> dict:
    """将 Step2 模式产物标准化为 next_input/input.csv。

    Args:
        output_root: 新工作区输出根目录。
        filled_merged_csv: OCR 回填模式生成的 filled_merged.csv，优先使用。
        fallback_csv: directory 模式生成的 admission_wide.csv。
        summary: 可选 summary 信息，会写入 next_input/summary.json。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.normalize_outputs(output_root, filled_merged_csv=filled_merged_csv, fallback_csv=fallback_csv, summary=summary)
        return _trace(output_root, "normalize_step2_outputs", locals(), ok("Step2 next_input 已生成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "normalize_step2_outputs", locals(), repair("Step2 输出标准化失败。", [str(exc)], {"output_root": output_root}), started)


def validate_step2_output(next_input_csv: str, next_summary_json: str | None = None, reason: str = "") -> dict:
    """校验 Step2 next_input/input.csv 是否可交给 Step3。

    Args:
        next_input_csv: normalize_step2_outputs 返回的 input.csv 路径。
        next_summary_json: normalize_step2_outputs 返回的 summary.json 路径。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    result = logic.validate_output(next_input_csv, next_summary_json)
    payload = ok("Step2 输出校验通过。", result["details"]) if result["passed"] else repair("Step2 输出校验失败。", result["issues"], result["details"])
    # next_input_csv 已在 step2_results 下时，可以推导 output_root；不能推导时不写 trace。
    output_root = None
    parts = Path(next_input_csv).parts
    if "step2_results" in parts:
        idx = parts.index("step2_results")
        output_root = str(Path(*parts[:idx])) if idx else None
    if output_root:
        logic.append_tool_trace(output_root, "validate_step2_output", {"next_input_csv": next_input_csv, "next_summary_json": next_summary_json}, payload, time.time())
    return payload


TOOL_FUNCTIONS = [
    resolve_step2_input,
    scan_step2_input,
    select_step2_pipeline,
    collect_directory_payloads,
    extract_entities,
    cluster_entity_names,
    build_admission_wide,
    run_ocr,
    extract_ocr_entities,
    fill_table_from_entities,
    normalize_step2_outputs,
    validate_step2_output,
]
