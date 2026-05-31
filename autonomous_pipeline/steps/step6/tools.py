from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from autonomous_pipeline.contracts import failed, ok, repair

from . import logic


def _step_output_root(output_root: str | None) -> str | None:
    if not output_root:
        return None
    path = Path(output_root).expanduser()
    if path.name == "step6_results":
        return str(path)
    if path.name.startswith("step6"):
        return str(path.parent / "step6_results")
    return str(path / "step6_results")


def _pipeline_output_root(output_root: str | None) -> str | None:
    if not output_root:
        return None
    path = Path(output_root).expanduser()
    return str(path.parent if path.name.startswith("step6") else path)


def _trace(output_root: str | None, tool: str, args: dict[str, Any], payload: dict[str, Any], started: float) -> dict[str, Any]:
    step_root = _step_output_root(output_root)
    if step_root:
        trace_path = logic.append_tool_trace(step_root, tool, args, payload, started)
        payload.setdefault("artifacts", {})["tool_trace_path"] = trace_path
    return payload


def _infer_output_root_from_step6_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if "step6_results" not in parts:
        return None
    idx = parts.index("step6_results")
    return str(Path(*parts[:idx])) if idx else None


def resolve_step6_inputs(input_path: str, output_root: str | None = None, passed_patients_json: str | None = None, selection_report_path: str | None = None, reason: str = "") -> dict:
    """解析 Step6 输入，定位 Step5 next_input 中的 filtered.csv 和 passed_patients.json。

    Args:
        input_path: Step5 next_input 目录、Step5 输出目录，或 filtered.csv 路径。
        output_root: 新工作区输出根目录；提供时写入 Step6 tool_trace。
        passed_patients_json: 可选，显式指定 Step5 通过患者名单。
        selection_report_path: 可选，显式指定 Step3 selection_report。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.resolve_step6_inputs(input_path, passed_patients_json=passed_patients_json, selection_report_path=selection_report_path)
        return _trace(output_root, "resolve_step6_inputs", locals(), ok("Step6 输入路径已解析。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "resolve_step6_inputs", locals(), repair("Step6 输入路径解析失败。", [str(exc)], {"input_path": input_path}), started)


def load_task_context(input_csv: str, output_root: str, task_text: str = "", passed_patients_json: str | None = None, selection_report_path: str | None = None, reason: str = "") -> dict:
    """加载 Step6 任务上下文，列出自动特征列、标签候选列和 ICD 编码列。

    Args:
        input_csv: Step5 next_input/filtered.csv。
        output_root: 新工作区输出根目录。
        task_text: 用户任务目标，例如 肝病诊断。
        passed_patients_json: Step5 通过患者名单；提供时只分析通过患者。
        selection_report_path: 可选 Step3 selection_report。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.load_task_context(input_csv, _pipeline_output_root(output_root) or output_root, task_text=task_text, passed_patients_json=passed_patients_json, selection_report_path=selection_report_path)
        return _trace(output_root, "load_task_context", locals(), ok("Step6 任务上下文加载完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "load_task_context", locals(), failed("Step6 任务上下文加载失败。", [str(exc)], {"input_csv": input_csv}), started)


def create_icd_binary_label(input_csv: str, icd_codes_column: str, output_root: str, positive_icd_prefixes: list[str] | None = None, label_name_positive: str = "liver_disease", label_name_negative: str = "no_liver_disease", passed_patients_json: str | None = None, task_name: str = "liver_disease_icd_binary", reason: str = "") -> dict:
    """基于 ICD 编码列创建疾病二分类标签，并保存带标签中间表。

    Args:
        input_csv: Step5 next_input/filtered.csv。
        icd_codes_column: ICD 编码来源列，例如 structured__diagnoses_icd__icd_code。
        output_root: 新工作区输出根目录。
        positive_icd_prefixes: 阳性 ICD 前缀；为空时使用肝病默认前缀。
        label_name_positive: 阳性标签名。
        label_name_negative: 阴性标签名。
        passed_patients_json: Step5 通过患者名单；提供时只保留通过患者。
        task_name: 输出任务名。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.create_icd_binary_label(input_csv, icd_codes_column, _pipeline_output_root(output_root) or output_root, positive_icd_prefixes=positive_icd_prefixes, label_name_positive=label_name_positive, label_name_negative=label_name_negative, passed_patients_json=passed_patients_json, task_name=task_name)
        status = ok("Step6 ICD 二分类标签已生成。", artifacts) if artifacts.get("positive_count", 0) > 0 and artifacts.get("negative_count", 0) > 0 else repair("Step6 ICD 标签类别不足。", ["阳性或阴性样本为 0，无法训练二分类模型"], artifacts)
        return _trace(output_root, "create_icd_binary_label", locals(), status, started)
    except Exception as exc:
        return _trace(output_root, "create_icd_binary_label", locals(), failed("Step6 ICD 标签生成失败。", [str(exc)], {"input_csv": input_csv, "icd_codes_column": icd_codes_column}), started)


def create_notnull_binary_label(input_csv: str, source_column: str, output_root: str, positive_condition: str = "not_null", label_name_positive: str = "positive", label_name_negative: str = "negative", passed_patients_json: str | None = None, task_name: str = "notnull_binary", reason: str = "") -> dict:
    """基于某列是否为空或数值条件创建二分类标签，并保存带标签中间表。

    Args:
        input_csv: Step5 next_input/filtered.csv。
        source_column: 标签来源列。
        output_root: 新工作区输出根目录。
        positive_condition: not_null / is_null / gt_0 / eq_0。
        label_name_positive: 阳性标签名。
        label_name_negative: 阴性标签名。
        passed_patients_json: Step5 通过患者名单；提供时只保留通过患者。
        task_name: 输出任务名。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.create_notnull_binary_label(input_csv, source_column, _pipeline_output_root(output_root) or output_root, positive_condition=positive_condition, label_name_positive=label_name_positive, label_name_negative=label_name_negative, passed_patients_json=passed_patients_json, task_name=task_name)
        status = ok("Step6 非空二分类标签已生成。", artifacts) if artifacts.get("positive_count", 0) > 0 and artifacts.get("negative_count", 0) > 0 else repair("Step6 非空标签类别不足。", ["阳性或阴性样本为 0，无法训练二分类模型"], artifacts)
        return _trace(output_root, "create_notnull_binary_label", locals(), status, started)
    except Exception as exc:
        return _trace(output_root, "create_notnull_binary_label", locals(), failed("Step6 非空标签生成失败。", [str(exc)], {"source_column": source_column}), started)


def discover_diagnosis_fields(input_csv: str, output_root: str, passed_patients_json: str | None = None, max_fields: int = 30, reason: str = "") -> dict:
    """识别可能包含疾病/诊断信息的字段，供 ICD-10 标准化和数据集增强使用。

    Args:
        input_csv: Step5 next_input/filtered.csv。
        output_root: 新工作区输出根目录。
        passed_patients_json: Step5 通过患者名单；提供时只分析通过患者。
        max_fields: 候选过多时最多保留的诊断字段数量。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.discover_diagnosis_fields(input_csv, _pipeline_output_root(output_root) or output_root, passed_patients_json=passed_patients_json, max_fields=max_fields)
        return _trace(output_root, "discover_diagnosis_fields", locals(), ok("Step6 诊断字段识别完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "discover_diagnosis_fields", locals(), repair("Step6 诊断字段识别失败。", [str(exc)], {"input_csv": input_csv}), started)


def extract_and_map_all_diagnosis_fields(input_csv: str, diagnosis_fields: list[str], output_root: str, passed_patients_json: str | None = None, max_unique_per_field: int = 200, reason: str = "") -> dict:
    """对诊断字段进行疾病内容提取，并映射到本地 ICD-10 标准库。

    Args:
        input_csv: Step5 next_input/filtered.csv。
        diagnosis_fields: discover_diagnosis_fields 返回的诊断字段列表。
        output_root: 新工作区输出根目录。
        passed_patients_json: Step5 通过患者名单；提供时只分析通过患者。
        max_unique_per_field: 每个字段最多处理的唯一值数量，默认沿用旧 Step7 的 200。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.extract_and_map_all_diagnosis_fields(input_csv, diagnosis_fields, _pipeline_output_root(output_root) or output_root, passed_patients_json=passed_patients_json, max_unique_per_field=max_unique_per_field)
        return _trace(output_root, "extract_and_map_all_diagnosis_fields", locals(), ok("Step6 诊断字段 ICD-10 映射完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "extract_and_map_all_diagnosis_fields", locals(), repair("Step6 诊断字段 ICD-10 映射失败。", [str(exc)], {"diagnosis_fields": diagnosis_fields}), started)


def map_diagnoses_to_icd10(diagnosis_names: list[str], output_root: str, reason: str = "") -> dict:
    """将一组诊断名称映射为 ICD-10 编码，可用于 label_ICD10 或增强字段。

    Args:
        diagnosis_names: 需要映射的诊断名称列表。
        output_root: 新工作区输出根目录。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.map_diagnoses_to_icd10(diagnosis_names, _pipeline_output_root(output_root) or output_root)
        return _trace(output_root, "map_diagnoses_to_icd10", locals(), ok("Step6 诊断名称 ICD-10 映射完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "map_diagnoses_to_icd10", locals(), repair("Step6 诊断名称 ICD-10 映射失败。", [str(exc)], {"diagnosis_names": diagnosis_names}), started)


def propose_label_mapping(input_csv: str, label_column: str, output_root: str, passed_patients_json: str | None = None, mode: str = "multiclass_enum", positive_values: list[str] | None = None, reason: str = "") -> dict:
    """根据标签列实际取值生成 label_mapping。

    Args:
        input_csv: Step5 next_input/filtered.csv。
        label_column: 现有标签列。
        output_root: 新工作区输出根目录。
        passed_patients_json: Step5 通过患者名单；提供时只分析通过患者。
        mode: multiclass_enum 或 binary_positive_vs_rest。
        positive_values: binary_positive_vs_rest 模式下的阳性取值。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.propose_label_mapping(input_csv, label_column, _pipeline_output_root(output_root) or output_root, passed_patients_json=passed_patients_json, mode=mode, positive_values=positive_values)
        return _trace(output_root, "propose_label_mapping", locals(), ok("Step6 标签映射已生成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "propose_label_mapping", locals(), repair("Step6 标签映射生成失败。", [str(exc)], {"label_column": label_column}), started)


def build_label_series(input_csv: str, label_column: str, label_mapping: dict[str, int], output_root: str, passed_patients_json: str | None = None, reason: str = "") -> dict:
    """验证标签列和 label_mapping 可映射出有效 y，不保存训练集。

    Args:
        input_csv: Step5 next_input/filtered.csv。
        label_column: 标签列。
        label_mapping: 类别值到整数 y 的映射。
        output_root: 新工作区输出根目录。
        passed_patients_json: Step5 通过患者名单；提供时只分析通过患者。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.build_label_series(input_csv, label_column, label_mapping, _pipeline_output_root(output_root) or output_root, passed_patients_json=passed_patients_json)
        return _trace(output_root, "build_label_series", locals(), ok("Step6 标签序列验证完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "build_label_series", locals(), repair("Step6 标签序列验证失败。", [str(exc)], {"label_column": label_column}), started)


def select_feature_columns(labeled_csv: str, output_root: str, label_column: str = "_icd_label", label_source_column: str | None = None, max_missing_rate: float = 0.70, reason: str = "") -> dict:
    """自动选择训练特征，排除 ID、时间列、标签列、ICD/标签泄漏源和高缺失列。

    Args:
        labeled_csv: create_*_label 生成的带标签 CSV。
        output_root: 新工作区输出根目录。
        label_column: 标签列名。
        label_source_column: 标签来源列；会被排除出特征。
        max_missing_rate: 特征最大允许缺失率，默认沿用旧 Step7 的 0.70。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.select_feature_columns(labeled_csv, _pipeline_output_root(output_root) or output_root, label_column=label_column, label_source_column=label_source_column, max_missing_rate=max_missing_rate)
        return _trace(output_root, "select_feature_columns", locals(), ok("Step6 特征列选择完成。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "select_feature_columns", locals(), failed("Step6 特征列选择失败。", [str(exc)], {"labeled_csv": labeled_csv}), started)


def create_icd_label(input_csv: str, label_source_column: str, output_root: str, task_text: str = "", passed_patients_json: str | None = None, reason: str = "") -> dict:
    """兼容旧测试/旧调用名：基于 ICD 编码列创建肝病二分类标签。

    Args:
        input_csv: 输入患者级 CSV。
        label_source_column: ICD 编码来源列。
        output_root: 新工作区输出根目录。
        task_text: 任务目标；当前兼容函数不使用。
        passed_patients_json: 可选 Step5 通过患者名单。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    return create_icd_binary_label(
        input_csv=input_csv,
        icd_codes_column=label_source_column,
        output_root=output_root,
        passed_patients_json=passed_patients_json,
        task_name="liver_disease_icd_binary",
        reason=reason,
    )


def format_and_save_ml_dataset(labeled_csv: str, feature_columns_json: str, output_root: str, label_column: str, label_mapping: dict[str, int], task_name: str = "ml_dataset", icd10_mapping: dict[str, str] | None = None, reason: str = "") -> dict:
    """导出 ML 数据集 CSV/JSON/JSONL 和模型配置。

    Args:
        labeled_csv: create_*_label 生成的带标签 CSV。
        feature_columns_json: select_feature_columns 生成的特征列 JSON。
        output_root: 新工作区输出根目录。
        label_column: 标签列名。
        label_mapping: 类别值到整数 y 的映射。
        task_name: 输出数据集任务名。
        icd10_mapping: 诊断名称到 ICD-10 编码的映射，可由 map_diagnoses_to_icd10 或 extract_and_map_all_diagnosis_fields 生成。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    try:
        artifacts = logic.format_and_save_ml_dataset(labeled_csv, feature_columns_json, _pipeline_output_root(output_root) or output_root, label_column=label_column, label_mapping=label_mapping, task_name=task_name, icd10_mapping=icd10_mapping)
        return _trace(output_root, "format_and_save_ml_dataset", locals(), ok("Step6 ML 数据集已保存。", artifacts), started)
    except Exception as exc:
        return _trace(output_root, "format_and_save_ml_dataset", locals(), repair("Step6 ML 数据集保存失败。", [str(exc)], {"labeled_csv": labeled_csv}), started)


def build_ml_dataset(labeled_csv: str, feature_columns_json: str, output_root: str, task_name: str = "ml_dataset", reason: str = "") -> dict:
    """兼容旧测试/旧调用名：根据带标签 CSV 和特征列导出 ML 数据集。

    Args:
        labeled_csv: create_icd_label/create_icd_binary_label 生成的带标签 CSV。
        feature_columns_json: select_feature_columns 生成的特征列 JSON。
        output_root: 新工作区输出根目录。
        task_name: 输出数据集任务名。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    import pandas as pd

    cols = list(pd.read_csv(labeled_csv, nrows=1).columns)
    if "_icd_label" in cols:
        return format_and_save_ml_dataset(
            labeled_csv,
            feature_columns_json,
            output_root,
            label_column="_icd_label",
            label_mapping={"no_liver_disease": 0, "liver_disease": 1},
            task_name=task_name,
            reason=reason,
        )
    if "_notnull_label" in cols:
        return format_and_save_ml_dataset(
            labeled_csv,
            feature_columns_json,
            output_root,
            label_column="_notnull_label",
            label_mapping={"negative": 0, "positive": 1},
            task_name=task_name,
            reason=reason,
        )
    return repair("Step6 兼容 build_ml_dataset 无法推断标签列。", ["缺少 _icd_label 或 _notnull_label"], {"labeled_csv": labeled_csv})


def validate_dataset(dataset_csv: str, model_config_json: str, reason: str = "") -> dict:
    """校验 ML 数据集是否有双类别标签，且特征列无标签/时间泄漏。

    Args:
        dataset_csv: format_and_save_ml_dataset 生成的 CSV。
        model_config_json: format_and_save_ml_dataset 生成的模型配置 JSON。
        reason: Agent 选择当前工具的中文理由，会打印到终端日志。
    """
    started = time.time()
    result = logic.validate_dataset(dataset_csv, model_config_json)
    payload = ok("Step6 数据集校验通过。", result) if result["passed"] else repair("Step6 数据集校验未通过。", result["issues"], result)
    output_root = _infer_output_root_from_step6_path(dataset_csv) or _infer_output_root_from_step6_path(model_config_json)
    return _trace(output_root, "validate_dataset", locals(), payload, started) if output_root else payload


TOOL_FUNCTIONS = [
    resolve_step6_inputs,
    load_task_context,
    create_icd_binary_label,
    create_notnull_binary_label,
    discover_diagnosis_fields,
    extract_and_map_all_diagnosis_fields,
    map_diagnoses_to_icd10,
    propose_label_mapping,
    build_label_series,
    select_feature_columns,
    format_and_save_ml_dataset,
    validate_dataset,
]
