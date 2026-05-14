#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 2-3 结果评估脚本（五层证据链框架）

基于《数据清洗评估_方法全景版_2026-04-24.pptx》重写，覆盖：
  A. 处理完整性（Processing Completeness）
  B. Schema 与约束检查（Schema & Constraint Validation）
  C. 统计质量（Statistical Quality）
  D. 语义一致性（Semantic Consistency）
  E. 结构化产出质量（Structured Output Quality）

支持两条流水线：
  - Standard pipeline: summary.json + entities.csv + patients.csv + merged_patients.csv
  - Liver/JSONL pipeline: liver_patients_note_processed_*.json + *.csv

用法：
    python evaluate_results.py                        # 最新 results_* 目录
    python evaluate_results.py results/results_xxx    # 指定目录
    python evaluate_results.py --save                 # 保存 JSON 报告
    python evaluate_results.py --detail               # 详细诊断
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def find_latest_results(results_root: Path) -> Path:
    """查找最新的 results_* 目录"""
    subdirs = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("results_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No results_* directories found in {results_root}")
    return subdirs[0]


def detect_pipeline_type(results_dir: Path) -> str:
    """检测流水线类型：standard 或 liver"""
    has_liver = any(f.name.startswith("liver_patients_note") for f in results_dir.iterdir())
    return "liver" if has_liver else "standard"


def load_standard_files(results_dir: Path) -> Dict[str, Path]:
    """加载 standard pipeline 输出文件"""
    found = {}
    for f in results_dir.iterdir():
        stem = f.stem
        if f.suffix == ".json" and stem.startswith("summary"):
            found["summary"] = f
        elif f.suffix == ".csv" and stem.startswith("entities"):
            found["entities"] = f
        elif f.suffix == ".csv" and stem.startswith("patients"):
            found["patients"] = f
        elif f.suffix == ".csv" and stem.startswith("merged"):
            found["merged"] = f
    missing = [k for k in ("summary", "entities", "patients") if k not in found]
    if missing:
        raise FileNotFoundError(f"Missing standard pipeline files: {missing}")
    return found


def load_liver_files(results_dir: Path) -> Dict[str, Path]:
    """加载 liver pipeline 输出文件"""
    found = {}
    for f in results_dir.iterdir():
        if f.name.startswith("liver_patients_note_processed"):
            if f.suffix == ".json":
                found["liver_json"] = f
            elif f.suffix == ".csv":
                found["liver_csv"] = f
    if not found.get("liver_json") or not found.get("liver_csv"):
        raise FileNotFoundError("Missing liver pipeline files")
    return found


# ═══════════════════════════════════════════════════════════════════════════
# A. 处理完整性评估（Processing Completeness）
# ═══════════════════════════════════════════════════════════════════════════

def eval_processing_completeness_standard(summary: Dict) -> Dict:
    """Standard pipeline 处理完整性"""
    stats = summary.get("statistics", {})
    total_files = stats.get("total_files", 0)
    processed = stats.get("processed_files", 0)
    failed = stats.get("failed_files", 0)

    failed_patients = [
        pid for pid, pdata in summary.get("patients", {}).items()
        if pdata.get("failed_count", 0) > 0
    ]

    return {
        "total_patients": stats.get("total_patients", 0),
        "total_files": total_files,
        "processed_files": processed,
        "failed_files": failed,
        "success_rate_pct": round(processed / total_files * 100, 1) if total_files else 0,
        "failed_patient_ids": failed_patients,
    }


def eval_processing_completeness_liver(liver_data: Dict) -> Dict:
    """Liver pipeline 处理完整性"""
    stats = liver_data.get("statistics", {})
    total = stats.get("total_rows", 0)
    skipped = stats.get("skipped_rows", 0)
    extracted_texts = stats.get("extracted_texts", 0)
    extracted_entities = stats.get("extracted_entities", 0)

    return {
        "total_rows": total,
        "skipped_rows": skipped,
        "processed_rows": total - skipped,
        "success_rate_pct": round((total - skipped) / total * 100, 1) if total else 0,
        "extracted_texts": extracted_texts,
        "extracted_entities": extracted_entities,
    }


# ═══════════════════════════════════════════════════════════════════════════
# B. Schema 与约束检查（Schema & Constraint Validation）
# ═══════════════════════════════════════════════════════════════════════════

def eval_schema_constraints_standard(entities_df: pd.DataFrame, merged_df: Optional[pd.DataFrame]) -> Dict:
    """Standard pipeline Schema 约束检查"""
    required_cols = ["patient_id", "category", "name", "value", "unit", "duration", "original_text", "source_file"]
    missing_cols = [c for c in required_cols if c not in entities_df.columns]

    # 类型检查
    type_violations = []
    if "patient_id" in entities_df.columns and not pd.api.types.is_string_dtype(entities_df["patient_id"]):
        type_violations.append("patient_id should be string")
    if "category" in entities_df.columns and not pd.api.types.is_string_dtype(entities_df["category"]):
        type_violations.append("category should be string")

    # 枚举约束：category 应在预期范围内
    expected_categories = {"Symptom", "Disease", "Test", "LabValue", "Medication", "Procedure", "Finding", "Other"}
    if "category" in entities_df.columns:
        invalid_categories = set(entities_df["category"].dropna().unique()) - expected_categories
    else:
        invalid_categories = set()

    # 非空约束
    null_violations = {}
    for col in ["patient_id", "category", "name"]:
        if col in entities_df.columns:
            null_count = entities_df[col].isna().sum()
            if null_count > 0:
                null_violations[col] = int(null_count)

    return {
        "missing_required_columns": missing_cols,
        "type_violations": type_violations,
        "invalid_categories": list(invalid_categories),
        "null_violations": null_violations,
        "schema_valid": len(missing_cols) == 0 and len(type_violations) == 0 and len(invalid_categories) == 0,
    }


def eval_schema_constraints_liver(liver_df: pd.DataFrame) -> Dict:
    """Liver pipeline Schema 约束检查"""
    expected_cols = ["病历", "诊断列表", "手术与操作", "住院用药", "就诊文本", "实验室检验", "生命体征", "体格测量"]
    missing_cols = [c for c in expected_cols if c not in liver_df.columns]

    # 检查 entity 列
    entity_cols = [c for c in liver_df.columns if c.startswith("就诊文本_")]
    has_entity_json = "就诊文本_entities_json" in liver_df.columns
    has_entity_count = "就诊文本_entity_count" in liver_df.columns

    return {
        "missing_expected_columns": missing_cols,
        "entity_columns_count": len(entity_cols),
        "has_entity_json": has_entity_json,
        "has_entity_count": has_entity_count,
        "schema_valid": len(missing_cols) == 0 and has_entity_json and has_entity_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C. 统计质量评估（Statistical Quality）
# ═══════════════════════════════════════════════════════════════════════════

def eval_statistical_quality_standard(entities_df: pd.DataFrame, merged_df: Optional[pd.DataFrame]) -> Dict:
    """Standard pipeline 统计质量评估"""
    total = len(entities_df)
    if total == 0:
        return {"total_entities": 0}

    # 完整性：value 缺失率
    value_missing_rate = (entities_df["value"].isna() | (entities_df["value"].astype(str).str.strip() == "")).sum() / total

    # 唯一性：重复实体（同一患者、同一 name）
    duplicate_count = entities_df.groupby(["patient_id", "name"]).size().gt(1).sum()

    # 异常值检测：数值型 value 的异常值比例
    numeric_df = entities_df[entities_df["category"].isin(["LabValue", "Test"])].copy()
    outlier_count = 0
    if len(numeric_df) > 0:
        numeric_df["value_numeric"] = pd.to_numeric(numeric_df["value"], errors="coerce")
        valid_numeric = numeric_df["value_numeric"].dropna()
        if len(valid_numeric) > 0:
            q1, q3 = valid_numeric.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = ((valid_numeric < q1 - 1.5 * iqr) | (valid_numeric > q3 + 1.5 * iqr)).sum()

    # 单位覆盖率
    unit_coverage = numeric_df["unit"].notna().sum() / len(numeric_df) if len(numeric_df) > 0 else None

    # 分布统计
    category_dist = entities_df["category"].value_counts().to_dict()
    per_patient = entities_df.groupby("patient_id").size()

    return {
        "total_entities": total,
        "value_missing_rate_pct": round(value_missing_rate * 100, 1),
        "duplicate_entity_count": int(duplicate_count),
        "outlier_count": int(outlier_count),
        "unit_coverage_pct": round(unit_coverage * 100, 1) if unit_coverage is not None else None,
        "category_distribution": category_dist,
        "entities_per_patient": {
            "mean": round(per_patient.mean(), 1),
            "min": int(per_patient.min()),
            "max": int(per_patient.max()),
            "std": round(per_patient.std(), 1) if len(per_patient) > 1 else 0.0,
        },
    }


def eval_statistical_quality_liver(liver_df: pd.DataFrame) -> Dict:
    """Liver pipeline 统计质量评估"""
    total = len(liver_df)
    if total == 0:
        return {"total_rows": 0}

    # 完整性：核心字段缺失率
    core_fields = ["病历", "诊断列表", "就诊文本"]
    missing_rates = {field: round(liver_df[field].isna().sum() / total * 100, 1) for field in core_fields if field in liver_df.columns}

    # entity_count 分布
    if "就诊文本_entity_count" in liver_df.columns:
        entity_counts = liver_df["就诊文本_entity_count"].dropna()
        entity_stats = {
            "mean": round(entity_counts.mean(), 1),
            "min": int(entity_counts.min()),
            "max": int(entity_counts.max()),
            "std": round(entity_counts.std(), 1) if len(entity_counts) > 1 else 0.0,
        }
    else:
        entity_stats = None

    return {
        "total_rows": total,
        "core_field_missing_rates_pct": missing_rates,
        "entity_count_stats": entity_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# D. 语义一致性评估（Semantic Consistency）
# ═══════════════════════════════════════════════════════════════════════════

def eval_semantic_consistency_standard(entities_df: pd.DataFrame) -> Dict:
    """Standard pipeline 语义一致性评估"""
    issues = []

    # 检查 name 与 category 的语义匹配（简单规则）
    if "category" in entities_df.columns and "name" in entities_df.columns:
        # Drug 类实体 name 不应包含检验关键词
        drug_df = entities_df[entities_df["category"] == "Drug"]
        test_keywords = {"检验", "检查", "化验", "test", "assay", "count", "level"}
        suspicious_drug = drug_df[
            drug_df["name"].str.lower().apply(
                lambda n: any(kw in n for kw in test_keywords)
            )
        ]
        if len(suspicious_drug) > 0:
            issues.append(f"Drug 类实体中疑似含检验项目名称：{len(suspicious_drug)} 条")

        # LabValue / Test 类实体 value 应可转为数值或标准定性词
        lab_df = entities_df[entities_df["category"].isin(["LabValue", "Test"])].copy()
        if len(lab_df) > 0 and "value" in lab_df.columns:
            non_empty = lab_df[lab_df["value"].notna() & (lab_df["value"].astype(str).str.strip() != "")]
            if len(non_empty) > 0:
                numeric_parseable = pd.to_numeric(non_empty["value"], errors="coerce").notna()
                qualitative_ok = non_empty["value"].astype(str).str.strip().isin(
                    {"阴性", "阳性", "正常", "异常", "偏高", "偏低", "negative", "positive", "normal", "abnormal"}
                )
                bad = (~numeric_parseable) & (~qualitative_ok)
                if bad.sum() > 0:
                    issues.append(f"LabValue/Test 中 value 既非数值也非标准定性词：{int(bad.sum())} 条")

    # original_text 与 name 交叉验证：只对应该直接出现在原文中的类别做字面匹配检查
    # LabValue/Test/Finding 的 name 是 LLM 从上下文推断的标准化名称，不要求字面匹配
    text_mismatch = 0
    literal_categories = {"Disease", "Drug", "Symptom", "Treatment", "Anatomy"}
    if "original_text" in entities_df.columns and "name" in entities_df.columns and "category" in entities_df.columns:
        literal_sample = entities_df[
            entities_df["category"].isin(literal_categories)
        ].dropna(subset=["original_text", "name"])

        if len(literal_sample) > 0:
            # 取最多 500 条样本
            literal_sample = literal_sample.head(500)
            text_mismatch = int(
                literal_sample.apply(
                    lambda r: r["name"].lower() not in r["original_text"].lower()
                    and len(r["name"]) > 2,
                    axis=1,
                ).sum()
            )
            if text_mismatch > len(literal_sample) * 0.3:
                issues.append(
                    f"Disease/Drug/Symptom 等类别中 name 未出现在 original_text 的比例偏高：{text_mismatch}/{len(literal_sample)}"
                )

    return {
        "issues": issues,
        "name_not_in_text_count": text_mismatch,
        "literal_categories_checked": list(literal_categories),
        "semantic_ok": len(issues) == 0,
    }


def eval_semantic_consistency_liver(liver_df: pd.DataFrame) -> Dict:
    """Liver pipeline 语义一致性评估"""
    issues = []

    # 诊断列表字段：应为列表且含 icd_code
    if "诊断列表" in liver_df.columns:
        bad_diag = 0
        for val in liver_df["诊断列表"].dropna():
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
                if not isinstance(parsed, list):
                    bad_diag += 1
                elif parsed and "icd_code" not in parsed[0]:
                    bad_diag += 1
            except Exception:
                bad_diag += 1
        if bad_diag > 0:
            issues.append(f"诊断列表字段格式异常（非列表或缺 icd_code）：{bad_diag} 行")

    # entities_json 应为合法 JSON 列表
    if "就诊文本_entities_json" in liver_df.columns:
        bad_json = 0
        for val in liver_df["就诊文本_entities_json"].dropna():
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
                if not isinstance(parsed, list):
                    bad_json += 1
            except Exception:
                bad_json += 1
        if bad_json > 0:
            issues.append(f"就诊文本_entities_json 解析失败或非列表：{bad_json} 行")

    # entity_count 与 entities_json 实际长度一致性
    mismatch_count = 0
    if "就诊文本_entities_json" in liver_df.columns and "就诊文本_entity_count" in liver_df.columns:
        for _, row in liver_df.dropna(subset=["就诊文本_entities_json", "就诊文本_entity_count"]).iterrows():
            try:
                parsed = json.loads(row["就诊文本_entities_json"]) if isinstance(row["就诊文本_entities_json"], str) else row["就诊文本_entities_json"]
                if isinstance(parsed, list) and len(parsed) != int(row["就诊文本_entity_count"]):
                    mismatch_count += 1
            except Exception:
                pass
        if mismatch_count > 0:
            issues.append(f"entity_count 与 entities_json 实际条数不一致：{mismatch_count} 行")

    return {
        "issues": issues,
        "entity_count_mismatch": mismatch_count,
        "semantic_ok": len(issues) == 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# E. 结构化产出质量（Structured Output Quality）
# ═══════════════════════════════════════════════════════════════════════════

def eval_structured_output_quality_standard(
    entities_df: pd.DataFrame,
    merged_df: Optional[pd.DataFrame],
    summary: Dict,
) -> Dict:
    """Standard pipeline 结构化产出质量评估"""
    result = {}

    # 实体覆盖率：每个患者应有至少一个实体
    if "patient_id" in entities_df.columns:
        patients_with_entities = set(entities_df["patient_id"].dropna().unique())
        total_patients = summary.get("statistics", {}).get("total_patients", len(patients_with_entities))
        result["patient_entity_coverage_pct"] = round(
            len(patients_with_entities) / total_patients * 100, 1
        ) if total_patients > 0 else 0

    # 高置信度分类占比（非 Other 类别）
    if "category" in entities_df.columns:
        high_conf = entities_df[entities_df["category"] != "Other"]
        result["high_confidence_category_pct"] = round(
            len(high_conf) / len(entities_df) * 100, 1
        ) if len(entities_df) > 0 else 0

    # 关系覆盖：summary 中 relations 字段
    total_relations = sum(
        len(pdata.get("relations", [])) if isinstance(pdata, dict) else 0
        for pdata in summary.get("patients", {}).values()
    )
    result["total_relations_extracted"] = total_relations

    # merged_patients 的患者数与 patients.csv 一致性
    if merged_df is not None and "patient_id" in entities_df.columns:
        merged_patients = set(merged_df["patient_id"].dropna().unique()) if "patient_id" in merged_df.columns else set()
        entity_patients = set(entities_df["patient_id"].dropna().unique())
        result["merged_vs_entities_patient_overlap_pct"] = round(
            len(merged_patients & entity_patients) / len(entity_patients) * 100, 1
        ) if entity_patients else 0

    return result


def eval_structured_output_quality_liver(liver_df: pd.DataFrame, liver_json: Dict) -> Dict:
    """Liver pipeline 结构化产出质量评估"""
    result = {}
    total = len(liver_df)

    # 实体抽取覆盖率：有就诊文本但 entity_count=0 的行
    if "就诊文本" in liver_df.columns and "就诊文本_entity_count" in liver_df.columns:
        has_text = liver_df["就诊文本"].notna() & (liver_df["就诊文本"].astype(str).str.strip() != "")
        zero_entity = (liver_df["就诊文本_entity_count"].fillna(0) == 0)
        missed = int((has_text & zero_entity).sum())
        result["text_with_zero_entities"] = missed
        result["entity_extraction_coverage_pct"] = round(
            (has_text.sum() - missed) / has_text.sum() * 100, 1
        ) if has_text.sum() > 0 else 0

    # category 多样性（各行中出现的 entity category 种类）
    category_cols = [c for c in liver_df.columns if c.startswith("就诊文本_") and c not in (
        "就诊文本_entities_json", "就诊文本_entity_count", "就诊文本_impression"
    )]
    result["entity_category_columns"] = category_cols

    # impression 覆盖率
    if "就诊文本_impression" in liver_df.columns:
        imp_coverage = liver_df["就诊文本_impression"].notna() & (
            liver_df["就诊文本_impression"].astype(str).str.strip() != ""
        )
        result["impression_coverage_pct"] = round(imp_coverage.sum() / total * 100, 1) if total > 0 else 0

    # 总实体数与统计一致性
    stats_entities = liver_json.get("statistics", {}).get("extracted_entities", None)
    if stats_entities is not None and "就诊文本_entity_count" in liver_df.columns:
        actual_total = int(liver_df["就诊文本_entity_count"].fillna(0).sum())
        result["entity_count_stats_match"] = actual_total == stats_entities
        result["entity_count_actual"] = actual_total
        result["entity_count_reported"] = stats_entities

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 汇总打分
# ═══════════════════════════════════════════════════════════════════════════

def compute_overall_score(report: Dict, pipeline: str) -> Dict:
    """基于五层评估结果计算综合得分（百分制）"""
    scores = {}

    A = report.get("A_processing_completeness", {})
    B = report.get("B_schema_constraints", {})
    C = report.get("C_statistical_quality", {})
    D = report.get("D_semantic_consistency", {})
    E = report.get("E_structured_output_quality", {})

    # A: 成功率直接作为分数
    scores["A"] = A.get("success_rate_pct", 0)

    # B: schema_valid = 100，否则按缺失项扣分
    if B.get("schema_valid", False):
        scores["B"] = 100.0
    else:
        deductions = (
            len(B.get("missing_required_columns", []) or B.get("missing_expected_columns", [])) * 10
            + len(B.get("type_violations", [])) * 5
            + len(B.get("invalid_categories", [])) * 5
            + len(B.get("null_violations", {})) * 5
        )
        scores["B"] = max(0.0, 100.0 - deductions)

    # C: 基于缺失率和异常值扣分
    if pipeline == "standard":
        missing_penalty = C.get("value_missing_rate_pct", 0)
        outlier_total = C.get("total_entities", 1)
        outlier_pct = C.get("outlier_count", 0) / outlier_total * 100 if outlier_total > 0 else 0
        scores["C"] = max(0.0, 100.0 - missing_penalty * 0.5 - outlier_pct * 0.3)
    else:
        core_missing = C.get("core_field_missing_rates_pct", {})
        avg_missing = sum(core_missing.values()) / len(core_missing) if core_missing else 0
        scores["C"] = max(0.0, 100.0 - avg_missing)

    # D: 语义问题数量扣分
    issue_count = len(D.get("issues", []))
    scores["D"] = max(0.0, 100.0 - issue_count * 15)

    # E: 覆盖率平均
    coverage_keys = ["patient_entity_coverage_pct", "entity_extraction_coverage_pct",
                     "high_confidence_category_pct", "impression_coverage_pct"]
    coverage_vals = [E[k] for k in coverage_keys if k in E]
    scores["E"] = round(sum(coverage_vals) / len(coverage_vals), 1) if coverage_vals else 50.0

    weights = {"A": 0.25, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.15}
    overall = sum(scores[k] * weights[k] for k in scores)

    return {
        "layer_scores": {k: round(scores[k], 1) for k in scores},
        "weights": weights,
        "overall_score": round(overall, 1),
        "grade": "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 60 else "D",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 报告打印
# ═══════════════════════════════════════════════════════════════════════════

def print_report(report: Dict, detail: bool = False) -> None:
    sep = "═" * 70
    thin = "─" * 70
    print(f"\n{sep}")
    print(f"  Agent 2-3 评估报告  |  流水线: {report['pipeline'].upper()}  |  目录: {report['results_dir']}")
    print(sep)

    score_info = report.get("overall_score", {})
    layer = score_info.get("layer_scores", {})
    print(f"\n【综合得分】{score_info.get('overall_score', 'N/A')} / 100  (等级: {score_info.get('grade', '?')})")
    print(f"  A.处理完整性={layer.get('A','?')}  B.Schema={layer.get('B','?')}  "
          f"C.统计质量={layer.get('C','?')}  D.语义={layer.get('D','?')}  E.产出质量={layer.get('E','?')}")
    print(thin)

    A = report.get("A_processing_completeness", {})
    print(f"\nA. 处理完整性")
    if report["pipeline"] == "standard":
        print(f"   患者数={A.get('total_patients')}  文件总数={A.get('total_files')}  "
              f"成功={A.get('processed_files')}  失败={A.get('failed_files')}  "
              f"成功率={A.get('success_rate_pct')}%")
        if A.get("failed_patient_ids"):
            print(f"   失败患者: {A['failed_patient_ids']}")
    else:
        print(f"   总行数={A.get('total_rows')}  跳过={A.get('skipped_rows')}  "
              f"处理={A.get('processed_rows')}  成功率={A.get('success_rate_pct')}%")
        print(f"   抽取文本={A.get('extracted_texts')}  抽取实体={A.get('extracted_entities')}")

    B = report.get("B_schema_constraints", {})
    print(f"\nB. Schema 约束  {'✓ 通过' if B.get('schema_valid') else '✗ 存在问题'}")
    if detail or not B.get("schema_valid"):
        for k, v in B.items():
            if k != "schema_valid" and v:
                print(f"   {k}: {v}")

    C = report.get("C_statistical_quality", {})
    print(f"\nC. 统计质量")
    if report["pipeline"] == "standard":
        print(f"   实体总数={C.get('total_entities')}  value缺失率={C.get('value_missing_rate_pct')}%  "
              f"重复实体={C.get('duplicate_entity_count')}  异常值={C.get('outlier_count')}")
        print(f"   单位覆盖率={C.get('unit_coverage_pct')}%")
        if detail:
            print(f"   类别分布: {C.get('category_distribution')}")
            print(f"   每患者实体数: {C.get('entities_per_patient')}")
    else:
        print(f"   总行数={C.get('total_rows')}  核心字段缺失率={C.get('core_field_missing_rates_pct')}")
        if C.get("entity_count_stats"):
            print(f"   实体数统计: {C.get('entity_count_stats')}")

    D = report.get("D_semantic_consistency", {})
    print(f"\nD. 语义一致性  {'✓ 无问题' if D.get('semantic_ok') else '✗ 存在问题'}")
    for issue in D.get("issues", []):
        print(f"   ⚠ {issue}")
    if detail and report["pipeline"] == "standard":
        checked_cats = D.get("literal_categories_checked", [])
        print(f"   字面溯源检查范围: {', '.join(checked_cats)}")
        print(f"   该范围内 name 未出现在原文中的条数: {D.get('name_not_in_text_count')}")

    E = report.get("E_structured_output_quality", {})
    print(f"\nE. 结构化产出质量")
    for k, v in E.items():
        print(f"   {k}: {v}")

    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agent 2-3 结果评估（五层证据链框架）")
    parser.add_argument("results_dir", nargs="?", default=None, help="指定 results_* 目录路径")
    parser.add_argument("--save", action="store_true", help="保存 JSON 报告到 results_dir")
    parser.add_argument("--detail", action="store_true", help="输出详细诊断信息")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_root = script_dir / "results"

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = find_latest_results(results_root)

    print(f"评估目录: {results_dir}")
    pipeline = detect_pipeline_type(results_dir)
    print(f"检测到流水线类型: {pipeline}")

    report: Dict[str, Any] = {
        "pipeline": pipeline,
        "results_dir": str(results_dir),
    }

    if pipeline == "standard":
        files = load_standard_files(results_dir)
        with open(files["summary"]) as f:
            summary = json.load(f)
        entities_df = pd.read_csv(files["entities"])
        patients_df = pd.read_csv(files["patients"])
        merged_df = pd.read_csv(files["merged"]) if "merged" in files else None

        report["A_processing_completeness"] = eval_processing_completeness_standard(summary)
        report["B_schema_constraints"] = eval_schema_constraints_standard(entities_df, merged_df)
        report["C_statistical_quality"] = eval_statistical_quality_standard(entities_df, merged_df)
        report["D_semantic_consistency"] = eval_semantic_consistency_standard(entities_df)
        report["E_structured_output_quality"] = eval_structured_output_quality_standard(entities_df, merged_df, summary)

    else:  # liver
        files = load_liver_files(results_dir)
        with open(files["liver_json"]) as f:
            liver_json = json.load(f)
        liver_df = pd.read_csv(files["liver_csv"])

        report["A_processing_completeness"] = eval_processing_completeness_liver(liver_json)
        report["B_schema_constraints"] = eval_schema_constraints_liver(liver_df)
        report["C_statistical_quality"] = eval_statistical_quality_liver(liver_df)
        report["D_semantic_consistency"] = eval_semantic_consistency_liver(liver_df)
        report["E_structured_output_quality"] = eval_structured_output_quality_liver(liver_df, liver_json)

    report["overall_score"] = compute_overall_score(report, pipeline)
    print_report(report, detail=args.detail)

    if args.save:
        out_path = results_dir / "evaluation_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"报告已保存至: {out_path}")


if __name__ == "__main__":
    main()
