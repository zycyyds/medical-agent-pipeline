# -*- coding: utf-8 -*-
"""
抽取结果评估脚本

针对 entities CSV 的五个维度评估：
  A. 覆盖完整性   — 患者/文件覆盖率，每患者实体数分布
  B. 格式合规性   — LabValue.value 纯数字、Finding.unit 为空、必填字段非空
  C. 去重质量     — 跨文件重复率（同患者同名同值）
  D. 语义一致性   — LabValue unit 覆盖率、value 数值范围合理性（IQR 异常值）
  E. 结构完整性   — Finding/LabValue 配对率（每个 Finding 体位应有对应 LabValue）

用法：
  python evaluate_extraction.py <entities_csv> [--out report.json]
"""
import re
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# A. 覆盖完整性
# ---------------------------------------------------------------------------

def eval_coverage(df: pd.DataFrame) -> dict:
    total_patients = df["patient_id"].nunique()
    entities_per_patient = df.groupby("patient_id").size()

    # 每患者文件数（source_file 去重）
    files_per_patient = df.groupby("patient_id")["source_file"].nunique()

    return {
        "total_patients": total_patients,
        "total_entities": len(df),
        "entities_per_patient": {
            "mean": round(entities_per_patient.mean(), 1),
            "min": int(entities_per_patient.min()),
            "max": int(entities_per_patient.max()),
            "std": round(entities_per_patient.std(), 1),
            "detail": entities_per_patient.to_dict(),
        },
        "files_per_patient": {
            "mean": round(files_per_patient.mean(), 1),
            "detail": files_per_patient.to_dict(),
        },
        "category_distribution": df["category"].value_counts().to_dict(),
        "score": 100.0,  # 只要有输出就满分，具体缺失由 B/C/D 扣分
    }


# ---------------------------------------------------------------------------
# B. 格式合规性
# ---------------------------------------------------------------------------

def eval_format(df: pd.DataFrame) -> dict:
    issues = []
    deductions = 0.0

    lv = df[df["category"] == "LabValue"].copy()
    finding = df[df["category"] == "Finding"].copy()

    # B1: LabValue.value 必须是纯数字字符串
    def is_numeric_str(v):
        if pd.isna(v):
            return False
        return bool(re.fullmatch(r"[+-]?\d+\.?\d*", str(v).strip()))

    bad_value = lv[~lv["value"].apply(is_numeric_str)]
    b1_rate = len(bad_value) / max(len(lv), 1)
    if len(bad_value) > 0:
        issues.append(f"LabValue.value 含非数字: {len(bad_value)} 条")
        deductions += min(20, b1_rate * 100)

    # B2: Finding.unit 必须为空
    bad_finding_unit = finding[finding["unit"].notna() & (finding["unit"].astype(str).str.strip() != "")]
    b2_rate = len(bad_finding_unit) / max(len(finding), 1)
    if len(bad_finding_unit) > 0:
        issues.append(f"Finding.unit 非空: {len(bad_finding_unit)} 条")
        deductions += min(15, b2_rate * 100)

    # B3: name/category 必填字段不能为空
    null_name = df["name"].isna().sum()
    null_cat = df["category"].isna().sum()
    if null_name > 0:
        issues.append(f"name 字段为空: {null_name} 条")
        deductions += 5
    if null_cat > 0:
        issues.append(f"category 字段为空: {null_cat} 条")
        deductions += 5

    # B4: category 只能是合法值
    valid_cats = {"LabValue", "Test", "Finding", "Disease", "Symptom", "Drug", "Treatment", "Anatomy", "Other"}
    invalid_cat = df[~df["category"].isin(valid_cats)]
    if len(invalid_cat) > 0:
        issues.append(f"非法 category 值: {invalid_cat['category'].unique().tolist()}")
        deductions += 10

    score = max(0.0, 100.0 - deductions)
    return {
        "labvalue_bad_value_count": len(bad_value),
        "finding_bad_unit_count": len(bad_finding_unit),
        "null_name_count": int(null_name),
        "null_category_count": int(null_cat),
        "invalid_category_values": invalid_cat["category"].unique().tolist() if len(invalid_cat) > 0 else [],
        "issues": issues,
        "score": round(score, 1),
    }


# ---------------------------------------------------------------------------
# C. 去重质量
# ---------------------------------------------------------------------------

def eval_dedup(df: pd.DataFrame) -> dict:
    issues = []

    # 同患者、同 category、同 name、同 value → 跨文件重复
    key_cols = ["patient_id", "category", "name", "value"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_count = dup_mask.sum()
    dup_rate = dup_count / max(len(df), 1)

    # 按患者统计重复数
    dup_per_patient = df[dup_mask].groupby("patient_id").size().to_dict()

    if dup_count > 0:
        issues.append(f"跨文件重复实体: {dup_count} 条 ({dup_rate:.1%})")

    # 去重后实体数
    deduped_count = len(df.drop_duplicates(subset=key_cols))

    # 扣分：重复率每 10% 扣 5 分，上限 30 分
    deductions = min(30, (dup_rate * 100 // 10) * 5)
    score = max(0.0, 100.0 - deductions)

    return {
        "total_entities": len(df),
        "duplicate_count": int(dup_count),
        "duplicate_rate_pct": round(dup_rate * 100, 1),
        "deduped_entity_count": int(deduped_count),
        "duplicate_per_patient": {str(k): int(v) for k, v in dup_per_patient.items()},
        "issues": issues,
        "score": round(score, 1),
    }


# ---------------------------------------------------------------------------
# D. 语义一致性
# ---------------------------------------------------------------------------

def eval_semantic(df: pd.DataFrame) -> dict:
    issues = []
    deductions = 0.0

    lv = df[df["category"] == "LabValue"].copy()

    # D1: LabValue unit 覆盖率（增益类无单位是正常的，排除）
    gain_pattern = re.compile(r"增益|gain", re.IGNORECASE)
    lv_need_unit = lv[~lv["name"].str.contains(gain_pattern, na=False)]
    unit_missing = lv_need_unit["unit"].isna() | (lv_need_unit["unit"].astype(str).str.strip() == "")
    unit_missing_rate = unit_missing.mean() if len(lv_need_unit) > 0 else 0.0
    if unit_missing_rate > 0.1:
        issues.append(f"LabValue unit 缺失率（排除增益类）: {unit_missing_rate:.1%}")
        deductions += min(15, unit_missing_rate * 50)

    # D2: 数值异常值检测（IQR 方法，按 name 分组）
    outlier_details = []
    lv_numeric = lv.copy()
    lv_numeric["_num"] = pd.to_numeric(lv_numeric["value"], errors="coerce")
    lv_numeric = lv_numeric.dropna(subset=["_num"])

    for name, grp in lv_numeric.groupby("name"):
        if len(grp) < 4:
            continue
        q1, q3 = grp["_num"].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        outliers = grp[(grp["_num"] < q1 - 3 * iqr) | (grp["_num"] > q3 + 3 * iqr)]
        for _, row in outliers.iterrows():
            outlier_details.append({
                "patient_id": str(row["patient_id"]),
                "name": name,
                "value": row["value"],
                "unit": row.get("unit", ""),
            })

    if outlier_details:
        issues.append(f"数值异常值（IQR×3）: {len(outlier_details)} 条")
        deductions += min(10, len(outlier_details) * 0.5)

    # D3: Test.value 应为定性词，不应含纯数字
    test_df = df[df["category"] == "Test"].copy()
    test_numeric_val = test_df[test_df["value"].astype(str).str.fullmatch(r"[+-]?\d+\.?\d*", na=False)]
    if len(test_numeric_val) > 0:
        issues.append(f"Test.value 为纯数字（应为定性词）: {len(test_numeric_val)} 条")
        deductions += min(10, len(test_numeric_val) * 0.5)

    score = max(0.0, 100.0 - deductions)
    return {
        "labvalue_unit_missing_rate_pct": round(unit_missing_rate * 100, 1),
        "labvalue_unit_missing_count": int(unit_missing.sum()),
        "outlier_count": len(outlier_details),
        "outlier_details": outlier_details[:20],  # 最多展示 20 条
        "test_numeric_value_count": int(len(test_numeric_val)),
        "issues": issues,
        "score": round(score, 1),
    }


# ---------------------------------------------------------------------------
# E. 结构完整性（Finding ↔ LabValue 配对）
# ---------------------------------------------------------------------------

def eval_structure(df: pd.DataFrame) -> dict:
    issues = []
    deductions = 0.0

    finding = df[df["category"] == "Finding"].copy()
    lv = df[df["category"] == "LabValue"].copy()

    # 只检查"有速度值意义"的 Finding：即 LabValue 中存在以该体位名开头的慢相角速度条目
    # 静态位置试验、Dix-Hallpike试验（无速度）等不强制要求配对
    speed_lv_names = set(lv[lv["name"].str.contains("慢相角速度|角速度", na=False)]["name"].tolist())

    paired = 0
    unpaired_findings = []

    unique_findings = finding.drop_duplicates(subset=["patient_id", "name"])
    for _, row in unique_findings.iterrows():
        pid = row["patient_id"]
        pos_name = str(row["name"])
        patient_lv = lv[lv["patient_id"] == pid]

        # 该体位是否有对应 LabValue（name 包含体位名）
        match = patient_lv[patient_lv["name"].str.contains(re.escape(pos_name), na=False)]
        if len(match) > 0:
            paired += 1
        else:
            # 判断是否属于"应有速度值"的体位（同患者有任意以该体位名开头的速度 LabValue）
            has_speed_lv = any(pos_name in n for n in speed_lv_names)
            if has_speed_lv:
                unpaired_findings.append({
                    "patient_id": str(pid),
                    "finding_name": pos_name,
                    "finding_value": str(row.get("value", "")),
                })

    total_findings = max(len(unique_findings), 1)
    # 配对率基于"有速度值意义"的 Finding（已配对 + 未配对）
    speed_findings_total = paired + len(unpaired_findings)
    pair_rate = paired / max(speed_findings_total, 1)

    if pair_rate < 0.7 and speed_findings_total > 0:
        issues.append(f"有速度意义的 Finding 无对应 LabValue: {len(unpaired_findings)}/{speed_findings_total}")
        deductions += min(20, (1 - pair_rate) * 30)

    # Disease 实体数（诊断覆盖）
    disease_count = len(df[df["category"] == "Disease"])
    disease_per_patient = df[df["category"] == "Disease"].groupby("patient_id").size()
    patients_without_disease = df["patient_id"].nunique() - len(disease_per_patient)
    if patients_without_disease > 0:
        issues.append(f"{patients_without_disease} 个患者无 Disease 实体")
        deductions += patients_without_disease * 2

    score = max(0.0, 100.0 - deductions)
    return {
        "total_findings": int(len(finding)),
        "unique_findings": int(len(unique_findings)),
        "speed_findings_total": int(speed_findings_total),
        "findings_with_labvalue": int(paired),
        "finding_labvalue_pair_rate_pct": round(pair_rate * 100, 1),
        "unpaired_findings_count": len(unpaired_findings),
        "unpaired_findings_sample": unpaired_findings[:10],
        "disease_entity_count": int(disease_count),
        "patients_without_disease": int(patients_without_disease),
        "issues": issues,
        "score": round(score, 1),
    }


# ---------------------------------------------------------------------------
# 综合评分
# ---------------------------------------------------------------------------

WEIGHTS = {"A": 0.20, "B": 0.25, "C": 0.20, "D": 0.20, "E": 0.15}


def compute_overall(layer_scores: dict) -> dict:
    overall = sum(layer_scores[k] * WEIGHTS[k] for k in WEIGHTS)
    grade = "A" if overall >= 90 else "B" if overall >= 80 else "C" if overall >= 70 else "D"
    return {
        "layer_scores": layer_scores,
        "weights": WEIGHTS,
        "overall_score": round(overall, 1),
        "grade": grade,
    }


# ---------------------------------------------------------------------------
# 打印报告
# ---------------------------------------------------------------------------

def print_report(report: dict):
    print("\n" + "=" * 60)
    print("  抽取结果评估报告")
    print("=" * 60)

    labels = {
        "A": "覆盖完整性",
        "B": "格式合规性",
        "C": "去重质量",
        "D": "语义一致性",
        "E": "结构完整性",
    }

    for layer, label in labels.items():
        data = report[f"layer_{layer}"]
        score = data["score"]
        bar = "█" * int(score // 5) + "░" * (20 - int(score // 5))
        print(f"\n[{layer}] {label}  {bar}  {score:.1f}/100")
        for issue in data.get("issues", []):
            print(f"    ⚠  {issue}")
        if layer == "A":
            dist = data["category_distribution"]
            print(f"    实体总数: {data['total_entities']}  患者数: {data['total_patients']}")
            print(f"    类别分布: " + "  ".join(f"{k}:{v}" for k, v in dist.items()))
            ep = data["entities_per_patient"]
            print(f"    每患者实体: 均值={ep['mean']} 最小={ep['min']} 最大={ep['max']}")
        if layer == "C":
            print(f"    重复实体: {data['duplicate_count']} 条 ({data['duplicate_rate_pct']}%)")
            print(f"    去重后: {data['deduped_entity_count']} 条")
        if layer == "D":
            print(f"    LabValue unit 缺失: {data['labvalue_unit_missing_count']} 条 ({data['labvalue_unit_missing_rate_pct']}%)")
            print(f"    数值异常值: {data['outlier_count']} 条")
        if layer == "E":
            print(f"    Finding↔LabValue 配对率（速度类）: {data['finding_labvalue_pair_rate_pct']}%  ({data['findings_with_labvalue']}/{data['speed_findings_total']})")
            print(f"    Disease 实体数: {data['disease_entity_count']}")

    ov = report["overall"]
    print(f"\n{'=' * 60}")
    print(f"  综合得分: {ov['overall_score']:.1f} / 100   等级: {ov['grade']}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def evaluate(csv_path: str, out_path: str | None = None):
    df = pd.read_csv(csv_path)

    # 统一 patient_id 为字符串
    df["patient_id"] = df["patient_id"].astype(str)

    layer_A = eval_coverage(df)
    layer_B = eval_format(df)
    layer_C = eval_dedup(df)
    layer_D = eval_semantic(df)
    layer_E = eval_structure(df)

    layer_scores = {
        "A": layer_A["score"],
        "B": layer_B["score"],
        "C": layer_C["score"],
        "D": layer_D["score"],
        "E": layer_E["score"],
    }

    report = {
        "csv_path": csv_path,
        "layer_A": layer_A,
        "layer_B": layer_B,
        "layer_C": layer_C,
        "layer_D": layer_D,
        "layer_E": layer_E,
        "overall": compute_overall(layer_scores),
    }

    print_report(report)

    if out_path:
        Path(out_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"报告已保存: {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估医学实体抽取结果")
    parser.add_argument("csv", help="entities CSV 路径")
    parser.add_argument("--out", default=None, help="输出 JSON 报告路径（可选）")
    args = parser.parse_args()
    evaluate(args.csv, args.out)
