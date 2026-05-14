# -*- coding: utf-8 -*-
"""
STEP 7: Phenotyping Agent — 临床表型标注
==========================================
基于 AgentScope 框架，使用 Phenotyping Agent 实现：

输入：干净 + 有置信度的数据（来自 STEP 6）
输出：标准化表型标签

流程概述：
1. 规则匹配：根据已知的医学知识和诊断标准，初步判断可能的临床表型
2. 知识推理：利用医学知识库进行更深层次的关系推断，找到潜在的疾病关联
3. 模型预测：应用机器学习模型对病症进行预测，获取额外的证据支持
4. 决策融合：综合规则匹配、知识推理和模型预测的结果，做出最终的表型确认
"""

import asyncio
import json
import math
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

# ============================================================
# 配置区 —— 请根据实际情况修改
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = (
    os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or "https://api.openai.com/v1"
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

# 数据文件路径（来自 STEP 6）
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_CSV = os.path.join(DATA_DIR, "input_cleaned_38.csv")
FILTERED_CSV = os.path.join(
    DATA_DIR, "input_cleaned_38_filtered_20260302_142104.csv",
)
SELECTION_REPORT_JSON = os.path.join(
    DATA_DIR, "input_cleaned_38_selection_report_20260302_142104.json",
)

# Step6 输出（一致性报告）
STEP6_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output",
)

# Step7 输出
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_step7",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        return df
    for col in ("subject_id", "hadm_id", "record_index", "id"):
        if col in df.columns:
            return df.rename(columns={col: "patient_id"})
    return df


# ============================================================
# 标准表型定义（前庭功能检查相关临床表型）
# ============================================================
PHENOTYPE_DEFINITIONS = {
    "BPPV_posterior": {
        "name": "后半规管良性阵发性位置性眩晕(pc-BPPV)",
        "icd10": "H81.1",
        "description": "Dix-Hallpike试验阳性，可见扭转+上跳眼震，有潜伏期和疲劳性",
    },
    "BPPV_horizontal": {
        "name": "水平半规管良性阵发性位置性眩晕(hc-BPPV)",
        "icd10": "H81.1",
        "description": "Roll-test阳性，可见水平方向变向性眼震",
    },
    "BPPV_anterior": {
        "name": "前半规管良性阵发性位置性眩晕(ac-BPPV)",
        "icd10": "H81.1",
        "description": "深悬头位试验阳性，可见下跳+扭转眼震",
    },
    "unilateral_vestibular_hypofunction": {
        "name": "单侧前庭功能减退",
        "icd10": "H83.2",
        "description": "单侧温度试验反应显著低于对侧（CP值>25%），自发眼震可能存在",
    },
    "bilateral_vestibular_hypofunction": {
        "name": "双侧前庭功能减退",
        "icd10": "H83.2",
        "description": "双侧温度试验反应均减弱，双侧总慢相角速度之和<20度/秒",
    },
    "central_vestibular_disorder": {
        "name": "中枢性前庭功能障碍",
        "icd10": "H81.4",
        "description": "视跟踪I型，方向改变性眼震，视动眼震异常等中枢性体征",
    },
    "vestibular_neuritis": {
        "name": "前庭神经炎",
        "icd10": "H81.2",
        "description": "急性单侧前庭功能减退，自发水平-扭转眼震，温度试验单侧减弱",
    },
    "meniere_disease": {
        "name": "梅尼埃病",
        "icd10": "H81.0",
        "description": "反复发作性眩晕伴波动性听力下降，温度试验可异常",
    },
    "normal_vestibular": {
        "name": "前庭功能正常",
        "icd10": "Z03.3",
        "description": "所有前庭功能检查结果在正常范围内",
    },
    "positional_nystagmus": {
        "name": "位置性眼震(待分类)",
        "icd10": "H81.3",
        "description": "静态位置试验中发现位置性眼震，需进一步鉴别诊断",
    },
}


# ============================================================
# 辅助函数
# ============================================================

def _extract_numeric(text: str) -> float | None:
    """从文本中提取数值（如 '10度/秒' → 10.0）。"""
    if not text or pd.isna(text):
        return None
    m = re.search(r"(\d+\.?\d*)", str(text))
    return float(m.group(1)) if m else None


def _has_nystagmus(text: str) -> bool:
    """判断是否存在眼震（非正常）。"""
    if not text or pd.isna(text):
        return False
    s = str(text).strip()
    if s in ("未见眼震", "未做", "未见异常", "未见明显异常", ""):
        return False
    if "未见" in s:
        return False
    return True


def _get_nystagmus_type(text: str) -> str:
    """从眼震描述中提取眼震类型。"""
    if not text or pd.isna(text):
        return "无"
    s = str(text).strip()
    if not _has_nystagmus(s):
        return "无"
    types = []
    if "扭转" in s:
        if "顺时针" in s:
            types.append("顺时针扭转")
        elif "逆时针" in s:
            types.append("逆时针扭转")
        else:
            types.append("扭转")
    if "上跳" in s:
        types.append("上跳")
    if "下跳" in s:
        types.append("下跳")
    if "水平" in s:
        if "左" in s:
            types.append("左水平")
        elif "右" in s:
            types.append("右水平")
        else:
            types.append("水平")
    return "+".join(types) if types else "眼震(未分类)"


def _compute_cp_value(
    r50: float | None, l50: float | None,
    r24: float | None, l24: float | None,
) -> dict[str, Any]:
    """计算前庭双温试验的CP值（Canal Paresis）。

    CP = |((R50+R24)-(L50+L24)) / (R50+R24+L50+L24)| × 100%
    """
    vals = [r50, l50, r24, l24]
    if any(v is None for v in vals):
        return {"cp_value": None, "total_response": None, "side": None}

    right_sum = r50 + r24
    left_sum = l50 + l24
    total = right_sum + left_sum

    if total == 0:
        return {"cp_value": None, "total_response": total, "side": None}

    cp = abs(right_sum - left_sum) / total * 100
    weaker_side = "右侧" if right_sum < left_sum else "左侧"

    return {
        "cp_value": round(cp, 1),
        "total_response": round(total, 1),
        "side": weaker_side if cp > 25 else "对称",
        "right_sum": round(right_sum, 1),
        "left_sum": round(left_sum, 1),
    }


def _load_step6_report() -> dict | None:
    """加载 Step6 的一致性报告。"""
    for fname in sorted(os.listdir(STEP6_OUTPUT_DIR), reverse=True):
        if fname.endswith(".json") and "consistency" in fname:
            path = os.path.join(STEP6_OUTPUT_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


# ============================================================
# 工具函数定义
# ============================================================

def load_patient_data() -> ToolResponse:
    """加载所有患者数据（含置信度信息），返回按患者分组的数据概览。
    这是进行表型标注的基础数据。"""
    df = _read_data(RAW_CSV)
    step6_report = _load_step6_report()

    output_lines = ["=== 患者数据概览（含置信度） ==="]
    output_lines.append(
        f"共 {df['patient_id'].nunique()} 位患者, {len(df)} 条记录\n"
    )

    if step6_report:
        output_lines.append(
            f"Step6 整体平均置信度: "
            f"{step6_report.get('summary', {})}"
        )
        output_lines.append("")

    for pid in sorted(df["patient_id"].unique()):
        rows = df[df["patient_id"] == pid]
        output_lines.append(f"--- 患者 {pid} ({len(rows)} 条记录) ---")
        for _, row in rows.iterrows():
            fname = row.get("filename", "N/A")
            output_lines.append(f"  [{fname}]")
            for col in df.columns:
                if col not in ["patient_id", "folder_category", "filename"]:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        col_short = col.replace("Extracted_Test__", "T:")
                        col_short = col_short.replace(
                            "Extracted_Symptom__", "S:",
                        )
                        output_lines.append(f"    {col_short} = {val}")
        output_lines.append("")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def rule_based_matching() -> ToolResponse:
    """【规则匹配】根据已知的医学知识和诊断标准，初步判断每位患者可能的临床表型。

    诊断规则包括：
    - BPPV：基于 Dix-Hallpike/Roll-test/深悬头位试验结果
    - 单侧前庭功能减退：基于 CP 值计算
    - 中枢性病变：基于视跟踪分型和眼震模式
    - 正常：所有检查结果均在正常范围内
    """
    df = _read_data(RAW_CSV)
    results = {}

    for pid in sorted(df["patient_id"].unique()):
        rows = df[df["patient_id"] == pid]
        patient_result = {
            "patient_id": pid,
            "matched_phenotypes": [],
            "evidence": [],
            "rule_scores": {},
        }

        # 合并该患者所有记录的信息（取最信息丰富的）
        all_data = {}
        for _, row in rows.iterrows():
            for col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    if col not in all_data or all_data[col] == "未做":
                        all_data[col] = str(val).strip()

        # ---------- 规则 1: 后半规管 BPPV ----------
        score_bppv_pc = 0.0
        evidence_bppv_pc = []

        # Dix-Hallpike 相关
        dh_cols = [
            c for c in df.columns
            if "Dix-Hallpike" in c and "坐起后" in c
        ]
        for col in dh_cols:
            val = all_data.get(col, "")
            if _has_nystagmus(val):
                ntype = _get_nystagmus_type(val)
                if "扭转" in ntype and "上跳" in ntype:
                    score_bppv_pc += 0.4
                    evidence_bppv_pc.append(
                        f"{col}: {val} → 典型pc-BPPV眼震模式"
                    )
                elif "扭转" in ntype or "上跳" in ntype:
                    score_bppv_pc += 0.2
                    evidence_bppv_pc.append(f"{col}: {val} → 部分符合")

        # Dix-Hallpike悬头位
        dh_head_cols = [
            c for c in df.columns
            if "Dix-Hallpike" in c and "悬头位" in c
            and "坐起后" not in c
        ]
        for col in dh_head_cols:
            val = all_data.get(col, "")
            if _has_nystagmus(val):
                ntype = _get_nystagmus_type(val)
                if "扭转" in ntype:
                    score_bppv_pc += 0.2
                    evidence_bppv_pc.append(f"{col}: {val}")

        if score_bppv_pc > 0:
            score_bppv_pc = min(score_bppv_pc, 1.0)
            patient_result["rule_scores"]["BPPV_posterior"] = round(
                score_bppv_pc, 2,
            )
            patient_result["evidence"].extend(evidence_bppv_pc)

        # ---------- 规则 2: 水平半规管 BPPV ----------
        score_bppv_hc = 0.0
        evidence_bppv_hc = []

        roll_cols = [c for c in df.columns if "Roll-test" in c]
        roll_nystagmus = {}
        for col in roll_cols:
            val = all_data.get(col, "")
            if _has_nystagmus(val):
                ntype = _get_nystagmus_type(val)
                roll_nystagmus[col] = {"val": val, "type": ntype}

        if len(roll_nystagmus) >= 2:
            has_horizontal = any(
                "水平" in v["type"] for v in roll_nystagmus.values()
            )
            if has_horizontal:
                score_bppv_hc += 0.5
                for col, info in roll_nystagmus.items():
                    evidence_bppv_hc.append(
                        f"{col}: {info['val']} → 水平方向眼震"
                    )

        if score_bppv_hc > 0:
            patient_result["rule_scores"]["BPPV_horizontal"] = round(
                score_bppv_hc, 2,
            )
            patient_result["evidence"].extend(evidence_bppv_hc)

        # ---------- 规则 3: 前半规管 BPPV ----------
        score_bppv_ac = 0.0
        evidence_bppv_ac = []

        deep_cols = [c for c in df.columns if "深悬头位" in c]
        for col in deep_cols:
            val = all_data.get(col, "")
            if _has_nystagmus(val):
                ntype = _get_nystagmus_type(val)
                if "下跳" in ntype:
                    score_bppv_ac += 0.3
                    evidence_bppv_ac.append(
                        f"{col}: {val} → 下跳眼震(ac-BPPV特征)"
                    )

        if score_bppv_ac > 0:
            patient_result["rule_scores"]["BPPV_anterior"] = round(
                min(score_bppv_ac, 1.0), 2,
            )
            patient_result["evidence"].extend(evidence_bppv_ac)

        # ---------- 规则 4: 单侧/双侧前庭功能减退 ----------
        r50 = _extract_numeric(
            all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧50°C", ""),
        )
        l50 = _extract_numeric(
            all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧50°C", ""),
        )
        r24 = _extract_numeric(
            all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧24°C", ""),
        )
        l24 = _extract_numeric(
            all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧24°C", ""),
        )

        caloric = _compute_cp_value(r50, l50, r24, l24)
        if caloric["cp_value"] is not None:
            if caloric["cp_value"] > 25:
                score_uvh = min(caloric["cp_value"] / 100 + 0.3, 1.0)
                patient_result["rule_scores"][
                    "unilateral_vestibular_hypofunction"
                ] = round(score_uvh, 2)
                patient_result["evidence"].append(
                    f"CP值={caloric['cp_value']}%（>25%），"
                    f"{caloric['side']}减弱，"
                    f"右侧总和={caloric.get('right_sum')}，"
                    f"左侧总和={caloric.get('left_sum')}"
                )

            if caloric["total_response"] is not None:
                if caloric["total_response"] < 20:
                    patient_result["rule_scores"][
                        "bilateral_vestibular_hypofunction"
                    ] = 0.7
                    patient_result["evidence"].append(
                        f"双侧总反应={caloric['total_response']}度/秒（<20），"
                        f"提示双侧前庭功能减退"
                    )

        # ---------- 规则 5: 中枢性前庭障碍 ----------
        score_central = 0.0
        evidence_central = []

        tracking = all_data.get("Extracted_Test__视跟踪水平方向分型", "")
        if "I型" in tracking and "II" not in tracking and "III" not in tracking:
            score_central += 0.4
            evidence_central.append(f"视跟踪分型={tracking}（I型提示中枢性病变）")

        spontaneous = all_data.get("Extracted_Symptom__自发眼震", "")
        if _has_nystagmus(spontaneous):
            ntype = _get_nystagmus_type(spontaneous)
            if "垂直" in ntype or "下跳" in ntype or "上跳" in ntype:
                score_central += 0.3
                evidence_central.append(
                    f"自发眼震: {spontaneous} → 垂直性自发眼震提示中枢性"
                )

        optokinetic = all_data.get("Extracted_Test__视动眼震水平方向", "")
        if optokinetic and "异常" in optokinetic:
            score_central += 0.3
            evidence_central.append(f"视动眼震: {optokinetic}")

        if score_central > 0:
            patient_result["rule_scores"]["central_vestibular_disorder"] = (
                round(min(score_central, 1.0), 2)
            )
            patient_result["evidence"].extend(evidence_central)

        # ---------- 规则 6: 位置性眼震 ----------
        static_cols = [
            c for c in df.columns
            if "静态位置试验眼震" in c and "Symptom" in c
        ]
        has_positional = False
        for col in static_cols:
            val = all_data.get(col, "")
            if _has_nystagmus(val):
                has_positional = True
                patient_result["evidence"].append(
                    f"位置性眼震: {col}={val}"
                )
        if has_positional:
            patient_result["rule_scores"]["positional_nystagmus"] = 0.5

        # ---------- 规则 7: 前庭功能正常 ----------
        # 如果没有任何阳性发现
        if not patient_result["rule_scores"]:
            patient_result["rule_scores"]["normal_vestibular"] = 0.8
            patient_result["evidence"].append("所有检查结果未见明显异常")

        # 确定主匹配表型
        if patient_result["rule_scores"]:
            best = max(
                patient_result["rule_scores"],
                key=patient_result["rule_scores"].get,
            )
            patient_result["matched_phenotypes"].append(best)

        results[str(pid)] = patient_result

    # 生成输出
    output_lines = ["=== 规则匹配结果 ===\n"]
    for pid, res in results.items():
        output_lines.append(f"--- 患者 {pid} ---")
        output_lines.append(f"  匹配表型: {res['matched_phenotypes']}")
        output_lines.append("  规则评分:")
        for pheno, score in sorted(
            res["rule_scores"].items(),
            key=lambda x: -x[1],
        ):
            name = PHENOTYPE_DEFINITIONS.get(pheno, {}).get("name", pheno)
            output_lines.append(f"    {name}: {score:.2f}")
        output_lines.append("  证据:")
        for e in res["evidence"]:
            output_lines.append(f"    • {e}")
        output_lines.append("")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def knowledge_reasoning() -> ToolResponse:
    """【知识推理】利用医学知识库进行更深层次的关系推断，
    找到潜在的疾病关联。将检测指标与疾病之间的知识图谱关系进行推理。

    知识库包括：
    - 前庭功能检查各指标的临床意义
    - 不同前庭疾病的鉴别诊断要点
    - 眼震模式与病变部位的对应关系
    """
    df = _read_data(RAW_CSV)

    # 构建知识推理结果
    knowledge_base = {
        "眼震模式→病变部位": {
            "扭转+上跳眼震": "后半规管受累（典型BPPV）",
            "下跳眼震": "前半规管或小脑蚓部病变",
            "水平方向变向性眼震": "水平半规管受累",
            "纯垂直眼震": "脑干或小脑病变（中枢性）",
            "顺时针/逆时针扭转眼震": "耳石器或前庭核病变",
        },
        "温度试验→前庭功能": {
            "CP>25%": "单侧半规管功能低下",
            "CP>50%": "严重的单侧半规管麻痹",
            "双侧总反应<20度/秒": "双侧前庭功能减退",
            "总反应正常+CP正常": "外周前庭功能正常",
        },
        "视跟踪分型→中枢性": {
            "I型": "高度提示中枢性病变（脑干/小脑）",
            "II型": "轻度异常，可见于中枢或外周",
            "III型": "正常或轻度异常",
        },
        "鉴别诊断要点": {
            "BPPV vs 中枢性位置性眩晕":
                "BPPV有潜伏期和疲劳性，中枢性无潜伏期无疲劳性",
            "前庭神经炎 vs 脑卒中":
                "前庭神经炎HIT阳性，脑卒中HIT阴性",
            "梅尼埃病 vs BPPV":
                "梅尼埃病有波动性听力下降和耳鸣",
        },
    }

    results = {}
    for pid in sorted(df["patient_id"].unique()):
        rows = df[df["patient_id"] == pid]
        inferences = []

        # 收集该患者的全部数据
        all_data = {}
        for _, row in rows.iterrows():
            for col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    if col not in all_data or all_data[col] == "未做":
                        all_data[col] = str(val).strip()

        # 推理1: 眼震模式分析
        nystagmus_patterns = []
        for col, val in all_data.items():
            if _has_nystagmus(val) and col != "patient_id":
                ntype = _get_nystagmus_type(val)
                if ntype != "无":
                    nystagmus_patterns.append(
                        {"field": col, "value": val, "type": ntype},
                    )

        if nystagmus_patterns:
            # 分析眼震模式
            has_torsional_up = any(
                "扭转" in p["type"] and "上跳" in p["type"]
                for p in nystagmus_patterns
            )
            has_down_beat = any(
                "下跳" in p["type"] for p in nystagmus_patterns
            )
            has_horizontal = any(
                "水平" in p["type"] for p in nystagmus_patterns
            )

            if has_torsional_up:
                inferences.append({
                    "inference": "扭转+上跳眼震模式提示后半规管受累",
                    "related_disease": "pc-BPPV",
                    "confidence": 0.8,
                    "knowledge_source": "眼震模式→病变部位",
                })
            if has_down_beat:
                inferences.append({
                    "inference": "下跳眼震提示前半规管或中枢病变可能",
                    "related_disease": "ac-BPPV 或 中枢性前庭障碍",
                    "confidence": 0.6,
                    "knowledge_source": "眼震模式→病变部位",
                })
            if has_horizontal:
                inferences.append({
                    "inference": "水平方向眼震提示水平半规管受累",
                    "related_disease": "hc-BPPV 或 前庭神经炎",
                    "confidence": 0.5,
                    "knowledge_source": "眼震模式→病变部位",
                })

        # 推理2: 温度试验分析
        r50 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧50°C", "",
            ),
        )
        l50 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧50°C", "",
            ),
        )
        r24 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧24°C", "",
            ),
        )
        l24 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧24°C", "",
            ),
        )
        caloric = _compute_cp_value(r50, l50, r24, l24)

        if caloric["cp_value"] is not None:
            if caloric["cp_value"] > 25:
                inferences.append({
                    "inference": (
                        f"CP值={caloric['cp_value']}%，"
                        f"{caloric['side']}半规管功能减弱"
                    ),
                    "related_disease": "单侧前庭功能减退/前庭神经炎",
                    "confidence": 0.7,
                    "knowledge_source": "温度试验→前庭功能",
                })

                # 进一步推理：如果CP>25%且有自发眼震
                spontaneous = all_data.get("Extracted_Symptom__自发眼震", "")
                if _has_nystagmus(spontaneous):
                    inferences.append({
                        "inference": "CP值异常+自发眼震，提示急性前庭损伤",
                        "related_disease": "前庭神经炎",
                        "confidence": 0.65,
                        "knowledge_source": "鉴别诊断要点",
                    })

        # 推理3: 视跟踪分型
        tracking = all_data.get("Extracted_Test__视跟踪水平方向分型", "")
        if "I型" in tracking and "II" not in tracking and "III" not in tracking:
            inferences.append({
                "inference": "视跟踪I型高度提示中枢性前庭病变",
                "related_disease": "中枢性前庭功能障碍",
                "confidence": 0.75,
                "knowledge_source": "视跟踪分型→中枢性",
            })

        # 推理4: 综合推断
        if not inferences:
            inferences.append({
                "inference": "未发现明显异常指标，前庭功能可能正常",
                "related_disease": "前庭功能正常",
                "confidence": 0.7,
                "knowledge_source": "排除法",
            })

        results[str(pid)] = inferences

    # 生成输出
    output_lines = ["=== 知识推理结果 ===\n"]
    output_lines.append(
        "基于医学知识库进行深层次疾病关联推理:\n"
    )
    for pid, infers in results.items():
        output_lines.append(f"--- 患者 {pid} ---")
        for inf in infers:
            output_lines.append(f"  推理: {inf['inference']}")
            output_lines.append(f"    关联疾病: {inf['related_disease']}")
            output_lines.append(f"    置信度: {inf['confidence']:.2f}")
            output_lines.append(f"    知识来源: {inf['knowledge_source']}")
        output_lines.append("")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def model_prediction() -> ToolResponse:
    """【模型预测】应用基于特征工程的评分模型对每位患者的病症进行预测，
    为每种可能的表型计算预测概率，获取额外的证据支持。

    采用加权特征评分方法，基于关键指标组合进行量化评估。"""
    df = _read_data(RAW_CSV)

    results = {}
    for pid in sorted(df["patient_id"].unique()):
        rows = df[df["patient_id"] == pid]

        # 提取特征
        features = {}
        all_data = {}
        for _, row in rows.iterrows():
            for col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    if col not in all_data or all_data[col] == "未做":
                        all_data[col] = str(val).strip()

        # 特征 1: 眼震严重度评分 (0-1)
        nystagmus_count = 0
        nystagmus_speed_sum = 0.0
        for col, val in all_data.items():
            if _has_nystagmus(val):
                nystagmus_count += 1
                spd = _extract_numeric(val)
                if spd is not None:
                    nystagmus_speed_sum += spd
        features["nystagmus_severity"] = min(
            nystagmus_count / 10.0, 1.0,
        )
        features["nystagmus_speed_avg"] = (
            nystagmus_speed_sum / max(nystagmus_count, 1)
        )

        # 特征 2: 温度试验异常度
        r50 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧50°C", "",
            ),
        )
        l50 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧50°C", "",
            ),
        )
        r24 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧24°C", "",
            ),
        )
        l24 = _extract_numeric(
            all_data.get(
                "Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧24°C", "",
            ),
        )
        caloric = _compute_cp_value(r50, l50, r24, l24)
        features["cp_value"] = caloric.get("cp_value", 0) or 0
        features["total_caloric"] = caloric.get("total_response", 0) or 0

        # 特征 3: 视跟踪评分
        tracking = all_data.get("Extracted_Test__视跟踪水平方向分型", "")
        tracking_score = 0
        if "I型" in tracking and "II" not in tracking:
            tracking_score = 1.0
        elif "II型" in tracking:
            tracking_score = 0.3
        elif "III型" in tracking:
            tracking_score = 0.1
        features["tracking_abnormality"] = tracking_score

        # 特征 4: 左右眼不对称度
        left_speed = _extract_numeric(
            all_data.get("Extracted_Test__视动眼震左眼速度", ""),
        )
        right_speed = _extract_numeric(
            all_data.get("Extracted_Test__视动眼震右眼速度", ""),
        )
        if left_speed and right_speed:
            ratio = abs(left_speed - right_speed) / max(
                left_speed, right_speed, 0.01,
            )
            features["eye_asymmetry"] = ratio
        else:
            features["eye_asymmetry"] = 0

        # 特征 5: Dix-Hallpike阳性指标
        dh_positive = 0
        for col in df.columns:
            if "Dix-Hallpike" in col:
                if _has_nystagmus(all_data.get(col, "")):
                    dh_positive += 1
        features["dh_positive_count"] = dh_positive

        # 特征 6: Roll-test 阳性指标
        roll_positive = 0
        for col in df.columns:
            if "Roll-test" in col:
                if _has_nystagmus(all_data.get(col, "")):
                    roll_positive += 1
        features["roll_positive_count"] = roll_positive

        # 特征 7: 位置性眼震指标
        positional_positive = 0
        for col in df.columns:
            if "静态位置试验" in col:
                if _has_nystagmus(all_data.get(col, "")):
                    positional_positive += 1
        features["positional_count"] = positional_positive

        # ---------- 基于特征的多表型预测 ----------
        predictions = {}

        # BPPV_posterior 预测
        p_bppv_pc = (
            0.35 * min(features["dh_positive_count"] / 3.0, 1.0)
            + 0.25 * features["nystagmus_severity"]
            + 0.20 * (1.0 if features["cp_value"] < 25 else 0.3)
            + 0.20 * (1.0 - features["tracking_abnormality"])
        )
        predictions["BPPV_posterior"] = round(p_bppv_pc, 3)

        # BPPV_horizontal 预测
        p_bppv_hc = (
            0.40 * min(features["roll_positive_count"] / 3.0, 1.0)
            + 0.25 * features["nystagmus_severity"]
            + 0.15 * (1.0 if features["cp_value"] < 25 else 0.3)
            + 0.20 * (1.0 - features["tracking_abnormality"])
        )
        predictions["BPPV_horizontal"] = round(p_bppv_hc, 3)

        # BPPV_anterior 预测
        deep_positive = sum(
            1 for col in df.columns
            if "深悬头位" in col and _has_nystagmus(all_data.get(col, ""))
        )
        p_bppv_ac = (
            0.40 * min(deep_positive / 2.0, 1.0)
            + 0.25 * features["nystagmus_severity"]
            + 0.15 * (1.0 if features["cp_value"] < 25 else 0.3)
            + 0.20 * (1.0 - features["tracking_abnormality"])
        )
        predictions["BPPV_anterior"] = round(p_bppv_ac, 3)

        # 单侧前庭功能减退
        p_uvh = (
            0.50 * min(features["cp_value"] / 50.0, 1.0)
            + 0.20 * features["nystagmus_severity"]
            + 0.15 * features["eye_asymmetry"]
            + 0.15 * (1.0 - features["tracking_abnormality"])
        )
        predictions["unilateral_vestibular_hypofunction"] = round(p_uvh, 3)

        # 中枢性前庭障碍
        p_central = (
            0.40 * features["tracking_abnormality"]
            + 0.25 * features["nystagmus_severity"]
            + 0.20 * features["eye_asymmetry"]
            + 0.15 * min(features["positional_count"] / 5.0, 1.0)
        )
        predictions["central_vestibular_disorder"] = round(p_central, 3)

        # 正常
        p_normal = max(
            0,
            1.0
            - features["nystagmus_severity"] * 0.4
            - features["tracking_abnormality"] * 0.3
            - min(features["cp_value"] / 50.0, 1.0) * 0.3,
        )
        predictions["normal_vestibular"] = round(p_normal, 3)

        results[str(pid)] = {
            "features": features,
            "predictions": predictions,
        }

    # 生成输出
    output_lines = ["=== 模型预测结果 ===\n"]
    for pid, res in results.items():
        output_lines.append(f"--- 患者 {pid} ---")
        output_lines.append("  关键特征:")
        for k, v in res["features"].items():
            output_lines.append(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        output_lines.append("  预测概率:")
        sorted_pred = sorted(
            res["predictions"].items(), key=lambda x: -x[1],
        )
        for pheno, prob in sorted_pred:
            name = PHENOTYPE_DEFINITIONS.get(pheno, {}).get("name", pheno)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            output_lines.append(f"    {name}: {prob:.3f} |{bar}|")
        output_lines.append("")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def decision_fusion() -> ToolResponse:
    """【决策融合】综合规则匹配、知识推理和模型预测三方面的结果，
    通过加权投票机制做出最终的表型确认。

    融合策略：
    - 规则匹配权重: 0.40（基于明确的诊断标准）
    - 知识推理权重: 0.30（基于医学知识库关联）
    - 模型预测权重: 0.30（基于特征量化评分）

    输出每位患者的最终标准化表型标签。"""
    df = _read_data(RAW_CSV)
    step6_report = _load_step6_report()

    # 权重配置
    W_RULE = 0.40
    W_KNOWLEDGE = 0.30
    W_MODEL = 0.30

    final_results = {}

    for pid in sorted(df["patient_id"].unique()):
        rows = df[df["patient_id"] == pid]
        all_data = {}
        for _, row in rows.iterrows():
            for col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    if col not in all_data or all_data[col] == "未做":
                        all_data[col] = str(val).strip()

        # ---- 获取规则匹配得分 ----
        rule_scores = _get_rule_scores_for_patient(pid, df)

        # ---- 获取知识推理得分 ----
        knowledge_scores = _get_knowledge_scores_for_patient(pid, df, all_data)

        # ---- 获取模型预测得分 ----
        model_scores = _get_model_scores_for_patient(pid, df, all_data)

        # ---- 加权融合 ----
        all_phenotypes = set(
            list(rule_scores.keys())
            + list(knowledge_scores.keys())
            + list(model_scores.keys())
        )

        fused_scores = {}
        for pheno in all_phenotypes:
            r = rule_scores.get(pheno, 0)
            k = knowledge_scores.get(pheno, 0)
            m = model_scores.get(pheno, 0)
            fused = W_RULE * r + W_KNOWLEDGE * k + W_MODEL * m
            fused_scores[pheno] = {
                "fused_score": round(fused, 3),
                "rule_score": round(r, 3),
                "knowledge_score": round(k, 3),
                "model_score": round(m, 3),
            }

        # 排序，选出最终表型
        sorted_phenotypes = sorted(
            fused_scores.items(),
            key=lambda x: -x[1]["fused_score"],
        )

        primary_phenotype = sorted_phenotypes[0][0] if sorted_phenotypes else "unknown"
        primary_score = sorted_phenotypes[0][1]["fused_score"] if sorted_phenotypes else 0

        # 次要表型（融合得分 > 0.3 且不是主表型）
        secondary = [
            (p, s) for p, s in sorted_phenotypes[1:]
            if s["fused_score"] > 0.3
        ]

        final_results[str(pid)] = {
            "primary_phenotype": primary_phenotype,
            "primary_phenotype_name": PHENOTYPE_DEFINITIONS.get(
                primary_phenotype, {},
            ).get("name", primary_phenotype),
            "primary_score": primary_score,
            "icd10": PHENOTYPE_DEFINITIONS.get(
                primary_phenotype, {},
            ).get("icd10", "N/A"),
            "secondary_phenotypes": [
                {
                    "phenotype": p,
                    "name": PHENOTYPE_DEFINITIONS.get(p, {}).get("name", p),
                    "score": s["fused_score"],
                }
                for p, s in secondary
            ],
            "all_scores": fused_scores,
        }

    # 生成输出
    output_lines = [
        "=" * 60,
        "  决策融合结果 — 最终标准化表型标签",
        "=" * 60,
        f"  融合权重: 规则={W_RULE}, 知识={W_KNOWLEDGE}, 模型={W_MODEL}\n",
    ]

    for pid, res in final_results.items():
        output_lines.append(f"╔══ 患者 {pid} {'═' * 40}")
        output_lines.append(
            f"║ 主要表型: {res['primary_phenotype_name']}"
        )
        output_lines.append(
            f"║ ICD-10:   {res['icd10']}"
        )
        output_lines.append(
            f"║ 融合置信度: {res['primary_score']:.3f}"
        )
        if res["secondary_phenotypes"]:
            output_lines.append("║ 次要表型:")
            for sp in res["secondary_phenotypes"]:
                output_lines.append(
                    f"║   - {sp['name']}: {sp['score']:.3f}"
                )
        output_lines.append("║ 各方法评分:")
        for pheno, scores in sorted(
            res["all_scores"].items(),
            key=lambda x: -x[1]["fused_score"],
        ):
            name = PHENOTYPE_DEFINITIONS.get(pheno, {}).get("name", pheno)
            output_lines.append(
                f"║   {name}: "
                f"融合={scores['fused_score']:.3f} "
                f"(规则={scores['rule_score']:.3f}, "
                f"知识={scores['knowledge_score']:.3f}, "
                f"模型={scores['model_score']:.3f})"
            )
        output_lines.append(f"╚{'═' * 50}\n")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def save_phenotyping_report(report_content: str) -> ToolResponse:
    """将最终的表型标注报告保存到文件。

    Args:
        report_content (str):
            要保存的报告内容（纯文本格式）。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 文本版
    report_path = os.path.join(
        OUTPUT_DIR, f"phenotyping_report_{timestamp}.txt",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    # JSON 结构化版（自动生成）
    json_path = os.path.join(
        OUTPUT_DIR, f"phenotyping_report_{timestamp}.json",
    )

    df = _read_data(RAW_CSV)
    # 重新执行融合以获取结构化结果
    structured = _generate_structured_output(df)
    structured["timestamp"] = datetime.now().isoformat()
    structured["step"] = "STEP7_Phenotyping_表型标注"
    structured["report_text"] = report_content

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    # CSV 标签版（方便后续使用）
    csv_path = os.path.join(
        OUTPUT_DIR, f"phenotype_labels_{timestamp}.csv",
    )
    label_rows = []
    for pid_str, info in structured.get("patients", {}).items():
        label_rows.append({
            "patient_id": int(pid_str),
            "primary_phenotype": info.get("primary_phenotype", ""),
            "primary_phenotype_name": info.get("primary_phenotype_name", ""),
            "icd10": info.get("icd10", ""),
            "confidence": info.get("primary_score", 0),
            "secondary_phenotypes": "; ".join(
                f"{s['name']}({s['score']:.3f})"
                for s in info.get("secondary_phenotypes", [])
            ),
        })
    if label_rows:
        pd.DataFrame(label_rows).to_csv(csv_path, index=False)

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"表型标注报告已保存:\n"
                    f"  文本版: {report_path}\n"
                    f"  JSON版: {json_path}\n"
                    f"  标签CSV: {csv_path}"
                ),
            ),
        ],
    )


# ============================================================
# 决策融合的内部辅助函数（供 decision_fusion 复用）
# ============================================================

def _get_rule_scores_for_patient(
    pid: int, df: pd.DataFrame,
) -> dict[str, float]:
    """内部：获取某位患者的规则匹配得分。"""
    rows = df[df["patient_id"] == pid]
    all_data = {}
    for _, row in rows.iterrows():
        for col in df.columns:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                if col not in all_data or all_data[col] == "未做":
                    all_data[col] = str(val).strip()

    scores = {}

    # BPPV_posterior
    s = 0.0
    for col in df.columns:
        if "Dix-Hallpike" in col and "坐起后" in col:
            val = all_data.get(col, "")
            if _has_nystagmus(val):
                ntype = _get_nystagmus_type(val)
                if "扭转" in ntype and "上跳" in ntype:
                    s += 0.4
                elif "扭转" in ntype or "上跳" in ntype:
                    s += 0.2
    for col in df.columns:
        if "Dix-Hallpike" in col and "悬头位" in col and "坐起后" not in col:
            val = all_data.get(col, "")
            if _has_nystagmus(val) and "扭转" in _get_nystagmus_type(val):
                s += 0.2
    if s > 0:
        scores["BPPV_posterior"] = min(s, 1.0)

    # BPPV_horizontal
    s = 0.0
    for col in df.columns:
        if "Roll-test" in col:
            val = all_data.get(col, "")
            if _has_nystagmus(val) and "水平" in _get_nystagmus_type(val):
                s += 0.25
    if s > 0:
        scores["BPPV_horizontal"] = min(s, 1.0)

    # BPPV_anterior
    s = 0.0
    for col in df.columns:
        if "深悬头位" in col:
            val = all_data.get(col, "")
            if _has_nystagmus(val) and "下跳" in _get_nystagmus_type(val):
                s += 0.3
    if s > 0:
        scores["BPPV_anterior"] = min(s, 1.0)

    # 单侧前庭功能减退
    r50 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧50°C", ""),
    )
    l50 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧50°C", ""),
    )
    r24 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧24°C", ""),
    )
    l24 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧24°C", ""),
    )
    cal = _compute_cp_value(r50, l50, r24, l24)
    if cal["cp_value"] is not None and cal["cp_value"] > 25:
        scores["unilateral_vestibular_hypofunction"] = min(
            cal["cp_value"] / 100 + 0.3, 1.0,
        )

    # 中枢性
    s = 0.0
    tracking = all_data.get("Extracted_Test__视跟踪水平方向分型", "")
    if "I型" in tracking and "II" not in tracking and "III" not in tracking:
        s += 0.5
    spontaneous = all_data.get("Extracted_Symptom__自发眼震", "")
    if _has_nystagmus(spontaneous):
        s += 0.3
    if s > 0:
        scores["central_vestibular_disorder"] = min(s, 1.0)

    # 正常
    if not scores:
        scores["normal_vestibular"] = 0.8

    return scores


def _get_knowledge_scores_for_patient(
    pid: int, df: pd.DataFrame, all_data: dict,
) -> dict[str, float]:
    """内部：获取某位患者的知识推理得分。"""
    scores = {}

    # 眼震模式分析
    has_torsional_up = False
    has_down_beat = False
    has_horizontal = False
    for col, val in all_data.items():
        if _has_nystagmus(val):
            ntype = _get_nystagmus_type(val)
            if "扭转" in ntype and "上跳" in ntype:
                has_torsional_up = True
            if "下跳" in ntype:
                has_down_beat = True
            if "水平" in ntype:
                has_horizontal = True

    if has_torsional_up:
        scores["BPPV_posterior"] = 0.8
    if has_down_beat:
        scores["BPPV_anterior"] = scores.get("BPPV_anterior", 0) + 0.6
    if has_horizontal:
        scores["BPPV_horizontal"] = scores.get("BPPV_horizontal", 0) + 0.5
        scores["vestibular_neuritis"] = 0.3

    # 温度试验
    r50 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧50°C", ""),
    )
    l50 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧50°C", ""),
    )
    r24 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧24°C", ""),
    )
    l24 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧24°C", ""),
    )
    cal = _compute_cp_value(r50, l50, r24, l24)
    if cal["cp_value"] is not None and cal["cp_value"] > 25:
        scores["unilateral_vestibular_hypofunction"] = 0.7

    # 视跟踪
    tracking = all_data.get("Extracted_Test__视跟踪水平方向分型", "")
    if "I型" in tracking and "II" not in tracking and "III" not in tracking:
        scores["central_vestibular_disorder"] = (
            scores.get("central_vestibular_disorder", 0) + 0.75
        )

    if not scores:
        scores["normal_vestibular"] = 0.7

    # 归一化到 0-1
    for k in scores:
        scores[k] = min(scores[k], 1.0)

    return scores


def _get_model_scores_for_patient(
    pid: int, df: pd.DataFrame, all_data: dict,
) -> dict[str, float]:
    """内部：获取某位患者的模型预测得分。"""
    # 提取特征
    nystagmus_count = 0
    for col, val in all_data.items():
        if _has_nystagmus(val):
            nystagmus_count += 1
    severity = min(nystagmus_count / 10.0, 1.0)

    r50 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧50°C", ""),
    )
    l50 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧50°C", ""),
    )
    r24 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度右侧24°C", ""),
    )
    l24 = _extract_numeric(
        all_data.get("Extracted_Test__前庭双温评价-眼震方向慢向角速度左侧24°C", ""),
    )
    cal = _compute_cp_value(r50, l50, r24, l24)
    cp_val = cal.get("cp_value", 0) or 0

    tracking = all_data.get("Extracted_Test__视跟踪水平方向分型", "")
    tracking_score = 0
    if "I型" in tracking and "II" not in tracking:
        tracking_score = 1.0
    elif "II型" in tracking:
        tracking_score = 0.3
    elif "III型" in tracking:
        tracking_score = 0.1

    dh_positive = sum(
        1 for col in df.columns
        if "Dix-Hallpike" in col
        and _has_nystagmus(all_data.get(col, ""))
    )
    roll_positive = sum(
        1 for col in df.columns
        if "Roll-test" in col
        and _has_nystagmus(all_data.get(col, ""))
    )
    deep_positive = sum(
        1 for col in df.columns
        if "深悬头位" in col
        and _has_nystagmus(all_data.get(col, ""))
    )

    left_s = _extract_numeric(
        all_data.get("Extracted_Test__视动眼震左眼速度", ""),
    )
    right_s = _extract_numeric(
        all_data.get("Extracted_Test__视动眼震右眼速度", ""),
    )
    eye_asym = 0
    if left_s and right_s:
        eye_asym = abs(left_s - right_s) / max(left_s, right_s, 0.01)

    scores = {}
    scores["BPPV_posterior"] = round(
        0.35 * min(dh_positive / 3.0, 1.0)
        + 0.25 * severity
        + 0.20 * (1.0 if cp_val < 25 else 0.3)
        + 0.20 * (1.0 - tracking_score),
        3,
    )
    scores["BPPV_horizontal"] = round(
        0.40 * min(roll_positive / 3.0, 1.0)
        + 0.25 * severity
        + 0.15 * (1.0 if cp_val < 25 else 0.3)
        + 0.20 * (1.0 - tracking_score),
        3,
    )
    scores["BPPV_anterior"] = round(
        0.40 * min(deep_positive / 2.0, 1.0)
        + 0.25 * severity
        + 0.15 * (1.0 if cp_val < 25 else 0.3)
        + 0.20 * (1.0 - tracking_score),
        3,
    )
    scores["unilateral_vestibular_hypofunction"] = round(
        0.50 * min(cp_val / 50.0, 1.0)
        + 0.20 * severity
        + 0.15 * eye_asym
        + 0.15 * (1.0 - tracking_score),
        3,
    )
    scores["central_vestibular_disorder"] = round(
        0.40 * tracking_score
        + 0.25 * severity
        + 0.20 * eye_asym
        + 0.15 * min(
            sum(
                1 for col in df.columns
                if "静态位置" in col
                and _has_nystagmus(all_data.get(col, ""))
            ) / 5.0,
            1.0,
        ),
        3,
    )
    scores["normal_vestibular"] = round(
        max(
            0,
            1.0
            - severity * 0.4
            - tracking_score * 0.3
            - min(cp_val / 50.0, 1.0) * 0.3,
        ),
        3,
    )

    return scores


def _generate_structured_output(df: pd.DataFrame) -> dict:
    """生成结构化的表型标注结果（供保存使用）。"""
    W_RULE = 0.40
    W_KNOWLEDGE = 0.30
    W_MODEL = 0.30

    patients = {}
    for pid in sorted(df["patient_id"].unique()):
        rows = df[df["patient_id"] == pid]
        all_data = {}
        for _, row in rows.iterrows():
            for col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    if col not in all_data or all_data[col] == "未做":
                        all_data[col] = str(val).strip()

        rule_s = _get_rule_scores_for_patient(pid, df)
        know_s = _get_knowledge_scores_for_patient(pid, df, all_data)
        model_s = _get_model_scores_for_patient(pid, df, all_data)

        all_phenos = set(
            list(rule_s.keys()) + list(know_s.keys()) + list(model_s.keys())
        )
        fused = {}
        for p in all_phenos:
            fused[p] = round(
                W_RULE * rule_s.get(p, 0)
                + W_KNOWLEDGE * know_s.get(p, 0)
                + W_MODEL * model_s.get(p, 0),
                3,
            )

        sorted_p = sorted(fused.items(), key=lambda x: -x[1])
        primary = sorted_p[0][0] if sorted_p else "unknown"
        secondary = [
            {"phenotype": p, "name": PHENOTYPE_DEFINITIONS.get(p, {}).get("name", p), "score": s}
            for p, s in sorted_p[1:]
            if s > 0.3
        ]

        patients[str(pid)] = {
            "primary_phenotype": primary,
            "primary_phenotype_name": PHENOTYPE_DEFINITIONS.get(
                primary, {},
            ).get("name", primary),
            "icd10": PHENOTYPE_DEFINITIONS.get(primary, {}).get("icd10", "N/A"),
            "primary_score": sorted_p[0][1] if sorted_p else 0,
            "secondary_phenotypes": secondary,
            "all_scores": fused,
        }

    return {
        "total_patients": len(patients),
        "phenotype_definitions": PHENOTYPE_DEFINITIONS,
        "patients": patients,
    }


# ============================================================
# 创建模型和Agent
# ============================================================

def create_model() -> OpenAIChatModel:
    """创建 OpenAI 兼容模型（通过云雾 API）。"""
    return OpenAIChatModel(
        model_name=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        stream=True,
        client_kwargs={"base_url": OPENAI_BASE_URL},
        generate_kwargs={
            "temperature": 0.2,
            "max_tokens": 4096,
        },
    )


def create_phenotyping_agent() -> ReActAgent:
    """创建 Phenotyping Agent（表型标注智能体）。"""
    toolkit = Toolkit()
    toolkit.register_tool_function(load_patient_data)
    toolkit.register_tool_function(rule_based_matching)
    toolkit.register_tool_function(knowledge_reasoning)
    toolkit.register_tool_function(model_prediction)
    toolkit.register_tool_function(decision_fusion)
    toolkit.register_tool_function(save_phenotyping_report)

    model = create_model()

    agent = ReActAgent(
        name="PhenotypingAgent",
        sys_prompt="""你是一位经验丰富的前庭功能疾病临床表型标注专家（Phenotyping Agent）。

你的任务是对干净且带有置信度的医疗数据进行标准化临床表型标注。

## 可用的标准化表型
- **后半规管BPPV (pc-BPPV)**：Dix-Hallpike试验阳性，扭转+上跳眼震
- **水平半规管BPPV (hc-BPPV)**：Roll-test阳性，水平方向变向性眼震
- **前半规管BPPV (ac-BPPV)**：深悬头位试验阳性，下跳+扭转眼震
- **单侧前庭功能减退**：CP值>25%，温度试验不对称
- **双侧前庭功能减退**：双侧温度试验均减弱
- **中枢性前庭功能障碍**：视跟踪I型、视动异常等中枢体征
- **前庭神经炎**：急性单侧前庭功能减退+自发眼震
- **梅尼埃病**：反复发作眩晕+波动性听力下降
- **前庭功能正常**：所有检查正常
- **位置性眼震(待分类)**：仅有位置试验阳性

## 工作流程（严格按以下顺序执行）
1. 首先调用 `load_patient_data` 查看数据
2. 调用 `rule_based_matching` 进行规则匹配
3. 调用 `knowledge_reasoning` 进行知识推理
4. 调用 `model_prediction` 进行模型预测
5. 调用 `decision_fusion` 进行决策融合，得出最终表型
6. 综合分析四步的结果，撰写详细的中文表型标注报告，内容包括：
   - 数据概况
   - 每位患者的表型判定过程（规则→知识→模型→融合）
   - 最终的标准化表型标签（含ICD-10编码）
   - 置信度评估
   - 临床建议
7. 调用 `save_phenotyping_report` 保存报告

请使用中文撰写报告，确保每位患者都有明确的表型标签。
""",
        model=model,
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=15,
    )

    return agent


# ============================================================
# 主程序入口
# ============================================================

async def main() -> None:
    """主程序：运行 Phenotyping Agent 表型标注流程。"""
    print("=" * 60)
    print("  STEP 7: Phenotyping Agent — 临床表型标注")
    print("=" * 60)
    print(f"  模型: {MODEL_NAME}")
    print(f"  API: {OPENAI_BASE_URL}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    agent = create_phenotyping_agent()

    task_msg = Msg(
        name="user",
        role="user",
        content=(
            "请对眼震检查医疗数据执行完整的临床表型标注分析。\n\n"
            "严格按照以下步骤执行：\n"
            "1. 加载患者数据\n"
            "2. 执行规则匹配（基于诊断标准）\n"
            "3. 执行知识推理（基于医学知识库）\n"
            "4. 执行模型预测（基于特征评分模型）\n"
            "5. 执行决策融合（综合三方结果）\n"
            "6. 撰写完整的中文表型标注报告，为每位患者给出：\n"
            "   - 标准化表型标签\n"
            "   - ICD-10编码\n"
            "   - 判定依据（规则/知识/模型三方证据）\n"
            "   - 置信度\n"
            "   - 临床建议\n"
            "7. 保存报告\n"
        ),
    )

    result = await agent(task_msg)

    print("\n" + "=" * 60)
    print("  Phenotyping Agent 分析完成！")
    print("=" * 60)
    if result and result.content:
        content = result.content
        if isinstance(content, list):
            for block in content:
                if hasattr(block, "text"):
                    print(block.text)
                elif isinstance(block, dict) and "text" in block:
                    print(block["text"])
                else:
                    print(block)
        elif isinstance(content, str):
            print(content)
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
