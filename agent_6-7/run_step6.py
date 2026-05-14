# -*- coding: utf-8 -*-
"""
STEP 6: 一致性验证与置信度估计
=================================
基于 AgentScope 框架，使用多Agent协作实现：
1. 构建统一事件时间线：将来自不同模态的事件按患者对齐到统一时间轴
2. 执行多维度一致性校验：检测逻辑矛盾、时间冲突或跨模态语义不一致
3. 计算字段级与记录级置信度：为每个字段和整条记录赋予可信度分数

输入：清洗数据或任务子集（CSV + 筛选报告JSON）
输出：数据一致性报告 + 整体清洗置信度 + 异常记录列表
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd

from agentscope.agent import ReActAgent, AgentBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

# ============================================================
# 配置区 —— 请根据实际情况修改
# ============================================================
# OpenAI 兼容 API（云雾）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = (
    os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or "https://api.openai.com/v1"  # OpenAI compatible base URL
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

# 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_CSV = os.path.join(DATA_DIR, "input_cleaned_38.csv")
FILTERED_CSV = os.path.join(
    DATA_DIR, "input_cleaned_38_filtered_20260302_142104.csv"
)
SELECTION_REPORT_JSON = os.path.join(
    DATA_DIR, "input_cleaned_38_selection_report_20260302_142104.json"
)

# 输出路径
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 工具函数定义
# ============================================================

def load_raw_data() -> ToolResponse:
    """加载原始（未筛选）的CSV数据，返回数据概览。"""
    df = pd.read_csv(RAW_CSV)
    summary_lines = [
        f"原始数据共 {len(df)} 行, {len(df.columns)} 列",
        f"患者ID列表: {sorted(df['patient_id'].unique().tolist())}",
        f"列名: {list(df.columns)}",
        "",
        "前5行数据预览:",
        df.head().to_string(index=False),
    ]
    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(summary_lines))],
    )


def load_filtered_data() -> ToolResponse:
    """加载经过筛选的CSV数据，返回数据概览。"""
    df = pd.read_csv(FILTERED_CSV)
    summary_lines = [
        f"筛选后数据共 {len(df)} 行, {len(df.columns)} 列",
        f"列名: {list(df.columns)}",
        "",
        "前5行数据预览:",
        df.head().to_string(index=False),
    ]
    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(summary_lines))],
    )


def load_selection_report() -> ToolResponse:
    """加载筛选报告JSON文件，返回关键信息摘要。"""
    with open(SELECTION_REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary_parts = [
        f"报告时间: {report.get('timestamp', 'N/A')}",
        f"任务描述: {report.get('task_text', 'N/A')}",
        f"最终保留列: {report.get('final_columns', [])}",
        f"必须保留列: {report.get('agent_decisions', {}).get('must_have_columns', [])}",
        f"丢弃列: {report.get('agent_decisions', {}).get('drop_columns', [])}",
        f"警告信息: {report.get('warnings', [])}",
    ]

    # 添加每列的决策理由
    reasons = report.get("agent_decisions", {}).get("reason_by_column", {})
    if reasons:
        summary_parts.append("\n各列决策理由:")
        for col, reason in reasons.items():
            summary_parts.append(f"  - {col}: {reason}")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(summary_parts))],
    )


def build_patient_timeline() -> ToolResponse:
    """构建统一事件时间线：将来自不同模态（如检验报告、影像记录、电子病历）的事件
    按患者对齐到统一时间轴。返回每位患者的数据记录汇总。"""
    df = pd.read_csv(RAW_CSV)
    timeline = {}
    for pid in sorted(df["patient_id"].unique()):
        patient_rows = df[df["patient_id"] == pid]
        records = []
        for _, row in patient_rows.iterrows():
            record = {
                "filename": row.get("filename", "N/A"),
                "folder_category": row.get("folder_category", "N/A"),
            }
            # 收集非空字段
            for col in df.columns:
                if col not in ["patient_id", "folder_category", "filename"]:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        record[col] = str(val)
            records.append(record)
        timeline[str(pid)] = records

    output_lines = ["=== 患者时间线汇总 ==="]
    for pid, records in timeline.items():
        output_lines.append(f"\n--- 患者 {pid} ({len(records)} 条记录) ---")
        for i, rec in enumerate(records, 1):
            output_lines.append(f"  记录{i} [{rec.get('filename', '?')}]:")
            for k, v in rec.items():
                if k not in ["filename", "folder_category"]:
                    output_lines.append(f"    {k}: {v}")
    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def detect_inconsistencies() -> ToolResponse:
    """检测多维度一致性问题：检测同一患者不同记录间的逻辑矛盾、数值冲突和跨模态语义不一致。
    自动对比同一患者多次检查结果，识别潜在冲突。"""
    df = pd.read_csv(RAW_CSV)
    issues = []

    for pid in sorted(df["patient_id"].unique()):
        patient_rows = df[df["patient_id"] == pid]
        if len(patient_rows) <= 1:
            continue

        patient_issues = []

        # 1. 检查分型不一致
        tracking_col = "Extracted_Test__视跟踪水平方向分型"
        if tracking_col in df.columns:
            vals = patient_rows[tracking_col].dropna().unique()
            if len(vals) > 1:
                patient_issues.append(
                    f"  [分型不一致] 视跟踪水平方向分型存在多个值: {list(vals)}"
                )

        # 2. 检查双温评价数值差异过大
        temp_cols = [
            c for c in df.columns if "前庭双温评价" in str(c)
        ]
        for col in temp_cols:
            vals = patient_rows[col].dropna().tolist()
            numeric_vals = []
            for v in vals:
                v_str = str(v)
                # 提取数值
                num_str = ""
                for ch in v_str:
                    if ch.isdigit() or ch == ".":
                        num_str += ch
                    elif num_str:
                        break
                if num_str:
                    try:
                        numeric_vals.append(float(num_str))
                    except ValueError:
                        pass
            if len(numeric_vals) >= 2:
                max_v = max(numeric_vals)
                min_v = min(numeric_vals)
                if max_v > 0 and (max_v - min_v) / max_v > 0.5:
                    col_short = col.replace("Extracted_Test__", "")
                    patient_issues.append(
                        f"  [数值差异大] {col_short}: "
                        f"范围 {min_v}-{max_v}（差异>{50}%）"
                    )

        # 3. 检查试验结果矛盾（一个记录为"未见"，另一个记录有异常）
        test_cols = [
            c for c in df.columns
            if "试验眼震" in str(c) and "Extracted_Test__" in str(c)
        ]
        for col in test_cols:
            vals = patient_rows[col].dropna().tolist()
            has_normal = any("未见" in str(v) for v in vals)
            has_abnormal = any(
                str(v).strip() not in ["", "未见眼震", "未做"]
                and "未见" not in str(v)
                for v in vals
            )
            if has_normal and has_abnormal:
                col_short = col.replace("Extracted_Test__", "")
                normal_vals = [str(v) for v in vals if "未见" in str(v)]
                abnormal_vals = [
                    str(v) for v in vals
                    if str(v).strip() not in ["", "未见眼震", "未做"]
                    and "未见" not in str(v)
                ]
                patient_issues.append(
                    f"  [结果矛盾] {col_short}: "
                    f"部分记录为正常({normal_vals}), "
                    f"但部分记录有异常({abnormal_vals})"
                )

        # 4. 检查左右眼速度不对称
        left_col = "Extracted_Test__视动眼震左眼速度"
        right_col = "Extracted_Test__视动眼震右眼速度"
        if left_col in df.columns and right_col in df.columns:
            for _, row in patient_rows.iterrows():
                left_v = row.get(left_col)
                right_v = row.get(right_col)
                if pd.notna(left_v) and pd.notna(right_v):
                    try:
                        lv = float(left_v)
                        rv = float(right_v)
                        ratio = max(lv, rv) / max(min(lv, rv), 0.01)
                        if ratio > 1.5:
                            patient_issues.append(
                                f"  [左右不对称] {row.get('filename', '?')}: "
                                f"左眼={lv}, 右眼={rv}, 比值={ratio:.2f}"
                            )
                    except (ValueError, TypeError):
                        pass

        if patient_issues:
            issues.append(f"\n患者 {pid}:")
            issues.extend(patient_issues)

    if not issues:
        result_text = "未检测到明显的一致性问题。"
    else:
        result_text = "=== 一致性检测结果 ===\n" + "\n".join(issues)

    return ToolResponse(
        content=[TextBlock(type="text", text=result_text)],
    )


def compute_field_confidence() -> ToolResponse:
    """计算字段级置信度：基于缺失率、值分布一致性、跨记录一致性等维度，
    为每个字段计算置信度分数（0-1之间，1为最可信）。"""
    df = pd.read_csv(RAW_CSV)

    with open(SELECTION_REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)

    field_scores = {}
    columns_profile = report.get("schema_profile", {}).get("columns", {})

    for col in df.columns:
        if col in ["patient_id", "folder_category"]:
            continue

        score = 1.0
        reasons = []

        # 1. 缺失率惩罚
        missing_rate = df[col].isna().mean()
        if missing_rate > 0:
            penalty = missing_rate * 0.4
            score -= penalty
            reasons.append(f"缺失率={missing_rate:.2%}(扣{penalty:.2f})")

        # 2. 跨记录一致性
        for pid in df["patient_id"].unique():
            subset = df[df["patient_id"] == pid][col].dropna()
            if len(subset) > 1:
                unique_vals = subset.nunique()
                if unique_vals > 1:
                    inconsistency = (unique_vals - 1) / len(subset)
                    penalty = inconsistency * 0.3
                    score -= penalty
                    reasons.append(
                        f"患者{pid}存在{unique_vals}个不同值(扣{penalty:.2f})"
                    )

        # 3. 是否在筛选报告中被标记为有问题
        col_profile = columns_profile.get(col, {})
        if col_profile.get("is_potential_phi", False):
            score -= 0.1
            reasons.append("潜在PHI字段(扣0.10)")

        # 4. 是否在丢弃列表中
        drop_cols = report.get("agent_decisions", {}).get("drop_columns", [])
        if col in drop_cols:
            score -= 0.15
            reasons.append("在筛选丢弃列表中(扣0.15)")

        score = max(0.0, min(1.0, score))
        field_scores[col] = {
            "confidence": round(score, 3),
            "reasons": reasons,
        }

    # 生成输出
    output_lines = ["=== 字段级置信度评分 ===\n"]
    sorted_fields = sorted(
        field_scores.items(), key=lambda x: x[1]["confidence"]
    )
    for col, info in sorted_fields:
        output_lines.append(
            f"  {col}: {info['confidence']:.3f}"
        )
        if info["reasons"]:
            for r in info["reasons"]:
                output_lines.append(f"    - {r}")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def compute_record_confidence() -> ToolResponse:
    """计算记录级置信度：为每条记录（每行）赋予整体可信度分数。
    综合考虑该行的缺失情况、该行与同患者其他记录的一致性。"""
    df = pd.read_csv(RAW_CSV)
    data_cols = [
        c for c in df.columns
        if c not in ["patient_id", "folder_category", "filename"]
    ]

    record_scores = []
    for idx, row in df.iterrows():
        score = 1.0
        reasons = []
        pid = row["patient_id"]
        fname = row.get("filename", f"row_{idx}")

        # 1. 行级缺失率
        non_null_count = row[data_cols].notna().sum()
        total_count = len(data_cols)
        completeness = non_null_count / total_count if total_count > 0 else 0
        if completeness < 1.0:
            penalty = (1 - completeness) * 0.4
            score -= penalty
            reasons.append(
                f"完整度={completeness:.2%}(扣{penalty:.2f})"
            )

        # 2. 与同患者其他记录的一致性
        sibling_rows = df[
            (df["patient_id"] == pid) & (df.index != idx)
        ]
        if len(sibling_rows) > 0:
            conflicts = 0
            for col in data_cols:
                my_val = row[col]
                if pd.isna(my_val):
                    continue
                sibling_vals = sibling_rows[col].dropna()
                for sv in sibling_vals:
                    if str(my_val).strip() != str(sv).strip():
                        # 排除正常不同（如未做 vs 未见）
                        if not (
                            {"未做", "未见眼震"} & {
                                str(my_val).strip(), str(sv).strip()
                            }
                        ):
                            conflicts += 1
            if conflicts > 0:
                penalty = min(conflicts * 0.02, 0.3)
                score -= penalty
                reasons.append(
                    f"与同患者记录有{conflicts}处差异(扣{penalty:.2f})"
                )

        # 3. 是否有"未做"标记（表示数据不完整但不影响可信度太多）
        undone_count = sum(
            1 for col in data_cols
            if str(row.get(col, "")).strip() == "未做"
        )
        if undone_count > 0:
            penalty = undone_count * 0.02
            score -= penalty
            reasons.append(f"有{undone_count}项标记为'未做'(扣{penalty:.2f})")

        score = max(0.0, min(1.0, score))
        record_scores.append({
            "patient_id": pid,
            "filename": fname,
            "confidence": round(score, 3),
            "reasons": reasons,
        })

    # 生成输出
    output_lines = ["=== 记录级置信度评分 ===\n"]
    sorted_records = sorted(record_scores, key=lambda x: x["confidence"])
    for rec in sorted_records:
        output_lines.append(
            f"  患者{rec['patient_id']} [{rec['filename']}]: "
            f"{rec['confidence']:.3f}"
        )
        if rec["reasons"]:
            for r in rec["reasons"]:
                output_lines.append(f"    - {r}")

    # 汇总
    avg_score = sum(r["confidence"] for r in record_scores) / len(
        record_scores
    )
    anomaly_records = [r for r in record_scores if r["confidence"] < 0.6]
    output_lines.append(f"\n整体平均置信度: {avg_score:.3f}")
    output_lines.append(f"异常记录数（置信度<0.6）: {len(anomaly_records)}")

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def compare_raw_and_filtered() -> ToolResponse:
    """对比原始数据和筛选后数据的差异，分析筛选操作的合理性。"""
    raw_df = pd.read_csv(RAW_CSV)
    filtered_df = pd.read_csv(FILTERED_CSV)

    raw_cols = set(raw_df.columns)
    filtered_cols = set(filtered_df.columns)

    removed_cols = raw_cols - filtered_cols
    kept_cols = raw_cols & filtered_cols
    new_cols = filtered_cols - raw_cols

    output_lines = [
        "=== 原始数据 vs 筛选数据对比 ===",
        f"原始数据: {len(raw_df)}行 x {len(raw_df.columns)}列",
        f"筛选数据: {len(filtered_df)}行 x {len(filtered_df.columns)}列",
        f"\n被移除的列 ({len(removed_cols)}):",
    ]
    for c in sorted(removed_cols):
        output_lines.append(f"  - {c}")

    output_lines.append(f"\n保留的列 ({len(kept_cols)}):")
    for c in sorted(kept_cols):
        output_lines.append(f"  + {c}")

    if new_cols:
        output_lines.append(f"\n新增的列 ({len(new_cols)}):")
        for c in sorted(new_cols):
            output_lines.append(f"  * {c}")

    # 数据一致性检查
    output_lines.append("\n=== 保留列的数据一致性检查 ===")
    common_cols = list(kept_cols)
    for col in sorted(common_cols):
        if col in raw_df.columns and col in filtered_df.columns:
            # 检查筛选数据中的值是否都来自原始数据
            filtered_vals = set(filtered_df[col].dropna().astype(str))
            raw_vals = set(raw_df[col].dropna().astype(str))
            extra = filtered_vals - raw_vals
            if extra:
                output_lines.append(
                    f"  [异常] {col}: 筛选数据中存在原始数据没有的值: {extra}"
                )

    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(output_lines))],
    )


def save_report(report_content: str) -> ToolResponse:
    """将最终的一致性验证与置信度报告保存到文件。

    Args:
        report_content (str):
            要保存的报告内容（纯文本格式）。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        OUTPUT_DIR, f"consistency_confidence_report_{timestamp}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    # 同时保存结构化JSON版本
    json_path = os.path.join(
        OUTPUT_DIR, f"consistency_confidence_report_{timestamp}.json"
    )

    # 自动生成结构化数据
    df = pd.read_csv(RAW_CSV)
    structured_report = {
        "timestamp": datetime.now().isoformat(),
        "step": "STEP6_一致性验证与置信度估计",
        "input_files": {
            "raw_csv": RAW_CSV,
            "filtered_csv": FILTERED_CSV,
            "selection_report": SELECTION_REPORT_JSON,
        },
        "summary": {
            "total_patients": int(df["patient_id"].nunique()),
            "total_records": len(df),
            "total_columns": len(df.columns),
        },
        "report_text": report_content,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_report, f, ensure_ascii=False, indent=2)

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"报告已保存:\n  文本版: {report_path}\n  JSON版: {json_path}",
            )
        ],
    )


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


def create_consistency_agent() -> ReActAgent:
    """创建一致性验证与置信度估计的 ReAct 智能体。"""
    # 注册工具函数
    toolkit = Toolkit()
    toolkit.register_tool_function(load_raw_data)
    toolkit.register_tool_function(load_filtered_data)
    toolkit.register_tool_function(load_selection_report)
    toolkit.register_tool_function(build_patient_timeline)
    toolkit.register_tool_function(detect_inconsistencies)
    toolkit.register_tool_function(compute_field_confidence)
    toolkit.register_tool_function(compute_record_confidence)
    toolkit.register_tool_function(compare_raw_and_filtered)
    toolkit.register_tool_function(save_report)

    model = create_model()

    agent = ReActAgent(
        name="ConsistencyValidator",
        sys_prompt="""你是一位医疗数据质量分析专家，专门负责对多模态、时序性医疗数据进行一致性验证与置信度估计。

你的工作目标：
1. **构建统一事件时间线**：将来自不同模态（如检验报告、影像记录、电子病历）的事件按患者对齐到统一时间轴。
2. **执行多维度一致性校验**：检测逻辑矛盾、时间冲突或跨模态语义不一致。
3. **计算字段级与记录级置信度**：为每个字段和整条记录赋予可信度分数。

当前数据是眼震检查（Nystagmus Test）相关的前庭功能检测数据，包含多位患者的多次检查记录。
数据包括检验测试结果（Extracted_Test__*）和症状记录（Extracted_Symptom__*）两大类。

你的工作流程：
1. 首先加载并查看原始数据、筛选数据和筛选报告
2. 构建患者时间线，了解每位患者有多少条记录
3. 对比原始数据和筛选数据，了解筛选操作
4. 运行一致性检测，发现数据中的矛盾和冲突
5. 计算字段级和记录级的置信度分数
6. 综合所有分析，撰写完整的一致性验证报告
7. 保存最终报告

请注意：
- 眼震检查中，同一患者多次检查结果可能有合理差异（如视跟踪分型II型和III型可能是不同检测条件下的结果）
- 但如果同一检测项目（如Dix-Hallpike试验）出现"未见眼震"和有异常眼震的矛盾，则需重点关注
- 前庭双温评价的数值如果同一患者差异巨大，可能提示数据质量问题
- 请使用中文撰写报告
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
    """主程序：运行一致性验证与置信度估计流程。"""
    print("=" * 60)
    print("  STEP 6: 一致性验证与置信度估计")
    print("=" * 60)
    print(f"  模型: {MODEL_NAME}")
    print(f"  API: {OPENAI_BASE_URL}")
    print(f"  原始数据: {RAW_CSV}")
    print(f"  筛选数据: {FILTERED_CSV}")
    print(f"  筛选报告: {SELECTION_REPORT_JSON}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 创建Agent
    agent = create_consistency_agent()

    # 发送任务消息
    task_msg = Msg(
        name="user",
        role="user",
        content=(
            "请对当前的眼震检查医疗数据执行完整的一致性验证与置信度估计分析。\n\n"
            "具体要求：\n"
            "1. 加载所有数据文件（原始CSV、筛选CSV、筛选报告JSON）\n"
            "2. 构建患者时间线，对比原始数据和筛选数据\n"
            "3. 运行一致性检测，发现数据中的矛盾和冲突\n"
            "4. 计算字段级和记录级的置信度分数\n"
            "5. 撰写完整的中文分析报告，包括：\n"
            "   - 数据概览\n"
            "   - 一致性检测结果（发现的矛盾/冲突）\n"
            "   - 字段级置信度排名\n"
            "   - 记录级置信度排名\n"
            "   - 异常记录列表\n"
            "   - 整体清洗数据的置信度评估\n"
            "   - 建议和结论\n"
            "6. 将报告保存到文件\n"
        ),
    )

    # 运行Agent
    result = await agent(task_msg)
    print("\n" + "=" * 60)
    print("  Agent 分析完成！")
    print("=" * 60)
    if result and result.content:
        # 打印结果摘要
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
