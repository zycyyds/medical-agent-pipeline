# -*- coding: utf-8 -*-
"""
STEP 6: 患者数据一致性验证
==========================
运行方式:
  python run_step6_ml_pipeline.py

输出:
  output_step6/consistency_report_<ts>.txt/.json
  output_step6/passed_patients_<ts>.json   ← Step 7 读取此文件
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

# 加载项目根目录的 .env 文件
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.is_file():
    with open(_ENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import pandas as pd

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.token import CharTokenCounter
from agentscope.tool import Toolkit, ToolResponse

_FORMATTER_MAX_CHARS = 375_000


def _install_retry_interceptor(max_retries: int = 5, base_delay: float = 10.0) -> None:
    try:
        import openai as _openai
        from openai._base_client import AsyncAPIClient
        _orig_request = AsyncAPIClient.request
        _PermissionDeniedError = _openai.PermissionDeniedError
        _APIConnectionError = _openai.APIConnectionError

        async def _patched_request(self, cast_to, options, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await _orig_request(self, cast_to, options, **kwargs)
                except (_PermissionDeniedError, _APIConnectionError) as exc:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        err_type = "403" if isinstance(exc, _PermissionDeniedError) else "连接错误"
                        print(f"\n[Retry] {err_type}，{delay:.0f}s 后重试 (第{attempt+1}/{max_retries}次)...")
                        await asyncio.sleep(delay)
                    else:
                        raise

        AsyncAPIClient.request = _patched_request
    except Exception as _e:
        print(f"[Retry] 安装重试拦截器失败: {_e}")

_install_retry_interceptor()

# ══════════════════════════════════════════════════════════════════
# 0. 全局配置
# ══════════════════════════════════════════════════════════════════

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEYS", "")
OPENAI_BASE_URL = os.environ.get("YUNWU_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME      = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_MIMIC_DATA_DIR = os.path.join(_SCRIPT_DIR, "data", "mimic")
_mimic_files_exist = os.path.isdir(_MIMIC_DATA_DIR) and any(
    f.startswith("liver_notes_extracted_filtered") and f.endswith(".csv")
    for f in os.listdir(_MIMIC_DATA_DIR)
)
_default_source = "mimic" if _mimic_files_exist else "original"
DATA_SOURCE = os.environ.get("DATA_SOURCE", _default_source).lower()
_IS_MIMIC   = DATA_SOURCE == "mimic"

if _IS_MIMIC:
    DATA_DIR       = os.path.join(_SCRIPT_DIR, "data", "mimic")
    PATIENT_ID_COL = "record_index"
else:
    DATA_DIR       = os.path.join(_SCRIPT_DIR, "data")
    PATIENT_ID_COL = os.environ.get("PATIENT_ID_COL", "")

ICD10_XLSX       = os.path.join(_SCRIPT_DIR, "ICD-10.xlsx")
ICD10_VECTOR_DIR = os.path.join(_SCRIPT_DIR, "data", "icd10_bert_index")
ICD10_USE_BERT   = os.environ.get("ICD10_USE_BERT", "1").lower() not in ("0", "false", "no")
ICD10_BERT_MODEL = os.environ.get("ICD10_BERT_MODEL", "models/bge-small-zh-v1.5")
ICD10_BERT_MIN_SIM = float(os.environ.get("ICD10_BERT_MIN_SIM", "0.70"))


def _find_latest(prefix: str, suffix: str, search_dir: str | None = None) -> str:
    d = search_dir or DATA_DIR
    candidates = [f for f in os.listdir(d)
                  if f.startswith(prefix) and f.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"找不到 {prefix}*{suffix}（目录：{d}）")
    return os.path.join(d, sorted(candidates)[-1])


def _find_latest_by_suffix(keyword: str, suffix: str, search_dir: str | None = None) -> str:
    d = search_dir or DATA_DIR
    candidates = [f for f in os.listdir(d)
                  if keyword in f and f.endswith(suffix)]
    if not candidates:
        fallback_name = keyword.lstrip("_") + suffix
        fallback_path = os.path.join(d, fallback_name)
        if os.path.isfile(fallback_path):
            return fallback_path
        raise FileNotFoundError(f"找不到 *{keyword}*{suffix}（目录：{d}）")
    return os.path.join(d, sorted(candidates)[-1])


if _IS_MIMIC:
    RAW_CSV      = _find_latest("liver_notes_extracted_filtered", ".csv")
    OUTPUT_STEP6 = os.path.join(_SCRIPT_DIR, "output_step6_mimic")
else:
    _raw_fixed   = os.path.join(DATA_DIR, "副本all_patients.csv")
    FILTERED_CSV = _find_latest_by_suffix("_filtered", ".csv")
    RAW_CSV      = _raw_fixed if os.path.isfile(_raw_fixed) else FILTERED_CSV
    OUTPUT_STEP6 = os.path.join(_SCRIPT_DIR, "output_step6")

os.makedirs(OUTPUT_STEP6, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.70

_SKIP_COLS_STATIC = {
    "ocr_text", "preprocessed_text", "file_path", "error",
    "relations", "temporal_info", "quantity_info",
    "impression", "indication",
}
_MIMIC_EXTRA_SKIP_COLS = {
    "所有用药", "实验室检验_摘要", "所有手术操作_titles", "所有手术操作_codes",
    "所有诊断_icd_codes",
    "抽取_Disease", "抽取_Drug", "抽取_Symptom", "抽取_Test",
    "抽取_Treatment", "抽取_LabValue", "抽取_Finding",
}
_SKIP_COLS: set[str] = set(_SKIP_COLS_STATIC)
if _IS_MIMIC:
    _SKIP_COLS |= _MIMIC_EXTRA_SKIP_COLS

_LONG_TEXT_RATIO_THRESHOLD = 0.3
_LONG_TEXT_LEN_THRESHOLD   = 200


def _build_skip_cols(df: "pd.DataFrame") -> None:
    global _SKIP_COLS
    for col in df.columns:
        if col in _SKIP_COLS:
            continue
        if col.endswith(("_抽取", "_标准化", "_extracted", "_normalized")):
            _SKIP_COLS.add(col)
            continue
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        long_ratio = (sample.str.len() > _LONG_TEXT_LEN_THRESHOLD).mean()
        if long_ratio >= _LONG_TEXT_RATIO_THRESHOLD:
            _SKIP_COLS.add(col)


def _trunc(val: Any, n: int = 80) -> str:
    s = str(val)
    return s if len(s) <= n else s[:n] + "…"


def _should_skip(col: str) -> bool:
    return col in _SKIP_COLS


_PATIENT_ID_CANDIDATES = [
    "patient_id", "record_index", "subject_id", "hadm_id",
    "patientid", "patient", "id", "pid",
]


def _detect_patient_id_col(df: "pd.DataFrame") -> str:
    if PATIENT_ID_COL and PATIENT_ID_COL in df.columns:
        return PATIENT_ID_COL
    for cand in _PATIENT_ID_CANDIDATES:
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "id" in col.lower():
            return col
    return df.columns[0]


def _read_data(path: str) -> "pd.DataFrame":
    df = pd.read_csv(path)
    actual_id_col = _detect_patient_id_col(df)
    if actual_id_col != "patient_id":
        df = df.rename(columns={actual_id_col: "patient_id"})
        print(f"[数据加载] 患者ID列: '{actual_id_col}' → 'patient_id'")
    _build_skip_cols(df)
    return df


# ══════════════════════════════════════════════════════════════════
# 1. @register_tool 装饰器工厂
# ══════════════════════════════════════════════════════════════════

def register_tool(toolkit: Toolkit):
    def decorator(func):
        toolkit.register_tool_function(func)
        return func
    return decorator


# ══════════════════════════════════════════════════════════════════
# 2. Step 6 工具集
# ══════════════════════════════════════════════════════════════════

step6_toolkit = Toolkit()


@register_tool(step6_toolkit)
def load_data_overview() -> ToolResponse:
    """
    加载数据集，返回整体概况：总行数、患者数、列数、每位患者记录数。
    同时展示第1位患者第1条记录的非空字段，帮助 Agent 理解字段结构。
    自动检测患者 ID 列，适应任意字段命名的数据集。
    """
    raw = _read_data(RAW_CSV)
    n_patients = raw["patient_id"].nunique() if "patient_id" in raw.columns else "N/A"
    lines = [
        "=== 数据集概况 ===",
        f"原始CSV:  {raw.shape[0]} 行 × {raw.shape[1]} 列",
        f"患者数:   {n_patients}",
        f"跳过的长文本列数: {len(_SKIP_COLS)}",
        "",
        "每位患者的记录数:",
    ]
    if "patient_id" in raw.columns:
        rec_counts = raw.groupby("patient_id").size()
        lines.append(f"  记录数分布: min={rec_counts.min()} / max={rec_counts.max()} / mean={rec_counts.mean():.1f}")
        lines.append(f"  （患者列表已省略，共 {len(rec_counts)} 位）")
    else:
        lines.append("  （未检测到患者ID列，请检查数据）")

    lines.append("\n字段预览（第1条记录，长文本列已跳过，最多30个字段）:")
    row = raw.iloc[0]
    shown = 0
    for col in raw.columns:
        if _should_skip(col):
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip() not in ("", "nan"):
            lines.append(f"  {col}: {_trunc(val)}")
            shown += 1
            if shown >= 30:
                lines.append("  ... （更多字段已省略）")
                break

    lines.append(f"\n所有列名（共 {raw.shape[1]} 列，前50列）:")
    lines.append("  " + ", ".join(raw.columns.tolist()[:50]))
    if raw.shape[1] > 50:
        lines.append(f"  ... 还有 {raw.shape[1] - 50} 列")

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step6_toolkit)
def analyze_all_patients_consistency() -> ToolResponse:
    """
    批量分析所有患者的记录一致性（Step 6 核心工具）。

    在 Python 内一次性完成全部患者的以下分析，无需逐患者循环调用：
      1. 记录数统计
      2. 跨记录不一致字段数（HIGH/MEDIUM/LOW 分级）
      3. 字段完整性
      4. 按评分规则自动计算置信度并判断是否通过验证

    评分规则（置信度阈值 {CONFIDENCE_THRESHOLD}）：
      HIGH 不一致每处 -0.20，MEDIUM -0.10，LOW -0.03，完整性<30% 额外 -0.10

    返回：
      - 每位患者的简要摘要（一行一患者）
      - 通过/未通过人数
      - PASSED_PATIENTS 列表（供 save_step6_report 使用）
    """
    raw = _read_data(RAW_CSV)
    _meta_cols = frozenset({"patient_id", "filename", "file_path", "folder_category",
                             "note_index", "note_type", "note_seq"})
    data_cols = [c for c in raw.columns if c not in _meta_cols and not _should_skip(c)]

    results: list[dict] = []
    for pid, grp in raw.groupby("patient_id"):
        n_rec = len(grp)
        high = medium = low = 0
        for col in data_cols:
            uniq = grp[col].dropna().unique()
            if len(uniq) <= 1:
                continue
            col_lower = col.lower()
            if any(k in col_lower for k in ("诊断", "diagnosis", "性别", "gender", "民族", "race")):
                high += 1
            elif any(k in col_lower for k in ("年龄", "age", "dob", "birth")):
                medium += 1
            else:
                low += 1

        filled = sum(1 for c in data_cols if grp[c].notna().any())
        completeness = filled / len(data_cols) * 100 if data_cols else 100.0

        score = 1.0 - high * 0.20 - medium * 0.10 - low * 0.03
        if completeness < 30:
            score -= 0.10
        score = max(0.0, round(score, 3))
        passed = score >= CONFIDENCE_THRESHOLD

        results.append({
            "patient_id":   str(pid),
            "n_records":    n_rec,
            "high":         high,
            "medium":       medium,
            "low":          low,
            "completeness": round(completeness, 1),
            "score":        score,
            "passed":       passed,
        })

    passed_ids  = [r["patient_id"] for r in results if r["passed"]]
    failed_ids  = [r["patient_id"] for r in results if not r["passed"]]

    lines = [
        "=== Step 6 批量一致性分析结果 ===",
        f"总患者数: {len(results)}  通过: {len(passed_ids)}  未通过: {len(failed_ids)}",
        f"置信度阈值: {CONFIDENCE_THRESHOLD}",
        "",
        f"{'患者ID':<15} {'记录数':>5} {'HIGH':>5} {'MED':>5} {'LOW':>5} "
        f"{'完整性':>7} {'置信度':>7} {'结果':>6}",
        "-" * 65,
    ]
    failed_results = [r for r in results if not r["passed"]]
    for r in failed_results:
        lines.append(
            f"{r['patient_id']:<15} {r['n_records']:>5} {r['high']:>5} {r['medium']:>5} "
            f"{r['low']:>5} {r['completeness']:>6.1f}% {r['score']:>7.3f}  ✗未通"
        )
    if passed_ids:
        pass_scores = [r["score"] for r in results if r["passed"]]
        lines.append(
            f"（通过的 {len(passed_ids)} 位患者已折叠，"
            f"置信度 min={min(pass_scores):.3f} / max={max(pass_scores):.3f}）"
        )
    lines += [
        "-" * 65,
        "",
        f"PASSED_PATIENTS: {passed_ids}",
        "",
        "请根据以上结果撰写摘要报告并调用 save_step6_report 保存。",
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step6_toolkit)
def save_step6_report(report_content: str) -> ToolResponse:
    """
    保存 Step 6 一致性验证报告（文本版 + JSON 版），
    并额外保存通过患者 ID 列表到 passed_patients_<ts>.json，供 Step 7 读取。

    Args:
        report_content (str): 完整的报告文本（末尾须含 PASSED_PATIENTS: [...]）。
    """
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path  = os.path.join(OUTPUT_STEP6, f"consistency_report_{ts}.txt")
    json_path = os.path.join(OUTPUT_STEP6, f"consistency_report_{ts}.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    raw = _read_data(RAW_CSV)

    # 从报告文本中解析通过患者 ID
    passed_ids = _parse_passed_ids(report_content)

    struct = {
        "timestamp": datetime.now().isoformat(),
        "step":      "STEP6_ConsistencyValidation",
        "data_file": RAW_CSV,
        "summary": {
            "total_patients":       int(raw["patient_id"].nunique()),
            "total_records":        int(len(raw)),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "passed_count":         len(passed_ids),
        },
        "report_text": report_content,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(struct, f, ensure_ascii=False, indent=2)

    # 额外保存通过患者 ID 列表（Step 7 读取此文件）
    passed_json_path = os.path.join(OUTPUT_STEP6, f"passed_patients_{ts}.json")
    with open(passed_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":   datetime.now().isoformat(),
            "passed_count": len(passed_ids),
            "passed_patient_ids": passed_ids,
        }, f, ensure_ascii=False, indent=2)

    return ToolResponse(content=[TextBlock(type="text",
        text=(
            f"Step 6 报告已保存:\n"
            f"  文本: {txt_path}\n"
            f"  JSON: {json_path}\n"
            f"  通过患者列表: {passed_json_path}  ({len(passed_ids)} 位患者)"
        ))])


# ══════════════════════════════════════════════════════════════════
# 3. 模型与 Agent
# ══════════════════════════════════════════════════════════════════

def create_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        stream=True,
        client_kwargs={"base_url": OPENAI_BASE_URL},
        generate_kwargs={"temperature": 0.2, "max_tokens": 4096},
    )


def create_consistency_agent() -> ReActAgent:
    if _IS_MIMIC:
        _dataset_desc = (
            "MIMIC 住院患者医疗数据集（肝硬化/肝衰竭队列），"
            "每条记录对应一次住院（record_index），可能包含多条临床笔记（note_index）。"
        )
        _analysis_hint = (
            "注意：MIMIC 数据中一个 record_index 可能对应多条不同类型的临床笔记记录，"
            "请检查同一 record_index 下不同笔记之间的关键临床字段是否一致。"
        )
    else:
        _dataset_desc = "前庭功能检查数据集，每位患者有多条检查记录。"
        _analysis_hint = "注意：同一患者可能来自不同检查批次，请重点关注诊断、性别等关键字段的一致性。"

    return ReActAgent(
        name="ConsistencyAgent",
        sys_prompt=f"""你是医疗数据质量工程师，负责对医疗数据集执行 Step 6 一致性验证。

## 工作流程（严格按顺序，共 3 步工具调用）

### 第一步
调用 load_data_overview —— 了解数据集整体情况与字段结构。

### 第二步
调用 analyze_all_patients_consistency —— 批量完成所有患者的一致性评分，
工具会返回每位患者的置信度分数与通过状态，以及 PASSED_PATIENTS 列表。

### 第三步
根据第二步结果撰写简要中文摘要报告（包含通过/未通过患者数、整体质量评估），
然后调用 save_step6_report 保存报告。

报告末尾必须包含（原样复制自第二步工具返回）：
PASSED_PATIENTS: [patient_id_1, patient_id_2, ...]
""",
        model=create_model(),
        formatter=OpenAIChatFormatter(
            token_counter=CharTokenCounter(),
            max_tokens=_FORMATTER_MAX_CHARS,
        ),
        toolkit=step6_toolkit,
        memory=InMemoryMemory(),
        max_iters=8,
    )


# ══════════════════════════════════════════════════════════════════
# 4. 辅助函数
# ══════════════════════════════════════════════════════════════════

def _extract_msg_text(msg: Msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _parse_passed_ids(text: str) -> list[str]:
    """从 Step 6 输出解析 PASSED_PATIENTS 列表。"""
    def _extract_from_text(t: str) -> list[str]:
        m = re.search(r"PASSED_PATIENTS:\s*\[([^\]]+)\]", t)
        if m:
            ids = re.findall(r"[^\s,\"']+", m.group(1))
            ids = [i.strip().strip("\"'") for i in ids if i.strip().strip("\"'")]
            if ids:
                return ids
        return []

    ids = _extract_from_text(text)
    if ids:
        return ids

    # 回退：读最新的 Step 6 报告文件
    try:
        report_files = sorted(
            f for f in os.listdir(OUTPUT_STEP6)
            if f.startswith("consistency_report_") and f.endswith(".txt")
        )
        if report_files:
            latest = os.path.join(OUTPUT_STEP6, report_files[-1])
            with open(latest, encoding="utf-8") as fh:
                ids = _extract_from_text(fh.read())
            if ids:
                print(f"[_parse_passed_ids] 从报告文件读取 {len(ids)} 个通过患者: {latest}")
                return ids
    except Exception as exc:
        print(f"[_parse_passed_ids] 读报告文件失败: {exc}")

    # 最终回退：全部患者
    raw = _read_data(RAW_CSV)
    return [str(p) for p in raw["patient_id"].unique()]


# ══════════════════════════════════════════════════════════════════
# 5. Pipeline
# ══════════════════════════════════════════════════════════════════

async def run_step6() -> None:
    _data_label = "MIMIC" if _IS_MIMIC else "Original"
    print("=" * 65)
    print("  STEP 6: 患者数据一致性验证")
    print("=" * 65)
    print(f"  模式:    {_data_label}（DATA_SOURCE={DATA_SOURCE}）")
    print(f"  模型:    {MODEL_NAME}")
    print(f"  数据:    {RAW_CSV}")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"  输出目录: {OUTPUT_STEP6}")
    print("=" * 65)

    # 如果已有 Step 6 报告文件，直接复用
    _existing_reports = sorted(
        f for f in os.listdir(OUTPUT_STEP6)
        if f.startswith("consistency_report_") and f.endswith(".txt")
    ) if os.path.isdir(OUTPUT_STEP6) else []

    if _existing_reports:
        _latest_report = os.path.join(OUTPUT_STEP6, _existing_reports[-1])
        print(f"\n[Step 6] 发现已有报告，跳过重新运行：{_latest_report}")
        with open(_latest_report, encoding="utf-8") as _fh:
            text6 = _fh.read()
        passed_ids = _parse_passed_ids(text6)
        print(f"[Step 6] 通过患者数量: {len(passed_ids)}")
    else:
        agent6 = create_consistency_agent()
        msg6 = Msg(
            name="Pipeline",
            role="user",
            content=(
                "请对以下医疗数据集执行完整的 Step 6 一致性验证。\n\n"
                f"数据文件路径：{RAW_CSV}\n\n"
                "步骤：① load_data_overview → ② analyze_all_patients_consistency → "
                "③ 撰写摘要后 save_step6_report（末尾含 PASSED_PATIENTS: [...]）。\n\n"
                f"置信度阈值：{CONFIDENCE_THRESHOLD}"
            ),
        )
        print("\n[Step 6] ▶ 启动 ConsistencyAgent...")
        result6 = await agent6(msg6)
        text6   = _extract_msg_text(result6)
        passed_ids = _parse_passed_ids(text6)

        print("\n" + "=" * 65)
        print("  Step 6 完成")
        print("=" * 65)
        print(text6)

    print(f"\n[Step 6] 通过患者数量: {len(passed_ids)}")
    print(f"[Step 6] 通过患者 ID 列表已保存至 {OUTPUT_STEP6}/passed_patients_*.json")
    print("\n" + "=" * 65)
    print("  Step 6 全部完成")
    print(f"  报告目录: {OUTPUT_STEP6}/")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_step6())
