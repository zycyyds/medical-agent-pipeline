#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_table.py — 将 OCR 抽取实体与模板 CSV 合并，输出到 results 目录

用法：
  python fill_table.py --reorg reorganized_output --results reorganized_output/results_v2
  python fill_table.py  # 使用默认路径

流程：
  1. 扫描 reorganized_output/<patient_id>/table/副本病例数据_脱敏-总.csv（只读）
  2. 读取最新的 entities_*.csv（抽取实体长表）
  3. 用规则/LLM 将实体 name/value 匹配到模板的空列
  4. 只填写原来为空的字段，有值的字段保持不动
  5. 将合并后的结果写到 output_dir/<patient_id>/副本病例数据_脱敏-总.csv
     原始文件不做任何修改
"""

import argparse
import asyncio
import csv
import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import get_api_config

# ---------------------------------------------------------------------------
# LLM 客户端
# ---------------------------------------------------------------------------

_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        import openai
        cfg = get_api_config()
        _llm_client = openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
    return _llm_client


async def _call_llm(prompt: str) -> Optional[str]:
    try:
        cfg = get_api_config()
        if not cfg["api_key"]:
            return None
        client = _get_llm_client()
        resp = await client.chat.completions.create(
            model=cfg["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip() if resp.choices else None
    except Exception as e:
        print(f"  [LLM] 调用失败: {e}", file=sys.stderr)
        return None


def _parse_json(content: str) -> Optional[Any]:
    if not content:
        return None
    for marker in ["```json", "```"]:
        if marker in content:
            parts = content.split(marker)
            if len(parts) >= 2:
                try:
                    return json.loads(parts[1].split("```")[0].strip())
                except json.JSONDecodeError:
                    pass
    try:
        s, e = content.find("{"), content.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(content[s:e])
    except json.JSONDecodeError:
        pass
    try:
        s, e = content.find("["), content.rfind("]") + 1
        if s != -1 and e > s:
            return json.loads(content[s:e])
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# CSV / Excel 读写
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> Tuple[List[str], List[Dict]]:
    """返回 (columns, rows)，rows 是 dict 列表。"""
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                cols = list(reader.fieldnames or [])
                rows = [dict(r) for r in reader]
            return cols, rows
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解析文件编码: {path}")


def _read_xlsx(path: str) -> Tuple[List[str], List[Dict]]:
    """读取 Excel 文件，返回 (columns, rows)。"""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = iter(ws.iter_rows(values_only=True))
    header = [str(c) if c is not None else "" for c in next(rows_iter, [])]
    if not header:
        wb.close()
        raise ValueError(f"Excel 文件无表头: {path}")
    rows = []
    for raw in rows_iter:
        row = {h: (str(v) if v is not None else "") for h, v in zip(header, raw)}
        rows.append(row)
    wb.close()
    return header, rows


def _read_template(path: str) -> Tuple[List[str], List[Dict]]:
    """根据扩展名自动选择 CSV 或 Excel 读取。"""
    if path.lower().endswith((".xlsx", ".xls")):
        return _read_xlsx(path)
    return _read_csv(path)


def _write_csv(path: str, cols: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _write_xlsx_highlighted(
    path: str,
    cols: List[str],
    row: Dict,
    mapping: Dict[str, Dict],
) -> None:
    """
    写带高亮标注的 Excel 文件：
    - Sheet1 "填写结果"：
        - 新填入的单元格：黄色背景 + 加粗
        - 原始已有值的单元格：白色背景
        - 空单元格（未能填入）：浅灰背景
    - Sheet2 "溯源"：每个填入字段的来源实体、图片、原始 OCR 文本
    """
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment

    YELLOW = PatternFill("solid", fgColor="FFFF00")
    GRAY   = PatternFill("solid", fgColor="F2F2F2")
    WHITE  = PatternFill("solid", fgColor="FFFFFF")
    BOLD9  = Font(bold=True, size=9)
    NORM9  = Font(size=9)
    WRAP   = Alignment(wrap_text=True)

    wb = Workbook()

    # ── Sheet1：填写结果 ──────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "填写结果"
    filled_cols = set(mapping.keys())
    _EMPTY = {"", "nan", "NaN", "None", "none"}

    for col_idx, col_name in enumerate(cols, start=1):
        # 表头
        hc = ws1.cell(row=1, column=col_idx, value=col_name)
        hc.font = BOLD9
        hc.alignment = WRAP
        # 数据
        val = row.get(col_name, "")
        val_str = "" if str(val).strip() in _EMPTY else str(val).strip()
        dc = ws1.cell(row=2, column=col_idx, value=val_str)
        dc.alignment = WRAP
        if col_name in filled_cols:
            dc.fill = YELLOW
            dc.font = BOLD9
        elif val_str:
            dc.fill = WHITE
            dc.font = NORM9
        else:
            dc.fill = GRAY
            dc.font = NORM9

    for col_idx, col_name in enumerate(cols, start=1):
        ws1.column_dimensions[
            ws1.cell(row=1, column=col_idx).column_letter
        ].width = min(max(len(col_name) * 1.2, 8), 30)

    # ── Sheet2：溯源 ──────────────────────────────────────────────────
    ws2 = wb.create_sheet("溯源")
    headers = ["列名", "填入值", "实体名称", "实体类别", "来源图片/文件", "原始OCR文本"]
    for col_idx, h in enumerate(headers, start=1):
        hc = ws2.cell(row=1, column=col_idx, value=h)
        hc.font = BOLD9
        hc.alignment = WRAP

    for row_idx, (col_name, info) in enumerate(mapping.items(), start=2):
        ws2.cell(row=row_idx, column=1, value=col_name).alignment = WRAP
        ws2.cell(row=row_idx, column=2, value=info.get("value", "")).alignment = WRAP
        ws2.cell(row=row_idx, column=3, value=info.get("entity_name", "")).alignment = WRAP
        ws2.cell(row=row_idx, column=4, value=info.get("category", "")).alignment = WRAP
        ws2.cell(row=row_idx, column=5, value=info.get("source_file", "")).alignment = WRAP
        ws2.cell(row=row_idx, column=6, value=info.get("original_text", "")).alignment = WRAP

    ws2.column_dimensions["A"].width = 40
    ws2.column_dimensions["B"].width = 20
    ws2.column_dimensions["C"].width = 20
    ws2.column_dimensions["D"].width = 12
    ws2.column_dimensions["E"].width = 18
    ws2.column_dimensions["F"].width = 50

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
# 发现文件
# ---------------------------------------------------------------------------

def find_latest_entities_csv(results_dir: str) -> Optional[str]:
    """在 results_v2/ 下找最新的 entities_*.csv（非 long 版本）。"""
    pattern = os.path.join(results_dir, "**/entities_[0-9]*.csv")
    # 优先找非 long 版
    candidates = [p for p in glob.glob(pattern, recursive=True)
                  if "long" not in os.path.basename(p)]
    if not candidates:
        candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def find_patient_template_csvs(reorg_dir: str) -> Dict[str, str]:
    """返回 {patient_id: csv_or_xlsx_path}。"""
    result = {}
    for entry in sorted(os.listdir(reorg_dir)):
        full = os.path.join(reorg_dir, entry)
        if not os.path.isdir(full) or entry.startswith("_") or entry == "results_v2":
            continue
        table_dir = os.path.join(full, "table")
        if not os.path.isdir(table_dir):
            continue
        for fname in os.listdir(table_dir):
            if fname.endswith(".csv") or fname.endswith(".xlsx") or fname.endswith(".xls"):
                result[entry] = os.path.join(table_dir, fname)
                break
    return result


# ---------------------------------------------------------------------------
# 核心：LLM 映射
# ---------------------------------------------------------------------------

_EMPTY_VALUES = {"", "nan", "NaN", "None", "none", "NULL", "null"}


def _is_empty(val: Any) -> bool:
    return str(val).strip() in _EMPTY_VALUES


def _build_mapping_prompt(
    patient_id: str,
    entities: List[Dict],
    empty_cols: List[str],
) -> str:
    """构建让 LLM 做实体→列名映射的 prompt。"""
    ent_lines = []
    for i, e in enumerate(entities):
        name = e.get("name", "")
        val = e.get("value", "")
        unit = e.get("unit", "")
        cat = e.get("category", "")
        duration = e.get("duration", "")
        val_str = str(val) if not _is_empty(val) else "(无具体值)"
        unit_str = f" {unit}" if unit and not _is_empty(unit) else ""
        dur_str = f"，持续{duration}" if duration and not _is_empty(duration) else ""
        ent_lines.append(f"  [{i}] [{cat}] {name}: {val_str}{unit_str}{dur_str}")

    col_lines = [f"  [{j}] {c}" for j, c in enumerate(empty_cols)]

    prompt = f"""你是医学数据录入专家。请将下面的"抽取实体"匹配到"目标列"中，并输出填写值。

## 患者ID
{patient_id}

## 抽取到的实体（共{len(entities)}条）
{chr(10).join(ent_lines)}

## 需要填写的空列（共{len(empty_cols)}列，仅选出你能确定匹配的列）
{chr(10).join(col_lines)}

## 匹配规则
1. 只填写你**高置信度**确定能匹配的列，不确定的跳过
2. 列名含"慢相角速度"→ 填数值（°/s）；含"增益"→ 填增益数值；含"试验结果"→ 填阴性/阳性/未见眼震等
3. 列名含"自发眼震"→ 对应实体"自发眼震"的 value；含"扫视试验"→ 对应"扫视"的 value
4. 列名含"跟踪试验"→ 对应"视跟踪-水平方向"；含"视动眼震"→ 对应"视动眼震-水平方向"
5. 列名含"Dix-hallpike"→ 对应 "Dix-Hallpike" 相关实体；含"Roll试验"→ 对应"Roll-test"
6. 实验室检查：列名含"类风湿因子"→ 实体"风湿因子"value；含"C反应蛋白"→ 实体"超敏C反应蛋白"value
7. 若同一列有多个来源（多次检查），取**最后一次**（source_file 编号最大）的值
8. 若列名含"(详细)"或"其他"，填写原始文本描述

## 输出格式（JSON，不要输出其他内容）
{{
  "填写结果": [
    {{"列名": "精确的列名字符串", "值": "填入的值"}},
    ...
  ],
  "跳过": ["无法匹配的列名", ...]
}}
"""
    return prompt


async def _llm_map_entities_to_cols(
    patient_id: str,
    entities: List[Dict],
    empty_cols: List[str],
    chunk_size: int = 80,
) -> Dict[str, Dict]:
    """
    让 LLM 将实体映射到空列。
    返回 {col_name: {"value": str, "entity_name": str, "category": str,
                      "source_file": str, "original_text": str}}
    """
    if not entities or not empty_cols:
        return {}

    # 建立索引方便溯源：实体编号 → 实体 dict
    ent_index = {str(i): ent for i, ent in enumerate(entities)}
    result: Dict[str, Dict] = {}

    for i in range(0, len(empty_cols), chunk_size):
        batch_cols = empty_cols[i:i + chunk_size]
        prompt = _build_mapping_prompt(patient_id, entities, batch_cols)
        raw = await _call_llm(prompt)
        if not raw:
            continue
        parsed = _parse_json(raw)
        if not parsed or not isinstance(parsed, dict):
            continue
        for item in parsed.get("填写结果", []):
            col = item.get("列名", "").strip()
            val = item.get("值", "")
            if not col or not val or _is_empty(val):
                continue
            # LLM 可能返回实体编号（可选），用于精确溯源
            ent_idx = str(item.get("实体编号", ""))
            src_ent = ent_index.get(ent_idx, {})
            result[col] = {
                "value":         str(val).strip(),
                "entity_name":   src_ent.get("name", ""),
                "category":      src_ent.get("category", ""),
                "source_file":   src_ent.get("source_file", ""),
                "original_text": src_ent.get("original_text", ""),
            }

    return result


# ---------------------------------------------------------------------------
# 规则回退映射（无 LLM 时）
# ---------------------------------------------------------------------------

# 实体名到列名关键词的规则映射
_RULE_MAP = [
    # (实体name关键词, 列名关键词, 取entity的哪个字段)
    # ── 眼动检查 ──────────────────────────────────────────
    ("扫视", "扫视试验", "value"),
    ("视跟踪", "跟踪试验", "value"),          # 含 视跟踪水平方向 / 视跟踪-水平方向 / 视跟踪
    ("视动眼震", "视动眼震", "value"),         # 含 视动眼震水平方向 / 视动眼震-水平方向 / 视动眼震
    ("增益(左)", "增益(左)", "value"),
    ("增益(右)", "增益(右)", "value"),
    ("视动增益-左", "增益(左)", "value"),
    ("视动增益-右", "增益(右)", "value"),
    ("左侧视动眼震", "增益(左)", "value"),
    ("右侧视动眼震", "增益(右)", "value"),
    ("视动眼震左侧", "增益(左)", "value"),
    ("视动眼震右侧", "增益(右)", "value"),
    ("凝视", "凝视试验", "value"),
    ("自发眼震", "自发眼震", "value"),
    ("摇头眼震", "摇头眼震", "value"),
    ("固视抑制", "固视抑制", "value"),
    # ── 位置试验 ──────────────────────────────────────────
    ("Dix-Hallpike", "Dix-hallpike", "value"),
    ("Roll-test", "Roll试验", "value"),
    ("Roll试验", "Roll试验", "value"),
    ("深悬头位试验", "深悬头位", "value"),
    ("深悬头位", "深悬头位", "value"),
    # ── 动态位置试验 ──────────────────────────────────────
    ("动态位置试验", "动态位置试验", "value"),
    # ── 静态位置试验（整体结果，映射到"静态位置试验"列）──────
    ("静态位置试验", "眼震视图等检查-静态位置试验", "value"),
    # ── 静态位置试验各体位（短名和长名两种OCR写法）──────────
    ("静态位置试验坐位", "坐位", "value"),
    ("静态位置试验仰卧位", "仰卧位", "value"),
    ("静态位置试验悬头左侧", "左悬头", "value"),
    ("静态位置试验悬头位", "悬头", "value"),
    ("静态位置试验悬头右侧", "右悬头", "value"),
    ("静态位置试验坐起", "坐起", "value"),
    # 短名（OCR直接抽成体位名而不带前缀）
    ("坐位眼震", "坐位", "value"),
    ("左侧卧眼震", "左侧卧位", "value"),
    ("仰卧位眼震", "仰卧位", "value"),
    ("右侧卧眼震", "右侧卧位", "value"),
    ("悬头左侧眼震", "左侧悬头位", "value"),
    ("悬头右侧眼震", "右侧悬头位", "value"),
    ("悬头位眼震", "悬头", "value"),
    ("左侧卧位", "左侧卧位", "value"),
    ("右侧卧位", "右侧卧位", "value"),
    ("左侧卧", "左侧卧位", "value"),
    ("右侧卧", "右侧卧位", "value"),
    ("坐位", "坐位", "value"),
    ("仰卧位", "仰卧位", "value"),
    ("悬头左侧", "左侧悬头位", "value"),
    ("悬头右侧", "右侧悬头位", "value"),
    ("右侧悬头位", "右侧悬头位", "value"),
    ("左侧悬头位", "左侧悬头位", "value"),
    # ── 双温试验（温度→热/冷，侧别→左/右）─────────────────
    ("左侧50℃慢相速度", "左热", "value"),
    ("左侧热", "左热", "value"),
    ("右侧50℃慢相速度", "右热", "value"),
    ("右侧热", "右热", "value"),
    ("左侧24℃慢相速度", "左冷", "value"),
    ("左侧冷", "左冷", "value"),
    ("右侧24℃慢相速度", "右冷", "value"),
    ("右侧冷", "右冷", "value"),
    # 双温试验 OCR 多种写法
    ("右侧 50℃", "右热", "value"),
    ("左侧 50℃", "左热", "value"),
    ("右侧50℃", "右热", "value"),
    ("左侧50℃", "左热", "value"),
    ("50℃右侧", "右热", "value"),
    ("50℃左侧", "左热", "value"),
    ("右侧 24℃", "右冷", "value"),
    ("左侧 24℃", "左冷", "value"),
    ("右侧24℃", "右冷", "value"),
    ("左侧24℃", "左冷", "value"),
    ("24℃右侧", "右冷", "value"),
    ("24℃左侧", "左冷", "value"),
    ("温度右侧慢向角速度", "右热", "value"),   # 温度试验原始写法
    ("温度左侧慢向角速度", "左热", "value"),
    # ── CP 值（半规管轻瘫，括号格式多样）────────────────────
    ("CP(R)", "CP值", "value"),
    ("CP（R）", "CP值", "value"),
    ("CP(L)", "CP值", "value"),
    ("CP（L）", "CP值", "value"),
    ("CP(R）", "CP值", "value"),          # OCR 混合括号
    ("校正值CP", "CP值", "value"),
    ("CP右侧", "CP值", "value"),
    ("CP左侧", "CP值", "value"),
    ("半规管轻瘫值(CP)", "CP值", "value"),
    ("半规管轻瘫", "CP值", "value"),
    # ── 免疫/自身抗体 ─────────────────────────────────────
    ("风湿因子", "类风湿因子", "value"),
    ("类风湿因子", "类风湿因子", "value"),
    ("超敏C反应蛋白", "C反应蛋白", "value"),
    ("C反应蛋白", "C反应蛋白", "value"),
    ("甲状腺球蛋白抗体", "甲状腺球蛋白抗体", "value"),
    ("甲状腺过氧化物酶抗体", "甲状腺微粒抗体", "value"),
    ("抗心磷脂抗体IgM", "抗ACA", "value"),
    ("抗心磷脂抗体IgG", "抗ACA", "value"),
    ("抗心磷脂抗体IgA", "抗ACA", "value"),
    ("抗β2糖蛋白1抗体", "抗磷脂抗体", "value"),
    ("抗B2糖蛋白1抗体", "抗磷脂抗体", "value"),
]


def _rule_based_map(entities: List[Dict], empty_cols: List[str]) -> Dict[str, Dict]:
    """
    纯规则的实体→列名映射（LLM 不可用时使用）。
    返回 {col_name: {"value": str, "entity_name": str, "category": str,
                      "source_file": str, "original_text": str}}
    """
    result: Dict[str, Dict] = {}
    for ent in entities:
        name = str(ent.get("name", "")).strip()
        val = ent.get("value", "")
        if _is_empty(val):
            continue
        for ent_kw, col_kw, field in _RULE_MAP:
            if ent_kw.lower() in name.lower():
                for col in empty_cols:
                    if col_kw in col and col not in result:
                        result[col] = {
                            "value":         str(ent.get(field, val)).strip(),
                            "entity_name":   name,
                            "category":      ent.get("category", ""),
                            "source_file":   ent.get("source_file", ""),
                            "original_text": ent.get("original_text", ""),
                        }
                        break
    return result


# ---------------------------------------------------------------------------
# 主处理逻辑
# ---------------------------------------------------------------------------

async def fill_patient(
    patient_id: str,
    template_csv: str,
    entities: List[Dict],
    output_dir: str,
    use_llm: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    读取 template_csv（只读），将实体填入空列，把合并结果写到 output_dir。

    Args:
        output_dir: 结果写到此目录下的 <patient_id>/副本病例数据_脱敏-总.csv

    Returns:
        {"patient_id": ..., "filled": {col: val}, "output_path": str, "error": ...}
    """
    try:
        cols, rows = _read_template(template_csv)
    except Exception as e:
        return {"patient_id": patient_id, "error": str(e), "filled": {}, "output_path": ""}

    if not rows:
        return {"patient_id": patient_id, "error": "CSV 无数据行", "filled": {}, "output_path": ""}

    row = rows[0]  # 每个病人只有一行

    # 找出空列（排除单位列，用 rsplit 取最后一段判断，覆盖 .10/.24 等所有编号）
    def _is_unit_col(col: str) -> bool:
        return "单位" in col.rsplit("-", 1)[-1]

    empty_cols = [
        c for c in cols
        if _is_empty(row.get(c, ""))
        and not _is_unit_col(c)
        and c != "id"
    ]

    if verbose:
        print(f"  [fill] 患者 {patient_id}: {len(cols)} 列，{len(empty_cols)} 个空列，{len(entities)} 条实体")

    mapping: Dict[str, Dict] = {}
    if empty_cols and entities:
        # 规则优先；LLM 模式下补充规则未覆盖的语义相关列
        rule_mapping = _rule_based_map(entities, empty_cols)
        if use_llm:
            _EVALUABLE_PREFIXES = (
                "辅助检查-眼震视图等检查-",
                "辅助检查-实验室检查-",
                "辅助检查-耳科及眼科检查-",
                "床旁查体-",
                "第一次复诊-",
                "第二次复诊-",
                "第三次复诊-",
            )
            # 提取实体中出现的关键词（长度>=2的中文词）
            import re
            ent_keywords: set = set()
            for e in entities:
                for tok in re.findall(r'[一-鿿]{2,}|[A-Za-z]{3,}', str(e.get("name", ""))):
                    ent_keywords.add(tok.lower())

            rule_filled = set(rule_mapping)
            llm_candidate_cols = []
            seen_last: set = set()  # 同名尾段只发一列（去重复列结构）
            for c in empty_cols:
                if c in rule_filled:
                    continue
                if not any(c.startswith(pfx) for pfx in _EVALUABLE_PREFIXES):
                    continue
                # 取列名最后一段（去掉编号，如 .1/.2）
                last = re.sub(r'\.\d+$', '', c.rsplit("-", 1)[-1]).strip()
                if last in seen_last:
                    continue
                # 检查列名最后段是否与任意实体关键词有重叠
                col_kws = set(re.findall(r'[一-鿿]{2,}|[A-Za-z]{3,}', last.lower()))
                if col_kws & ent_keywords:
                    llm_candidate_cols.append(c)
                    seen_last.add(last)

            if verbose:
                print(f"  [llm] 候选列数: {len(llm_candidate_cols)}")
            if llm_candidate_cols:
                llm_mapping = await _llm_map_entities_to_cols(
                    patient_id, entities, llm_candidate_cols
                )
                mapping = {**llm_mapping, **rule_mapping}  # 规则覆盖 LLM
            else:
                mapping = rule_mapping
        else:
            mapping = rule_mapping

    if verbose and mapping:
        print(f"  [fill] 患者 {patient_id}: 映射 {len(mapping)} 个字段")
        for col, info in list(mapping.items())[:10]:
            print(f"    {col!r} ← {info['value']!r}  (来自 {info['source_file']})")

    # 确定输出路径（写到 output_dir，不碰原文件）
    out_fname = os.path.basename(template_csv)
    out_path = os.path.join(output_dir, patient_id, out_fname)

    if not dry_run:
        # 将映射值合并到行副本
        merged_row = dict(row)
        for col, info in mapping.items():
            if col in merged_row:
                merged_row[col] = info["value"]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        _write_csv(out_path, cols, [merged_row])

        # 同时输出带高亮标注 + 溯源 sheet 的 xlsx
        xlsx_path = os.path.splitext(out_path)[0] + "_highlighted.xlsx"
        _write_xlsx_highlighted(xlsx_path, cols, merged_row,
                                mapping=mapping)

    return {
        "patient_id": patient_id,
        "filled": mapping,
        "output_path": out_path,
        "template_csv": template_csv,
    }


async def run(
    reorg_dir: str,
    results_dir: Optional[str],
    output_dir: str,
    use_llm: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    patient_filter: Optional[List[str]] = None,
) -> None:
    # 找模板 CSV
    templates = find_patient_template_csvs(reorg_dir)
    if not templates:
        print(f"未找到模板 CSV（在 {reorg_dir} 下）", file=sys.stderr)
        return
    print(f"找到 {len(templates)} 个病人的模板 CSV")

    # 找实体 CSV
    search_dir = results_dir or os.path.join(reorg_dir, "results_v2")
    entities_csv = find_latest_entities_csv(search_dir)
    if not entities_csv:
        print(f"未找到 entities_*.csv（在 {search_dir} 下）", file=sys.stderr)
        return
    print(f"使用实体文件: {entities_csv}")

    # 读取实体
    try:
        _, ent_rows = _read_csv(entities_csv)
    except Exception as e:
        print(f"读取实体文件失败: {e}", file=sys.stderr)
        return
    print(f"共 {len(ent_rows)} 条实体记录")

    # 按 patient_id 分组
    ent_by_patient: Dict[str, List[Dict]] = {}
    for row in ent_rows:
        pid = str(row.get("patient_id", "")).strip()
        ent_by_patient.setdefault(pid, []).append(row)

    # 汇总宽表路径（所有病人合并到一个 CSV + 一个 Excel 单 Sheet）
    os.makedirs(output_dir, exist_ok=True)
    merged_csv  = os.path.join(output_dir, "filled_merged.csv")
    merged_xlsx = os.path.join(output_dir, "filled_merged_highlighted.xlsx")
    _merged_lock = asyncio.Lock()
    _merged_header_written = [False]

    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    _YELLOW = PatternFill("solid", fgColor="FFFF00")
    _GRAY   = PatternFill("solid", fgColor="F2F2F2")
    _WHITE  = PatternFill("solid", fgColor="FFFFFF")
    _BOLD9  = Font(bold=True, size=9)
    _NORM9  = Font(size=9)
    _WRAP   = Alignment(wrap_text=True)
    _merged_wb = Workbook()
    _merged_ws = _merged_wb.active
    _merged_ws.title = "填写结果汇总"
    _merged_row_idx = [2]   # 数据从第2行开始（第1行是表头）
    _EMPTY_VALS = {"", "nan", "NaN", "None", "none"}

    async def _fill_and_save(pid: str, csv_path: str, ents: List[Dict]) -> Dict:
        result = await fill_patient(pid, csv_path, ents,
                                    output_dir=output_dir,
                                    use_llm=use_llm,
                                    dry_run=dry_run,
                                    verbose=verbose)
        # 实时追加到汇总 CSV（跳过 Excel 写入，太慢）
        if not dry_run and not result.get("error") and result.get("output_path"):
            try:
                cols, rows = _read_csv(result["output_path"])
                if rows:
                    async with _merged_lock:
                        mode = "w" if not _merged_header_written[0] else "a"
                        with open(merged_csv, mode, encoding="utf-8-sig", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                            if not _merged_header_written[0]:
                                writer.writeheader()
                                _merged_header_written[0] = True
                            writer.writerows(rows)
            except Exception:
                pass
        return result

    # 并发处理所有病人
    tasks_list = []
    task_pids = []
    for pid, csv_path in templates.items():
        if patient_filter and pid not in patient_filter:
            continue
        ents = ent_by_patient.get(pid, [])
        tasks_list.append(_fill_and_save(pid, csv_path, ents))
        task_pids.append(pid)

    import tqdm as _tqdm_module
    bar = _tqdm_module.tqdm(total=len(tasks_list), desc="回填进度", unit="人", dynamic_ncols=True)
    fill_results = []
    for coro in asyncio.as_completed(tasks_list):
        result = await coro
        fill_results.append(result)
        bar.update(1)
        bar.set_postfix_str(f"最近: {result.get('patient_id', '')}", refresh=False)
    bar.close()

    # 汇总打印
    total_filled = 0
    for r in fill_results:
        pid = r["patient_id"]
        n = len(r.get("filled", {}))
        total_filled += n
        if r.get("error"):
            print(f"  ✗ 患者 {pid}: {r['error']}")
        elif n > 0:
            print(f"  ✓ 患者 {pid}: 填写 {n} 个字段 → {r.get('output_path', '')}")
        else:
            print(f"  - 患者 {pid}: 无可填字段")

    mode = "[演习，未写入]" if dry_run else "[已写入]"
    print(f"\n完成 {mode}：共填写 {total_filled} 个字段，涉及 {len(fill_results)} 个病人")
    if not dry_run and _merged_header_written[0]:
        print(f"汇总宽表: {merged_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fill_table",
        description="将 OCR 抽取实体与模板 CSV 合并，输出到 results 目录",
    )
    p.add_argument("--reorg", default="reorganized_output",
                   help="reorganized_output 目录路径（默认: reorganized_output）")
    p.add_argument("--results", default=None,
                   help="entities_*.csv 所在的 results 目录（默认: --reorg/results_v2）")
    p.add_argument("--output", default=None,
                   help="合并 CSV 写出目录（默认: --results 同级的 filled_tables/）")
    p.add_argument("--patient", nargs="*", metavar="ID",
                   help="只处理指定病人 ID（不指定则全部处理）")
    p.add_argument("--no-llm", dest="no_llm", action="store_true",
                   help="使用规则回退（不调用 LLM）")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="演习模式：只打印，不写文件")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="显示详细日志")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    reorg = args.reorg
    if not os.path.isabs(reorg):
        reorg = os.path.join(_HERE, reorg)
    results = args.results
    if results and not os.path.isabs(results):
        results = os.path.join(_HERE, results)

    # output_dir 默认放在 results 同级的 filled_tables/ 下
    if args.output:
        output_dir = args.output if os.path.isabs(args.output) else os.path.join(_HERE, args.output)
    else:
        base = results or os.path.join(reorg, "results_v2")
        output_dir = os.path.join(os.path.dirname(base), "filled_tables")

    asyncio.run(run(
        reorg_dir=reorg,
        results_dir=results,
        output_dir=output_dir,
        use_llm=not args.no_llm,
        dry_run=args.dry_run,
        verbose=args.verbose,
        patient_filter=args.patient or None,
    ))


if __name__ == "__main__":
    main()
