#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — 评估 fill_table 的填写质量

两个维度：

1. 遮盖测试（Mask Test）
   - 从原始 CSV 中找出"本来就有人工填写值"的字段作为 ground truth
   - 临时清空这些字段，跑 fill，与 ground truth 比对
   - 输出：precision / recall / F1，以及逐字段明细

2. 覆盖率报告（Coverage Report）
   - 统计抽取的实体中有多少成功映射到列（mapping rate）
   - 统计哪些原本为空、且有对应实体关键词的列仍未被填入（漏填）
   - 统计哪些实体没有任何一列与之匹配（漏映射）

用法：
  python evaluate.py
  python evaluate.py --patient 36907
  python evaluate.py --no-llm
  python evaluate.py --output eval_results/
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fill_table import (
    _is_empty, _read_csv, _write_csv, _write_xlsx_highlighted,
    _rule_based_map, find_latest_entities_csv, find_patient_template_csvs,
)


# ---------------------------------------------------------------------------
# 遮盖测试
# ---------------------------------------------------------------------------

# 单位占位列不算 ground truth
_UNIT_SUFFIXES = ("单位", "单位.1", "单位.2", "单位.3", "单位.4",
                  "单位.5", "单位.6", "单位.7", "单位.8", "单位.9")

_SKIP_COLS = {"id"}

# 只评估这些前缀下的字段（OCR 应能覆盖的范围）
_EVALUABLE_PREFIXES = (
    "辅助检查-眼震视图等检查-",
    "辅助检查-实验室检查-",
    "辅助检查-耳科及眼科检查-",
    "床旁查体-",
    "第一次复诊-",
    "第二次复诊-",
    "第三次复诊-",
)

# 值为是/否/0/1 的勾选型字段不参与评估（无法从 OCR 抽取）
_CHECKBOX_VALUES = {"是", "否", "0", "1", "0.0", "1.0", "阴性", "阳性"}


def _is_evaluable_col(col: str, val: str) -> bool:
    """判断该列是否应纳入遮盖测试评估范围。"""
    if not any(col.startswith(pfx) for pfx in _EVALUABLE_PREFIXES):
        return False
    # 过滤所有单位占位列（列名末段含"单位"的，包括 "单位.10" 等）
    last = col.rsplit("-", 1)[-1]   # 取最后一个"-"之后的部分
    if "单位" in last:
        return False
    if val.strip() in _CHECKBOX_VALUES:
        return False
    return True


def _extract_ground_truth(cols: List[str], row: Dict) -> Dict[str, str]:
    """
    从原始行中提取有人工填写值的字段作为 ground truth。
    只保留 OCR 理论上应能填的列（辅助检查、床旁查体等）。
    返回 {col: original_value}（只含非空、可评估列）。
    """
    gt = {}
    for c in cols:
        if c in _SKIP_COLS:
            continue
        val = str(row.get(c, "")).strip()
        if _is_empty(val):
            continue
        if _is_evaluable_col(c, val):
            gt[c] = val
    return gt


def _normalize(val: str) -> str:
    """宽松归一化：去空格、转小写、去单位符号，用于比对。"""
    v = val.strip().lower()
    for ch in ["°/s", "°", "%", " ", "°"]:
        v = v.replace(ch, "")
    return v


def _values_match(pred: str, gold: str) -> bool:
    """判断预测值和 ground truth 是否匹配（宽松比对）。"""
    if pred == gold:
        return True
    pn, gn = _normalize(pred), _normalize(gold)
    if pn == gn:
        return True
    # 数值近似（±5%）
    try:
        pf, gf = float(pn), float(gn)
        if gf != 0 and abs(pf - gf) / abs(gf) < 0.05:
            return True
    except ValueError:
        pass
    # 子串包含
    if pn in gn or gn in pn:
        return True
    return False


async def evaluate_patient(
    patient_id: str,
    template_csv: str,
    entities: List[Dict],
    use_llm: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    对单个病人做遮盖测试 + 覆盖率分析。
    """
    cols, rows = _read_csv(template_csv)
    if not rows:
        return {"patient_id": patient_id, "error": "CSV 无数据行"}
    row = rows[0]

    # 1. 提取 ground truth（原始有值的列）
    gt = _extract_ground_truth(cols, row)

    # 2. 遮盖：清空 gt 列，构造"仅空列"行
    masked_row = dict(row)
    for c in gt:
        masked_row[c] = ""
    all_cols_set = set(cols)
    empty_cols_after_mask = [
        c for c in cols
        if _is_empty(masked_row.get(c, ""))
        and not any(c.endswith(sfx) for sfx in _UNIT_SUFFIXES)
        and c not in _SKIP_COLS
    ]

    # 3. 跑映射（只用规则，保证评估可复现；LLM 模式下也可选）
    if use_llm:
        import re
        from fill_table import _llm_map_entities_to_cols, _rule_based_map as _rb
        rule_mapping = _rb(entities, empty_cols_after_mask)
        # 对 LLM 候选列做关键词过滤（同 fill_table.py 策略）
        ent_keywords: set = set()
        for e in entities:
            for tok in re.findall(r'[一-鿿]{2,}|[A-Za-z]{3,}', str(e.get("name", ""))):
                ent_keywords.add(tok.lower())
        rule_filled = set(rule_mapping)
        llm_cols: List[str] = []
        seen_last: set = set()
        for c in empty_cols_after_mask:
            if c in rule_filled:
                continue
            last = re.sub(r'\.\d+$', '', c.rsplit("-", 1)[-1]).strip()
            if last in seen_last:
                continue
            col_kws = set(re.findall(r'[一-鿿]{2,}|[A-Za-z]{3,}', last.lower()))
            if col_kws & ent_keywords:
                llm_cols.append(c)
                seen_last.add(last)
        llm_mapping = await _llm_map_entities_to_cols(patient_id, entities, llm_cols) if llm_cols else {}
        mapping = {**llm_mapping, **rule_mapping}
    else:
        mapping = _rule_based_map(entities, empty_cols_after_mask)

    # 4. 计算 precision / recall
    #    只评估"gt 与 empty_cols_after_mask 的交集"——即原来有值、被我们遮盖、系统尝试填的列
    evaluable_cols = [c for c in gt if c in set(empty_cols_after_mask)]

    tp, fp, fn = 0, 0, 0
    details: List[Dict] = []

    # TP/FN：系统预测了 evaluable_cols 中的哪些，对了多少
    for col in evaluable_cols:
        gold = gt[col]
        if col in mapping:
            pred = mapping[col]["value"]
            correct = _values_match(pred, gold)
            if correct:
                tp += 1
                status = "TP"
            else:
                fp += 1   # 填了但填错
                status = "FP(错误)"
            details.append({
                "patient_id": patient_id, "列名": col,
                "ground_truth": gold, "预测值": pred,
                "状态": status,
                "来源实体": mapping[col].get("entity_name", ""),
                "来源图片": mapping[col].get("source_file", ""),
            })
        else:
            fn += 1   # 应填但未填
            details.append({
                "patient_id": patient_id, "列名": col,
                "ground_truth": gold, "预测值": "",
                "状态": "FN(漏填)",
                "来源实体": "", "来源图片": "",
            })

    # FP：系统填了但 ground truth 中没有（映射到的是本来就空的列，无法判断对错，跳过）

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1        = (2 * precision * recall / (precision + recall)
                 if precision is not None and recall is not None
                 and (precision + recall) > 0 else None)

    # 5. 覆盖率：实体→列 的映射率
    total_entities = len(entities)
    mapped_entities = len({
        info.get("entity_name", "") for info in mapping.values()
        if info.get("entity_name")
    })

    # 漏映射的实体（有值但没有任何列与之匹配）
    mapped_ent_names = {info.get("entity_name", "") for info in mapping.values()}
    unmapped_entities = [
        e for e in entities
        if not _is_empty(e.get("value", ""))
        and str(e.get("name", "")) not in mapped_ent_names
    ]

    if verbose:
        print(f"  [eval] 患者 {patient_id}: GT={len(gt)} evaluable={len(evaluable_cols)} "
              f"TP={tp} FP={fp} FN={fn} "
              f"P={precision:.2f} R={recall:.2f}" if precision else
              f"  [eval] 患者 {patient_id}: GT={len(gt)} evaluable={len(evaluable_cols)} "
              f"TP={tp} FP={fp} FN={fn}")

    return {
        "patient_id":        patient_id,
        "gt_total":          len(gt),
        "evaluable_cols":    len(evaluable_cols),
        "tp": tp, "fp": fp, "fn": fn,
        "precision":         precision,
        "recall":            recall,
        "f1":                f1,
        "total_entities":    total_entities,
        "mapped_entities":   mapped_entities,
        "unmapped_entities": [{"name": e.get("name"), "value": e.get("value"),
                               "source_file": e.get("source_file")}
                              for e in unmapped_entities],
        "details":           details,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _write_eval_xlsx(path: str, all_results: List[Dict]) -> None:
    """将评估结果写成 Excel：Sheet1 汇总，Sheet2 逐字段明细，Sheet3 漏映射实体。"""
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment

    GREEN  = PatternFill("solid", fgColor="C6EFCE")
    RED    = PatternFill("solid", fgColor="FFC7CE")
    ORANGE = PatternFill("solid", fgColor="FFEB9C")
    BOLD   = Font(bold=True)
    WRAP   = Alignment(wrap_text=True)

    wb = Workbook()

    # ── Sheet1：汇总 ────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "汇总"
    summary_headers = [
        "患者ID", "GT列数", "可评估列数",
        "TP(正确)", "FP(填错)", "FN(漏填)",
        "Precision", "Recall", "F1",
        "总实体数", "已映射实体数", "漏映射实体数",
    ]
    for ci, h in enumerate(summary_headers, 1):
        c = ws1.cell(row=1, column=ci, value=h)
        c.font = BOLD
        c.alignment = WRAP

    for ri, r in enumerate(all_results, 2):
        if r.get("error"):
            ws1.cell(row=ri, column=1, value=r["patient_id"])
            ws1.cell(row=ri, column=2, value=r["error"])
            continue
        vals = [
            r["patient_id"], r["gt_total"], r["evaluable_cols"],
            r["tp"], r["fp"], r["fn"],
            f"{r['precision']:.3f}" if r["precision"] is not None else "N/A",
            f"{r['recall']:.3f}"    if r["recall"]    is not None else "N/A",
            f"{r['f1']:.3f}"        if r["f1"]        is not None else "N/A",
            r["total_entities"], r["mapped_entities"],
            len(r["unmapped_entities"]),
        ]
        for ci, v in enumerate(vals, 1):
            ws1.cell(row=ri, column=ci, value=v)

    ws1.column_dimensions["A"].width = 12
    for ci in range(2, len(summary_headers) + 1):
        ws1.column_dimensions[
            ws1.cell(row=1, column=ci).column_letter
        ].width = 14

    # ── Sheet2：逐字段明细 ──────────────────────────────────────────
    ws2 = wb.create_sheet("逐字段明细")
    detail_headers = ["患者ID", "列名", "ground_truth", "预测值", "状态", "来源实体", "来源图片"]
    for ci, h in enumerate(detail_headers, 1):
        c = ws2.cell(row=1, column=ci, value=h)
        c.font = BOLD
    ri = 2
    for r in all_results:
        for d in r.get("details", []):
            row_vals = [d.get(k, "") for k in
                        ["patient_id", "列名", "ground_truth", "预测值", "状态", "来源实体", "来源图片"]]
            for ci, v in enumerate(row_vals, 1):
                cell = ws2.cell(row=ri, column=ci, value=v)
                cell.alignment = WRAP
            # 颜色
            status = d.get("状态", "")
            fill = GREEN if "TP" in status else (RED if "FP" in status else ORANGE)
            for ci in range(1, len(detail_headers) + 1):
                ws2.cell(row=ri, column=ci).fill = fill
            ri += 1

    ws2.column_dimensions["A"].width = 10
    ws2.column_dimensions["B"].width = 40
    ws2.column_dimensions["C"].width = 20
    ws2.column_dimensions["D"].width = 20
    ws2.column_dimensions["E"].width = 14
    ws2.column_dimensions["F"].width = 20
    ws2.column_dimensions["G"].width = 18

    # ── Sheet3：漏映射实体 ──────────────────────────────────────────
    ws3 = wb.create_sheet("漏映射实体")
    unmap_headers = ["患者ID", "实体名称", "值", "来源图片"]
    for ci, h in enumerate(unmap_headers, 1):
        ws3.cell(row=1, column=ci, value=h).font = BOLD
    ri = 2
    for r in all_results:
        for e in r.get("unmapped_entities", []):
            ws3.cell(row=ri, column=1, value=r["patient_id"])
            ws3.cell(row=ri, column=2, value=e.get("name", ""))
            ws3.cell(row=ri, column=3, value=e.get("value", ""))
            ws3.cell(row=ri, column=4, value=e.get("source_file", ""))
            ri += 1
    ws3.column_dimensions["A"].width = 10
    ws3.column_dimensions["B"].width = 25
    ws3.column_dimensions["C"].width = 20
    ws3.column_dimensions["D"].width = 18

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wb.save(path)


def _print_summary(all_results: List[Dict]) -> None:
    valid = [r for r in all_results if not r.get("error")]
    if not valid:
        print("没有有效结果")
        return

    total_tp = sum(r["tp"] for r in valid)
    total_fp = sum(r["fp"] for r in valid)
    total_fn = sum(r["fn"] for r in valid)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p and micro_r else None)

    total_ents    = sum(r["total_entities"] for r in valid)
    mapped_ents   = sum(r["mapped_entities"] for r in valid)
    unmapped_ents = sum(len(r["unmapped_entities"]) for r in valid)

    print("\n" + "="*60)
    print("遮盖测试结果（Micro 汇总）")
    print("="*60)
    print(f"  TP (正确填入): {total_tp}")
    print(f"  FP (填错):     {total_fp}")
    print(f"  FN (漏填):     {total_fn}")
    print(f"  Precision:     {micro_p:.3f}" if micro_p is not None else "  Precision:     N/A")
    print(f"  Recall:        {micro_r:.3f}" if micro_r is not None else "  Recall:        N/A")
    print(f"  F1:            {micro_f1:.3f}" if micro_f1 is not None else "  F1:            N/A")
    print()
    print("覆盖率")
    print(f"  总实体数:       {total_ents}")
    print(f"  已映射实体:     {mapped_ents}  ({mapped_ents/total_ents*100:.1f}%)" if total_ents else "")
    print(f"  漏映射实体:     {unmapped_ents}")
    print()
    print("逐病人：")
    for r in valid:
        p = f"{r['precision']:.3f}" if r["precision"] is not None else " N/A"
        rc = f"{r['recall']:.3f}"   if r["recall"]    is not None else " N/A"
        print(f"  {r['patient_id']}: P={p}  R={rc}  "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  "
              f"漏映射实体={len(r['unmapped_entities'])}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def run_evaluation(
    reorg_dir: str,
    results_dir: Optional[str],
    output_dir: str,
    use_llm: bool = False,
    verbose: bool = False,
    patient_filter: Optional[List[str]] = None,
) -> None:
    templates = find_patient_template_csvs(reorg_dir)
    if not templates:
        print(f"未找到模板 CSV（在 {reorg_dir} 下）", file=sys.stderr)
        return

    search_dir = results_dir or os.path.join(reorg_dir, "results_v2")
    entities_csv = find_latest_entities_csv(search_dir)
    if not entities_csv:
        print(f"未找到 entities_*.csv（在 {search_dir} 下）", file=sys.stderr)
        return
    print(f"实体文件: {entities_csv}")

    _, ent_rows = _read_csv(entities_csv)
    ent_by_patient: Dict[str, List[Dict]] = {}
    for row in ent_rows:
        pid = str(row.get("patient_id", "")).strip()
        ent_by_patient.setdefault(pid, []).append(row)

    tasks = []
    for pid, csv_path in templates.items():
        if patient_filter and pid not in patient_filter:
            continue
        ents = ent_by_patient.get(pid, [])
        tasks.append(evaluate_patient(pid, csv_path, ents,
                                      use_llm=use_llm, verbose=verbose))

    all_results = await asyncio.gather(*tasks)
    all_results = list(all_results)

    _print_summary(all_results)

    # 数据充足性提示
    total_gt = sum(r.get("evaluable_cols", 0) for r in all_results if not r.get("error"))
    if total_gt < 20:
        print("\n⚠️  注意：当前原始 CSV 中可评估字段数量极少（共 %d 列）。" % total_gt)
        print("   遮盖测试需要原始 CSV 中已有人工填写的参考值才能评估准确率。")
        print("   建议：手工填写至少 10-20 个关键字段后再运行评估；")
        print('   或参考"覆盖率"部分了解漏抽取/漏映射情况。')

    # 写 Excel 报告
    os.makedirs(output_dir, exist_ok=True)
    xlsx_path = os.path.join(output_dir, "evaluation_report.xlsx")
    _write_eval_xlsx(xlsx_path, all_results)
    print(f"\n详细报告已保存: {xlsx_path}")

    # 写 JSON 明细（方便程序读取）
    json_path = os.path.join(output_dir, "evaluation_detail.json")
    # unmapped_entities 已是 list[dict]，details 也是，可直接序列化
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON 明细已保存: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate",
        description="评估 fill_table 的填写质量（遮盖测试 + 覆盖率报告）",
    )
    p.add_argument("--reorg", default="reorganized_output")
    p.add_argument("--results", default=None,
                   help="entities_*.csv 所在目录（默认: --reorg/results_v2）")
    p.add_argument("--output", default="eval_results",
                   help="报告输出目录（默认: eval_results/）")
    p.add_argument("--patient", nargs="*", metavar="ID")
    p.add_argument("--llm", action="store_true",
                   help="用 LLM 映射（默认规则，保证可复现）")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    reorg = args.reorg if os.path.isabs(args.reorg) else os.path.join(_HERE, args.reorg)
    results = args.results
    if results and not os.path.isabs(results):
        results = os.path.join(_HERE, results)
    output = args.output if os.path.isabs(args.output) else os.path.join(_HERE, args.output)

    asyncio.run(run_evaluation(
        reorg_dir=reorg,
        results_dir=results,
        output_dir=output,
        use_llm=args.llm,
        verbose=args.verbose,
        patient_filter=args.patient or None,
    ))


if __name__ == "__main__":
    main()
