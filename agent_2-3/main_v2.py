#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medical_data_cleaner_v2 命令行入口

用法示例：
  # 自动选择目录处理或 OCR 回填
  python main_v2.py --input reorganized_output --output results/

  # 批量目录处理
  python main_v2.py --input data/patients/ --output results/ --batch

  # 交互模式（循环输入文件路径）
  python main_v2.py --interactive

  # 不使用 LLM（仅规则处理）
  python main_v2.py --input data/sample.csv --no-llm

  # 详细日志
  python main_v2.py --input data/sample.csv --verbose

  # OCR 回填流水线：OCR + LLM 抽取 + 回填
  python main_v2.py --mode ocr_fill --input 归档/output/step1_results --output results-all/ --ocr-workers 64 --concurrency 100

  # 流水线（使用已有 OCR 缓存，跳过 OCR 步骤）
  python main_v2.py --pipeline --input 归档/output/step1_results --output results-all/ --cache ocr_cache.json --concurrency 100

  # 流水线（跳过回填步骤）
  python main_v2.py --pipeline --input 归档/output/step1_results --output results-all/ --no-fill
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

# 确保包路径可访问
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)
_AS_ROOT = os.path.abspath(os.path.join(_EXAMPLES, "../../"))
_AS_SRC = os.path.join(_AS_ROOT, "src")
for _p in [_HERE, _AS_SRC, _AS_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PROJECT_ROOT = Path(_HERE).parent
_STEP23_DIR = _PROJECT_ROOT / "step-2-3"
_ORCHESTRATOR_DIR = _PROJECT_ROOT / "main_orchestrator"
_EXECUTION_DIR = _ORCHESTRATOR_DIR / "execution"
_CORE_DIR = _ORCHESTRATOR_DIR / "core"
for _p in (_STEP23_DIR, _ORCHESTRATOR_DIR, _EXECUTION_DIR, _CORE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def get_default_input_path() -> str:
    return str(_PROJECT_ROOT / "agent_2-3" / "data_input")


def get_default_output_dir() -> str:
    return str(_PROJECT_ROOT / "program" / "output" / "step2_3_results")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical_data_cleaner_v2",
        description="基于 ReAct 智能体的医疗数据清洗系统 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="兼容参数，等价于 --mode ocr_fill：依次执行 OCR → LLM 抽取 → 回填表格",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "directory", "ocr_fill"],
        default="auto",
        help="处理模式：auto 根据输入自动选择；directory 用于 MIMIC/结构化目录；ocr_fill 用于图片 OCR 回填",
    )
    parser.add_argument(
        "--cache",
        metavar="PATH",
        default=None,
        help="[流水线模式] 已有 OCR 缓存 JSON 路径，指定后跳过 OCR 步骤",
    )
    parser.add_argument(
        "--ocr-workers",
        metavar="N",
        type=int,
        default=64,
        help="[流水线模式] OCR 线程数（默认: 64）",
    )
    parser.add_argument(
        "--no-fill",
        dest="no_fill",
        action="store_true",
        help="[流水线模式] 跳过回填步骤",
    )
    parser.add_argument(
        "--input", "-i",
        metavar="PATH",
        default=None,
        help="输入目录路径；Step2-3 不再支持单文件输入",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=get_default_output_dir(),
        help="输出目录（默认: program/output/step2_3_results）",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量模式：将 --input 目录下每个子目录视为独立患者组",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互模式：循环读取输入路径，输入 q/quit/exit 退出",
    )
    parser.add_argument(
        "--no-llm",
        dest="no_llm",
        action="store_true",
        help="禁用 LLM（使用规则推断处理计划）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志（包含 ReAct 思考链）",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="覆盖配置文件中的模型名称",
    )
    parser.add_argument(
        "--concurrency", "-c",
        metavar="N",
        type=int,
        default=8,
        help="最大并发 LLM API 调用数（默认: 8）",
    )
    parser.add_argument(
        "--ocr-concurrency",
        metavar="N",
        type=int,
        default=0,
        help="本地 OCR 推理并发数（默认: 0=自动取 CPU 核数）",
    )
    return parser


def _print_result(result: dict, verbose: bool = False) -> None:
    """格式化打印处理结果。"""
    status = result.get("status", "unknown")
    symbol = "✓" if status == "success" else "✗"
    print(f"\n{symbol} 处理{status}")

    if result.get("output_dir"):
        print(f"  输出目录: {result['output_dir']}")
    if result.get("output_path"):
        print(f"  输出文件: {result['output_path']}")
    if result.get("output_json"):
        print(f"  结果 JSON: {result['output_json']}")
    if result.get("output_entities_csv"):
        print(f"  实体 CSV: {result['output_entities_csv']}")
    if result.get("output_patients_csv"):
        print(f"  患者统计 CSV: {result['output_patients_csv']}")
    if result.get("output_csv"):
        print(f"  输出 CSV: {result['output_csv']}")

    summary = result.get("summary", {})
    if summary:
        if summary.get("total_patients") is not None:
            print(f"  患者数: {summary['total_patients']}")
        if summary.get("row_count") is not None:
            print(f"  行数: {summary['row_count']}")
        if summary.get("entity_count") is not None:
            print(f"  实体数: {summary['entity_count']}")

    # 直接从 statistics 和 patients 读取
    stats = result.get("statistics", {})
    if stats:
        print(f"  文件: {stats.get('processed_files', 0)}/{stats.get('total_files', 0)} 成功, {stats.get('failed_files', 0)} 失败")
    patients = result.get("patients", {})
    if patients:
        total_entities = sum(len(p.get("merged_entities", [])) for p in patients.values())
        print(f"  抽取实体总数: {total_entities}")

    if verbose:
        plan = result.get("plan")
        if plan:
            print("\n  [处理计划]")
            print(f"    数据类型: {plan.get('data_type', '?')}")
            if plan.get("reasoning"):
                print(f"    推理: {plan['reasoning']}")
        if result.get("message"):
            print(f"\n  [Agent 回复]\n  {result['message'][:500]}")

    if result.get("error"):
        print(f"  错误: {result['error']}", file=sys.stderr)


async def _process_directory(
    input_path: str,
    output_dir: str,
    use_llm: bool,
    verbose: bool,
    model_name: str | None,
    concurrency: int = 8,
    ocr_concurrency: int = 0,
    mode: str = "auto",
) -> dict:
    """处理单个目录输入路径，返回结果 dict。"""
    from handoffs import run_step23_handoff

    if not os.path.isdir(input_path):
        return {
            "status": "failed",
            "success": False,
            "error": f"Step2-3 现在只接受目录输入: {input_path}",
        }
    result = await run_step23_handoff(
        task_type="step2_3_only",
        input_path=input_path,
        output_root=output_dir,
        mode=mode,
        concurrency=concurrency,
        use_llm=use_llm,
    )
    payload = result.to_dict()
    payload["status"] = "success" if payload.get("status") == "SUCCESS" else "failed"
    payload["success"] = result.status.value == "SUCCESS"
    return payload


async def _interactive_loop(
    output_dir: str,
    use_llm: bool,
    verbose: bool,
    model_name: str | None,
    concurrency: int = 8,
    ocr_concurrency: int = 0,
) -> None:
    """交互模式：循环读取文件路径并处理。"""
    print("医疗数据清洗系统 v2 — 交互模式")
    print("输入文件/目录路径开始处理，输入 q/quit/exit 退出\n")

    while True:
        try:
            raw = input("请输入路径 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出")
            break

        if not raw or raw.lower() in ("q", "quit", "exit"):
            print("已退出")
            break

        if not os.path.exists(raw):
            print(f"路径不存在: {raw}", file=sys.stderr)
            continue

        print(f"\n正在处理: {raw}")
        result = await _process_directory(raw, output_dir, use_llm, verbose, model_name, concurrency, ocr_concurrency)
        _print_result(result, verbose)
        print()


async def _batch_mode(
    input_dir: str,
    output_dir: str,
    use_llm: bool,
    verbose: bool,
    model_name: str | None,
    concurrency: int = 8,
    ocr_concurrency: int = 0,
) -> None:
    """批量模式：遍历 input_dir 的直接子项逐个处理。"""
    if not os.path.isdir(input_dir):
        print(f"批量模式要求 --input 为目录: {input_dir}", file=sys.stderr)
        sys.exit(1)

    entries = sorted(os.listdir(input_dir))
    total = len(entries)
    print(f"批量模式：共 {total} 个条目\n")

    success, failed = 0, 0
    for idx, entry in enumerate(entries, 1):
        path = os.path.join(input_dir, entry)
        sub_out = os.path.join(output_dir, entry)
        os.makedirs(sub_out, exist_ok=True)

        print(f"[{idx}/{total}] {entry}")
        try:
            result = await _process_directory(path, sub_out, use_llm, verbose, model_name, concurrency, ocr_concurrency)
            _print_result(result, verbose)
            if result.get("status") == "success":
                success += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  处理失败: {exc}", file=sys.stderr)
            failed += 1

    print(f"\n批量处理完成：成功 {success}，失败 {failed}")



# ---------------------------------------------------------------------------
# 流水线模式（委托给 ReactMedicalAgent pipeline_mode=True）
# ---------------------------------------------------------------------------

async def _pipeline_mode(
    input_dir: str,
    output_dir: str,
    cache_path: str | None,
    ocr_workers: int,
    concurrency: int,
    no_fill: bool,
) -> None:
    """流水线模式：通过 ReactMedicalAgent 依次执行 OCR → LLM 抽取 → 回填。"""
    from handoffs import run_step23_handoff

    result = await run_step23_handoff(
        task_type="step2_3_only",
        input_path=input_dir,
        output_root=output_dir,
        mode="ocr_fill",
        cache_path=cache_path,
        ocr_workers=ocr_workers,
        concurrency=concurrency,
        use_llm=not no_fill,
    )
    _print_result({"status": "success" if result.status.value == "SUCCESS" else "failed", **result.to_dict()}, verbose=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # 覆盖环境变量（model）
    if args.model:
        os.environ["OPENAI_MODEL_NAME"] = args.model

    if not args.input and not args.interactive:
        args.input = get_default_input_path()

    os.makedirs(args.output, exist_ok=True)
    use_llm = not args.no_llm
    ocr_c = args.ocr_concurrency

    if args.pipeline or args.mode == "ocr_fill":
        if not args.input:
            parser.error("流水线模式需要 --input 目录")
        if not os.path.isdir(args.input):
            print(f"错误：流水线模式要求 --input 为目录: {args.input}", file=sys.stderr)
            sys.exit(1)
        asyncio.run(
            _pipeline_mode(
                input_dir=args.input,
                output_dir=args.output,
                cache_path=args.cache,
                ocr_workers=args.ocr_workers,
                concurrency=args.concurrency,
                no_fill=args.no_fill,
            )
        )
    elif args.interactive:
        asyncio.run(
            _interactive_loop(args.output, use_llm, args.verbose, args.model, args.concurrency, ocr_c)
        )
    elif args.batch:
        if not args.input:
            parser.error("批量模式需要 --input 目录")
        asyncio.run(
            _batch_mode(args.input, args.output, use_llm, args.verbose, args.model, args.concurrency, ocr_c)
        )
    elif args.input:
        if not os.path.exists(args.input):
            print(f"错误：路径不存在: {args.input}", file=sys.stderr)
            sys.exit(1)
        result = asyncio.run(
            _process_directory(args.input, args.output, use_llm, args.verbose, args.model, args.concurrency, ocr_c, mode=args.mode)
        )
        _print_result(result, args.verbose)
        sys.exit(0 if result.get("status") == "success" else 1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
