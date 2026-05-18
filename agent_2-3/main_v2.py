#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medical_data_cleaner_v2 命令行入口

用法示例：
  # 单文件处理
  python main_v2.py --input reorganized_output --output results/

  # 批量目录处理
  python main_v2.py --input data/patients/ --output results/ --batch

  # 交互模式（循环输入文件路径）
  python main_v2.py --interactive

  # 不使用 LLM（仅规则处理）
  python main_v2.py --input data/sample.csv --no-llm

  # 详细日志
  python main_v2.py --input data/sample.csv --verbose

  # 一体化流水线：OCR + LLM 抽取 + 回填
  python main_v2.py --pipeline --input 归档/output/step1_results --output results-all/ --ocr-workers 64 --concurrency 100

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
        help="流水线模式：依次执行 OCR → LLM 抽取 → 回填表格",
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
        help="输入文件或目录路径",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default="./output_v2",
        help="输出目录（默认: ./output_v2）",
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


async def _process_single(
    input_path: str,
    output_dir: str,
    use_llm: bool,
    verbose: bool,
    model_name: str | None,
    concurrency: int = 8,
    ocr_concurrency: int = 0,
) -> dict:
    """处理单个输入路径，返回结果 dict。"""
    from agents_v2.react_agent import ReactMedicalAgent

    agent = ReactMedicalAgent(
        name="MedicalAgent",
        use_llm=use_llm,
        verbose=verbose,
        output_dir=output_dir,
        concurrency=concurrency,
        ocr_concurrency=ocr_concurrency,
    )

    from agentscope.message import Msg
    msg = Msg(
        name="user",
        content=input_path,
        role="user",
    )
    response = await agent.reply(msg)
    content = response.content if hasattr(response, "content") else str(response)
    metadata = response.metadata if hasattr(response, "metadata") and response.metadata else {}

    # metadata 是完整结果 dict，补充 status 和 message 后返回
    if metadata:
        result = dict(metadata)
        result["status"] = "success" if metadata.get("success") else "failed"
        result["message"] = content if isinstance(content, str) else str(content)
        return result

    # 降级：尝试解析 JSON 摘要
    try:
        return json.loads(content)
    except Exception:
        return {"status": "success", "message": content}


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
        result = await _process_single(raw, output_dir, use_llm, verbose, model_name, concurrency, ocr_concurrency)
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
            result = await _process_single(path, sub_out, use_llm, verbose, model_name, concurrency, ocr_concurrency)
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
    from agents_v2.react_agent import ReactMedicalAgent
    from agentscope.message import Msg

    agent = ReactMedicalAgent(
        name="PipelineAgent",
        use_llm=True,
        output_dir=output_dir,
        concurrency=concurrency,
        pipeline_mode=True,
        ocr_workers=ocr_workers,
        ocr_cache=cache_path,
        no_fill=no_fill,
    )
    msg = Msg(name="user", content=input_dir, role="user")
    await agent.reply(msg)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # 覆盖环境变量（model）
    if args.model:
        os.environ["OPENAI_MODEL_NAME"] = args.model

    os.makedirs(args.output, exist_ok=True)
    use_llm = not args.no_llm
    ocr_c = args.ocr_concurrency

    if args.pipeline:
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
            _process_single(args.input, args.output, use_llm, args.verbose, args.model, args.concurrency, ocr_c)
        )
        _print_result(result, args.verbose)
        sys.exit(0 if result.get("status") == "success" else 1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
