#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medical_data_cleaner_v2 命令行入口

用法示例：
  # 单文件处理
  python main.py --input reorganized_output --output results/

  # 批量目录处理
  python main.py --input data/patients/ --output results/ --batch

  # 交互模式（循环输入文件路径）
  python main.py --interactive

  # 不使用 LLM（仅规则处理）
  python main.py --input data/sample.csv --no-llm

  # 详细日志
  python main.py --input data/sample.csv --verbose
"""
import argparse
import asyncio
import json
import os
import sys

# 确保包路径可访问
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
_EXAMPLES = os.path.dirname(_HERE)
_AS_ROOT = os.path.abspath(os.path.join(_EXAMPLES, "../../"))
_AS_SRC = os.path.join(_AS_ROOT, "src")
for _p in [_HERE, _AS_SRC, _AS_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_default_input_path() -> str:
    return os.path.join(_HERE, "data_input")


def get_default_output_dir() -> str:
    return os.path.join(PROJECT_ROOT, "program", "output", "step2_3_results")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical_data_cleaner_v2",
        description="基于 ReAct 智能体的医疗数据清洗系统 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        metavar="PATH",
        help=f"输入文件或目录路径（默认: {get_default_input_path()}）",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=get_default_output_dir(),
        help=f"输出目录（默认: {get_default_output_dir()}）",
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
) -> dict:
    """处理单个输入路径，返回结果 dict。"""
    from agents.react_agent import ReactMedicalAgent

    agent = ReactMedicalAgent(
        name="MedicalAgent",
        use_llm=use_llm,
        verbose=verbose,
        output_dir=output_dir,
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
        result = await _process_single(raw, output_dir, use_llm, verbose, model_name)
        _print_result(result, verbose)
        print()


async def _batch_mode(
    input_dir: str,
    output_dir: str,
    use_llm: bool,
    verbose: bool,
    model_name: str | None,
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
            result = await _process_single(path, sub_out, use_llm, verbose, model_name)
            _print_result(result, verbose)
            if result.get("status") == "success":
                success += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  处理失败: {exc}", file=sys.stderr)
            failed += 1

    print(f"\n批量处理完成：成功 {success}，失败 {failed}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # 覆盖环境变量（model）
    if args.model:
        os.environ["OPENAI_MODEL_NAME"] = args.model

    os.makedirs(args.output, exist_ok=True)
    use_llm = not args.no_llm
    input_path = args.input or get_default_input_path()

    if args.interactive:
        asyncio.run(
            _interactive_loop(args.output, use_llm, args.verbose, args.model)
        )
    elif args.batch:
        asyncio.run(
            _batch_mode(input_path, args.output, use_llm, args.verbose, args.model)
        )
    else:
        if not os.path.exists(input_path):
            print(f"错误：路径不存在: {input_path}", file=sys.stderr)
            sys.exit(1)
        result = asyncio.run(
            _process_single(input_path, args.output, use_llm, args.verbose, args.model)
        )
        _print_result(result, args.verbose)
        sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
