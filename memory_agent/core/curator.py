from __future__ import annotations

import asyncio
import ast
import json
import re
from typing import Any, Callable

from agentscope.model import OpenAIChatModel

from .models import (
    CuratorDelta,
    CuratorOperation,
    ReflectionResult,
    SECTION_ORDER,
    normalize_section,
)
from .playbook import ACEPlaybookManager, normalize_bullet_content
from .reflector import build_safe_fallback_rule
from .trace_parser import normalize_trace_payload


_STEP_NAME_TO_SOURCE_STEP = {
    "数据感知与模态识别": "step1",
    "解析与结构化抽取": "step2",
    "Step2_3 医学数据清洗与标准化": "step2_3",
    "语义标准化与本体映射": "step3",
    "数据质量检测与自动修复": "step4",
    "任务导向裁剪": "step5",
    "一致性验证与置信度估计": "step6",
    "表型/知识确认": "step7",
}


def infer_source_step(trace_data: dict[str, Any] | None) -> str:
    steps = (trace_data or {}).get("steps") or []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        step_name = str(step.get("step_name") or "").strip()
        if not step_name:
            continue
        if step_name in _STEP_NAME_TO_SOURCE_STEP:
            return _STEP_NAME_TO_SOURCE_STEP[step_name]

        normalized = step_name.lower().replace(" ", "")
        if "step2_3" in normalized:
            return "step2_3"

        match = re.search(r"step[_\s-]*([1-7])", normalized, re.IGNORECASE)
        if match:
            return f"step{match.group(1)}"
    return ""


def _contains_cjk(text: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _stringify_model_content(content: Any) -> str:
    if isinstance(content, list):
        text_parts: list[str] = []
        fallback_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "thinking":
                    continue
                elif block_type == "tool_result":
                    fallback_parts.append(str(block.get("output", "")))
                else:
                    fallback_parts.append(str(block))
            else:
                fallback_parts.append(str(block))
        if any(part.strip() for part in text_parts):
            return "\n".join(part for part in text_parts if part).strip()
        return "\n".join(part for part in fallback_parts if part).strip()
    return str(content or "").strip()


def _iter_json_candidates(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    candidates: list[str] = [raw]

    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        candidates.append(match.group(1).strip())
    else:
        match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())

    brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    normalized_quotes = (
        raw.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    if normalized_quotes != raw:
        candidates.append(normalized_quotes)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
        trimmed_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", stripped)
        if trimmed_trailing_commas not in seen:
            seen.add(trimmed_trailing_commas)
            deduped.append(trimmed_trailing_commas)
    return deduped


def _extract_json_object(text: str) -> dict[str, Any]:
    for candidate in _iter_json_candidates(text):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


async def _translate_delta_to_chinese(model: Any, parsed: dict[str, Any]) -> dict[str, Any]:
    operations = parsed.get("operations", [])
    needs_translation = False

    if str(parsed.get("reasoning") or "").strip() and not _contains_cjk(parsed.get("reasoning")):
        needs_translation = True
    if not needs_translation:
        for item in operations:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if content and not _contains_cjk(content):
                needs_translation = True
                break

    if not needs_translation:
        return parsed

    translation_prompt = f"""把下面 JSON 中给人看的自然语言字符串翻译成中文，并保持 JSON 结构不变。

要求：
1. 只翻译 `reasoning` 和 `operations[].content` 这类自然语言字段。
2. 不要修改 `type`、`section` 等枚举值。
3. 工具名、路径、文件名可以保留原样。
4. 保持原意，不要扩写。
5. 只输出 JSON，不要附带解释。

原始 JSON：
{json.dumps(parsed, ensure_ascii=False, indent=2)}
"""

    response = model([{"role": "user", "content": translation_prompt}])
    if asyncio.iscoroutine(response):
        response = await response
    translated = _extract_json_object(
        _stringify_model_content(response.content if hasattr(response, "content") else response)
    )
    if not isinstance(translated, dict) or not translated:
        return parsed

    merged = dict(parsed)
    translated_reasoning = str(translated.get("reasoning") or "").strip()
    if translated_reasoning and _contains_cjk(translated_reasoning):
        merged["reasoning"] = translated_reasoning

    translated_operations = translated.get("operations", [])
    if isinstance(translated_operations, list):
        merged_ops: list[dict[str, Any]] = []
        for index, item in enumerate(operations):
            if not isinstance(item, dict):
                continue
            translated_item = translated_operations[index] if index < len(translated_operations) else {}
            merged_item = dict(item)
            translated_content = (
                str(translated_item.get("content") or "").strip()
                if isinstance(translated_item, dict)
                else ""
            )
            if translated_content and _contains_cjk(translated_content):
                merged_item["content"] = translated_content
            merged_ops.append(merged_item)
        merged["operations"] = merged_ops

    return merged


class ACECurator:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        seed: int,
        enable_thinking: bool | None = False,
        model_cls=OpenAIChatModel,
        model_factory: Callable[[], Any] | None = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.model_cls = model_cls
        self.model_factory = model_factory

    def _build_model(self):
        if self.model_factory is not None:
            return self.model_factory()
        return self.model_cls(
            model_name=self.model_name,
            enable_thinking=self.enable_thinking,
            options={
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": 4096,
            },
        )

    async def propose_delta(
        self,
        manager: ACEPlaybookManager,
        reflection: ReflectionResult,
        trace_json: str | dict[str, Any],
    ) -> CuratorDelta:
        trace_data = normalize_trace_payload(trace_json)
        model = self._build_model()

        input_text = str(trace_data.get("input_text") or "")
        output_text = str(trace_data.get("orchestrator_summary") or "")
        current_playbook, _, _ = manager.format_playbook(query_text=input_text, max_bullets=30)
        reflection_text = json.dumps(reflection.to_dict(), ensure_ascii=False, indent=2)
        sections_text = ", ".join(SECTION_ORDER)
        prompt = f"""你是 ACE 框架中的 Curator。你的职责是把最近一次 reflection、任务输入、最终输出摘要，以及当前 playbook 上下文，转成可追加的增量规则集合。
你不能重写整个 playbook，也不能泛泛总结本轮任务；你只能提出未来还能复用、且当前 playbook 中缺失的新条目。

你的目标不是“写一条总结”，而是“沉淀一组以后能直接指导执行的策略”。
如果证据支持，本轮可以新增多条策略，不必限制为一条；只有当完整 trace 里确实只稳定支持一个结论时，才只输出一条。

硬性要求：
1. 只输出新的增量操作，不要重复已有条目，不要改写已有条目。
2. 只允许 `ADD` 操作。
3. `section` 只能从以下集合中选择：{sections_text}
4. `content` 必须使用中文。
5. 每条 `content` 必须是原子化、命令式、可执行、可长期复用的规则，最好同时包含“触发条件/场景”与“应该怎么做”。
6. 每条规则必须尽量具体，能直接指导下一次执行；不要写成空泛口号。
7. 一次可以输出多条策略。通常输出 1-5 条；如果证据非常充分，也可以更多，但不要为了凑数硬拆。
8. 多条策略之间不要重复、不要同义改写、不要只是在措辞上轻微变化。
9. 不要输出总结、统计、路径清单、文件名清单、病例事实复述、一次性数值结论。
10. 不要直接照抄 reflection 原文；要把它提炼成未来任务可复用的策略。
11. 如果当前 playbook 已经有语义高度相近的规则，就不要再新增。
12. 输出必须是 JSON，不要附带 markdown、解释段落或代码块。

优先提炼什么：
- 工具调用顺序与依赖关系
- 工具输入约束、边界条件、前置检查
- 工具返回结果的校验方式
- 面向用户输出时必须遵守的输出契约
- 多模态/多文件场景下的路由、汇总、验证规则
- 明确可复用的失败模式与纠正动作

什么样的规则是好的：
- 好："当工具返回的是结构化字段时，最终结论必须只引用已返回字段，不得补写 trace 中不存在的统计项。"
- 好："当目录同时包含图片与表格时，先按主处理链完成图片/文本抽取，再把表格结果作为补充证据合并，避免把辅助文件误当主输入。"
- 好："当交互式步骤尚未结束时，不要提前生成整轮总结；应等用户输入 quit 后再输出会话摘要。"
- 不好："这次做得还可以。"
- 不好："需要更认真检查数据。"
- 不好："本次处理了 3 个文件，结果正常。"
- 不好："输出目录是 /path/to/output。"

任务上下文：
输入: {input_text}
输出摘要: {output_text}

Recent Reflection:
{reflection_text}

Current Playbook Stats:
{manager.get_stats_text()}

Current Playbook:
{current_playbook}

输出格式：
{{
  "reasoning": "为什么需要这些增量，以及为什么这些规则在未来可复用",
  "operations": [
    {{
      "type": "ADD",
      "section": "tool_usage",
      "content": "新增规则1"
    }},
    {{
      "type": "ADD",
      "section": "output_contracts",
      "content": "新增规则2"
    }}
  ]
}}

如果没有任何真正新增且可复用的规则，请输出：
{{
  "reasoning": "当前 reflection 未提供可安全新增的规则，或当前 playbook 已覆盖相同经验。",
  "operations": []
}}
"""

        response = model([{"role": "user", "content": prompt}])
        if asyncio.iscoroutine(response):
            response = await response

        parsed = _extract_json_object(
            _stringify_model_content(response.content if hasattr(response, "content") else response)
        )
        parsed = await _translate_delta_to_chinese(model, parsed)

        operations = []
        for item in parsed.get("operations", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").upper() != "ADD":
                continue
            content = normalize_bullet_content(item.get("content"))
            if not content:
                continue
            section = normalize_section(item.get("section"), content)
            operations.append(CuratorOperation(type="ADD", section=section, content=content))

        if not operations:
            fallback_content = normalize_bullet_content(reflection.key_insight)
            if fallback_content:
                operations = [
                    CuratorOperation(
                        type="ADD",
                        section=normalize_section(reflection.key_insight_section, fallback_content),
                        content=fallback_content,
                    )
                ]

        if not operations:
            fallback_section, fallback_content = build_safe_fallback_rule(trace_data)
            operations = [
                CuratorOperation(
                    type="ADD",
                    section=fallback_section,
                    content=fallback_content,
                )
            ]

        return CuratorDelta(
            reasoning=str(parsed.get("reasoning", "")).strip() or "基于已验证 reflection 生成增量更新。",
            operations=operations,
        )

    def apply_delta(
        self,
        manager: ACEPlaybookManager,
        delta: CuratorDelta,
        source: str = "curator",
        trace_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_ids = []
        skipped_duplicates = 0
        applied_operations = []
        source_step = infer_source_step(trace_data)

        for op in delta.operations:
            if op.type != "ADD":
                continue
            metadata = {"source": source}
            if source_step:
                metadata["source_step"] = source_step
            result = manager.add_bullet(
                content=op.content,
                section=op.section,
                metadata=metadata,
            )
            applied_operations.append(op.to_dict())
            if result.get("created"):
                created_ids.append(result["bullet_id"])
            elif result.get("reason") == "duplicate":
                skipped_duplicates += 1

        return {
            "operation": "curator_apply_delta",
            "reasoning": delta.reasoning,
            "applied_operations": applied_operations,
            "new_bullet_ids": created_ids,
            "new_bullets": len(created_ids),
            "skipped_duplicates": skipped_duplicates,
        }


MemoryCurator = ACECurator
