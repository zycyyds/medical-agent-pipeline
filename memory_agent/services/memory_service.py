from __future__ import annotations

import os
from typing import Any

from memory_agent.config import Config, config
from memory_agent.services.experience_memory_service import ExperienceMemoryService
from memory_agent.services.rule_memory_service import RuleMemoryService


class MemoryService:
    """Unified service used by the main pipeline and standalone MemoryAgent."""

    def __init__(
        self,
        cfg: Config = config,
        rule_service: RuleMemoryService | None = None,
        experience_service: ExperienceMemoryService | None = None,
    ) -> None:
        self.config = cfg
        self.rule_memory = rule_service or RuleMemoryService(cfg)
        self.experience_memory = experience_service or ExperienceMemoryService(cfg)

    async def get_context(self, query_text: str = "", task_spec: Any | None = None) -> tuple[str, list[str]]:
        self._print_progress("[MemoryAgent] pre-run: 检索 ReMe 规则记忆和 Step 经验记忆...")
        warnings: list[str] = []
        used_ids: list[str] = []

        rule_text, rule_ids, rule_warnings = await self.rule_memory.search_rules(query_text=query_text)
        task_text, tool_text, experience_ids, experience_warnings = await self.experience_memory.retrieve(
            query_text=query_text,
            task_spec=task_spec,
        )
        used_ids.extend(rule_ids)
        used_ids.extend(experience_ids)
        warnings.extend(rule_warnings)
        warnings.extend(experience_warnings)

        context = self._format_context(rule_text, task_text, tool_text, warnings)
        self._print_progress(f"[MemoryAgent] pre-run: 完成，memory_ids={len(used_ids)}, warnings={len(warnings)}")
        return context, used_ids

    async def get_rule_context(self, query_text: str = "") -> tuple[str, list[str]]:
        self._print_progress("[MemoryAgent] pre-run: 检索 ReMeLight 规则记忆...")
        rule_text, rule_ids, warnings = await self.rule_memory.search_rules(query_text=query_text)
        context = self._format_rule_context(rule_text, warnings)
        self._print_progress(f"[MemoryAgent] pre-run: 规则记忆完成，memory_ids={len(rule_ids)}, warnings={len(warnings)}")
        return context, rule_ids

    async def get_step_context(
        self,
        step_name: str,
        query_text: str = "",
        task_spec: Any | None = None,
    ) -> tuple[str, list[str]]:
        self._print_progress(f"[MemoryAgent] pre-step: 检索 {step_name} Step 经验记忆...")
        task_text, tool_text, experience_ids, warnings = await self.experience_memory.retrieve(
            query_text=query_text,
            task_spec=task_spec,
            step_name=step_name,
        )
        context = self._format_experience_context(step_name, task_text, tool_text, warnings)
        self._print_progress(
            f"[MemoryAgent] pre-step: {step_name} 经验记忆完成，memory_ids={len(experience_ids)}, warnings={len(warnings)}"
        )
        return context, experience_ids

    async def report_result(
        self,
        trace_payload: dict[str, Any] | str,
        used_memory_ids: list[str] | None = None,
    ) -> str:
        self._print_progress("[MemoryAgent] post-run: 写入 ReMeLight 规则记忆和 Step 经验记忆...")
        self._print_progress("[MemoryAgent] post-run: Rule Memory start")
        rule_summary, rule_warnings = await self.rule_memory.summarize_trace(trace_payload)
        self._print_progress(f"[MemoryAgent] post-run: Rule Memory done，warnings={len(rule_warnings)}")
        self._print_progress("[MemoryAgent] post-run: Task/Tool Memory start")
        experience_summary, experience_warnings = await self.experience_memory.record_from_trace(trace_payload)
        self._print_progress(f"[MemoryAgent] post-run: Task/Tool Memory done，warnings={len(experience_warnings)}")
        warnings = rule_warnings + experience_warnings
        lines = [
            "### MemoryAgent ReMe 写入总结",
            "",
            f"- Rule Memory: {rule_summary}",
            f"- Experience Memory: {experience_summary}",
            f"- used_memory_ids: {used_memory_ids or []}",
        ]
        if warnings:
            lines.append("- Warnings:")
            lines.extend(f"  - {item}" for item in warnings)
        summary = "\n".join(lines)
        self._print_progress(f"[MemoryAgent] post-run: 完成，warnings={len(warnings)}")
        return summary

    def status(self) -> dict[str, Any]:
        return {
            "reme_enabled": self.config.REME_ENABLED,
            "reme_backend": self.config.REME_BACKEND,
            "rule_memory": self.rule_memory.status(),
            "experience_memory": self.experience_memory.status(),
        }

    @staticmethod
    def _format_context(rule_text: str, task_text: str, tool_text: str, warnings: list[str]) -> str:
        sections = [
            "# Memory Context",
            "",
            "## Rule Memory",
            rule_text.strip() or "No related rule memory found.",
            "",
            "## Similar Step Experience",
            task_text.strip() or "No related Step experience found.",
            "",
            "## Tool Memory",
            tool_text.strip() or "No related tool memory found.",
        ]
        if warnings:
            sections.extend(["", "## Memory Warnings"])
            sections.extend(f"- {item}" for item in warnings)
        return "\n".join(sections).strip()

    @staticmethod
    def _format_rule_context(rule_text: str, warnings: list[str]) -> str:
        sections = [
            "# Memory Context",
            "",
            "## Rule Memory",
            rule_text.strip() or "No related rule memory found.",
        ]
        if warnings:
            sections.extend(["", "## Memory Warnings"])
            sections.extend(f"- {item}" for item in warnings)
        return "\n".join(sections).strip()

    @staticmethod
    def _format_experience_context(step_name: str, task_text: str, tool_text: str, warnings: list[str]) -> str:
        sections = [
            f"# Step Memory Context: {step_name}",
            "",
            "## Similar Step Experience",
            task_text.strip() or "No related Step experience found.",
            "",
            "## Tool Memory",
            tool_text.strip() or "No related tool memory found.",
        ]
        if warnings:
            sections.extend(["", "## Memory Warnings"])
            sections.extend(f"- {item}" for item in warnings)
        return "\n".join(sections).strip()

    @staticmethod
    def _print_progress(message: str) -> None:
        if os.environ.get("MEMORY_PROGRESS_ENABLED", "true").lower() in {"0", "false", "no", "off"}:
            return
        print(message)


_DEFAULT_MEMORY_SERVICE: MemoryService | None = None


def get_default_memory_service() -> MemoryService:
    global _DEFAULT_MEMORY_SERVICE
    if _DEFAULT_MEMORY_SERVICE is None:
        _DEFAULT_MEMORY_SERVICE = MemoryService()
    return _DEFAULT_MEMORY_SERVICE
