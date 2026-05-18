from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentscope.message import Msg

from memory_agent.config import Config, config
from memory_agent.services.utils import suppress_external_output, tool_response_to_text


class RuleMemoryService:
    """ReMeLight-backed replacement for the old Playbook rule memory."""

    def __init__(self, cfg: Config = config, reme_light_cls: Any | None = None) -> None:
        self.config = cfg
        self.root = cfg.resolve_project_path(cfg.REME_LIGHT_ROOT)
        self._reme_light_cls = reme_light_cls

    def _load_reme_light_cls(self):
        if self._reme_light_cls is not None:
            return self._reme_light_cls
        from reme.reme_light import ReMeLight

        return ReMeLight

    def _create_reme(self):
        cls = self._load_reme_light_cls()
        embedding_enabled = bool(self.config.EMBEDDING_API_KEY and self.config.EMBEDDING_MODEL)
        with suppress_external_output():
            return cls(
                working_dir=str(self.root),
                llm_api_key=self.config.LLM_API_KEY or None,
                llm_base_url=self.config.LLM_BASE_URL or None,
                embedding_api_key=self.config.EMBEDDING_API_KEY or None,
                embedding_base_url=self.config.EMBEDDING_BASE_URL or None,
                default_as_llm_config={
                    "model_name": self.config.get_llm_model(),
                    "params": {
                        "temperature": self.config.LLM_TEMPERATURE,
                        "seed": self.config.LLM_SEED,
                    },
                },
                default_embedding_model_config={
                    "model_name": self.config.EMBEDDING_MODEL,
                    "params": {"dimensions": self.config.EMBEDDING_DIMENSIONS},
                },
                default_file_store_config={
                    "fts_enabled": True,
                    "vector_enabled": embedding_enabled,
                },
                enable_load_env=False,
            )

    def ensure_store(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        memory_file = self.root / "MEMORY.md"
        if not memory_file.exists():
            memory_file.write_text("# MemoryAgent Rule Memory\n\n", encoding="utf-8")

    async def search_rules(self, query_text: str, max_results: int = 5) -> tuple[str, list[str], list[str]]:
        if not self.config.REME_ENABLED:
            return "ReMeLight rule memory disabled.", [], []
        self.ensure_store()

        reme = self._create_reme()
        try:
            with suppress_external_output():
                await reme.start()
                response = await reme.memory_search(query=query_text or "pipeline memory", max_results=max_results)
            text = tool_response_to_text(response)
            memory_ids = self._extract_memory_ids(text)
            if not text or text == "[]":
                text = "No related rule memory found."
            return text, [f"rule:{item}" for item in memory_ids], []
        except Exception as exc:
            return "Rule memory unavailable.", [], [f"ReMeLight rule memory unavailable: {type(exc).__name__}: {exc}"]
        finally:
            try:
                with suppress_external_output():
                    await reme.close()
            except Exception:
                pass

    async def summarize_trace(self, trace_payload: dict[str, Any] | str) -> tuple[str, list[str]]:
        if not self.config.REME_ENABLED:
            return "ReMeLight rule memory disabled.", []
        self.ensure_store()
        payload = trace_payload if isinstance(trace_payload, dict) else {"raw_trace_text": str(trace_payload)}
        summary = self._build_rule_summary(payload)
        messages = [
            Msg(
                name="orchestrator",
                role="user",
                content=summary,
            )
        ]
        reme = self._create_reme()
        try:
            with suppress_external_output():
                await reme.start()
                result = await reme.summary_memory(messages=messages, language="zh", add_thinking_block=False)
            return str(result or "ReMeLight rule summary completed."), []
        except Exception as exc:
            return "Rule memory write skipped.", [f"ReMeLight summary_memory unavailable: {type(exc).__name__}: {exc}"]
        finally:
            try:
                with suppress_external_output():
                    await reme.close()
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        self.ensure_store()
        return {
            "enabled": self.config.REME_ENABLED,
            "backend": "reme_light",
            "root": str(self.root),
            "memory_file": str(self.root / "MEMORY.md"),
            "memory_file_exists": (self.root / "MEMORY.md").exists(),
        }

    @staticmethod
    def _extract_memory_ids(search_text: str) -> list[str]:
        try:
            data = json.loads(search_text)
        except Exception:
            return []
        ids: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    candidate = item.get("id") or item.get("path") or item.get("file_path")
                    if candidate:
                        ids.append(str(candidate))
        return ids

    @staticmethod
    def _build_rule_summary(payload: dict[str, Any]) -> str:
        success = bool(payload.get("success"))
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        failure_signals = payload.get("failure_signals") or []
        lines = [
            "请把这次多智能体流水线执行结果沉淀为可复用规则记忆。",
            "只记录未来能复用的流程约束、失败教训、验证口径和工具使用经验；不要保存完整医疗数据或完整脚本。",
            "",
            f"任务输入: {payload.get('input_text', '')}",
            f"执行状态: {'SUCCESS' if success else 'FAILED'}",
            f"主链路摘要: {payload.get('orchestrator_summary', '')}",
        ]
        if failure_signals:
            lines.append(f"失败信号: {failure_signals}")
        for idx, step in enumerate(steps[:8], 1):
            if isinstance(step, dict):
                result = step.get("result") if isinstance(step.get("result"), dict) else {}
                lines.append(
                    f"Step {idx}: state={step.get('state', '')}, "
                    f"status={result.get('status', '')}, summary={result.get('summary', '')}"
                )
        return "\n".join(lines)
