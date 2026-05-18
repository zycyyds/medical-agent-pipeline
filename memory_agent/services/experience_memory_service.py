from __future__ import annotations

import csv
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from agentscope.embedding import OpenAITextEmbedding
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from memory_agent.config import Config, config
from memory_agent.services.utils import compact_json_payload, sha256_file, suppress_external_output, tool_response_to_text


class ExperienceMemoryService:
    """Vector ReMe service for Step/task/tool experience memory."""

    MAX_SAMPLE_FILES = 20
    MAX_SCHEMA_FILES = 8

    STEP1_TOOL_NAMES = [
        "scan_source_files_tool",
        "build_records_parallel_tool",
        "validate_records_tool",
        "write_generated_reorganizer_tool",
        "run_reorganize_from_records_tool",
        "validate_step1_output_tool",
    ]
    STEP4_TOOL_NAMES = [
        "resolve_step4_input_tool",
        "resolve_step4_task_text_tool",
        "run_step4_column_selection_tool",
        "normalize_step4_outputs_tool",
        "validate_step4_output_tool",
    ]

    def __init__(
        self,
        cfg: Config = config,
        task_memory_cls: Any | None = None,
        tool_memory_cls: Any | None = None,
    ) -> None:
        self.config = cfg
        self.root = cfg.resolve_project_path(cfg.REME_VECTOR_ROOT)
        self._task_memory_cls = task_memory_cls
        self._tool_memory_cls = tool_memory_cls

    def _enabled(self) -> bool:
        return bool(
            self.config.REME_ENABLED
            and self.config.EMBEDDING_API_KEY
            and self.config.EMBEDDING_MODEL
            and self.config.LLM_API_KEY
        )

    def ensure_store(self) -> None:
        (self.root / "task").mkdir(parents=True, exist_ok=True)
        (self.root / "tool").mkdir(parents=True, exist_ok=True)

    def _load_memory_classes(self):
        if self._task_memory_cls is not None and self._tool_memory_cls is not None:
            return self._task_memory_cls, self._tool_memory_cls
        from agentscope.memory import ReMeTaskLongTermMemory, ReMeToolLongTermMemory

        return self._task_memory_cls or ReMeTaskLongTermMemory, self._tool_memory_cls or ReMeToolLongTermMemory

    def _create_model(self) -> OpenAIChatModel:
        return OpenAIChatModel(
            model_name=self.config.get_llm_model(),
            api_key=self.config.LLM_API_KEY,
            stream=False,
            client_kwargs={"base_url": self.config.LLM_BASE_URL or "https://api.openai.com/v1"},
            generate_kwargs={
                "temperature": self.config.LLM_TEMPERATURE,
                "seed": self.config.LLM_SEED,
            },
        )

    def _create_embedding(self) -> OpenAITextEmbedding:
        kwargs: dict[str, Any] = {}
        if self.config.EMBEDDING_BASE_URL:
            kwargs["base_url"] = self.config.EMBEDDING_BASE_URL
        return OpenAITextEmbedding(
            model_name=self.config.EMBEDDING_MODEL,
            api_key=self.config.EMBEDDING_API_KEY,
            dimensions=self.config.EMBEDDING_DIMENSIONS,
            **kwargs,
        )

    def _memory_kwargs(self, store_dir: Path) -> dict[str, Any]:
        return {
            "agent_name": "MemoryAgent",
            "user_name": "new_pipeline",
            "run_name": "main_pipeline",
            "model": self._create_model(),
            "embedding_model": self._create_embedding(),
            "vector_store.default.backend": "local",
            "vector_store.default.params.store_dir": str(store_dir),
        }

    async def retrieve(self, query_text: str, task_spec: Any | None = None) -> tuple[str, str, list[str], list[str]]:
        if not self._enabled():
            return "", "", [], ["ReMe vector memory disabled: missing LLM or embedding config."]
        self.ensure_store()
        task_cls, tool_cls = self._load_memory_classes()
        warnings: list[str] = []
        task_text = ""
        tool_text = ""
        task_query = self._build_task_query(query_text, task_spec)
        try:
            with suppress_external_output():
                task_memory = task_cls(**self._memory_kwargs(self.root / "task"))
                async with task_memory:
                    task_text = await task_memory.retrieve(
                        Msg(name="user", role="user", content=task_query),
                        limit=3,
                    )
        except Exception as exc:
            warnings.append(f"ReMe Task memory unavailable: {type(exc).__name__}: {exc}")

        try:
            with suppress_external_output():
                tool_memory = tool_cls(**self._memory_kwargs(self.root / "tool"))
                async with tool_memory:
                    response = await tool_memory.retrieve_from_memory(
                        self._select_tool_names(query_text, task_spec),
                        limit=3,
                    )
            tool_text = tool_response_to_text(response)
        except Exception as exc:
            warnings.append(f"ReMe Tool memory unavailable: {type(exc).__name__}: {exc}")

        return task_text.strip(), tool_text.strip(), [], warnings

    async def record_from_trace(self, trace_payload: dict[str, Any] | str) -> tuple[str, list[str]]:
        if not self._enabled():
            return "ReMe vector memory disabled.", ["ReMe vector memory disabled: missing LLM or embedding config."]
        self.ensure_store()
        payload = trace_payload if isinstance(trace_payload, dict) else {"raw_trace_text": str(trace_payload)}
        experience_payloads = [
            item for item in (
                self._extract_step1_payload(payload),
                self._extract_step4_payload(payload),
            )
            if item
        ]
        if not experience_payloads:
            return "No Step experience to record.", []

        task_cls, tool_cls = self._load_memory_classes()
        warnings: list[str] = []
        summaries: list[str] = []

        try:
            with suppress_external_output():
                task_memory = task_cls(**self._memory_kwargs(self.root / "task"))
                async with task_memory:
                    for experience in experience_payloads:
                        response = await task_memory.record_to_memory(
                            thinking=self._task_memory_thinking(experience),
                            content=[compact_json_payload(experience, max_chars=5000)],
                            score=1.0 if self._experience_success(experience) else 0.0,
                        )
                        summaries.append(
                            tool_response_to_text(response)
                            or f"{self._experience_label(experience)} task memory recorded."
                        )
        except Exception as exc:
            warnings.append(f"ReMe Task memory record failed: {type(exc).__name__}: {exc}")

        try:
            tool_events: list[dict[str, Any]] = []
            for experience in experience_payloads:
                tool_events.extend(self._build_tool_events(experience, self._experience_success(experience)))
            if tool_events:
                with suppress_external_output():
                    tool_memory = tool_cls(**self._memory_kwargs(self.root / "tool"))
                    async with tool_memory:
                        response = await tool_memory.record_to_memory(
                            thinking="记录 Step1/Step4 细粒度工具调用结果，供未来工具参数和失败处理参考。",
                            content=[json.dumps(item, ensure_ascii=False) for item in tool_events],
                        )
                summaries.append(tool_response_to_text(response) or "Step tool memory recorded.")
        except Exception as exc:
            warnings.append(f"ReMe Tool memory record failed: {type(exc).__name__}: {exc}")

        return "\n".join(summaries) if summaries else "No vector memory recorded.", warnings

    def status(self) -> dict[str, Any]:
        self.ensure_store()
        return {
            "enabled": self._enabled(),
            "backend": "agentscope_reme_task_tool",
            "root": str(self.root),
            "task_store": str(self.root / "task"),
            "tool_store": str(self.root / "tool"),
            "embedding_model": self.config.EMBEDDING_MODEL,
            "embedding_dimensions": self.config.EMBEDDING_DIMENSIONS,
        }

    @staticmethod
    def _build_task_query(query_text: str, task_spec: Any | None) -> str:
        spec_text = ""
        task_type = ""
        if task_spec is not None:
            if hasattr(task_spec, "to_dict"):
                spec = task_spec.to_dict()
                task_type = str(spec.get("task_type") or "")
                spec_text = compact_json_payload(spec, max_chars=1200)
            else:
                spec_text = str(task_spec)
                task_type = spec_text
        focus = "Step4 任务驱动列筛选经验检索" if "step4" in task_type or "step2_3" in task_type else "Step1 数据重组任务经验检索"
        return f"{focus}。\n用户任务: {query_text}\nTaskSpec: {spec_text}"

    def _select_tool_names(self, query_text: str, task_spec: Any | None) -> list[str]:
        spec_text = ""
        if task_spec is not None:
            if hasattr(task_spec, "to_dict"):
                spec_text = compact_json_payload(task_spec.to_dict(), max_chars=800)
            else:
                spec_text = str(task_spec)
        text = f"{query_text} {spec_text}".lower()
        wants_step4 = any(token in text for token in ["step4", "step-4", "step 4", "resume_from_step2_3", "列筛选", "任务裁剪"])
        wants_step1 = any(token in text for token in ["step1", "step-1", "step 1", "records", "重组", "reorganize"])
        if wants_step4 and not wants_step1:
            return list(self.STEP4_TOOL_NAMES)
        if wants_step1 and not wants_step4:
            return list(self.STEP1_TOOL_NAMES)
        return list(dict.fromkeys(self.STEP1_TOOL_NAMES + self.STEP4_TOOL_NAMES))

    def _extract_step1_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "STEP1" not in str(step.get("state", "")):
                continue
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
            input_path = artifacts.get("input_path") or payload.get("input_text") or ""
            input_profile = self._safe_input_profile(str(input_path))
            script_path = str(artifacts.get("generated_script_path") or "")
            script_info = {"path": script_path, "hash": "", "size": 0}
            if script_path:
                path = Path(script_path)
                if path.exists() and path.is_file():
                    script_info = {"path": str(path), "hash": sha256_file(path), "size": path.stat().st_size}
            return {
                "memory_type": "step1_experience",
                "input_profile": input_profile,
                "status": result.get("status"),
                "summary": result.get("summary"),
                "records_count": artifacts.get("records_count"),
                "modality_counts": artifacts.get("modality_counts") or artifacts.get("sample_modality_counts"),
                "split_tables": artifacts.get("split_tables"),
                "written_files": artifacts.get("written_files"),
                "validator_status": artifacts.get("status"),
                "repair_ticket": payload.get("repair_ticket") or payload.get("failure_signals"),
                "generated_script": script_info,
                "parallel": {
                    "max_workers": artifacts.get("max_workers"),
                    "parallel_enabled": artifacts.get("parallel_enabled"),
                },
            }
        return {}

    def _extract_step4_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "STEP4" not in str(step.get("state", "")):
                continue
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
            input_path = str(artifacts.get("input_csv_path") or artifacts.get("input_path") or "")
            report_path = str(
                artifacts.get("selection_report_path")
                or artifacts.get("next_selection_report")
                or ""
            )
            report = self._safe_load_json(report_path)
            report_summary = self._summarize_step4_selection_report(report)
            return {
                "memory_type": "step4_experience",
                "input_profile": self._safe_tabular_profile(input_path),
                "status": result.get("status"),
                "summary": result.get("summary"),
                "task_text": artifacts.get("task_text") or report.get("task_text") or "",
                "selection_report_path": report_path,
                "filtered_csv_path": artifacts.get("filtered_csv_path") or artifacts.get("next_input_csv") or "",
                "column_counts": report_summary.get("column_counts") or {
                    "original_column_count": artifacts.get("original_column_count"),
                    "final_column_count": artifacts.get("final_column_count"),
                    "selector_candidate_column_count": artifacts.get("selector_candidate_column_count"),
                },
                "task_spec": report_summary.get("task_spec"),
                "term_profile": report_summary.get("term_profile"),
                "final_columns": report_summary.get("final_columns"),
                "structured_keep_overrides": report_summary.get("structured_keep_overrides"),
                "hard_gate_overrides": report_summary.get("hard_gate_overrides"),
                "warnings": report_summary.get("warnings"),
                "agent_execution": report_summary.get("agent_execution"),
                "repair_ticket": payload.get("repair_ticket") or payload.get("failure_signals"),
            }
        return {}

    @staticmethod
    def _normalize_suffix(path: Path) -> str:
        return path.suffix.lower() or "<no_suffix>"

    @staticmethod
    def _stable_json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _iter_visible_files(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if not path.exists() or not path.is_dir():
            return []
        return [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file()
            and not any(part.startswith(".") for part in item.parts)
            and "reorganized_output" not in item.parts
        ]

    @staticmethod
    def _relative_display(path: Path, root: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return path.name

    @staticmethod
    def _read_tabular_header(path: Path) -> list[str]:
        try:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                return [str(item).strip() for item in next(reader, []) if str(item).strip()]
        except Exception:
            return []

    @classmethod
    def _build_schema_summary(cls, files: list[Path], root: Path) -> dict[str, list[str]]:
        schema_summary: dict[str, list[str]] = {}
        for path in files:
            if path.suffix.lower() not in {".csv", ".tsv"}:
                continue
            header = cls._read_tabular_header(path)
            if header:
                schema_summary[cls._relative_display(path, root)] = header[:50]
            if len(schema_summary) >= cls.MAX_SCHEMA_FILES:
                break
        return schema_summary

    @classmethod
    def _safe_input_profile(cls, input_path: str) -> dict[str, Any]:
        try:
            path = Path(input_path).expanduser() if input_path else Path()
            exists = bool(input_path and path.exists())
            path_type = "missing"
            if input_path and path.is_file():
                path_type = "file"
            elif input_path and path.is_dir():
                path_type = "directory"

            files = cls._iter_visible_files(path)
            suffix_counts: dict[str, int] = {}
            total_size = 0
            for file_path in files:
                suffix = cls._normalize_suffix(file_path)
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
                try:
                    total_size += file_path.stat().st_size
                except OSError:
                    pass

            root = path if path.is_dir() else path.parent
            sample_files = [cls._relative_display(item, root) for item in files[: cls.MAX_SAMPLE_FILES]]
            schema_summary = cls._build_schema_summary(files, root)
            profile = {
                "input_path": str(path) if input_path else input_path,
                "exists": exists,
                "path_type": path_type,
                "file_count": len(files),
                "total_size": total_size,
                "suffix_counts": dict(sorted(suffix_counts.items())),
                "sample_files": sample_files,
                "schema_summary": schema_summary,
            }
            profile["profile_hash"] = hashlib.sha256(
                cls._stable_json(
                    {
                        "path_type": profile["path_type"],
                        "file_count": profile["file_count"],
                        "suffix_counts": profile["suffix_counts"],
                        "schema_summary": profile["schema_summary"],
                    }
                ).encode("utf-8")
            ).hexdigest()
            return profile
        except Exception as exc:
            return {"input_path": input_path, "profile_error": f"{type(exc).__name__}: {exc}"}

    def _build_tool_events(self, experience: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        if experience.get("memory_type") == "step4_experience":
            return self._build_step4_tool_events(experience, success)
        return self._build_step1_tool_events(experience, success)

    def _build_step1_tool_events(self, step1_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        common_output = {
            "status": step1_payload.get("status"),
            "records_count": step1_payload.get("records_count"),
            "written_files": step1_payload.get("written_files"),
            "summary": step1_payload.get("summary"),
        }
        return [
            {
                "create_time": now,
                "tool_name": "build_records_parallel_tool",
                "input": {"input_profile": step1_payload.get("input_profile")},
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "tool_name": "run_reorganize_from_records_tool",
                "input": {"generated_script": step1_payload.get("generated_script")},
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    def _build_step4_tool_events(self, step4_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        common_output = {
            "status": step4_payload.get("status"),
            "summary": step4_payload.get("summary"),
            "column_counts": step4_payload.get("column_counts"),
            "final_column_count": (step4_payload.get("column_counts") or {}).get("final_column_count"),
        }
        task_text = str(step4_payload.get("task_text") or "")
        report_paths = {
            "selection_report_path": step4_payload.get("selection_report_path"),
            "filtered_csv_path": step4_payload.get("filtered_csv_path"),
        }
        return [
            {
                "create_time": now,
                "tool_name": "resolve_step4_input_tool",
                "input": {"input_profile": step4_payload.get("input_profile")},
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "tool_name": "resolve_step4_task_text_tool",
                "input": {"task_text": task_text[:600]},
                "output": compact_json_payload({"task_text": task_text[:600]}, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "tool_name": "run_step4_column_selection_tool",
                "input": {
                    "input_profile": step4_payload.get("input_profile"),
                    "task_spec": step4_payload.get("task_spec"),
                    "term_profile": step4_payload.get("term_profile"),
                },
                "output": compact_json_payload(common_output, max_chars=1200),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "tool_name": "normalize_step4_outputs_tool",
                "input": report_paths,
                "output": compact_json_payload(report_paths, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "tool_name": "validate_step4_output_tool",
                "input": report_paths,
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    @staticmethod
    def _experience_label(experience: dict[str, Any]) -> str:
        return "Step4" if experience.get("memory_type") == "step4_experience" else "Step1"

    def _task_memory_thinking(self, experience: dict[str, Any]) -> str:
        if experience.get("memory_type") == "step4_experience":
            return "记录 Step4 列筛选案例，供未来相似任务复用词表、结构化字段保护和最终列选择经验。"
        return "记录 Step1 执行案例，供未来相似输入目录和 records 生成任务检索。"

    @staticmethod
    def _experience_success(experience: dict[str, Any]) -> bool:
        return str(experience.get("status") or "").upper() == "SUCCESS"

    @staticmethod
    def _safe_load_json(path_text: str) -> dict[str, Any]:
        if not path_text:
            return {}
        try:
            path = Path(path_text)
            if not path.is_file():
                return {}
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _bounded_list(values: Any, max_items: int = 80) -> list[Any]:
        if not isinstance(values, list):
            return []
        return values[:max_items]

    def _summarize_step4_selection_report(self, report: dict[str, Any]) -> dict[str, Any]:
        if not report:
            return {}
        agent_execution = report.get("agent_execution") if isinstance(report.get("agent_execution"), dict) else {}
        sanitized_agent_execution = {
            key: value
            for key, value in agent_execution.items()
            if "raw_response" not in key
        }
        agent_decisions = report.get("agent_decisions") if isinstance(report.get("agent_decisions"), dict) else {}
        hard_gate_overrides = report.get("hard_gate_overrides")
        if isinstance(hard_gate_overrides, dict):
            compact_hard_gates: Any = {
                key: self._bounded_list(value, 80) if isinstance(value, list) else value
                for key, value in hard_gate_overrides.items()
            }
        else:
            compact_hard_gates = self._bounded_list(hard_gate_overrides, 80)
        return {
            "task_spec": report.get("task_spec") if isinstance(report.get("task_spec"), dict) else {},
            "term_profile": report.get("term_profile") if isinstance(report.get("term_profile"), dict) else {},
            "column_counts": report.get("column_counts") if isinstance(report.get("column_counts"), dict) else {},
            "final_columns": self._bounded_list(report.get("final_columns"), 200),
            "structured_keep_overrides": self._bounded_list(agent_decisions.get("structured_keep_overrides"), 120),
            "hard_gate_overrides": compact_hard_gates,
            "warnings": self._bounded_list(report.get("warnings"), 40),
            "agent_execution": sanitized_agent_execution,
        }

    @staticmethod
    def _safe_tabular_profile(input_path: str) -> dict[str, Any]:
        path = Path(input_path) if input_path else Path()
        profile: dict[str, Any] = {
            "input_path": input_path,
            "exists": bool(input_path and path.exists()),
            "path_type": "file" if input_path and path.is_file() else "directory" if input_path and path.is_dir() else "missing",
        }
        try:
            if not path.is_file():
                return profile
            suffix = path.suffix.lower()
            if suffix == ".csv":
                columns = list(pd.read_csv(path, nrows=0).columns)
            elif suffix in {".xlsx", ".xls"}:
                columns = list(pd.read_excel(path, nrows=0).columns)
            else:
                columns = []
            profile.update(
                {
                    "suffix": suffix,
                    "file_size": path.stat().st_size,
                    "column_count": len(columns),
                    "sample_columns": columns[:40],
                    "profile_hash": hashlib.sha256(
                        json.dumps(
                            {
                                "path_name": path.name,
                                "suffix": suffix,
                                "file_size": path.stat().st_size,
                                "columns": columns,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
            return profile
        except Exception as exc:
            profile["profile_error"] = f"{type(exc).__name__}: {exc}"
            return profile
