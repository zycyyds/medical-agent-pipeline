from __future__ import annotations

import csv
import json
import hashlib
import os
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
    TOOL_EVENT_MAX_CHARS = 1600

    STEP1_TOOL_NAMES = [
        "scan_source_files_tool",
        "build_records_parallel_tool",
        "validate_records_tool",
        "write_generated_reorganizer_tool",
        "run_reorganize_from_records_tool",
        "validate_step1_output_tool",
    ]
    STEP23_TOOL_NAMES = [
        "resolve_step23_input_tool",
        "scan_step23_input_tool",
        "run_step23_ocr_tool",
        "run_step23_extraction_tool",
        "run_step23_fill_table_tool",
        "normalize_step23_outputs_tool",
        "validate_step23_output_tool",
    ]
    STEP4_TOOL_NAMES = [
        "resolve_step4_input_tool",
        "resolve_step4_task_text_tool",
        "run_step4_column_selection_tool",
        "normalize_step4_outputs_tool",
        "validate_step4_output_tool",
    ]
    STEP5_TOOL_NAMES = [
        "resolve_step5_input_tool",
        "profile_step5_table_tool",
        "classify_step5_column_risks_tool",
        "run_step5_data_cleaning_tool",
        "normalize_step5_outputs_tool",
        "validate_step5_output_tool",
    ]
    STEP6_TOOL_NAMES = [
        "load_data_overview",
        "analyze_all_patients_consistency",
        "save_step6_report",
    ]
    STEP7_TOOL_NAMES = [
        "load_task_context",
        "get_column_distribution",
        "discover_diagnosis_fields",
        "create_icd_binary_label",
        "create_notnull_binary_label",
        "propose_label_mapping",
        "build_label_series",
        "format_and_save_ml_dataset",
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

    async def retrieve(
        self,
        query_text: str,
        task_spec: Any | None = None,
        step_name: str | None = None,
    ) -> tuple[str, str, list[str], list[str]]:
        if not self._enabled():
            return "", "", [], ["ReMe vector memory disabled: missing LLM or embedding config."]
        self.ensure_store()
        task_cls, tool_cls = self._load_memory_classes()
        warnings: list[str] = []
        task_text = ""
        tool_text = ""
        task_query = self._build_task_query(query_text, task_spec, step_name=step_name)
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
                        self._select_tool_names(query_text, task_spec, step_name=step_name),
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
                self._extract_step23_payload(payload),
                self._extract_step4_payload(payload),
                self._extract_step5_payload(payload),
                self._extract_step6_payload(payload),
                self._extract_step7_payload(payload),
            )
            if item
        ]
        if not experience_payloads:
            return "No Step experience to record.", []

        task_cls, tool_cls = self._load_memory_classes()
        warnings: list[str] = []
        summaries: list[str] = []

        self._print_progress("[MemoryAgent] post-run: Task Memory start")
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
        self._print_progress(f"[MemoryAgent] post-run: Task Memory done，warnings={len(warnings)}")

        tool_warning_count_before = len(warnings)
        self._print_progress("[MemoryAgent] post-run: Tool Memory start")
        try:
            tool_events: list[dict[str, Any]] = []
            for experience in experience_payloads:
                tool_events.extend(self._build_tool_events(experience, self._experience_success(experience)))
            if tool_events:
                with suppress_external_output():
                    tool_memory = tool_cls(**self._memory_kwargs(self.root / "tool"))
                    async with tool_memory:
                        response = await tool_memory.record_to_memory(
                            thinking="记录 Step1/Step2-3/Step4/Step5/Step6/Step7 细粒度工具调用结果，供未来工具参数和失败处理参考。",
                            content=[self._tool_event_to_memory_text(item) for item in tool_events],
                        )
                summaries.append(tool_response_to_text(response) or "Step tool memory recorded.")
        except Exception as exc:
            warnings.append(f"ReMe Tool memory record failed: {type(exc).__name__}: {exc}")
        self._print_progress(
            f"[MemoryAgent] post-run: Tool Memory done，warnings={len(warnings) - tool_warning_count_before}"
        )

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
    def _print_progress(message: str) -> None:
        if os.environ.get("MEMORY_PROGRESS_ENABLED", "true").lower() in {"0", "false", "no", "off"}:
            return
        print(message)

    @staticmethod
    def _normalize_step_name(step_name: str | None) -> str:
        text = str(step_name or "").lower().replace("-", "_")
        if "step2_3" in text or "step23" in text or "step 2_3" in text:
            return "step2_3"
        if "step1" in text or "step 1" in text:
            return "step1"
        if "step4" in text or "step 4" in text:
            return "step4"
        if "step5" in text or "step 5" in text:
            return "step5"
        if "step6" in text or "step 6" in text:
            return "step6"
        if "step7" in text or "step 7" in text:
            return "step7"
        return ""

    @staticmethod
    def _step_focus(step_name: str) -> str:
        if step_name == "step5":
            return "Step5 数据清洗和质量修复经验检索"
        if step_name == "step6":
            return "Step6 患者一致性验证和通过名单报告经验检索"
        if step_name == "step7":
            return "Step7 ML 训练数据集生成、标签构造和 ICD 标准化经验检索"
        if step_name == "step4":
            return "Step4 任务驱动列筛选经验检索"
        if step_name == "step2_3":
            return "Step2-3 OCR、结构化抽取、目录处理和宽表回填经验检索"
        return "Step1 数据重组任务经验检索"

    def _build_task_query(self, query_text: str, task_spec: Any | None, step_name: str | None = None) -> str:
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
        query = f"{query_text} {task_type} {spec_text}".lower()
        normalized_step = self._normalize_step_name(step_name)
        wants_step4 = any(token in query for token in ["step4", "step-4", "step 4", "resume_from_step2_3", "列筛选", "任务裁剪"])
        wants_step5 = any(token in query for token in ["step5", "step-5", "step 5", "resume_from_step4", "数据清洗", "质量修复"])
        wants_step6 = any(token in query for token in ["step6", "step-6", "step 6", "resume_from_step5", "一致性验证", "一致性检查"])
        wants_step7 = any(token in query for token in ["step7", "step-7", "step 7", "resume_from_step6", "ml训练", "机器学习", "训练数据集"])
        wants_step23 = any(token in query for token in ["step2_3_only", "step2-3", "step 2-3", "agent2-3", "agent_2-3", "ocr", "回填", "抽取", "目录处理"])
        if normalized_step:
            focus = self._step_focus(normalized_step)
        elif wants_step7:
            focus = self._step_focus("step7")
        elif wants_step6:
            focus = self._step_focus("step6")
        elif wants_step5:
            focus = self._step_focus("step5")
        elif wants_step4:
            focus = self._step_focus("step4")
        elif wants_step23:
            focus = self._step_focus("step2_3")
        else:
            focus = self._step_focus("step1")
        return f"{focus}。\n用户任务: {query_text}\nTaskSpec: {spec_text}"

    def _select_tool_names(
        self,
        query_text: str,
        task_spec: Any | None,
        step_name: str | None = None,
    ) -> list[str]:
        normalized_step = self._normalize_step_name(step_name)
        if normalized_step == "step5":
            return list(self.STEP5_TOOL_NAMES)
        if normalized_step == "step6":
            return list(self.STEP6_TOOL_NAMES)
        if normalized_step == "step7":
            return list(self.STEP7_TOOL_NAMES)
        if normalized_step == "step4":
            return list(self.STEP4_TOOL_NAMES)
        if normalized_step == "step2_3":
            return list(self.STEP23_TOOL_NAMES)
        if normalized_step == "step1":
            return list(self.STEP1_TOOL_NAMES)

        spec_text = ""
        if task_spec is not None:
            if hasattr(task_spec, "to_dict"):
                spec_text = compact_json_payload(task_spec.to_dict(), max_chars=800)
            else:
                spec_text = str(task_spec)
        text = f"{query_text} {spec_text}".lower()
        wants_step4 = any(token in text for token in ["step4", "step-4", "step 4", "resume_from_step2_3", "列筛选", "任务裁剪"])
        wants_step5 = any(token in text for token in ["step5", "step-5", "step 5", "resume_from_step4", "数据清洗", "质量修复"])
        wants_step6 = any(token in text for token in ["step6", "step-6", "step 6", "resume_from_step5", "一致性验证", "一致性检查"])
        wants_step7 = any(token in text for token in ["step7", "step-7", "step 7", "resume_from_step6", "ml训练", "机器学习", "训练数据集"])
        wants_step23 = any(token in text for token in ["step2_3_only", "step2-3", "step 2-3", "agent2-3", "agent_2-3", "ocr", "回填", "抽取", "目录处理"])
        wants_step1 = any(token in text for token in ["step1", "step-1", "step 1", "records", "重组", "reorganize"])
        if wants_step7:
            return list(self.STEP7_TOOL_NAMES)
        if wants_step6 and not wants_step5 and not wants_step4 and not wants_step1 and not wants_step23:
            return list(self.STEP6_TOOL_NAMES)
        if wants_step5 and not wants_step4 and not wants_step1 and not wants_step23:
            return list(self.STEP5_TOOL_NAMES)
        if wants_step23 and not wants_step4 and not wants_step1:
            return list(self.STEP23_TOOL_NAMES)
        if wants_step4 and not wants_step1:
            return list(self.STEP4_TOOL_NAMES)
        if wants_step1 and not wants_step4 and not wants_step23:
            return list(self.STEP1_TOOL_NAMES)
        return list(
            dict.fromkeys(
                self.STEP1_TOOL_NAMES
                + self.STEP23_TOOL_NAMES
                + self.STEP4_TOOL_NAMES
                + self.STEP5_TOOL_NAMES
                + self.STEP6_TOOL_NAMES
                + self.STEP7_TOOL_NAMES
            )
        )

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

    def _extract_step23_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "STEP2_3" not in str(step.get("state", "")):
                continue
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
            input_path = str(artifacts.get("input_path") or "")
            return {
                "memory_type": "step23_experience",
                "input_profile": self._safe_input_profile(input_path),
                "status": result.get("status"),
                "summary": result.get("summary"),
                "mode": artifacts.get("mode"),
                "results_dir": artifacts.get("results_dir"),
                "ocr_cache_path": artifacts.get("ocr_cache_path"),
                "entities_csv": artifacts.get("entities_csv"),
                "patients_csv": artifacts.get("patients_csv"),
                "filled_tables_dir": artifacts.get("filled_tables_dir"),
                "filled_merged_csv": artifacts.get("filled_merged_csv"),
                "next_input_csv": artifacts.get("next_input_csv"),
                "next_summary_json": artifacts.get("next_summary_json"),
                "counts": {
                    "total_patients": artifacts.get("total_patients"),
                    "total_files": artifacts.get("total_files"),
                    "image_files": artifacts.get("image_files"),
                    "text_files": artifacts.get("text_files"),
                    "total_images": artifacts.get("total_images"),
                    "successful_ocr": artifacts.get("successful_ocr"),
                    "failed_ocr": artifacts.get("failed_ocr"),
                    "total_ocr_texts": artifacts.get("total_ocr_texts"),
                    "extracted_texts": artifacts.get("extracted_texts"),
                    "failed_texts": artifacts.get("failed_texts"),
                    "total_entities": artifacts.get("total_entities"),
                    "output_row_count": artifacts.get("output_row_count"),
                    "output_column_count": artifacts.get("output_column_count"),
                },
                "warnings": artifacts.get("warnings") or [],
                "repair_ticket": payload.get("repair_ticket") or payload.get("failure_signals"),
            }
        return {}

    def _extract_step5_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "STEP5" not in str(step.get("state", "")):
                continue
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
            input_path = str(artifacts.get("input_csv_path") or artifacts.get("input_path") or "")
            return {
                "memory_type": "step5_experience",
                "input_profile": self._safe_tabular_profile(input_path),
                "status": result.get("status"),
                "summary": result.get("summary"),
                "cleaned_csv_path": artifacts.get("cleaned_csv_path"),
                "next_input_csv": artifacts.get("next_input_csv"),
                "data_quality_report_path": artifacts.get("data_quality_report_path"),
                "column_risk_report_path": artifacts.get("column_risk_report_path"),
                "cleaning_changes_path": artifacts.get("cleaning_changes_path"),
                "risk_counts": artifacts.get("risk_counts"),
                "changed_cell_count": artifacts.get("changed_cell_count"),
                "row_count": artifacts.get("output_row_count") or artifacts.get("row_count"),
                "column_count": artifacts.get("output_column_count") or artifacts.get("column_count"),
                "warnings": artifacts.get("warnings") or [],
                "repair_ticket": payload.get("repair_ticket") or payload.get("failure_signals"),
            }
        return {}

    def _extract_step6_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "STEP6" not in str(step.get("state", "")):
                continue
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
            input_path = str(artifacts.get("input_csv") or artifacts.get("input_path") or "")
            passed_json_path = str(artifacts.get("passed_patients_json") or "")
            passed_payload = self._safe_load_json(passed_json_path)
            passed_count = artifacts.get("passed_patient_count")
            if passed_count in (None, ""):
                passed_count = passed_payload.get("passed_count")
            return {
                "memory_type": "step6_experience",
                "input_profile": self._safe_tabular_profile(input_path),
                "status": result.get("status"),
                "summary": result.get("summary"),
                "input_csv": artifacts.get("input_csv") or input_path,
                "raw_csv": artifacts.get("raw_csv") or input_path,
                "output_step6": artifacts.get("output_step6"),
                "step6_report_txt": artifacts.get("step6_report_txt"),
                "step6_report_json": artifacts.get("step6_report_json"),
                "passed_patients_json": passed_json_path,
                "passed_patient_count": passed_count,
                "passed_patient_ids_sample": artifacts.get("passed_patient_ids_sample") or [],
                "memory_context_used": bool(artifacts.get("memory_context_used")),
                "repair_ticket": payload.get("repair_ticket") or payload.get("failure_signals"),
            }
        return {}

    def _extract_step7_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "STEP7" not in str(step.get("state", "")):
                continue
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
            input_path = str(artifacts.get("input_csv") or artifacts.get("input_path") or "")
            dataset_csvs = artifacts.get("step7_dataset_csvs") if isinstance(artifacts.get("step7_dataset_csvs"), list) else []
            dataset_jsons = artifacts.get("step7_dataset_jsons") if isinstance(artifacts.get("step7_dataset_jsons"), list) else []
            model_configs = artifacts.get("model_config_jsons") if isinstance(artifacts.get("model_config_jsons"), list) else []
            return {
                "memory_type": "step7_experience",
                "input_profile": self._safe_tabular_profile(input_path),
                "status": result.get("status"),
                "summary": result.get("summary"),
                "input_csv": artifacts.get("input_csv") or input_path,
                "selection_report": artifacts.get("selection_report"),
                "passed_patients_json": artifacts.get("passed_patients_json"),
                "passed_patient_count": artifacts.get("passed_patient_count"),
                "output_step7": artifacts.get("output_step7"),
                "step7_dataset_count": artifacts.get("step7_dataset_count") or len(dataset_csvs),
                "step7_dataset_csvs": [str(item) for item in dataset_csvs[:5]],
                "step7_dataset_jsons": [str(item) for item in dataset_jsons[:5]],
                "model_config_jsons": [str(item) for item in model_configs[:5]],
                "memory_context_used": bool(artifacts.get("memory_context_used")),
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

    @staticmethod
    def _compact_profile_for_tool(profile: Any) -> dict[str, Any]:
        if not isinstance(profile, dict):
            return {}
        compact: dict[str, Any] = {}
        for key in (
            "input_path",
            "exists",
            "path_type",
            "suffix",
            "file_size",
            "file_count",
            "total_size",
            "row_count",
            "column_count",
            "profile_hash",
            "profile_error",
        ):
            if key in profile:
                compact[key] = profile.get(key)
        suffix_counts = profile.get("suffix_counts")
        if isinstance(suffix_counts, dict):
            compact["suffix_counts"] = suffix_counts
        schema_summary = profile.get("schema_summary")
        if isinstance(schema_summary, dict):
            compact["schema_file_count"] = len(schema_summary)
        sample_files = profile.get("sample_files")
        if isinstance(sample_files, list):
            compact["sample_file_count"] = len(sample_files)
        return compact

    @staticmethod
    def _compact_term_profile_for_tool(term_profile: Any) -> dict[str, Any]:
        if not isinstance(term_profile, dict):
            return {}
        compact: dict[str, Any] = {}
        if term_profile.get("task_domain"):
            compact["task_domain"] = term_profile.get("task_domain")
        for key, value in term_profile.items():
            if isinstance(value, list):
                compact[f"{key}_count"] = len(value)
        return compact

    @staticmethod
    def _compact_warnings(values: Any, max_items: int = 8) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(item)[:300] for item in values[:max_items]]

    def _tool_event_to_memory_text(self, event: dict[str, Any]) -> str:
        text = compact_json_payload(event, max_chars=self.TOOL_EVENT_MAX_CHARS)
        if len(text) <= self.TOOL_EVENT_MAX_CHARS:
            return text
        suffix = "\n...<truncated>"
        return text[: max(0, self.TOOL_EVENT_MAX_CHARS - len(suffix))] + suffix

    def _build_tool_events(self, experience: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        if experience.get("memory_type") == "step7_experience":
            return self._build_step7_tool_events(experience, success)
        if experience.get("memory_type") == "step6_experience":
            return self._build_step6_tool_events(experience, success)
        if experience.get("memory_type") == "step5_experience":
            return self._build_step5_tool_events(experience, success)
        if experience.get("memory_type") == "step23_experience":
            return self._build_step23_tool_events(experience, success)
        if experience.get("memory_type") == "step4_experience":
            return self._build_step4_tool_events(experience, success)
        return self._build_step1_tool_events(experience, success)

    def _build_step1_tool_events(self, step1_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compact_profile = self._compact_profile_for_tool(step1_payload.get("input_profile"))
        common_output = {
            "status": step1_payload.get("status"),
            "step_name": "step1",
            "records_count": step1_payload.get("records_count"),
            "written_files": step1_payload.get("written_files"),
            "summary": step1_payload.get("summary"),
            "warnings": self._compact_warnings(step1_payload.get("repair_ticket")),
        }
        return [
            {
                "create_time": now,
                "step_name": "step1",
                "tool_name": "build_records_parallel_tool",
                "input": {"input_profile": compact_profile},
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step1",
                "tool_name": "run_reorganize_from_records_tool",
                "input": {"generated_script": step1_payload.get("generated_script")},
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    def _build_step23_tool_events(self, step23_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compact_profile = self._compact_profile_for_tool(step23_payload.get("input_profile"))
        common_output = {
            "status": step23_payload.get("status"),
            "step_name": "step2_3",
            "summary": step23_payload.get("summary"),
            "counts": step23_payload.get("counts"),
            "next_input_csv": step23_payload.get("next_input_csv"),
            "warnings": self._compact_warnings(step23_payload.get("warnings")),
        }
        paths = {
            "entities_csv": step23_payload.get("entities_csv"),
            "patients_csv": step23_payload.get("patients_csv"),
            "filled_merged_csv": step23_payload.get("filled_merged_csv"),
            "next_input_csv": step23_payload.get("next_input_csv"),
        }
        return [
            {
                "create_time": now,
                "step_name": "step2_3",
                "tool_name": "resolve_step23_input_tool",
                "input": {"input_profile": compact_profile},
                "output": compact_json_payload({"mode": step23_payload.get("mode")}, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step2_3",
                "tool_name": "scan_step23_input_tool",
                "input": {"profile_hash": compact_profile.get("profile_hash"), "input_path": compact_profile.get("input_path")},
                "output": compact_json_payload({"counts": step23_payload.get("counts")}, max_chars=900),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step2_3",
                "tool_name": "run_step23_ocr_tool",
                "input": {"profile_hash": compact_profile.get("profile_hash"), "input_path": compact_profile.get("input_path")},
                "output": compact_json_payload(
                    {
                        "ocr_cache_path": step23_payload.get("ocr_cache_path"),
                        "total_images": (step23_payload.get("counts") or {}).get("total_images"),
                        "successful_ocr": (step23_payload.get("counts") or {}).get("successful_ocr"),
                        "failed_ocr": (step23_payload.get("counts") or {}).get("failed_ocr"),
                    },
                    max_chars=900,
                ),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step2_3",
                "tool_name": "run_step23_extraction_tool",
                "input": {"ocr_cache_path": step23_payload.get("ocr_cache_path")},
                "output": compact_json_payload(
                    {
                        "entities_csv": step23_payload.get("entities_csv"),
                        "patients_csv": step23_payload.get("patients_csv"),
                        "total_entities": (step23_payload.get("counts") or {}).get("total_entities"),
                    },
                    max_chars=900,
                ),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step2_3",
                "tool_name": "run_step23_fill_table_tool",
                "input": {"paths": paths},
                "output": compact_json_payload(common_output, max_chars=1000),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step2_3",
                "tool_name": "normalize_step23_outputs_tool",
                "input": {"paths": paths},
                "output": compact_json_payload({"next_input_csv": step23_payload.get("next_input_csv")}, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step2_3",
                "tool_name": "validate_step23_output_tool",
                "input": {"next_input_csv": step23_payload.get("next_input_csv")},
                "output": compact_json_payload(common_output, max_chars=1000),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    def _build_step4_tool_events(self, step4_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compact_profile = self._compact_profile_for_tool(step4_payload.get("input_profile"))
        common_output = {
            "status": step4_payload.get("status"),
            "step_name": "step4",
            "summary": step4_payload.get("summary"),
            "column_counts": step4_payload.get("column_counts"),
            "final_column_count": (step4_payload.get("column_counts") or {}).get("final_column_count"),
            "warnings": self._compact_warnings(step4_payload.get("warnings")),
        }
        task_text = str(step4_payload.get("task_text") or "")
        report_paths = {
            "selection_report_path": step4_payload.get("selection_report_path"),
            "filtered_csv_path": step4_payload.get("filtered_csv_path"),
        }
        return [
            {
                "create_time": now,
                "step_name": "step4",
                "tool_name": "resolve_step4_input_tool",
                "input": {"input_profile": compact_profile},
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step4",
                "tool_name": "resolve_step4_task_text_tool",
                "input": {"task_text": task_text[:600]},
                "output": compact_json_payload({"task_text": task_text[:600]}, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step4",
                "tool_name": "run_step4_column_selection_tool",
                "input": {
                    "input_profile": compact_profile,
                    "task_spec": step4_payload.get("task_spec"),
                    "term_counts": self._compact_term_profile_for_tool(step4_payload.get("term_profile")),
                },
                "output": compact_json_payload(common_output, max_chars=1200),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step4",
                "tool_name": "normalize_step4_outputs_tool",
                "input": report_paths,
                "output": compact_json_payload(report_paths, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step4",
                "tool_name": "validate_step4_output_tool",
                "input": report_paths,
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    def _build_step5_tool_events(self, step5_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compact_profile = self._compact_profile_for_tool(step5_payload.get("input_profile"))
        common_output = {
            "status": step5_payload.get("status"),
            "step_name": "step5",
            "summary": step5_payload.get("summary"),
            "risk_counts": step5_payload.get("risk_counts"),
            "changed_cell_count": step5_payload.get("changed_cell_count"),
            "next_input_csv": step5_payload.get("next_input_csv"),
            "warnings": self._compact_warnings(step5_payload.get("warnings")),
        }
        paths = {
            "cleaned_csv_path": step5_payload.get("cleaned_csv_path"),
            "data_quality_report_path": step5_payload.get("data_quality_report_path"),
            "column_risk_report_path": step5_payload.get("column_risk_report_path"),
            "cleaning_changes_path": step5_payload.get("cleaning_changes_path"),
            "next_input_csv": step5_payload.get("next_input_csv"),
        }
        return [
            {
                "create_time": now,
                "step_name": "step5",
                "tool_name": "resolve_step5_input_tool",
                "input": {"input_profile": compact_profile},
                "output": compact_json_payload(common_output, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step5",
                "tool_name": "profile_step5_table_tool",
                "input": {"profile_hash": compact_profile.get("profile_hash"), "input_path": compact_profile.get("input_path")},
                "output": compact_json_payload(
                    {"row_count": step5_payload.get("row_count"), "column_count": step5_payload.get("column_count")},
                    max_chars=800,
                ),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step5",
                "tool_name": "classify_step5_column_risks_tool",
                "input": {"profile_hash": compact_profile.get("profile_hash"), "input_path": compact_profile.get("input_path")},
                "output": compact_json_payload({"risk_counts": step5_payload.get("risk_counts")}, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step5",
                "tool_name": "run_step5_data_cleaning_tool",
                "input": paths,
                "output": compact_json_payload(common_output, max_chars=1000),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step5",
                "tool_name": "normalize_step5_outputs_tool",
                "input": paths,
                "output": compact_json_payload({"next_input_csv": step5_payload.get("next_input_csv")}, max_chars=800),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step5",
                "tool_name": "validate_step5_output_tool",
                "input": {"next_input_csv": step5_payload.get("next_input_csv")},
                "output": compact_json_payload(common_output, max_chars=1000),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    def _build_step6_tool_events(self, step6_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compact_profile = self._compact_profile_for_tool(step6_payload.get("input_profile"))
        common_output = {
            "status": step6_payload.get("status"),
            "step_name": "step6",
            "summary": step6_payload.get("summary"),
            "passed_patient_count": step6_payload.get("passed_patient_count"),
            "step6_report_json": step6_payload.get("step6_report_json"),
            "passed_patients_json": step6_payload.get("passed_patients_json"),
            "warnings": self._compact_warnings(step6_payload.get("repair_ticket")),
        }
        report_paths = {
            "step6_report_txt": step6_payload.get("step6_report_txt"),
            "step6_report_json": step6_payload.get("step6_report_json"),
            "passed_patients_json": step6_payload.get("passed_patients_json"),
        }
        return [
            {
                "create_time": now,
                "step_name": "step6",
                "tool_name": "load_data_overview",
                "input": {"input_profile": compact_profile},
                "output": compact_json_payload(
                    {
                        "row_count": compact_profile.get("row_count"),
                        "column_count": compact_profile.get("column_count"),
                    },
                    max_chars=800,
                ),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step6",
                "tool_name": "analyze_all_patients_consistency",
                "input": {"profile_hash": compact_profile.get("profile_hash"), "input_path": compact_profile.get("input_path")},
                "output": compact_json_payload(common_output, max_chars=900),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step6",
                "tool_name": "save_step6_report",
                "input": report_paths,
                "output": compact_json_payload(common_output, max_chars=900),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    def _build_step7_tool_events(self, step7_payload: dict[str, Any], success: bool) -> list[dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compact_profile = self._compact_profile_for_tool(step7_payload.get("input_profile"))
        common_output = {
            "status": step7_payload.get("status"),
            "step_name": "step7",
            "summary": step7_payload.get("summary"),
            "passed_patient_count": step7_payload.get("passed_patient_count"),
            "step7_dataset_count": step7_payload.get("step7_dataset_count"),
            "output_step7": step7_payload.get("output_step7"),
            "warnings": self._compact_warnings(step7_payload.get("repair_ticket")),
        }
        dataset_paths = {
            "step7_dataset_csvs": step7_payload.get("step7_dataset_csvs"),
            "step7_dataset_jsons": step7_payload.get("step7_dataset_jsons"),
            "model_config_jsons": step7_payload.get("model_config_jsons"),
        }
        return [
            {
                "create_time": now,
                "step_name": "step7",
                "tool_name": "load_task_context",
                "input": {
                    "input_profile": compact_profile,
                    "selection_report": step7_payload.get("selection_report"),
                    "passed_patients_json": step7_payload.get("passed_patients_json"),
                },
                "output": compact_json_payload(
                    {
                        "row_count": compact_profile.get("row_count"),
                        "column_count": compact_profile.get("column_count"),
                        "passed_patient_count": step7_payload.get("passed_patient_count"),
                    },
                    max_chars=900,
                ),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step7",
                "tool_name": "discover_diagnosis_fields",
                "input": {"profile_hash": compact_profile.get("profile_hash"), "input_path": compact_profile.get("input_path")},
                "output": compact_json_payload(common_output, max_chars=900),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
            {
                "create_time": now,
                "step_name": "step7",
                "tool_name": "format_and_save_ml_dataset",
                "input": dataset_paths,
                "output": compact_json_payload(common_output, max_chars=900),
                "token_cost": 0,
                "success": success,
                "time_cost": 0.0,
            },
        ]

    @staticmethod
    def _experience_label(experience: dict[str, Any]) -> str:
        memory_type = experience.get("memory_type")
        if memory_type == "step7_experience":
            return "Step7"
        if memory_type == "step6_experience":
            return "Step6"
        if memory_type == "step4_experience":
            return "Step4"
        if memory_type == "step5_experience":
            return "Step5"
        if memory_type == "step23_experience":
            return "Step2-3"
        return "Step1"

    def _task_memory_thinking(self, experience: dict[str, Any]) -> str:
        if experience.get("memory_type") == "step7_experience":
            return "记录 Step7 ML 数据集生成案例，供未来标签构造、ICD 标准化和输出验收经验复用。"
        if experience.get("memory_type") == "step6_experience":
            return "记录 Step6 一致性验证案例，供未来患者 ID 映射、通过名单报告和输出验收经验复用。"
        if experience.get("memory_type") == "step4_experience":
            return "记录 Step4 列筛选案例，供未来相似任务复用词表、结构化字段保护和最终列选择经验。"
        if experience.get("memory_type") == "step5_experience":
            return "记录 Step5 数据清洗案例，供未来相似列风险分类、清洗规则和 high-risk 脚本经验复用。"
        if experience.get("memory_type") == "step23_experience":
            return "记录 Step2-3 OCR、结构化抽取和宽表回填案例，供未来相似输入结构和工具参数复用。"
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
