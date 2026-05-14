from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


SECTION_ORDER = [
    "strategies_and_hard_rules",
    "modality_rules",
    "tool_usage",
    "validation_checklist",
    "data_organization",
    "failure_patterns",
    "domain_heuristics",
    "output_contracts",
]


SECTION_TITLES = {
    "strategies_and_hard_rules": "策略与硬规则",
    "modality_rules": "模态规则",
    "tool_usage": "工具使用",
    "validation_checklist": "验证检查表",
    "data_organization": "数据整理",
    "failure_patterns": "失败模式",
    "domain_heuristics": "领域启发",
    "output_contracts": "输出约定",
}


SECTION_PREFIX = {
    "strategies_and_hard_rules": "str",
    "modality_rules": "mod",
    "tool_usage": "tool",
    "validation_checklist": "chk",
    "data_organization": "data",
    "failure_patterns": "fail",
    "domain_heuristics": "dom",
    "output_contracts": "out",
}


SECTION_ALIASES = {
    "general": "strategies_and_hard_rules",
    "strategy": "strategies_and_hard_rules",
    "rule": "strategies_and_hard_rules",
    "domain_heuristic": "domain_heuristics",
    "domain_heuristics": "domain_heuristics",
    "tool": "tool_usage",
    "tools": "tool_usage",
    "validation": "validation_checklist",
    "checklist": "validation_checklist",
    "output": "output_contracts",
    "modality": "modality_rules",
    "data": "data_organization",
    "failure": "failure_patterns",
}


def normalize_section(section: str | None, content: str = "", bullet_type: str | None = None) -> str:
    raw = str(section or bullet_type or "").strip().lower()
    if raw in SECTION_TITLES:
        return raw
    if raw in SECTION_ALIASES:
        return SECTION_ALIASES[raw]

    text = str(content or "").lower()
    if any(word in text for word in ("ocr+figure", "模态", "figure", "ocr")):
        return "modality_rules"
    if any(word in text for word in ("验证", "复核", "人工确认", "check", "校验")):
        return "validation_checklist"
    if any(word in text for word in ("工具", "调用", "api", "路径", "json", "gui", "gradio")):
        return "tool_usage"
    if any(word in text for word in ("整理", "归档", "目录", "data/", "output/")):
        return "data_organization"
    if any(word in text for word in ("失败", "报错", "错误", "异常", "误判")):
        return "failure_patterns"
    if any(word in text for word in ("输出", "报告", "字段", "schema", "格式")):
        return "output_contracts"
    if any(word in text for word in ("病例", "眼位", "眼震", "实验室")):
        return "domain_heuristics"
    return "strategies_and_hard_rules"

@dataclass
class Bullet:
    id: str = ""
    content: str = ""
    section: str = "strategies_and_hard_rules"
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: str = ""
    last_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "active"

    def __post_init__(self) -> None:
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_used:
            self.last_used = now
        self.section = normalize_section(self.section, self.content)

    @property
    def score(self) -> int:
        return int(self.helpful_count) - int(self.harmful_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "section": self.section,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "metadata": self.metadata,
            "status": self.status,
        }


@dataclass
class ExecutionTrace:
    raw_text: str
    input_text: str
    output_text: str
    tool_sequence: list[str] = field(default_factory=list)
    used_bullet_ids: list[str] = field(default_factory=list)
    success: bool = True
    duration: float = 0.0
    trace_length: int = 0
    failure_signals: list[str] = field(default_factory=list)
    feedback_quality: str = "weak"

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "parse_execution_trace",
            "input_text": self.input_text,
            "output_text": self.output_text,
            "success": self.success,
            "tool_sequence": self.tool_sequence,
            "used_bullet_ids": self.used_bullet_ids,
            "duration": self.duration,
            "trace_length": self.trace_length,
            "failure_signals": self.failure_signals,
            "feedback_quality": self.feedback_quality,
        }


@dataclass
class BulletTag:
    id: str
    tag: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "tag": self.tag}


@dataclass
class ReflectionResult:
    reasoning: str = ""
    error_identification: str = ""
    root_cause_analysis: str = ""
    correct_approach: str = ""
    key_insight: str = ""
    key_insight_section: str = ""
    bullet_tags: list[BulletTag] = field(default_factory=list)
    is_success: bool = True

    def to_dict(self) -> dict[str, Any]:
        helpful = [item.id for item in self.bullet_tags if item.tag == "helpful"]
        harmful = [item.id for item in self.bullet_tags if item.tag == "harmful"]
        neutral = [item.id for item in self.bullet_tags if item.tag == "neutral"]
        return {
            "operation": "analyze",
            "analysis": self.reasoning,
            "error_identification": self.error_identification,
            "root_cause_analysis": self.root_cause_analysis,
            "correct_approach": self.correct_approach,
            "key_insight": self.key_insight,
            "key_insight_section": self.key_insight_section or (
                normalize_section(None, self.key_insight) if self.key_insight else ""
            ),
            "is_success": self.is_success,
            "helpful_bullet_ids": helpful,
            "harmful_bullet_ids": harmful,
            "neutral_bullet_ids": neutral,
            "bullet_tags": [item.to_dict() for item in self.bullet_tags],
        }


@dataclass
class CuratorOperation:
    type: str
    section: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "section": self.section,
            "content": self.content,
        }


@dataclass
class CuratorDelta:
    reasoning: str = ""
    operations: list[CuratorOperation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "curator_delta",
            "reasoning": self.reasoning,
            "operations": [item.to_dict() for item in self.operations],
        }
