from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["SUCCESS", "NEEDS_REPAIR", "FAILED"]


@dataclass(slots=True)
class RunContext:
    input_path: str
    output_root: str
    task_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def step_output_dir(self, step_name: str) -> Path:
        return Path(self.output_root) / f"{step_name}_results"


@dataclass(slots=True)
class ToolResult:
    status: Status
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    next_recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "artifacts": self.artifacts,
            "issues": self.issues,
            "next_recommendation": self.next_recommendation,
        }


def ok(summary: str, artifacts: dict[str, Any] | None = None, recommendation: str = "") -> dict[str, Any]:
    return ToolResult("SUCCESS", summary, artifacts or {}, [], recommendation).to_dict()


def repair(summary: str, issues: list[str], artifacts: dict[str, Any] | None = None, recommendation: str = "") -> dict[str, Any]:
    return ToolResult("NEEDS_REPAIR", summary, artifacts or {}, issues, recommendation).to_dict()


def failed(summary: str, issues: list[str], artifacts: dict[str, Any] | None = None, recommendation: str = "") -> dict[str, Any]:
    return ToolResult("FAILED", summary, artifacts or {}, issues, recommendation).to_dict()
