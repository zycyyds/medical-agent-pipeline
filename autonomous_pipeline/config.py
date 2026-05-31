from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "program" / "output"

STEP_AGENT_KEYS = {
    "step1": "agent_1",
    "step2": "agent_2_3",
    "step3": "agent_4",
    "step4": "agent_5",
    "step5": "agent_6_7",
    "step6": "agent_6_7",
    "task_analysis": "task_analysis",
    "planner": "planner",
    "evaluator": "evaluator",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_model_config() -> dict[str, Any]:
    root_config = _load_yaml(PROJECT_ROOT / "config.yaml")
    if root_config:
        return root_config

    # Backward-compatible fallback for older checkouts. New workspaces should use
    # the single root-level config.yaml so API keys and model settings are in one place.
    config_dir = PROJECT_ROOT / "configs"
    base = _load_yaml(config_dir / "model_config.yaml")
    local = _load_yaml(config_dir / "model_config.local.yaml")
    return _deep_merge(base, local)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    seed: int | None = 666
    timeout: int = 120
    embedding: dict[str, Any] | None = None

    @property
    def has_credentials(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY") or self.api_key)


def get_agent_config(step_or_agent_key: str) -> AgentConfig:
    cfg = load_model_config()
    agents = cfg.get("agents", {}) or {}
    global_cfg = cfg.get("global", {}) or {}
    agent_key = STEP_AGENT_KEYS.get(step_or_agent_key, step_or_agent_key)
    raw = {**global_cfg, **(agents.get(agent_key, {}) or {})}
    return AgentConfig(
        api_key=str(os.environ.get("OPENAI_API_KEY") or raw.get("api_key") or ""),
        base_url=str(os.environ.get("OPENAI_API_BASE") or raw.get("base_url") or raw.get("api_base") or "https://api.openai.com/v1"),
        model=str(os.environ.get("MODEL_NAME") or raw.get("model") or raw.get("model_name") or "gpt-4.1-mini"),
        temperature=float(raw.get("temperature", raw.get("default_temperature", 0.0)) or 0.0),
        seed=int(raw.get("seed", raw.get("default_seed", 666))) if raw.get("seed", raw.get("default_seed", 666)) is not None else None,
        timeout=int(raw.get("timeout", 120) or 120),
        embedding=raw.get("embedding") if isinstance(raw.get("embedding"), dict) else None,
    )
