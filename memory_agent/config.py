"""
配置管理模块
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


@dataclass
class Config:
    """Memory Agent 配置"""

    # 基础路径
    BASE_DIR: Path = Path(__file__).parent

    # LLM 配置 (从全局配置加载)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "qwen3:8b"
    LLM_TEMPERATURE: float = 0.0
    LLM_SEED: int = 666

    # 记忆配置
    MILESTONE_THRESHOLDS: list[int] = None  # 触发深度反思的计数器阈值
    REME_ENABLED: bool = True
    REME_BACKEND: str = "real"
    REME_LIGHT_ROOT: str = "memory_agent/reme_memory/playbook_light"
    REME_VECTOR_ROOT: str = "memory_agent/reme_memory/vector"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # 指纹配置
    FINGERPRINT_ALGORITHM: str = "sha256"

    def __post_init__(self):
        if self.MILESTONE_THRESHOLDS is None:
            self.MILESTONE_THRESHOLDS = [5, 10, 50, 100]
            
        # 尝试加载全局配置
        try:
            import sys
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            _project_root = os.path.dirname(_this_dir)
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)
            from configs.loader import get_agent_config
            
            agent_cfg = get_agent_config("memory_agent")
            if agent_cfg:
                self.LLM_API_KEY = agent_cfg.get("api_key", self.LLM_API_KEY)
                self.LLM_BASE_URL = agent_cfg.get("base_url") or agent_cfg.get("api_base") or self.LLM_BASE_URL
                self.LLM_MODEL = agent_cfg.get("model_name", self.LLM_MODEL)
                self.LLM_TEMPERATURE = agent_cfg.get("temperature", self.LLM_TEMPERATURE)
                self.LLM_SEED = agent_cfg.get("seed", self.LLM_SEED)
                self.REME_ENABLED = _as_bool(agent_cfg.get("reme_enabled", self.REME_ENABLED))
                self.REME_BACKEND = str(agent_cfg.get("reme_backend", self.REME_BACKEND) or self.REME_BACKEND)
                self.REME_LIGHT_ROOT = str(agent_cfg.get("reme_light_root", self.REME_LIGHT_ROOT) or self.REME_LIGHT_ROOT)
                self.REME_VECTOR_ROOT = str(agent_cfg.get("reme_vector_root", self.REME_VECTOR_ROOT) or self.REME_VECTOR_ROOT)

                embedding_cfg = agent_cfg.get("embedding") if isinstance(agent_cfg.get("embedding"), dict) else {}
                if not embedding_cfg:
                    # Backward-compatible convenience: older configs only had embedding under agent_2_3.
                    embedding_cfg = get_agent_config("agent_2_3").get("embedding") or {}
                if isinstance(embedding_cfg, dict):
                    self.EMBEDDING_API_KEY = embedding_cfg.get("api_key", self.EMBEDDING_API_KEY)
                    self.EMBEDDING_BASE_URL = (
                        embedding_cfg.get("base_url")
                        or embedding_cfg.get("api_base")
                        or self.EMBEDDING_BASE_URL
                    )
                    self.EMBEDDING_MODEL = (
                        embedding_cfg.get("model_name")
                        or embedding_cfg.get("model")
                        or self.EMBEDDING_MODEL
                    )
                    self.EMBEDDING_DIMENSIONS = int(
                        embedding_cfg.get("dimensions", self.EMBEDDING_DIMENSIONS) or self.EMBEDDING_DIMENSIONS
                    )
        except Exception as e:
            print(f"提示: 无法加载全局配置 ({e})，使用默认配置。")

    def get_llm_model(self) -> str:
        return os.environ.get("MODEL_NAME") or self.LLM_MODEL

    @property
    def project_root(self) -> Path:
        return self.BASE_DIR.parent

    def resolve_project_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

# 全局配置实例
config = Config()
