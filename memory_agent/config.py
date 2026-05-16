"""
配置管理模块
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Memory Agent 配置"""

    # 基础路径
    BASE_DIR: Path = Path(__file__).parent
    MEMORY_BANK_PATH: Path = BASE_DIR / "memory_bank.json"

    # LLM 配置 (从全局配置加载)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "qwen3:8b"
    LLM_TEMPERATURE: float = 0.0
    LLM_SEED: int = 666
    LLM_ENABLE_THINKING: bool | None = False

    # 记忆配置
    MILESTONE_THRESHOLDS: list[int] = None  # 触发深度反思的计数器阈值

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
                self.LLM_ENABLE_THINKING = agent_cfg.get(
                    "enable_thinking",
                    self.LLM_ENABLE_THINKING,
                )
        except Exception as e:
            print(f"提示: 无法加载全局配置 ({e})，使用默认配置。")

    def get_memory_bank_path(self) -> str:
        """获取记忆库文件路径"""
        return str(self.MEMORY_BANK_PATH)

    def ensure_dirs(self):
        """确保所有必要目录存在"""
        self.MEMORY_BANK_PATH.parent.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = Config()
