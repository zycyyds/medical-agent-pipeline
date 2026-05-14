import os
import yaml
from pathlib import Path
from copy import deepcopy

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent

def load_model_config():
    """读取并解析模型配置文件"""
    config_path = get_project_root() / "configs" / "model_config.yaml"
    
    if not config_path.exists():
        # 如果配置文件不存在，返回一个空的结构或者报错
        print(f"警告: 配置文件 {config_path} 不存在，将使用代码中的默认值。")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"错误: 解析配置文件失败: {e}")
            return {}

# 导出配置
model_config = load_model_config()


AGENT_KEY_ALIASES = {
    "dataset_codegen_agent": "agent_1",
    "agent_1": "dataset_codegen_agent",
    "agent_2-3": "agent_2_3",
    "agent2_3": "agent_2_3",
    "step2_3_medical_data_cleaner": "agent_2_3",
    "step4_task_oriented_clipping": "agent_4",
    "step4_data_quality_repair": "agent_5",
    "step5_data_quality_repair": "agent_5",
    "step5_task_oriented_clipping": "agent_4",
    "step6_7_pipeline": "agent_6_7",
    "agent_6-7": "agent_6_7",
    "agent_6_7": "step6_7_pipeline",
}


def _resolve_agent_key(agent_key: str, agents_cfg: dict) -> str:
    if agent_key in agents_cfg:
        return agent_key
    alias = AGENT_KEY_ALIASES.get(agent_key)
    if alias and alias in agents_cfg:
        return alias
    return agent_key


def _with_legacy_aliases(cfg: dict) -> dict:
    normalized = deepcopy(cfg)
    if "model" in normalized and "model_name" not in normalized:
        normalized["model_name"] = normalized["model"]
    if "model_name" in normalized and "model" not in normalized:
        normalized["model"] = normalized["model_name"]
    if "base_url" in normalized and "api_base" not in normalized:
        normalized["api_base"] = normalized["base_url"]
    if "api_base" in normalized and "base_url" not in normalized:
        normalized["base_url"] = normalized["api_base"]

    embedding = normalized.get("embedding")
    if isinstance(embedding, dict):
        if "model" in embedding and "model_name" not in embedding:
            embedding["model_name"] = embedding["model"]
        if "model_name" in embedding and "model" not in embedding:
            embedding["model"] = embedding["model_name"]
        if "base_url" in embedding and "api_base" not in embedding:
            embedding["api_base"] = embedding["base_url"]
        if "api_base" in embedding and "base_url" not in embedding:
            embedding["base_url"] = embedding["api_base"]
    return normalized


def get_agent_config(agent_key: str):
    """根据智能体 key 获取其特定配置"""
    agents_cfg = model_config.get("agents", {})
    global_cfg = {
        key: value
        for key, value in (model_config.get("global", {}) or {}).items()
        if key in {"api_key", "base_url", "api_base", "model", "model_name", "timeout", "temperature", "seed"}
    }
    resolved_key = _resolve_agent_key(agent_key, agents_cfg)
    agent_cfg = agents_cfg.get(resolved_key, {})
    merged = {**global_cfg, **agent_cfg}
    return _with_legacy_aliases(merged)
