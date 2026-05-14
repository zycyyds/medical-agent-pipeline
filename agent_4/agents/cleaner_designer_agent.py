# agents/cleaner_designer_agent.py
import os
import sys
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.formatter import OllamaChatFormatter, OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OllamaChatModel, OpenAIChatModel


PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import get_agent_config


AGENT_CFG = get_agent_config("step5_data_quality_repair")
OLLAMA_MODEL_NAME = "qwen3.5:4b"
OLLAMA_TEMPERATURE = 0.0
OLLAMA_SEED = 666
OPENAI_MODEL_NAME = "gpt-4.1-mini"
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_TIMEOUT = 120
OPENAI_TEMPERATURE = 0.0


def _get_backend() -> str:
    backend = str(
        os.environ.get("STEP5_MODEL_BACKEND")
        or os.environ.get("STEP4_MODEL_BACKEND")
        or AGENT_CFG.get("backend")
        or "openai"
    ).strip().lower()
    return backend if backend in {"ollama", "openai"} else "ollama"


def _get_ollama_cfg() -> dict:
    cfg = AGENT_CFG.get("ollama") or {}
    return {
        "model_name": os.environ.get("STEP5_OLLAMA_MODEL")
        or os.environ.get("STEP4_OLLAMA_MODEL")
        or os.environ.get("OLLAMA_MODEL_NAME")
        or cfg.get("model_name")
        or AGENT_CFG.get("model_name", OLLAMA_MODEL_NAME),
        "temperature": float(
            os.environ.get("STEP5_OLLAMA_TEMPERATURE")
            or os.environ.get("STEP4_OLLAMA_TEMPERATURE")
            or os.environ.get("OLLAMA_TEMPERATURE")
            or cfg.get("temperature")
            or AGENT_CFG.get("temperature", OLLAMA_TEMPERATURE)
        ),
        "seed": int(
            os.environ.get("STEP5_OLLAMA_SEED")
            or os.environ.get("STEP4_OLLAMA_SEED")
            or os.environ.get("OLLAMA_SEED")
            or cfg.get("seed")
            or AGENT_CFG.get("seed", OLLAMA_SEED)
        ),
    }


def _get_openai_cfg() -> dict:
    cfg = AGENT_CFG.get("openai") or {}
    return {
        "model_name": os.environ.get("STEP5_OPENAI_MODEL")
        or os.environ.get("STEP4_OPENAI_MODEL")
        or cfg.get("model")
        or cfg.get("model_name")
        or AGENT_CFG.get("model")
        or AGENT_CFG.get("model_name")
        or OPENAI_MODEL_NAME,
        "api_key": os.environ.get("STEP5_OPENAI_API_KEY")
        or os.environ.get("STEP4_OPENAI_API_KEY")
        or cfg.get("api_key")
        or AGENT_CFG.get("api_key")
        or os.environ.get("OPENAI_API_KEY", ""),
        "api_base": os.environ.get("STEP5_OPENAI_API_BASE")
        or os.environ.get("STEP4_OPENAI_API_BASE")
        or cfg.get("base_url")
        or cfg.get("api_base")
        or AGENT_CFG.get("base_url")
        or AGENT_CFG.get("api_base")
        or os.environ.get("OPENAI_API_BASE")
        or OPENAI_API_BASE,
        "timeout": int(
            os.environ.get("STEP5_OPENAI_TIMEOUT")
            or os.environ.get("STEP4_OPENAI_TIMEOUT")
            or AGENT_CFG.get("timeout")
            or cfg.get("timeout")
            or os.environ.get("OPENAI_TIMEOUT")
            or OPENAI_TIMEOUT
        ),
        "temperature": float(
            os.environ.get("STEP5_OPENAI_TEMPERATURE")
            or os.environ.get("STEP4_OPENAI_TEMPERATURE")
            or cfg.get("temperature")
            or OPENAI_TEMPERATURE
        ),
    }


def _get_model_and_formatter():
    backend = _get_backend()
    if backend == "openai":
        cfg = _get_openai_cfg()
        if not cfg["api_key"]:
            raise RuntimeError("Step5 配置为 OpenAI 后端，但未设置 STEP5_OPENAI_API_KEY 或 OPENAI_API_KEY。")
        model = OpenAIChatModel(
            model_name=cfg["model_name"],
            api_key=cfg["api_key"],
            stream=True,
            client_kwargs={
                "base_url": cfg["api_base"],
                "timeout": cfg["timeout"],
            },
            generate_kwargs={
                "temperature": cfg["temperature"],
            },
        )
        return model, OpenAIChatFormatter()

    cfg = _get_ollama_cfg()
    model = OllamaChatModel(
        model_name=cfg["model_name"],
        options={
            "temperature": cfg["temperature"],
            "seed": cfg["seed"],
        },
    )
    return model, OllamaChatFormatter()


def _strip_thinking_from_msg(msg: Msg) -> Msg:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return msg
    content_blocks = [
        block for block in msg.get_content_blocks() if str(block.get("type") or "") != "thinking"
    ]
    sanitized_msg = Msg(
        name=msg.name,
        content=content_blocks,
        role=msg.role,
        metadata=msg.metadata,
        timestamp=msg.timestamp,
        invocation_id=msg.invocation_id,
    )
    sanitized_msg.id = msg.id
    return sanitized_msg


def _register_no_thinking_print_hook(agent: ReActAgent) -> ReActAgent:
    def _hook(_agent: ReActAgent, kwargs: dict):
        msg = kwargs.get("msg")
        if isinstance(msg, Msg):
            kwargs["msg"] = _strip_thinking_from_msg(msg)
        return kwargs

    agent.register_instance_hook("pre_print", "strip_thinking_for_console", _hook)
    return agent


def create_cleaner_designer_agent() -> ReActAgent:
    """
    创建一个通用 CSV 数据清洗设计智能体。
    - sys_prompt 为空，行为完全由 user prompt 控制
    - memory=None，避免重试或长链路时上下文累积导致性能下降
    - 可安全复用两套提示词：分析 + cleaner 生成
    """
    model, formatter = _get_model_and_formatter()
    agent = ReActAgent(
        name="MedicalDataCleanerExpert",
        sys_prompt="",
        model=model,
        formatter=formatter,
        memory=None,
    )

    return _register_no_thinking_print_hook(agent)
