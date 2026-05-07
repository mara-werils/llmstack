from llmstack.config.presets.chat import CHAT_PRESET
from llmstack.config.presets.rag import RAG_PRESET
from llmstack.config.presets.agent import AGENT_PRESET

PRESETS = {
    "chat": CHAT_PRESET,
    "rag": RAG_PRESET,
    "agent": AGENT_PRESET,
}

__all__ = ["PRESETS", "CHAT_PRESET", "RAG_PRESET", "AGENT_PRESET"]
