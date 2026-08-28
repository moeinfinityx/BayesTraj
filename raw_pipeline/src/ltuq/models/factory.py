from __future__ import annotations

from .base import ChatModelConfig, ChatModelInterface
from .openai import build_openai_chat_model
from .tool_emulation import ToolCallEmulationChatModelClient


def create_chat_model(config: ChatModelConfig) -> ChatModelInterface:
    if config.provider in {"openai", "openai-compatible", "azure-openai", "ollama", "vllm"}:
        client = build_openai_chat_model(config)
        if config.emulate_tool_calls:
            return ToolCallEmulationChatModelClient(client)
        return client
    raise ValueError(f"Unsupported model provider: {config.provider}")