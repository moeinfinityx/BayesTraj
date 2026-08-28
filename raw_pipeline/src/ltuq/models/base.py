from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import os
from typing import Any, Literal


ModelProvider = Literal["openai", "openai-compatible", "azure-openai", "ollama", "vllm"]


class ModelGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        provider: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.provider = provider
        self.retryable = retryable
        self.details = dict(details) if details is not None else {}

    def to_record_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "error_code": self.error_code,
            "retryable": self.retryable,
        }
        if self.provider is not None:
            metadata["provider"] = self.provider
        if self.details:
            metadata["details"] = dict(self.details)
        return metadata


class PromptFilteredError(ModelGenerationError):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="content_filter",
            provider=provider,
            retryable=False,
            details=details,
        )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class ChatModelConfig:
    provider: ModelProvider = "openai"
    model: str = "gpt-5.4"
    api_key: str | None = None
    emulate_tool_calls: bool = False
    max_tokens: int = 8192
    parallel_requests: int = 8
    seed: int | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None

    def __post_init__(self) -> None:
        resolved_api_key = self.api_key
        resolved_base_url = self.base_url
        resolved_azure_endpoint = self.azure_endpoint
        resolved_api_version = self.api_version
        resolved_deployment_name = self.deployment_name

        if resolved_api_key is None:
            if self.provider == "azure-openai":
                resolved_api_key = _first_env("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
            elif self.provider == "ollama":
                resolved_api_key = _first_env("OLLAMA_API_KEY", "OPENAI_API_KEY") or "ollama"
            elif self.provider == "vllm":
                resolved_api_key = _first_env("VLLM_API_KEY", "OPENAI_API_KEY") or "vllm"
            else:
                resolved_api_key = _first_env("OPENAI_API_KEY")

        if resolved_base_url is None:
            if self.provider in {"openai", "openai-compatible"}:
                resolved_base_url = _first_env("OPENAI_BASE_URL")
            elif self.provider == "ollama":
                resolved_base_url = _first_env("OLLAMA_BASE_URL", "OPENAI_BASE_URL")
            elif self.provider == "vllm":
                resolved_base_url = _first_env("VLLM_BASE_URL", "OPENAI_BASE_URL")

        if self.provider == "azure-openai":
            if resolved_azure_endpoint is None:
                resolved_azure_endpoint = _first_env("AZURE_OPENAI_ENDPOINT")
            if resolved_api_version is None:
                resolved_api_version = _first_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION")
            if resolved_deployment_name is None:
                resolved_deployment_name = _first_env(
                    "AZURE_OPENAI_DEPLOYMENT",
                    "AZURE_OPENAI_DEPLOYMENT_NAME",
                )

        object.__setattr__(self, "api_key", resolved_api_key)
        object.__setattr__(self, "base_url", resolved_base_url)
        object.__setattr__(self, "azure_endpoint", resolved_azure_endpoint)
        object.__setattr__(self, "api_version", resolved_api_version)
        object.__setattr__(self, "deployment_name", resolved_deployment_name)

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.parallel_requests <= 0:
            raise ValueError("parallel_requests must be positive")
        if self.seed is not None and isinstance(self.seed, bool):
            raise ValueError("seed must be an integer when provided")
        if self.provider in {"openai-compatible", "ollama", "vllm"} and not self.base_url:
            raise ValueError(f"{self.provider} provider requires base_url")
        if self.provider == "azure-openai":
            if not self.azure_endpoint:
                raise ValueError("azure-openai provider requires azure_endpoint")
            if not self.api_version:
                raise ValueError("azure-openai provider requires api_version")


class ChatModelInterface(ABC):
    model: str

    @abstractmethod
    async def ainference(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def sample_many(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        raise NotImplementedError

    async def acompletion_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def acompletion_with_tools_many(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        n: int = 1,
        tool_choice: Any | None = None,
    ) -> list[dict[str, Any]]:
        if n <= 0:
            raise ValueError("n must be positive")
        return [
            await self.acompletion_with_tools(messages, tools, temperature=temperature, tool_choice=tool_choice)
            for _ in range(n)
        ]

    async def acompletion_with_tools_detailed(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def acompletion_with_tools_many_detailed(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        n: int = 1,
        tool_choice: Any | None = None,
    ) -> list[dict[str, Any]]:
        if n <= 0:
            raise ValueError("n must be positive")
        return [
            await self.acompletion_with_tools_detailed(
                messages,
                tools,
                temperature=temperature,
                tool_choice=tool_choice,
            )
            for _ in range(n)
        ]

    async def aclose(self) -> None:
        return None


async def close_chat_model(client: Any) -> None:
    if client is None:
        return
    close_callable = getattr(client, "aclose", None)
    if not callable(close_callable):
        return
    result = close_callable()
    if inspect.isawaitable(result):
        await result
