from __future__ import annotations

import inspect
import json
from typing import Any

from ..experiments.tracking import ExperimentTracker
from .base import ChatModelInterface, close_chat_model


class RecordedChatModelClient(ChatModelInterface):
    def __init__(
        self,
        client: Any,
        tracker: ExperimentTracker,
        *,
        provider: str | None = None,
    ) -> None:
        self.wrapped_client = client
        self.tracker = tracker
        self.provider = provider
        self.model = str(getattr(client, "model", tracker.experiment.model))

    async def ainference(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
    ) -> str:
        generation_callable = getattr(self.wrapped_client, "ainference", None)
        if generation_callable is None:
            generation_callable = getattr(self.wrapped_client, "inference")

        result = generation_callable(history, temperature=temperature)
        if inspect.isawaitable(result):
            result = await result
        output = self._coerce_output_text(result)
        self.tracker.record_llm_call(
            messages=history,
            responses=[output],
            metadata={
                "call_type": "ainference",
                "model": self.model,
                "provider": self.provider,
                "temperature": temperature,
                "n": 1,
            },
        )
        return output

    async def sample_many(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        outputs: list[str]
        sample_many_callable = getattr(self.wrapped_client, "sample_many", None)
        if sample_many_callable is None:
            if n <= 0:
                raise ValueError("n must be positive")
            generation_callable = getattr(self.wrapped_client, "ainference", None)
            if generation_callable is None:
                generation_callable = getattr(self.wrapped_client, "inference")
            outputs = []
            for _ in range(n):
                result = generation_callable(history, temperature=temperature)
                if inspect.isawaitable(result):
                    result = await result
                outputs.append(self._coerce_output_text(result))
            call_type = "sample_many_fallback"
        else:
            result = sample_many_callable(history, temperature=temperature, n=n)
            if inspect.isawaitable(result):
                result = await result
            outputs = [self._coerce_output_text(output) for output in result]
            call_type = "sample_many"

        self.tracker.record_llm_call(
            messages=history,
            responses=outputs,
            metadata={
                "call_type": call_type,
                "model": self.model,
                "provider": self.provider,
                "temperature": temperature,
                "n": n,
            },
        )
        return outputs

    async def sample_many_detailed(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
        n: int = 1,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        detailed_callable = getattr(self.wrapped_client, "sample_many_detailed", None)
        if detailed_callable is None:
            outputs = await self.sample_many(history, temperature=temperature, n=n)
            return [{"output": output, "metadata": {}} for output in outputs]

        try:
            result = detailed_callable(
                history,
                temperature=temperature,
                n=n,
                extra_body=extra_body,
                max_tokens=max_tokens,
            )
        except TypeError:
            if extra_body is not None or max_tokens is not None:
                raise
            result = detailed_callable(history, temperature=temperature, n=n)
        if inspect.isawaitable(result):
            result = await result

        detailed_outputs: list[dict[str, Any]] = []
        responses: list[str] = []
        for item in result:
            if isinstance(item, dict):
                output = self._coerce_output_text(item)
                metadata = item.get("metadata")
                detailed_outputs.append(
                    {
                        "output": output,
                        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                    }
                )
                responses.append(output)
                continue

            output = self._coerce_output_text(item)
            detailed_outputs.append({"output": output, "metadata": {}})
            responses.append(output)

        self.tracker.record_llm_call(
            messages=history,
            responses=responses,
            metadata={
                "call_type": "sample_many_detailed",
                "model": self.model,
                "provider": self.provider,
                "temperature": temperature,
                "n": n,
                "extra_body": extra_body,
                "max_tokens": max_tokens,
            },
        )
        return detailed_outputs

    async def acompletion_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        tool_callable = getattr(self.wrapped_client, "acompletion_with_tools", None)
        if tool_callable is None:
            raise NotImplementedError("Wrapped model client does not support tool-calling completions.")

        result = tool_callable(messages, tools, temperature=temperature, tool_choice=tool_choice)
        if inspect.isawaitable(result):
            result = await result

        response_payload = dict(result) if isinstance(result, dict) else {"output": self._coerce_output_text(result)}
        self.tracker.record_llm_call(
            messages=messages,
            responses=[json.dumps(response_payload, ensure_ascii=True, sort_keys=True)],
            metadata={
                "call_type": "acompletion_with_tools",
                "model": self.model,
                "provider": self.provider,
                "temperature": temperature,
                "tool_count": len(tools),
                "tool_choice": tool_choice,
            },
        )
        return response_payload

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

        tool_many_callable = getattr(self.wrapped_client, "acompletion_with_tools_many", None)
        if tool_many_callable is None:
            return [
                await self.acompletion_with_tools(messages, tools, temperature=temperature, tool_choice=tool_choice)
                for _ in range(n)
            ]

        result = tool_many_callable(messages, tools, temperature=temperature, n=n, tool_choice=tool_choice)
        if inspect.isawaitable(result):
            result = await result

        payloads = [
            dict(item) if isinstance(item, dict) else {"output": self._coerce_output_text(item)}
            for item in result
        ]
        self.tracker.record_llm_call(
            messages=messages,
            responses=[json.dumps(payload, ensure_ascii=True, sort_keys=True) for payload in payloads],
            metadata={
                "call_type": "acompletion_with_tools_many",
                "model": self.model,
                "provider": self.provider,
                "temperature": temperature,
                "tool_count": len(tools),
                "n": n,
                "tool_choice": tool_choice,
            },
        )
        return payloads

    async def acompletion_with_tools_detailed(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        detailed_callable = getattr(self.wrapped_client, "acompletion_with_tools_detailed", None)
        if detailed_callable is None:
            payload = await self.acompletion_with_tools(messages, tools, temperature=temperature, tool_choice=tool_choice)
            payload["metadata"] = {"logprobs_unavailable": True}
            return payload

        result = detailed_callable(messages, tools, temperature=temperature, tool_choice=tool_choice)
        if inspect.isawaitable(result):
            result = await result

        response_payload: dict[str, Any] = dict(result) if isinstance(result, dict) else {"output": self._coerce_output_text(result)}
        self.tracker.record_llm_call(
            messages=messages,
            responses=[json.dumps(response_payload, ensure_ascii=True, sort_keys=True)],
            metadata={
                "call_type": "acompletion_with_tools_detailed",
                "model": self.model,
                "provider": self.provider,
                "temperature": temperature,
                "tool_count": len(tools),
                "tool_choice": tool_choice,
            },
        )
        return response_payload

    async def aclose(self) -> None:
        await close_chat_model(self.wrapped_client)

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

        detailed_many_callable = getattr(self.wrapped_client, "acompletion_with_tools_many_detailed", None)
        if detailed_many_callable is None:
            return [
                await self.acompletion_with_tools_detailed(
                    messages,
                    tools,
                    temperature=temperature,
                    tool_choice=tool_choice,
                )
                for _ in range(n)
            ]

        result = detailed_many_callable(messages, tools, temperature=temperature, n=n, tool_choice=tool_choice)
        if inspect.isawaitable(result):
            result = await result

        payloads = [
            dict(item) if isinstance(item, dict) else {"output": self._coerce_output_text(item)}
            for item in result
        ]
        self.tracker.record_llm_call(
            messages=messages,
            responses=[json.dumps(payload, ensure_ascii=True, sort_keys=True) for payload in payloads],
            metadata={
                "call_type": "acompletion_with_tools_many_detailed",
                "model": self.model,
                "provider": self.provider,
                "temperature": temperature,
                "tool_count": len(tools),
                "n": n,
                "tool_choice": tool_choice,
            },
        )
        return payloads

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped_client, name)

    def _coerce_output_text(self, output: Any) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            for key in ("content", "text", "response", "output"):
                value = output.get(key)
                if isinstance(value, str):
                    return value.strip()
        return str(output).strip()
