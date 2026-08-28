from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncAzureOpenAI, AsyncOpenAI, BadRequestError

from .base import ChatModelConfig, ChatModelInterface, ModelGenerationError, PromptFilteredError


LOGGER = logging.getLogger(__name__)

_LOGPROB_FLOOR_THRESHOLD = -1000.0
_THINK_START_MARKER = "<think"
_THINK_END_MARKER = "</think>"
_TOOL_DECISION_MARKERS = ("<tool_call", "<|tool_call", "<function=")


class _BaseOpenAIChatModelClient(ChatModelInterface):
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 8192,
        parallel_requests: int = 8,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.parallel_requests = parallel_requests
        self.seed = seed
        self._seed_counter = 0
        self.max_choices_per_request: int | None = None
        self.provider: str | None = None
        self._text_logprobs_supported: bool | None = None
        self._tool_logprobs_supported: bool | None = None
        self._request_semaphore: asyncio.Semaphore | None = None

    async def ainference(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
    ) -> str:
        outputs = await self.sample_many(history, temperature=temperature, n=1)
        return outputs[0]

    async def sample_many(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        if n <= 0:
            raise ValueError("n must be positive")

        if self._use_raw_completions_for_text_sampling():
            detailed = await self._sample_many_raw_completions_detailed(
                history,
                temperature=temperature,
                n=n,
            )
            return [str(item.get("output", "")).strip() for item in detailed]

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": history,
            "temperature": temperature,
        }
        return await self._collect_batched_outputs(
            total_count=n,
            request_kwargs=request_kwargs,
            completion_kind="chat",
            extract_batch=self._extract_chat_text_batch,
        )

    async def sample_many_detailed(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
        n: int = 1,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        if n <= 0:
            raise ValueError("n must be positive")

        if self._use_raw_completions_for_text_sampling() and not self._use_chat_for_structured_sampling(extra_body):
            return await self._sample_many_raw_completions_detailed(
                history,
                temperature=temperature,
                n=n,
                extra_body=extra_body,
                max_tokens=max_tokens,
            )

        if self._text_logprobs_supported is False and (extra_body is None and max_tokens is None):
            return [
                {"output": output, "metadata": {"logprobs_unavailable": True}}
                for output in await self.sample_many(history, temperature=temperature, n=n)
            ]

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": history,
            "temperature": temperature,
            "logprobs": True,
        }
        if extra_body is not None:
            request_kwargs["extra_body"] = dict(extra_body)

        try:
            outputs = await self._collect_batched_outputs(
                total_count=n,
                request_kwargs=request_kwargs,
                completion_kind="chat",
                extract_batch=self._extract_chat_detail_batch,
                token_budget=max_tokens,
            )
            self._text_logprobs_supported = True
        except Exception as exc:
            if not self._handle_logprobs_fallback(exc, kind="chat"):
                raise
            if extra_body is not None or max_tokens is not None:
                request_kwargs.pop("logprobs", None)
                outputs = await self._collect_batched_outputs(
                    total_count=n,
                    request_kwargs=request_kwargs,
                    completion_kind="chat",
                    extract_batch=self._extract_chat_detail_batch,
                    token_budget=max_tokens,
                )
                return [self._with_logprobs_unavailable_metadata(output) for output in outputs]
            return [
                {"output": output, "metadata": {"logprobs_unavailable": True}}
                for output in await self.sample_many(history, temperature=temperature, n=n)
            ]

        return outputs

    async def acompletion_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        return (
            await self.acompletion_with_tools_many(
                messages,
                tools,
                temperature=temperature,
                n=1,
                tool_choice=tool_choice,
            )
        )[0]

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

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice or "auto",
            "parallel_tool_calls": False,
            "temperature": temperature,
        }
        return await self._collect_batched_outputs(
            total_count=n,
            request_kwargs=request_kwargs,
            completion_kind="tool",
            extract_batch=self._extract_tool_payload_batch,
        )

    async def acompletion_with_tools_detailed(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        return (
            await self.acompletion_with_tools_many_detailed(
                messages,
                tools,
                temperature=temperature,
                n=1,
                tool_choice=tool_choice,
            )
        )[0]

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

        if self._tool_logprobs_supported is False:
            payloads = await self.acompletion_with_tools_many(
                messages,
                tools,
                temperature=temperature,
                n=n,
                tool_choice=tool_choice,
            )
            return [self._with_logprobs_unavailable_metadata(payload) for payload in payloads]

        try:
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice or "auto",
                "parallel_tool_calls": False,
                "temperature": temperature,
                "logprobs": True,
            }
            outputs = await self._collect_batched_outputs(
                total_count=n,
                request_kwargs=request_kwargs,
                completion_kind="tool",
                extract_batch=self._extract_tool_detail_batch,
            )
            if self._tool_logprobs_supported is not False:
                self._tool_logprobs_supported = True
            return outputs
        except Exception as exc:
            if not self._handle_logprobs_fallback(exc, kind="tool"):
                raise
            payloads = await self.acompletion_with_tools_many(
                messages,
                tools,
                temperature=temperature,
                n=n,
                tool_choice=tool_choice,
            )
            return [self._with_logprobs_unavailable_metadata(payload) for payload in payloads]

    @staticmethod
    def _with_logprobs_unavailable_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("metadata")
        merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        merged_metadata["logprobs_unavailable"] = True
        return {**payload, "metadata": merged_metadata}

    def _use_raw_completions_for_text_sampling(self) -> bool:
        """Use raw completions for GPT-OSS on vLLM to avoid Harmony chat parsing.

        vLLM's OpenAI chat endpoint parses GPT-OSS outputs through Harmony. For
        emulated tool calling we only need plain text JSON, so the raw
        completions endpoint is safer and avoids backend 500s when the model
        emits malformed Harmony headers.
        """
        provider = str(getattr(self, "provider", "") or "").lower()
        model_name = self.model.strip().lower()
        return provider == "vllm" and model_name.startswith(("gpt-oss", "gemma"))

    def _use_chat_for_structured_sampling(self, extra_body: Mapping[str, Any] | None) -> bool:
        """Keep GPT-OSS JSON-schema requests on vLLM's chat endpoint.

        GPT-OSS emits Harmony reasoning before its final answer.  vLLM's chat
        endpoint separates that reasoning from ``message.content`` and applies
        ``structured_outputs`` to the final content.  The raw completions
        endpoint exposes the reasoning stream as ordinary text, so the same
        schema request can return non-JSON prose and systematically exhaust
        structured-step retries.

        Plain GPT-OSS text sampling still uses raw completions, preserving the
        existing workaround for malformed Harmony headers in unconstrained
        emulated-tool requests.
        """
        provider = str(getattr(self, "provider", "") or "").lower()
        model_name = self.model.strip().lower()
        return (
            provider == "vllm"
            and model_name.startswith("gpt-oss")
            and isinstance(extra_body, Mapping)
            and isinstance(extra_body.get("structured_outputs"), Mapping)
        )

    def _render_raw_completion_prompt(self, history: Sequence[Mapping[str, Any]]) -> str:
        parts: list[str] = []
        for message in history:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "user")).strip() or "user"
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(str(getattr(part, "text", part)) for part in content)
            elif content is None:
                text = ""
            else:
                text = str(content)
            parts.append(f"{role.upper()}:\n{text.strip()}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    async def _sample_many_raw_completions_detailed(
        self,
        history: list[dict[str, str]],
        *,
        temperature: float,
        n: int,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self._render_raw_completion_prompt(history)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "logprobs": 1,
        }
        if extra_body is not None:
            request_kwargs["extra_body"] = dict(extra_body)

        outputs: list[dict[str, Any]] = []
        while len(outputs) < n:
            batch_sizes = self._batched_request_sizes(n - len(outputs))
            responses = await asyncio.gather(
                *[
                    self._request_raw_completion_batch(
                        request_kwargs=request_kwargs,
                        batch_size=batch_size,
                        token_budget=max_tokens or self.max_tokens,
                    )
                    for batch_size in batch_sizes
                ]
            )
            for response in responses:
                outputs.extend(self._extract_raw_completion_detail_batch(response))
        return outputs[:n]

    async def _request_raw_completion_batch(
        self,
        *,
        request_kwargs: dict[str, Any],
        batch_size: int,
        token_budget: int,
    ) -> Any:
        prepared_request_kwargs = self._prepare_request_kwargs(request_kwargs)
        if self.seed is not None and "seed" not in prepared_request_kwargs:
            prepared_request_kwargs["seed"] = self.seed + self._seed_counter
            self._seed_counter += 1
        async with self._get_request_semaphore():
            return await self._create_raw_completion(
                **prepared_request_kwargs,
                n=batch_size,
                max_tokens=max(1, int(token_budget)),
            )

    def _extract_raw_completion_detail_batch(self, response: Any) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for choice in getattr(response, "choices", []) or []:
            text = getattr(choice, "text", "")
            metadata = self._extract_choice_finish_metadata(choice)
            logprobs = getattr(choice, "logprobs", None)
            token_logprobs = getattr(logprobs, "token_logprobs", None)
            tokens = getattr(logprobs, "tokens", None)
            if isinstance(token_logprobs, list):
                clean_logprobs = [
                    float(value)
                    for value in token_logprobs
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                if clean_logprobs and not any(value <= _LOGPROB_FLOOR_THRESHOLD for value in clean_logprobs):
                    metadata["token_logprob_sum"] = float(sum(clean_logprobs))
                    metadata["token_count"] = len(clean_logprobs)
                    if isinstance(tokens, list):
                        token_items = [
                            (str(token), float(logprob))
                            for token, logprob in zip(tokens, token_logprobs)
                            if isinstance(logprob, (int, float)) and not isinstance(logprob, bool)
                        ]
                        split_logprobs = self._split_token_logprobs_by_output_role(token_items)
                        if split_logprobs is not None:
                            (
                                reasoning_logprob_sum,
                                reasoning_token_count,
                                decision_logprob_sum,
                                decision_token_count,
                            ) = split_logprobs
                            if reasoning_token_count > 0:
                                metadata["reasoning_token_logprob_sum"] = reasoning_logprob_sum
                                metadata["reasoning_token_count"] = reasoning_token_count
                            if decision_token_count > 0:
                                metadata["decision_token_logprob_sum"] = decision_logprob_sum
                                metadata["decision_token_count"] = decision_token_count
                elif clean_logprobs:
                    metadata["logprobs_unavailable"] = True
                    metadata["token_logprob_floor_detected"] = True
            outputs.append({"output": str(text).strip(), "metadata": metadata})
        return outputs

    def _get_request_semaphore(self) -> asyncio.Semaphore:
        if self._request_semaphore is None:
            self._request_semaphore = asyncio.Semaphore(self.parallel_requests)
        return self._request_semaphore

    def _batched_request_sizes(self, total_count: int) -> list[int]:
        remaining = total_count
        batch_sizes: list[int] = []
        while remaining > 0:
            batch_size = remaining
            if self.max_choices_per_request is not None:
                batch_size = min(batch_size, self.max_choices_per_request)
            batch_sizes.append(batch_size)
            remaining -= batch_size
        return batch_sizes

    async def _request_completion_batch(
        self,
        *,
        request_kwargs: dict[str, Any],
        batch_size: int,
        token_budget: int | None = None,
    ) -> Any:
        prepared_request_kwargs = dict(request_kwargs)
        if self.seed is not None and "seed" not in prepared_request_kwargs:
            prepared_request_kwargs["seed"] = self.seed + self._seed_counter
            self._seed_counter += 1
        async with self._get_request_semaphore():
            return await self._create_completion_with_token_fallback(
                **prepared_request_kwargs,
                n=batch_size,
                token_budget=token_budget or self.max_tokens,
            )

    async def _collect_batched_outputs(
        self,
        *,
        total_count: int,
        request_kwargs: dict[str, Any],
        completion_kind: str,
        extract_batch: Callable[[Any], list[Any]],
        token_budget: int | None = None,
    ) -> list[Any]:
        outputs: list[Any] = []
        while len(outputs) < total_count:
            batch_sizes = self._batched_request_sizes(total_count - len(outputs))
            responses = await asyncio.gather(
                *[
                    self._request_completion_batch(request_kwargs=request_kwargs, batch_size=batch_size)
                    if token_budget is None
                    else self._request_completion_batch(
                        request_kwargs=request_kwargs,
                        batch_size=batch_size,
                        token_budget=token_budget,
                    )
                    for batch_size in batch_sizes
                ]
            )
            for batch_size, response in zip(batch_sizes, responses):
                batch_outputs = extract_batch(response)
                self._validate_choice_batch(
                    requested_count=batch_size,
                    actual_count=len(batch_outputs),
                    completion_kind=completion_kind,
                )
                outputs.extend(batch_outputs)
        return outputs[:total_count]

    def _extract_chat_text_batch(self, response: Any) -> list[str]:
        batch_outputs = [self._extract_choice_text(choice) for choice in response.choices]
        self._log_truncated_choices(response, batch_outputs)
        return batch_outputs

    def _extract_chat_detail_batch(self, response: Any) -> list[dict[str, Any]]:
        batch_outputs = [self._extract_choice_detail(choice) for choice in response.choices]
        self._log_truncated_choices(response, [item.get("output", "") for item in batch_outputs])
        return batch_outputs

    def _extract_tool_payload_batch(self, response: Any) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for choice in response.choices:
            payload = self._extract_message_payload(choice.message)
            metadata = self._extract_choice_finish_metadata(choice)
            if metadata:
                payload["metadata"] = metadata
            outputs.append(payload)
        return outputs

    def _extract_tool_detail_batch(self, response: Any) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        saw_tool_logprobs = False
        saw_missing_tool_logprobs = False
        for choice in response.choices:
            payload = self._extract_message_payload(choice.message)
            metadata = self._extract_choice_metadata(choice)
            if "token_logprob_sum" in metadata:
                saw_tool_logprobs = True
            else:
                metadata["logprobs_unavailable"] = True
                if payload.get("tool_calls"):
                    saw_missing_tool_logprobs = True
            payload["metadata"] = metadata
            outputs.append(payload)

        if saw_missing_tool_logprobs and not saw_tool_logprobs:
            if self._tool_logprobs_supported is not False:
                LOGGER.info(
                    "Tool-call logprobs were omitted for model=%s despite requesting them; treating them as unavailable.",
                    self.model,
                )
            self._tool_logprobs_supported = False

        return outputs

    async def _create_completion_with_token_fallback(self, *, token_budget: int, **kwargs: Any) -> Any:
        prepared_kwargs = self._prepare_request_kwargs(kwargs)
        response = None
        last_error: TypeError | None = None
        token_budgets = self._token_budget_candidates(token_budget)
        for token_field in ("max_completion_tokens", "max_tokens"):
            for budget_index, current_budget in enumerate(token_budgets):
                try:
                    response = await self._create_completion_with_json_parse_retry(
                        request_kwargs=prepared_kwargs,
                        token_field=token_field,
                        token_budget=current_budget,
                    )
                    break
                except BadRequestError as exc:
                    if self._is_prompt_filtered_error(exc):
                        raise self._build_prompt_filtered_error(exc) from exc
                    if self._is_logprobs_unsupported_error(exc):
                        raise
                    if self._is_context_length_error(exc) and budget_index < len(token_budgets) - 1:
                        next_budget = token_budgets[budget_index + 1]
                        LOGGER.warning(
                            "OpenAI-compatible request exceeded context for model=%s with %s=%s; retrying with %s=%s.",
                            self.model,
                            token_field,
                            current_budget,
                            token_field,
                            next_budget,
                        )
                        continue
                    raise self._build_bad_request_generation_error(exc) from exc
                except TypeError as exc:
                    last_error = exc
                    break
            if response is not None:
                break

        if response is None:
            raise TypeError("OpenAI client does not accept configured token arguments") from last_error
        return response

    @staticmethod
    def _token_budget_candidates(token_budget: int) -> list[int]:
        budgets: list[int] = []
        current = max(1, int(token_budget))
        floor = 16
        while True:
            if current not in budgets:
                budgets.append(current)
            if current <= floor:
                break
            current = max(floor, current // 2)
        return budgets

    async def _create_completion_with_json_parse_retry(
        self,
        *,
        request_kwargs: dict[str, Any],
        token_field: str,
        token_budget: int,
    ) -> Any:
        prepared_request_kwargs = dict(request_kwargs)
        json_parse_retries = 0
        harmony_resample_retries = 0
        while True:
            try:
                return await self._create_completion(
                    **prepared_request_kwargs,
                    **{token_field: token_budget},
                )
            except BadRequestError as exc:
                if self._is_json_parse_error(exc):
                    if json_parse_retries == 0:
                        json_parse_retries += 1
                        LOGGER.warning(
                            "OpenAI returned a request JSON parse error for model=%s; retrying once.",
                            self.model,
                        )
                        continue
                    raise self._build_json_parse_generation_error(
                        exc,
                        request_kwargs=prepared_request_kwargs,
                        token_field=token_field,
                        token_budget=token_budget,
                    ) from exc
                raise
            except APIStatusError as exc:
                if not self._is_gptoss_harmony_parse_error(exc, prepared_request_kwargs):
                    raise
                if harmony_resample_retries >= 2:
                    raise
                harmony_resample_retries += 1
                previous_seed = prepared_request_kwargs.get("seed")
                if isinstance(previous_seed, int) and not isinstance(previous_seed, bool):
                    prepared_request_kwargs["seed"] = previous_seed + 1
                LOGGER.warning(
                    "vLLM failed to parse a structured GPT-OSS Harmony response for model=%s; "
                    "resampling with %s (attempt %s/2).",
                    self.model,
                    (
                        f"seed={prepared_request_kwargs['seed']}"
                        if "seed" in prepared_request_kwargs
                        else "a fresh unseeded draw"
                    ),
                    harmony_resample_retries,
                )

    def _is_gptoss_harmony_parse_error(
        self,
        exc: APIStatusError,
        request_kwargs: Mapping[str, Any],
    ) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code != 500:
            return False
        extra_body = request_kwargs.get("extra_body")
        if not self._use_chat_for_structured_sampling(extra_body if isinstance(extra_body, Mapping) else None):
            return False
        message = str(self._extract_error_payload(exc).get("message") or exc).lower()
        return "unexpected tokens remaining in message header" in message

    def _is_prompt_filtered_error(self, exc: BadRequestError) -> bool:
        error_payload = self._extract_error_payload(exc)
        error_code = error_payload.get("code")
        if error_code == "content_filter":
            return True
        inner_error = error_payload.get("innererror")
        if isinstance(inner_error, dict) and inner_error.get("code") == "ResponsibleAIPolicyViolation":
            return True
        message = error_payload.get("message")
        if isinstance(message, str) and "content management policy" in message.lower():
            return True
        return False

    def _build_prompt_filtered_error(self, exc: BadRequestError) -> PromptFilteredError:
        error_payload = self._extract_error_payload(exc)
        details: dict[str, Any] = {}

        inner_error = error_payload.get("innererror")
        if isinstance(inner_error, dict):
            content_filter_result = inner_error.get("content_filter_result")
            if isinstance(content_filter_result, dict):
                details["content_filter_result"] = content_filter_result
            if inner_error.get("code") is not None:
                details["innererror_code"] = inner_error.get("code")

        if error_payload.get("status") is not None:
            details["status"] = error_payload.get("status")
        if getattr(exc, "status_code", None) is not None:
            details["status_code"] = getattr(exc, "status_code")

        return PromptFilteredError(
            error_payload.get("message") or str(exc),
            provider=self.provider,
            details=details,
        )

    def _extract_error_payload(self, exc: BadRequestError) -> dict[str, Any]:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            if isinstance(body.get("code"), str) or isinstance(body.get("message"), str):
                return body
            error = body.get("error")
            if isinstance(error, dict):
                return error
        return {}

    def _is_json_parse_error(self, exc: BadRequestError) -> bool:
        error_payload = self._extract_error_payload(exc)
        message = error_payload.get("message")
        return isinstance(message, str) and "parse the json body" in message.lower()

    def _is_context_length_error(self, exc: BadRequestError) -> bool:
        error_payload = self._extract_error_payload(exc)
        message = error_payload.get("message")
        if not isinstance(message, str):
            return False
        lowered = message.lower()
        return "maximum context length" in lowered or "context length" in lowered

    def _build_json_parse_generation_error(
        self,
        exc: BadRequestError,
        *,
        request_kwargs: dict[str, Any],
        token_field: str,
        token_budget: int,
    ) -> ModelGenerationError:
        error_payload = self._extract_error_payload(exc)
        messages = request_kwargs.get("messages")
        tools = request_kwargs.get("tools")
        details = {
            "token_field": token_field,
            "token_budget": token_budget,
            "request_size_bytes": len(self._serialize_request_payload(request_kwargs)),
            "message_count": len(messages) if isinstance(messages, list) else None,
            "tool_count": len(tools) if isinstance(tools, list) else None,
        }
        return ModelGenerationError(
            error_payload.get("message") or str(exc),
            error_code="request_json_parse_error",
            provider=self.provider,
            retryable=False,
            details=details,
        )

    def _build_bad_request_generation_error(self, exc: BadRequestError) -> ModelGenerationError:
        error_payload = self._extract_error_payload(exc)
        details: dict[str, Any] = {}

        param = error_payload.get("param")
        if param is not None:
            details["param"] = param
        if error_payload.get("type") is not None:
            details["type"] = error_payload.get("type")
        if error_payload.get("status") is not None:
            details["status"] = error_payload.get("status")
        if getattr(exc, "status_code", None) is not None:
            details["status_code"] = getattr(exc, "status_code")

        inner_error = error_payload.get("innererror")
        if isinstance(inner_error, dict):
            details["innererror"] = inner_error

        error_code = error_payload.get("code")
        if not isinstance(error_code, str) or not error_code:
            error_code = "bad_request"

        return ModelGenerationError(
            error_payload.get("message") or str(exc),
            error_code=error_code,
            provider=self.provider,
            retryable=False,
            details=details,
        )

    def _prepare_request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            str(key): self._normalize_request_value(value)
            for key, value in kwargs.items()
        }
        try:
            self._serialize_request_payload(normalized)
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise ModelGenerationError(
                "OpenAI request payload could not be serialized to JSON.",
                error_code="invalid_request_payload",
                provider=self.provider,
                retryable=False,
                details={"serialization_error": str(exc)},
            ) from exc
        return normalized

    def _normalize_request_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._normalize_request_value(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_request_value(item) for item in value]
        if hasattr(value, "model_dump"):
            return self._normalize_request_value(value.model_dump())
        if hasattr(value, "to_dict"):
            return self._normalize_request_value(value.to_dict())
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _serialize_request_payload(self, payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def _is_logprobs_unsupported_error(self, exc: Exception) -> bool:
        if not isinstance(exc, BadRequestError):
            return False
        error_payload = self._extract_error_payload(exc)
        if error_payload.get("code") != "unsupported_parameter":
            return False
        param = error_payload.get("param")
        if isinstance(param, str) and param == "logprobs":
            return True
        message = error_payload.get("message")
        return isinstance(message, str) and "logprobs" in message.lower() and "not supported" in message.lower()

    def _handle_logprobs_fallback(self, exc: Exception, *, kind: str) -> bool:
        if self._is_logprobs_unsupported_error(exc):
            if kind == "chat":
                self._text_logprobs_supported = False
            else:
                self._tool_logprobs_supported = False
            LOGGER.info(
                "Logprobs unavailable for %s completions on model=%s; continuing without them.",
                kind,
                self.model,
            )
            return True

        LOGGER.warning(
            "Logprob-enabled %s completion failed for model=%s; preserving logprob requirement and propagating error: %s",
            kind,
            self.model,
            exc,
        )
        return False

    def _log_truncated_choices(self, response: Any, outputs: list[str]) -> None:
        truncated_indices = [
            index
            for index, choice in enumerate(response.choices)
            if getattr(choice, "finish_reason", None) == "length"
        ]
        if not truncated_indices:
            return

        empty_indices = [index for index in truncated_indices if index < len(outputs) and not outputs[index]]
        LOGGER.warning(
            "Chat completion truncated for model=%s max_tokens=%s truncated_choices=%s empty_truncated_choices=%s",
            self.model,
            self.max_tokens,
            truncated_indices,
            empty_indices,
        )

    def _validate_choice_batch(
        self,
        *,
        requested_count: int,
        actual_count: int,
        completion_kind: str,
    ) -> None:
        if actual_count <= 0:
            raise ModelGenerationError(
                f"{completion_kind.capitalize()} completion request returned no choices.",
                error_code="empty_choices",
                provider=self.provider,
                retryable=False,
                details={
                    "model": self.model,
                    "requested_count": requested_count,
                    "actual_count": actual_count,
                    "completion_kind": completion_kind,
                },
            )

        if actual_count == requested_count:
            return

        LOGGER.warning(
            "%s completion request returned %s choice(s) for model=%s after requesting %s; continuing until the requested sample count is satisfied.",
            completion_kind.capitalize(),
            actual_count,
            self.model,
            requested_count,
        )

    async def _create_completion(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def _create_raw_completion(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _extract_choice_text(self, choice: Any) -> str:
        return self._extract_message_text(choice.message)

    def _extract_message_text(self, message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        if isinstance(content, list):
            text_parts = [getattr(part, "text", "") for part in content]
            text = "".join(text_parts).strip()
            if text:
                return text

        for key in ("reasoning", "reasoning_content"):
            value = getattr(message, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            for key in ("reasoning", "reasoning_content"):
                value = model_extra.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    def _extract_choice_metadata(self, choice: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = self._extract_choice_finish_metadata(choice)
        logprob_container = getattr(choice, "logprobs", None)
        content_logprobs = getattr(logprob_container, "content", None)
        if isinstance(content_logprobs, list):
            token_items: list[tuple[str, float]] = []
            for token_info in content_logprobs:
                logprob = getattr(token_info, "logprob", None)
                if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
                    continue
                token = getattr(token_info, "token", "")
                token_items.append((token if isinstance(token, str) else "", float(logprob)))

            token_logprobs = [logprob for _, logprob in token_items]
            if token_logprobs:
                if any(logprob <= _LOGPROB_FLOOR_THRESHOLD for logprob in token_logprobs):
                    metadata["logprobs_unavailable"] = True
                    metadata["token_logprob_floor_detected"] = True
                else:
                    metadata["token_logprob_sum"] = float(sum(token_logprobs))
                    metadata["token_count"] = len(token_logprobs)
                    split_logprobs = self._split_token_logprobs_by_output_role(token_items)
                    if split_logprobs is not None:
                        (
                            reasoning_logprob_sum,
                            reasoning_token_count,
                            decision_logprob_sum,
                            decision_token_count,
                        ) = split_logprobs
                        if reasoning_token_count > 0:
                            metadata["reasoning_token_logprob_sum"] = reasoning_logprob_sum
                            metadata["reasoning_token_count"] = reasoning_token_count
                        if decision_token_count > 0:
                            metadata["decision_token_logprob_sum"] = decision_logprob_sum
                            metadata["decision_token_count"] = decision_token_count

        return metadata

    @staticmethod
    def _split_token_logprobs_by_output_role(
        token_items: list[tuple[str, float]],
    ) -> tuple[float, int, float, int] | None:
        """Split Qwen-style thinking tokens from final decision/tool-call tokens."""
        if not token_items or not any(token for token, _ in token_items):
            return None

        pieces = [token for token, _ in token_items]
        combined = "".join(pieces)
        if not combined:
            return None

        lowered = combined.lower()
        marker_positions = [
            position
            for marker in _TOOL_DECISION_MARKERS
            if (position := lowered.find(marker)) >= 0
        ]
        if marker_positions:
            decision_char_index = min(marker_positions)
        else:
            think_end_index = lowered.find(_THINK_END_MARKER)
            if think_end_index >= 0:
                decision_char_index = think_end_index + len(_THINK_END_MARKER)
            elif _THINK_START_MARKER in lowered:
                decision_char_index = len(combined)
            else:
                decision_char_index = 0

        reasoning_logprobs: list[float] = []
        decision_logprobs: list[float] = []
        cursor = 0
        for token, logprob in token_items:
            token_end = cursor + len(token)
            if token_end <= decision_char_index:
                reasoning_logprobs.append(logprob)
            else:
                decision_logprobs.append(logprob)
            cursor = token_end

        return (
            float(sum(reasoning_logprobs)),
            len(reasoning_logprobs),
            float(sum(decision_logprobs)),
            len(decision_logprobs),
        )

    @staticmethod
    def _extract_choice_finish_metadata(choice: Any) -> dict[str, Any]:
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason != "length":
            return {}
        return {"finish_reason": "length", "truncated": True}

    def _extract_choice_detail(self, choice: Any) -> dict[str, Any]:
        metadata = self._extract_choice_metadata(choice)

        return {
            "output": self._extract_choice_text(choice),
            "metadata": metadata,
        }

    def _extract_message_payload(self, message: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": str(getattr(message, "role", "assistant")),
        }
        content = self._extract_message_text(message)
        if content:
            payload["content"] = content
        elif content is not None:
            payload["content"] = str(content)

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            serialized_tool_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                if hasattr(tool_call, "model_dump"):
                    serialized_tool_calls.append(tool_call.model_dump())
                elif hasattr(tool_call, "to_dict"):
                    serialized_tool_calls.append(tool_call.to_dict())
                else:
                    serialized_tool_calls.append(dict(tool_call))
            payload["tool_calls"] = serialized_tool_calls
        return payload


class OpenAIChatModelClient(_BaseOpenAIChatModelClient):
    """OpenAI, vLLM, Ollama, or OpenAI-compatible async chat model client."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "openai",
        max_tokens: int = 8192,
        parallel_requests: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__(model, max_tokens=max_tokens, parallel_requests=parallel_requests, seed=seed)
        self.provider = provider
        if provider in {"ollama", "vllm"}:
            self.max_choices_per_request = 1
        self._client_base_urls = self._split_base_urls(base_url)
        self._clients = [
            AsyncOpenAI(base_url=endpoint_base_url, api_key=api_key)
            for endpoint_base_url in self._client_base_urls
        ]
        self._next_client_index = 0

    @property
    def client(self) -> Any:
        return self._clients[0]

    @client.setter
    def client(self, value: Any) -> None:
        self._clients = [value]
        self._client_base_urls = [None]
        self._next_client_index = 0

    def _split_base_urls(self, base_url: str | None) -> list[str | None]:
        if base_url is None:
            return [None]
        endpoints = [endpoint.strip() for endpoint in base_url.split(",") if endpoint.strip()]
        if not endpoints:
            return [base_url]
        return endpoints

    def _next_client(self) -> Any:
        return self._clients[self._next_client_indices()[0]]

    def _next_client_indices(self) -> list[int]:
        if len(self._clients) == 1:
            return [0]
        index = self._next_client_index
        self._next_client_index = (index + 1) % len(self._clients)
        return [(index + offset) % len(self._clients) for offset in range(len(self._clients))]

    def _client_endpoint_label(self, index: int) -> str:
        if 0 <= index < len(self._client_base_urls):
            return str(self._client_base_urls[index])
        return f"endpoint-{index}"

    def _is_retryable_endpoint_error(self, exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
            return isinstance(status_code, int) and (status_code in {408, 409, 429} or status_code >= 500)
        return False

    async def _create_completion(self, **kwargs: Any) -> Any:
        attempt_indices = self._next_client_indices()
        for attempt_position, client_index in enumerate(attempt_indices):
            try:
                return await self._clients[client_index].chat.completions.create(**kwargs)
            except Exception as exc:
                if attempt_position == len(attempt_indices) - 1 or not self._is_retryable_endpoint_error(exc):
                    raise
                LOGGER.warning(
                    "OpenAI-compatible endpoint %s failed for model=%s; retrying on another endpoint: %s",
                    self._client_endpoint_label(client_index),
                    self.model,
                    exc,
                )
        raise RuntimeError("unreachable")

    async def _create_raw_completion(self, **kwargs: Any) -> Any:
        attempt_indices = self._next_client_indices()
        for attempt_position, client_index in enumerate(attempt_indices):
            try:
                return await self._clients[client_index].completions.create(**kwargs)
            except Exception as exc:
                if attempt_position == len(attempt_indices) - 1 or not self._is_retryable_endpoint_error(exc):
                    raise
                LOGGER.warning(
                    "OpenAI-compatible raw completion endpoint %s failed for model=%s; retrying on another endpoint: %s",
                    self._client_endpoint_label(client_index),
                    self.model,
                    exc,
                )
        raise RuntimeError("unreachable")

    async def aclose(self) -> None:
        for client in self._clients:
            close_callable = getattr(client, "close", None)
            if callable(close_callable):
                await close_callable()


class AzureOpenAIChatModelClient(_BaseOpenAIChatModelClient):
    """Azure OpenAI async chat model client."""

    def __init__(
        self,
        model: str,
        *,
        azure_endpoint: str,
        api_version: str,
        api_key: str | None = None,
        deployment_name: str | None = None,
        max_tokens: int = 8192,
        parallel_requests: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__(deployment_name or model, max_tokens=max_tokens, parallel_requests=parallel_requests, seed=seed)
        self.base_model = model
        self.deployment_name = deployment_name or model
        self.max_choices_per_request = 8
        self.provider = "azure-openai"
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
        )

    async def _create_completion(self, **kwargs: Any) -> Any:
        return await self.client.chat.completions.create(**kwargs)

    async def aclose(self) -> None:
        await self.client.close()


def build_openai_chat_model(config: ChatModelConfig) -> ChatModelInterface:
    if config.provider == "azure-openai":
        return AzureOpenAIChatModelClient(
            config.model,
            azure_endpoint=config.azure_endpoint or "",
            api_version=config.api_version or "",
            api_key=config.api_key,
            deployment_name=config.deployment_name,
            max_tokens=config.max_tokens,
            parallel_requests=config.parallel_requests,
            seed=config.seed,
        )

    return OpenAIChatModelClient(
        config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        provider=config.provider,
        max_tokens=config.max_tokens,
        parallel_requests=config.parallel_requests,
        seed=config.seed,
    )
