from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Sequence

from ..models import ChatModelConfig, create_chat_model


ProbeKind = Literal["text", "tool"]

_TEXT_PROBE_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "Respond with a single short answer."},
    {"role": "user", "content": "Answer yes."},
]

_TOOL_PROBE_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Use the provided tool to answer. Do not answer in plain text."},
    {"role": "user", "content": "Call the tool with the answer yes."},
]

_TOOL_PROBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "commit_answer",
            "description": "Commit a final answer for the probe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
]


def create_runner_model_client(config: Any) -> Any:
    return create_chat_model(
        ChatModelConfig(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            emulate_tool_calls=getattr(config, "emulate_tool_calls", False),
            max_tokens=config.max_tokens,
            parallel_requests=getattr(config, "parallel_requests", 8),
            base_url=config.base_url,
            azure_endpoint=config.azure_endpoint,
            api_version=config.api_version,
            deployment_name=config.deployment_name,
        )
    )


def payload_contains_token_logprobs(payload: Any) -> bool:
    if isinstance(payload, dict):
        logprob_sum = payload.get("token_logprob_sum")
        token_count = payload.get("token_count")
        if isinstance(logprob_sum, (int, float)) and not isinstance(logprob_sum, bool):
            if isinstance(token_count, int) and not isinstance(token_count, bool) and token_count > 0:
                return True
        return any(payload_contains_token_logprobs(value) for value in payload.values())
    if isinstance(payload, list):
        return any(payload_contains_token_logprobs(item) for item in payload)
    return False


def payload_marks_logprobs_unavailable(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("logprobs_unavailable") is True or payload.get("token_logprob_floor_detected") is True:
            return True
        return any(payload_marks_logprobs_unavailable(value) for value in payload.values())
    if isinstance(payload, list):
        return any(payload_marks_logprobs_unavailable(item) for item in payload)
    return False


def record_has_missing_logprob_state(record: dict[str, Any], *, method: str = "uprop") -> bool:
    if str(record.get("method", "")) != method:
        return False
    if record.get("status") == "generation_failed":
        return False
    if payload_marks_logprobs_unavailable(record):
        return True
    if payload_contains_token_logprobs(record):
        return False
    return True


def filter_stale_records(
    records: Sequence[dict[str, Any]],
    *,
    refresh_missing_logprob_records: bool,
    resolve_uncertainty: Callable[[dict[str, Any]], float | None] | None = None,
    method: str = "uprop",
) -> tuple[list[dict[str, Any]], list[str]]:
    reusable_records: list[dict[str, Any]] = []
    dropped_sample_ids: list[str] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        if resolve_uncertainty is not None:
            record["uncertainty"] = resolve_uncertainty(record)
        if refresh_missing_logprob_records and record_has_missing_logprob_state(record, method=method):
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str):
                dropped_sample_ids.append(sample_id)
            continue
        reusable_records.append(record)

    return reusable_records, dropped_sample_ids


async def probe_structured_logprob_support(
    model_client: Any,
    *,
    probe_kind: ProbeKind,
    logger: logging.Logger | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> bool:
    try:
        if probe_kind == "text":
            detailed_callable = getattr(model_client, "sample_many_detailed", None)
            if detailed_callable is None:
                return False
            outputs = await detailed_callable(_TEXT_PROBE_MESSAGES, temperature=0.0, n=1)
        else:
            detailed_many_callable = getattr(model_client, "acompletion_with_tools_many_detailed", None)
            if detailed_many_callable is not None:
                outputs = await detailed_many_callable(
                    _TOOL_PROBE_MESSAGES,
                    _TOOL_PROBE_TOOLS,
                    temperature=0.0,
                    n=1,
                )
            else:
                detailed_callable = getattr(model_client, "acompletion_with_tools_detailed", None)
                if detailed_callable is None:
                    return False
                outputs = [await detailed_callable(_TOOL_PROBE_MESSAGES, _TOOL_PROBE_TOOLS, temperature=0.0)]
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Could not probe %s logprob support for model=%s provider=%s: %s",
                probe_kind,
                model,
                provider,
                exc,
            )
        return False

    for output in outputs:
        if not isinstance(output, dict):
            continue
        metadata = output.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if payload_contains_token_logprobs(metadata) or payload_marks_logprobs_unavailable(metadata):
            return True
    return False
