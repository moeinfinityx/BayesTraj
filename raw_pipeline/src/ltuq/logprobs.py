from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final


DEFAULT_LOGPROB_FALLBACK_STRATEGY: Final[str] = "token-probability-1.0"
FALLBACK_TOKEN_PROBABILITY: Final[float] = 1.0
_MIN_REASONABLE_AVG_TOKEN_LOGPROB: Final[float] = -100.0


TOKEN_LOGPROB_METHODS: Final[frozenset[str]] = frozenset(
    {"pe", "ppl", "se", "sd", "sentsar", "saup"}
)
LOGPROB_DERIVED_METHODS: Final[frozenset[str]] = frozenset(
    {"uprop"}
)


def method_uses_token_logprobs(method: Any) -> bool:
    return isinstance(method, str) and method in TOKEN_LOGPROB_METHODS


def method_logprob_requirement(method: Any) -> str:
    if not isinstance(method, str):
        return "not-needed"
    normalized = method.strip().lower()
    if normalized in TOKEN_LOGPROB_METHODS:
        return "required"
    if normalized in LOGPROB_DERIVED_METHODS:
        return "degrades"
    return "not-needed"


def model_supports_token_logprobs(provider: Any, model: Any) -> bool | None:
    if not isinstance(provider, str) or not isinstance(model, str):
        return None
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip().lower()
    if normalized_provider == "openai" and normalized_model == "gpt-4.1-nano":
        return True
    if normalized_model == "gpt-5-mini" and normalized_provider in {"openai", "azure-openai"}:
        return False
    return None


def resolve_logprob_fallback(fallback_strategy: str | None) -> float | None:
    if fallback_strategy is None:
        return None
    if fallback_strategy == DEFAULT_LOGPROB_FALLBACK_STRATEGY:
        return 0.0
    raise ValueError(f"Unsupported logprob fallback strategy '{fallback_strategy}'.")


def extract_valid_token_logprob_sum(
    metadata: Mapping[str, Any],
    *,
    logprob_sum_key: str = "token_logprob_sum",
    token_count_key: str = "token_count",
) -> float | None:
    logprob_sum = metadata.get(logprob_sum_key)
    if isinstance(logprob_sum, bool) or not isinstance(logprob_sum, (int, float)):
        return None

    resolved_logprob_sum = float(logprob_sum)
    if not math.isfinite(resolved_logprob_sum):
        return None

    if token_count_key in metadata:
        token_count = metadata.get(token_count_key)
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
            return None
        average_logprob = resolved_logprob_sum / float(token_count)
        if average_logprob < _MIN_REASONABLE_AVG_TOKEN_LOGPROB:
            return None

    return resolved_logprob_sum


def extract_valid_token_logprob_stats(
    metadata: Mapping[str, Any],
    *,
    logprob_sum_key: str = "token_logprob_sum",
    token_count_key: str = "token_count",
) -> tuple[float, int] | None:
    logprob_sum = extract_valid_token_logprob_sum(
        metadata,
        logprob_sum_key=logprob_sum_key,
        token_count_key=token_count_key,
    )
    token_count = metadata.get(token_count_key)
    if logprob_sum is None:
        return None
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
        return None
    return logprob_sum, token_count


def resolve_logprob_sums_from_metadata(
    sampled_output_metadata: Sequence[Mapping[str, Any] | object],
    *,
    logprob_sum_key: str = "token_logprob_sum",
    token_count_key: str = "token_count",
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> tuple[list[float] | None, bool]:
    logprob_sums: list[float] = []
    used_fallback = False
    for item in sampled_output_metadata:
        metadata = item if isinstance(item, Mapping) else {}
        logprob_sum = extract_valid_token_logprob_sum(
            metadata,
            logprob_sum_key=logprob_sum_key,
            token_count_key=token_count_key,
        )
        if logprob_sum is not None:
            logprob_sums.append(float(logprob_sum))
            continue

        fallback_logprob_sum = resolve_logprob_fallback(fallback_strategy)
        if fallback_logprob_sum is None:
            return None, used_fallback
        logprob_sums.append(fallback_logprob_sum)
        used_fallback = True
    return logprob_sums, used_fallback


def compute_predictive_entropy_from_metadata(
    sampled_output_metadata: Sequence[Mapping[str, Any] | object],
    *,
    logprob_sum_key: str = "token_logprob_sum",
    token_count_key: str = "token_count",
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> tuple[float | None, bool]:
    if not sampled_output_metadata:
        return 0.0, False

    logprob_sums, used_fallback = resolve_logprob_sums_from_metadata(
        sampled_output_metadata,
        logprob_sum_key=logprob_sum_key,
        token_count_key=token_count_key,
        fallback_strategy=fallback_strategy,
    )
    if logprob_sums is None:
        return None, used_fallback
    if not logprob_sums:
        return 0.0, used_fallback
    return float(-sum(logprob_sums) / len(logprob_sums)), used_fallback
