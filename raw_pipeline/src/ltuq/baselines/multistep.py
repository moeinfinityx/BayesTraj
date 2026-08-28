from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
import math

import numpy as np

from ..logprobs import (
    DEFAULT_LOGPROB_FALLBACK_STRATEGY,
    FALLBACK_TOKEN_PROBABILITY,
    compute_predictive_entropy_from_metadata,
    extract_valid_token_logprob_stats,
    method_uses_token_logprobs,
    resolve_logprob_fallback,
    resolve_logprob_sums_from_metadata,
)
from ..trajectory import TDPStepRecord, TrajectoryDependentDecisionProcess
from .uprop import default_decision_distance
from .saup import SAUP_BASELINE, compute_saup_diagnostics
from .semantic_nearest_neighbor_entropy import SNNE_BASELINE


BaselineName = str
AggregationName = str

OUTCOME_ENTROPY_BASELINE = "outcome-entropy"
_SUPPORTED_BASELINES = {
    "pe",
    "ls",
    "ppl",
    "se",
    "deg",
    "sd",
    "sentsar",
    SAUP_BASELINE,
    SNNE_BASELINE,
    OUTCOME_ENTROPY_BASELINE,
}
_SUPPORTED_AGGREGATIONS = {"mean"}
_SEMANTIC_EQUIVALENCE_THRESHOLD = 0.85
_SENTSAR_TEMPERATURE = 1e-3
_MIN_REASONABLE_AVG_TOKEN_LOGPROB = -100.0
_MAX_REASONABLE_PERPLEXITY = math.exp(-_MIN_REASONABLE_AVG_TOKEN_LOGPROB)


def _resolve_step_texts(step: TDPStepRecord) -> list[str]:
    if step.sampled_decisions:
        return list(step.sampled_decisions)
    return [step.realized_decision]


def _resolve_sampled_output_metadata(step: TDPStepRecord, expected_count: int) -> list[dict[str, object]]:
    raw_metadata = step.metadata.get("sampled_output_metadata")
    if not isinstance(raw_metadata, list):
        return [{} for _ in range(expected_count)]

    metadata: list[dict[str, object]] = []
    for item in raw_metadata[:expected_count]:
        metadata.append(dict(item) if isinstance(item, dict) else {})

    if len(metadata) < expected_count:
        metadata.extend({} for _ in range(expected_count - len(metadata)))
    return metadata


def _resolve_step_logprob_sums(
    step: TDPStepRecord,
    *,
    logprob_sum_key: str = "token_logprob_sum",
    token_count_key: str = "token_count",
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> tuple[list[float] | None, bool]:
    texts = _resolve_step_texts(step)
    metadata = _resolve_sampled_output_metadata(step, len(texts))
    return resolve_logprob_sums_from_metadata(
        metadata,
        logprob_sum_key=logprob_sum_key,
        token_count_key=token_count_key,
        fallback_strategy=fallback_strategy,
    )


def _resolve_step_ppl(
    step: TDPStepRecord,
    *,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> tuple[float | None, bool]:
    ppl = step.uncertainty_measurements.get("ppl")
    if isinstance(ppl, (int, float)) and math.isfinite(float(ppl)) and 0.0 < float(ppl) <= _MAX_REASONABLE_PERPLEXITY:
        return float(ppl), False

    metadata = step.metadata.get("chosen_output_metadata")
    if isinstance(metadata, dict):
        stats = extract_valid_token_logprob_stats(metadata)
        if stats is not None:
            logprob_sum, token_count = stats
            return float(math.exp(-float(logprob_sum) / float(token_count))), False

    chosen_index = step.metadata.get("chosen_output_index")
    texts = _resolve_step_texts(step)
    if isinstance(chosen_index, int) and 0 <= chosen_index < len(texts):
        sampled_metadata = _resolve_sampled_output_metadata(step, len(texts))
        if chosen_index < len(sampled_metadata):
            chosen_metadata = sampled_metadata[chosen_index]
            stats = extract_valid_token_logprob_stats(chosen_metadata)
            if stats is not None:
                logprob_sum, token_count = stats
                return float(math.exp(-float(logprob_sum) / float(token_count))), False

    fallback_logprob_sum = resolve_logprob_fallback(fallback_strategy)
    if fallback_logprob_sum is None:
        return None, False
    return float(math.exp(-fallback_logprob_sum)), True


def tdp_uses_logprob_fallback(
    tdp: TrajectoryDependentDecisionProcess,
    baseline_name: BaselineName,
    *,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> bool:
    resolved_name = validate_baseline_name(baseline_name)
    if not method_uses_token_logprobs(resolved_name) or fallback_strategy is None:
        return False

    if resolved_name == "ppl":
        return any(_resolve_step_ppl(step, fallback_strategy=fallback_strategy)[1] for step in tdp.steps)

    if resolved_name == SAUP_BASELINE:
        # SAUP deliberately requires the stored per-step uncertainty and does
        # not manufacture it from the neutral logprob fallback.
        return False

    return any(_resolve_step_logprob_sums(step, fallback_strategy=fallback_strategy)[1] for step in tdp.steps)


def _pairwise_similarity_matrix(
    texts: Sequence[str],
    distance_fn: Callable[[str, str], float],
) -> np.ndarray:
    size = len(texts)
    matrix = np.eye(size, dtype=float)
    for row_index in range(size):
        for col_index in range(row_index + 1, size):
            similarity = max(0.0, 1.0 - float(distance_fn(texts[row_index], texts[col_index])))
            matrix[row_index, col_index] = similarity
            matrix[col_index, row_index] = similarity
    return matrix


def _connected_components_from_threshold(similarity_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    node_count = int(similarity_matrix.shape[0])
    parent = list(range(node_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row_index in range(node_count):
        for col_index in range(row_index + 1, node_count):
            if similarity_matrix[row_index, col_index] >= threshold:
                union(row_index, col_index)

    components: dict[int, list[int]] = {}
    for node_index in range(node_count):
        root = find(node_index)
        components.setdefault(root, []).append(node_index)
    return list(components.values())


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    array = np.asarray(values, dtype=float)
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return -math.inf
    max_value = float(np.max(finite_values))
    return float(max_value + math.log(np.sum(np.exp(finite_values - max_value))))


def _resolve_realized_sample_index(step: TDPStepRecord, texts: Sequence[str]) -> int:
    chosen_index = step.metadata.get("chosen_output_index")
    if isinstance(chosen_index, int) and 0 <= chosen_index < len(texts):
        return chosen_index
    for index, text in enumerate(texts):
        if text == step.realized_decision:
            return index
    return 0


def validate_baseline_name(baseline_name: BaselineName) -> str:
    stripped = baseline_name.strip()
    normalized = stripped.lower()
    if normalized not in _SUPPORTED_BASELINES:
        raise ValueError(
            f"Unsupported baseline '{baseline_name}'. Expected one of {_SUPPORTED_BASELINES}."
        )
    return normalized


def validate_aggregation_name(aggregation: AggregationName) -> str:
    normalized = aggregation.strip().lower()
    if normalized not in _SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. Expected one of {_SUPPORTED_AGGREGATIONS}."
        )
    return normalized


def aggregate_step_uncertainties(
    step_scores: Sequence[float | None],
    aggregation: AggregationName = "mean",
) -> float | None:
    resolved_aggregation = validate_aggregation_name(aggregation)
    valid_scores = [float(score) for score in step_scores if isinstance(score, (int, float)) and math.isfinite(float(score))]
    if not valid_scores:
        return None

    return float(np.mean(np.asarray(valid_scores, dtype=float)))


def _looks_explanatory_answer(answer: str) -> bool:
    """Return whether an answer reads like prose instead of a compact final value."""
    if "\n" in answer or "\r" in answer:
        return True
    if any(marker in answer for marker in ("Thought:", "Action:", "Observation:", "```", "**")):
        return True
    lowered = answer.lower().strip()
    if lowered.startswith(("the ", "based on", "looking at", "i ", "we ")):
        return True
    return False


def normalize_answer_for_outcome(
    answer: str,
    *,
    prompt: str,
) -> tuple[str | None, str | None]:
    """Normalize a final answer before it becomes a generic outcome bucket.

    The shared baseline layer intentionally avoids prompt-keyword schema
    guessing. Environment-specific runners should provide evaluator-aware
    outcome buckets when they know the expected answer type.
    """
    del prompt
    normalized = " ".join(answer.strip().split())
    if not normalized:
        return None, "empty"
    if _looks_explanatory_answer(answer):
        return None, "explanatory-format"
    if len(normalized) > 200:
        return None, "too-long"

    return normalized, None


def normalize_tdp_outcome(tdp: TrajectoryDependentDecisionProcess) -> str:
    """Return the validated outcome bucket used by the outcome-entropy baseline."""
    prompt = tdp.prompt or ""

    def answer_bucket(raw_answer: str) -> str:
        normalized_answer, invalid_reason = normalize_answer_for_outcome(raw_answer, prompt=prompt)
        if invalid_reason is not None:
            return f"invalid-answer:{invalid_reason}"
        return f"answer:{normalized_answer}"

    answer = tdp.final_answer
    if isinstance(answer, str) and answer.strip():
        return answer_bucket(answer)

    hard_finalization = tdp.metadata.get("hard_finalization")
    if isinstance(hard_finalization, dict):
        hard_answer = hard_finalization.get("answer")
        if isinstance(hard_answer, str) and hard_answer.strip():
            return answer_bucket(hard_answer)

    status = tdp.metadata.get("status")
    if isinstance(status, str) and status.strip():
        return f"no-answer:{status.strip()}"

    result = tdp.metadata.get("result")
    if isinstance(result, dict):
        result_answer = result.get("answer")
        if isinstance(result_answer, str) and result_answer.strip():
            return answer_bucket(result_answer)
        error_code = result.get("error_code")
        if isinstance(error_code, str) and error_code.strip():
            return f"no-answer:error:{error_code.strip()}"

    return "no-answer:unknown"


def compute_outcome_entropy(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    *,
    outcome_bucket_fn: Callable[[TrajectoryDependentDecisionProcess], str] = normalize_tdp_outcome,
) -> tuple[float | None, dict[str, int]]:
    """Compute Shannon entropy over final trajectory outcome buckets."""
    if not tdps:
        return None, {}

    outcome_counts: dict[str, int] = {}
    for tdp in tdps:
        outcome = outcome_bucket_fn(tdp)
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    total = float(sum(outcome_counts.values()))
    if total <= 0.0:
        return None, outcome_counts

    entropy = 0.0
    for count in outcome_counts.values():
        probability = float(count) / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return float(entropy), outcome_counts


def compute_stepwise_pe(
    tdp: TrajectoryDependentDecisionProcess,
    uncertainty_key: str = "pe",
    *,
    logprob_sum_key: str = "token_logprob_sum",
    token_count_key: str = "token_count",
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> list[float | None]:
    del uncertainty_key
    step_scores: list[float | None] = []
    for step in tdp.steps:
        texts = _resolve_step_texts(step)
        metadata = _resolve_sampled_output_metadata(step, len(texts))
        predictive_entropy, _ = compute_predictive_entropy_from_metadata(
            metadata,
            logprob_sum_key=logprob_sum_key,
            token_count_key=token_count_key,
            fallback_strategy=fallback_strategy,
        )
        if predictive_entropy is None:
            step_scores.append(None)
            continue
        step_scores.append(float(predictive_entropy))
    return step_scores


def compute_stepwise_ls(
    tdp: TrajectoryDependentDecisionProcess,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
) -> list[float]:
    step_scores: list[float] = []
    for step in tdp.steps:
        decisions = list(step.sampled_decisions) or [step.realized_decision]
        if len(decisions) <= 1:
            step_scores.append(0.0)
            continue

        pairwise_distances = [
            float(distance_fn(left, right)) for left, right in combinations(decisions, 2)
        ]
        step_scores.append(float(np.mean(pairwise_distances)) if pairwise_distances else 0.0)
    return step_scores


def compute_stepwise_ppl(
    tdp: TrajectoryDependentDecisionProcess,
    *,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> list[float | None]:
    step_scores: list[float | None] = []
    for step in tdp.steps:
        ppl, _ = _resolve_step_ppl(step, fallback_strategy=fallback_strategy)
        if ppl is None:
            step_scores.append(None)
            continue
        step_scores.append(float(ppl))
    return step_scores


def compute_stepwise_se(
    tdp: TrajectoryDependentDecisionProcess,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    equivalence_threshold: float = _SEMANTIC_EQUIVALENCE_THRESHOLD,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> list[float | None]:
    step_scores: list[float | None] = []
    for step in tdp.steps:
        texts = _resolve_step_texts(step)
        if len(texts) <= 1:
            step_scores.append(0.0)
            continue

        logprob_sums, _ = _resolve_step_logprob_sums(step, fallback_strategy=fallback_strategy)
        if logprob_sums is None:
            step_scores.append(None)
            continue
        similarity_matrix = _pairwise_similarity_matrix(texts, distance_fn)
        clusters = _connected_components_from_threshold(similarity_matrix, equivalence_threshold)
        cluster_log_masses = [_logsumexp([logprob_sums[index] for index in cluster]) for cluster in clusters]
        step_scores.append(float(np.mean([-log_mass for log_mass in cluster_log_masses])))
    return step_scores


def compute_stepwise_deg(
    tdp: TrajectoryDependentDecisionProcess,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
) -> list[float | None]:
    step_scores: list[float | None] = []
    for step in tdp.steps:
        texts = _resolve_step_texts(step)
        if len(texts) <= 1:
            step_scores.append(0.0)
            continue

        similarity_matrix = _pairwise_similarity_matrix(texts, distance_fn)
        degree_uncertainties = np.sum(1.0 - similarity_matrix, axis=1)
        step_scores.append(float(np.mean(degree_uncertainties)))
    return step_scores


def compute_stepwise_sd(
    tdp: TrajectoryDependentDecisionProcess,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> list[float | None]:
    step_scores: list[float | None] = []
    for step in tdp.steps:
        texts = _resolve_step_texts(step)
        logprob_sums, _ = _resolve_step_logprob_sums(step, fallback_strategy=fallback_strategy)
        if logprob_sums is None:
            step_scores.append(None)
            continue
        target_index = _resolve_realized_sample_index(step, texts)
        target_text = texts[target_index]
        distances = np.asarray([float(distance_fn(target_text, text)) for text in texts], dtype=float)
        kernels = np.clip(1.0 - np.square(distances), 0.0, 1.0)
        denominator_log_mass = _logsumexp(logprob_sums)
        numerator_terms = [
            logprob_sum + math.log(kernel)
            for logprob_sum, kernel in zip(logprob_sums, kernels, strict=True)
            if kernel > 0.0
        ]
        numerator_log_mass = _logsumexp(numerator_terms)
        if not math.isfinite(numerator_log_mass) or not math.isfinite(denominator_log_mass):
            density = 0.0
        else:
            density = float(math.exp(numerator_log_mass - denominator_log_mass))
        step_scores.append(float(max(0.0, 1.0 - density)))
    return step_scores


def compute_stepwise_sentsar(
    tdp: TrajectoryDependentDecisionProcess,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    temperature: float = _SENTSAR_TEMPERATURE,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> list[float | None]:
    if temperature <= 0.0:
        raise ValueError("sentSAR temperature must be positive")

    step_scores: list[float | None] = []
    log_temperature = math.log(temperature)
    for step in tdp.steps:
        texts = _resolve_step_texts(step)
        logprob_sums, _ = _resolve_step_logprob_sums(step, fallback_strategy=fallback_strategy)
        if logprob_sums is None:
            step_scores.append(None)
            continue
        if len(texts) == 1:
            step_scores.append(float(-logprob_sums[0]))
            continue

        similarity_matrix = _pairwise_similarity_matrix(texts, distance_fn)
        sentence_energies: list[float] = []
        for row_index in range(len(texts)):
            relevance_terms = [
                logprob_sums[col_index] + math.log(similarity_matrix[row_index, col_index])
                for col_index in range(len(texts))
                if col_index != row_index and similarity_matrix[row_index, col_index] > 0.0
            ]
            relevance_log_mass = _logsumexp(relevance_terms)
            shifted_relevance = relevance_log_mass - log_temperature if math.isfinite(relevance_log_mass) else -math.inf
            total_log_mass = float(np.logaddexp(logprob_sums[row_index], shifted_relevance))
            sentence_energies.append(-total_log_mass)
        step_scores.append(float(np.mean(sentence_energies)))
    return step_scores


def build_tdp_baseline_step_scores(
    tdp: TrajectoryDependentDecisionProcess,
    baseline_name: BaselineName,
    *,
    uncertainty_key: str = "pe",
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> Sequence[float | None]:
    resolved_name = validate_baseline_name(baseline_name)
    if resolved_name == "pe":
        return compute_stepwise_pe(tdp, uncertainty_key=uncertainty_key, fallback_strategy=fallback_strategy)
    if resolved_name == "ls":
        return compute_stepwise_ls(tdp, distance_fn=distance_fn)
    if resolved_name == "ppl":
        return compute_stepwise_ppl(tdp, fallback_strategy=fallback_strategy)
    if resolved_name == "se":
        return compute_stepwise_se(tdp, distance_fn=distance_fn, fallback_strategy=fallback_strategy)
    if resolved_name == "deg":
        return compute_stepwise_deg(tdp, distance_fn=distance_fn)
    if resolved_name == "sd":
        return compute_stepwise_sd(tdp, distance_fn=distance_fn, fallback_strategy=fallback_strategy)
    if resolved_name == SAUP_BASELINE:
        diagnostics = compute_saup_diagnostics(
            tdp,
            resolved_name,
            distance_fn=distance_fn,
            uncertainty_key=uncertainty_key,
        )
        return [step.weighted_uncertainty for step in diagnostics.steps]
    if resolved_name == OUTCOME_ENTROPY_BASELINE:
        return []
    return compute_stepwise_sentsar(tdp, distance_fn=distance_fn, fallback_strategy=fallback_strategy)


def compute_tdp_baseline(
    tdp: TrajectoryDependentDecisionProcess,
    baseline_name: BaselineName,
    *,
    aggregation: AggregationName = "mean",
    uncertainty_key: str = "pe",
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> float | None:
    resolved_name = validate_baseline_name(baseline_name)
    if resolved_name == SAUP_BASELINE:
        return compute_saup_diagnostics(
            tdp,
            resolved_name,
            distance_fn=distance_fn,
            uncertainty_key=uncertainty_key,
        ).score
    step_scores = build_tdp_baseline_step_scores(
        tdp,
        baseline_name,
        uncertainty_key=uncertainty_key,
        distance_fn=distance_fn,
        fallback_strategy=fallback_strategy,
    )
    return aggregate_step_uncertainties(step_scores, aggregation=aggregation)


def compute_multitdp_baseline(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    baseline_name: BaselineName,
    *,
    aggregation: AggregationName = "mean",
    uncertainty_key: str = "pe",
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    fallback_strategy: str | None = DEFAULT_LOGPROB_FALLBACK_STRATEGY,
) -> float | None:
    if not tdps:
        return None

    tdp_scores = [
        compute_tdp_baseline(
            tdp,
            baseline_name,
            aggregation=aggregation,
            uncertainty_key=uncertainty_key,
            distance_fn=distance_fn,
            fallback_strategy=fallback_strategy,
        )
        for tdp in tdps
    ]
    valid_scores = [float(score) for score in tdp_scores if isinstance(score, (int, float)) and math.isfinite(float(score))]
    if not valid_scores:
        return None
    return float(np.mean(valid_scores))
