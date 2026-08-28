from __future__ import annotations

import math
from typing import Any, Sequence

from .baselines.multistep import (
    DEFAULT_LOGPROB_FALLBACK_STRATEGY,
    FALLBACK_TOKEN_PROBABILITY,
    OUTCOME_ENTROPY_BASELINE,
    build_tdp_baseline_step_scores,
    compute_tdp_baseline,
    tdp_uses_logprob_fallback,
)
from .logprobs import method_logprob_requirement, method_uses_token_logprobs
from .trajectory import (
    TDPCounterfactualBranch,
    TDPCounterfactualRecord,
    TDPStepRecord,
    TrajectoryDependentDecisionProcess,
)


_SUSPICIOUS_LOGPROB_UNCERTAINTY_THRESHOLD = 1e6
_MULTI_TRAJECTORY_METHODS = {
    "uprop",
    "pe",
    "ls",
    "ppl",
    "se",
    "deg",
    "sd",
    "sentsar",
    OUTCOME_ENTROPY_BASELINE,
}


def logprobs_required_for_method(method: Any) -> bool:
    return method_uses_token_logprobs(method)


def _payload_contains_token_logprobs(payload: Any) -> bool:
    if isinstance(payload, dict):
        logprob_sum = _coerce_uncertainty(payload.get("token_logprob_sum"))
        token_count = _coerce_uncertainty(payload.get("token_count"))
        if logprob_sum is not None and token_count is not None and token_count > 0:
            return True
        return any(_payload_contains_token_logprobs(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_contains_token_logprobs(item) for item in payload)
    return False


def _record_has_token_logprobs(record: dict[str, Any]) -> bool:
    return _payload_contains_token_logprobs(record)


def _record_has_explicit_unavailable_logprobs(record: dict[str, Any], *, requires_logprobs: bool) -> bool:
    estimate = record.get("estimate")
    if isinstance(estimate, dict):
        metadata = estimate.get("metadata")
        if isinstance(metadata, dict):
            if bool(metadata.get("uses_logprob_fallback") or metadata.get("uses_token_probability_fallback")):
                return True
            if requires_logprobs and metadata.get("uncertainty_available") is False:
                return True

    error = record.get("error")
    if requires_logprobs and isinstance(error, str) and "token_logprob_sum" in error:
        return True

    failure = record.get("failure")
    if requires_logprobs and isinstance(failure, dict):
        for key in ("message", "error", "detail"):
            value = failure.get(key)
            if isinstance(value, str) and "token_logprob_sum" in value:
                return True
    return False


def _coerce_correctness(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def unknown_correctness_is_failure_for_task_family(task_family: Any = None) -> bool:
    """Return whether unknown correctness should count as failure in UQ metrics.

    UQ success metrics now apply this policy uniformly: only explicit True is a
    success, while False and unknown correctness are failures.
    """
    return True


def _coerce_uncertainty(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _main_trajectory_tdp_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    estimate = record.get("estimate")
    if not isinstance(estimate, dict):
        return None
    tdps = estimate.get("tdps")
    if not isinstance(tdps, list):
        return None

    fallback: dict[str, Any] | None = None
    for payload in tdps:
        if not isinstance(payload, dict):
            continue
        if fallback is None:
            fallback = payload
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata.get("trajectory_index") == 0:
            return payload
    return fallback


def _main_trajectory_correctness(record: dict[str, Any]) -> tuple[bool | None, dict[str, Any] | None]:
    payload = _main_trajectory_tdp_payload(record)
    if payload is None:
        return None, None

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None, payload

    result = metadata.get("result")
    if isinstance(result, dict):
        normalized = _coerce_correctness(result.get("is_correct"))
        if normalized is not None:
            return normalized, payload
        nested = result.get("result")
        if isinstance(nested, bool):
            return nested, payload
    else:
        normalized = _coerce_correctness(result)
        if normalized is not None:
            return normalized, payload

    return None, payload


def _step_length_span(start_step_index: Any, stop_step_index: Any) -> int | None:
    if isinstance(start_step_index, bool) or isinstance(stop_step_index, bool):
        return None
    try:
        start = int(start_step_index)
        stop = int(stop_step_index)
    except (TypeError, ValueError):
        return None
    if stop < start:
        return None
    return (stop - start) + 1


def _trajectory_payload_step_count(payload: dict[str, Any]) -> int | None:
    steps = payload.get("steps")
    if isinstance(steps, list):
        return len(steps)

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        span_length = _step_length_span(metadata.get("start_step_index"), metadata.get("stop_step_index"))
        if span_length is not None:
            return span_length

        step_entropies = metadata.get("step_entropies")
        if isinstance(step_entropies, list):
            return len(step_entropies)

    return None


def normalize_multi_trajectory_record(record: dict[str, Any]) -> None:
    migrate_logprob_baseline_record(record)


def _sum_uncertainty_values(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    normalized = [_coerce_uncertainty(value) for value in values]
    valid = [value for value in normalized if value is not None]
    if not valid:
        return None
    return float(sum(valid))


def _deserialize_tdp_step(payload: Any) -> TDPStepRecord | None:
    if not isinstance(payload, dict):
        return None
    raw_measurements = payload.get("uncertainty_measurements")
    uncertainty_measurements = (
        {str(key): float(value) for key, value in raw_measurements.items() if isinstance(value, (int, float))}
        if isinstance(raw_measurements, dict)
        else {}
    )
    metadata = payload.get("metadata")
    counterfactual_records = _deserialize_counterfactual_records(payload.get("counterfactual_records"))
    return TDPStepRecord(
        index=int(payload.get("index", 0)),
        realized_decision=str(payload.get("realized_decision", "")),
        sampled_decisions=[str(item) for item in payload.get("sampled_decisions", [])],
        uncertainty_measurements=uncertainty_measurements,
        counterfactual_records=counterfactual_records,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _deserialize_counterfactual_records(payload: Any) -> list[TDPCounterfactualRecord]:
    if not isinstance(payload, list):
        return []
    records: list[TDPCounterfactualRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        branches_payload = item.get("branches")
        branches: list[TDPCounterfactualBranch] = []
        if isinstance(branches_payload, list):
            for branch_payload in branches_payload:
                if not isinstance(branch_payload, dict):
                    continue
                branches.append(
                    TDPCounterfactualBranch(
                        source_decision=str(branch_payload.get("source_decision", "")),
                        target_sampled_decisions=[
                            str(value) for value in branch_payload.get("target_sampled_decisions", [])
                        ],
                        target_sampled_output_metadata=[
                            dict(value) for value in branch_payload.get("target_sampled_output_metadata", []) if isinstance(value, dict)
                        ],
                        metadata=dict(branch_payload.get("metadata", {})) if isinstance(branch_payload.get("metadata"), dict) else {},
                    )
                )
        records.append(
            TDPCounterfactualRecord(
                source_step_index=int(item.get("source_step_index", 0)),
                realized_source_decision=str(item.get("realized_source_decision", "")),
                branches=branches,
                metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
            )
        )
    return records


def _deserialize_tdp(payload: Any) -> TrajectoryDependentDecisionProcess | None:
    if not isinstance(payload, dict):
        return None
    steps = [step for item in payload.get("steps", []) if (step := _deserialize_tdp_step(item)) is not None]
    metadata = payload.get("metadata")
    return TrajectoryDependentDecisionProcess(
        sample_id=str(payload.get("sample_id", "")),
        prompt=str(payload.get("prompt", "")),
        steps=steps,
        final_answer=str(payload.get("final_answer")) if payload.get("final_answer") is not None else None,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def migrate_logprob_baseline_record(record: dict[str, Any]) -> None:
    method = record.get("method")
    if not method_uses_token_logprobs(method):
        return

    estimate = record.get("estimate")
    if not isinstance(estimate, dict):
        return

    estimate_metadata = estimate.get("metadata")
    if not isinstance(estimate_metadata, dict):
        estimate_metadata = {}
        estimate["metadata"] = estimate_metadata

    tdps_payload = estimate.get("tdps")
    if not isinstance(tdps_payload, list) or not tdps_payload:
        return

    direct_uncertainty = _coerce_uncertainty(record.get("uncertainty"))
    total_uncertainty = _coerce_uncertainty(estimate.get("total_uncertainty"))
    tdp_scores_payload = estimate.get("tdp_scores")
    has_suspicious_tdp_score = isinstance(tdp_scores_payload, list) and any(
        (score := _coerce_uncertainty(value)) is not None
        and (not math.isfinite(score) or score > _SUSPICIOUS_LOGPROB_UNCERTAINTY_THRESHOLD)
        for value in tdp_scores_payload
    )

    normalized_method = str(method).strip().lower()
    needs_upgrade = any(
        estimate.get(key) is None for key in ("tdp_scores", "tdp_step_scores", "total_uncertainty")
    ) or "uses_logprob_fallback" not in estimate_metadata
    needs_upgrade = needs_upgrade or (
        normalized_method == "pe" and "pe_auxiliary_uncertainties" not in estimate_metadata
    )
    needs_upgrade = needs_upgrade or (
        direct_uncertainty is not None
        and (not math.isfinite(direct_uncertainty) or direct_uncertainty > _SUSPICIOUS_LOGPROB_UNCERTAINTY_THRESHOLD)
    )
    needs_upgrade = needs_upgrade or (
        total_uncertainty is not None
        and (not math.isfinite(total_uncertainty) or total_uncertainty > _SUSPICIOUS_LOGPROB_UNCERTAINTY_THRESHOLD)
    )
    needs_upgrade = needs_upgrade or has_suspicious_tdp_score
    if not needs_upgrade:
        return

    tdps = [tdp for item in tdps_payload if (tdp := _deserialize_tdp(item)) is not None]
    if not tdps:
        return

    aggregation = estimate.get("aggregation")
    if not isinstance(aggregation, str) or not aggregation:
        aggregation = str(estimate_metadata.get("aggregation") or "mean")

    tdp_step_scores = [
        list(
            build_tdp_baseline_step_scores(
                tdp,
                str(method),
                fallback_strategy=DEFAULT_LOGPROB_FALLBACK_STRATEGY,
            )
        )
        for tdp in tdps
    ]
    tdp_scores = [
        compute_tdp_baseline(
            tdp,
            str(method),
            aggregation=aggregation,
            fallback_strategy=DEFAULT_LOGPROB_FALLBACK_STRATEGY,
        )
        for tdp in tdps
    ]
    valid_tdp_scores = [float(score) for score in tdp_scores if isinstance(score, (int, float))]
    total_uncertainty = (sum(valid_tdp_scores) / len(valid_tdp_scores)) if valid_tdp_scores else None
    fallback_tdp_count = sum(
        1
        for tdp in tdps
        if tdp_uses_logprob_fallback(
            tdp,
            str(method),
            fallback_strategy=DEFAULT_LOGPROB_FALLBACK_STRATEGY,
        )
    )

    estimate["aggregation"] = aggregation
    estimate["tdp_step_scores"] = tdp_step_scores
    estimate["tdp_scores"] = tdp_scores
    estimate["total_uncertainty"] = total_uncertainty
    estimate_metadata.update(
        {
            "requires_logprobs": True,
            "uncertainty_available": total_uncertainty is not None,
            "uses_logprob_fallback": fallback_tdp_count > 0,
            "logprob_fallback_strategy": DEFAULT_LOGPROB_FALLBACK_STRATEGY,
            "logprob_fallback_tdp_count": fallback_tdp_count,
            "fallback_token_probability": FALLBACK_TOKEN_PROBABILITY if fallback_tdp_count > 0 else None,
        }
    )
    record["uncertainty"] = total_uncertainty


def _normalize_uncertainty_trace(values: Sequence[Any]) -> list[float]:
    return [value for item in values if (value := _coerce_uncertainty(item)) is not None]


def _extract_record_uncertainty_traces(record: dict[str, Any]) -> list[list[float]]:
    normalize_multi_trajectory_record(record)
    traces: list[list[float]] = []

    estimate = record.get("estimate")
    if isinstance(estimate, dict):
        step_estimates = estimate.get("step_estimates")
        if isinstance(step_estimates, list):
            trace = _normalize_uncertainty_trace(
                step.get("entropy") for step in step_estimates if isinstance(step, dict)
            )
            if trace:
                traces.append(trace)

        tdp_step_scores = estimate.get("tdp_step_scores")
        if isinstance(tdp_step_scores, list):
            for item in tdp_step_scores:
                if isinstance(item, list):
                    trace = _normalize_uncertainty_trace(item)
                    if trace:
                        traces.append(trace)
            if traces:
                return traces

        tdps = estimate.get("tdps")
        if isinstance(tdps, list):
            uncertainty_key = str(
                estimate.get("uncertainty_key")
                or (estimate.get("metadata") or {}).get("uncertainty_key")
                or "pe"
            )
            for tdp in tdps:
                if not isinstance(tdp, dict):
                    continue
                steps = tdp.get("steps")
                if not isinstance(steps, list):
                    continue
                trace: list[float] = []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    measurements = step.get("uncertainty_measurements") or {}
                    if isinstance(measurements, dict):
                        score = _coerce_uncertainty(measurements.get(uncertainty_key))
                        if score is not None:
                            trace.append(score)
                if trace:
                    traces.append(trace)
            if traces:
                return traces

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
        return traces

    metadata = trajectory.get("metadata")
    if isinstance(metadata, dict):
        trace = _normalize_uncertainty_trace(metadata.get("step_entropies") or [])
        if trace:
            traces.append(trace)
            return traces

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return traces
    trace = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_metadata = step.get("metadata")
        if not isinstance(step_metadata, dict):
            continue
        entropy = _coerce_uncertainty(step_metadata.get("pe"))
        if entropy is None:
            entropy = _coerce_uncertainty(step_metadata.get("estimated_entropy"))
        if entropy is not None:
            trace.append(entropy)
    if trace:
        traces.append(trace)
    return traces


def _aggregate_uncertainty_trace(trace: Sequence[float], aggregation: str) -> float | None:
    if not trace:
        return None
    if aggregation == "sum":
        return float(sum(trace))
    if aggregation == "mean":
        return float(sum(trace) / len(trace))
    return None


def _resolve_recomputed_uncertainty(record: dict[str, Any], aggregation: str) -> float | None:
    traces = _extract_record_uncertainty_traces(record)
    if not traces:
        return None
    aggregated = [
        score for trace in traces if (score := _aggregate_uncertainty_trace(trace, aggregation)) is not None
    ]
    if not aggregated:
        return None
    return float(sum(aggregated) / len(aggregated))


def resolve_record_uncertainty(record: dict[str, Any], *, uncertainty_aggregation: str = "native") -> float | None:
    normalize_multi_trajectory_record(record)

    if uncertainty_aggregation != "native":
        recomputed = _resolve_recomputed_uncertainty(record, uncertainty_aggregation)
        if recomputed is not None:
            return recomputed

    direct = _coerce_uncertainty(record.get("uncertainty"))
    if direct is not None:
        return direct

    recomputed_native = _resolve_recomputed_uncertainty(record, "sum")
    if recomputed_native is not None:
        return recomputed_native

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
        return None

    metadata = trajectory.get("metadata")
    if isinstance(metadata, dict):
        from_step_entropies = _sum_uncertainty_values(metadata.get("step_entropies"))
        if from_step_entropies is not None:
            return from_step_entropies

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
    step_entropies: list[float] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_metadata = step.get("metadata")
        if not isinstance(step_metadata, dict):
            continue
        entropy = _coerce_uncertainty(step_metadata.get("pe"))
        if entropy is None:
            entropy = _coerce_uncertainty(step_metadata.get("estimated_entropy"))
        if entropy is not None:
            step_entropies.append(entropy)
    if not step_entropies:
        return None
    return float(sum(step_entropies))


def _compute_average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        group_total = 0
        group_positives = 0
        while index < len(ranked) and ranked[index][0] == score:
            group_total += 1
            group_positives += ranked[index][1]
            index += 1
        true_positives += group_positives
        false_positives += group_total - group_positives
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def _compute_auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        group_total = 0
        group_positives = 0
        while index < len(ranked) and ranked[index][0] == score:
            group_total += 1
            group_positives += ranked[index][1]
            index += 1
        average_rank = rank + (group_total - 1) / 2
        positive_rank_sum += average_rank * group_positives
        rank += group_total
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _compute_best_threshold_accuracy(labels: list[int], uncertainties: list[float]) -> tuple[float | None, float | None]:
    if not labels or not uncertainties or len(labels) != len(uncertainties):
        return None, None

    unique_thresholds = sorted(set(uncertainties))
    candidates = [float("-inf"), *unique_thresholds]
    best_accuracy: float | None = None
    best_threshold: float | None = None
    for threshold in candidates:
        correct_predictions = 0
        for label, uncertainty in zip(labels, uncertainties):
            predicted_success = uncertainty <= threshold
            if predicted_success == bool(label):
                correct_predictions += 1
        accuracy = correct_predictions / len(labels)
        if best_accuracy is None or accuracy > best_accuracy or (accuracy == best_accuracy and threshold < (best_threshold if best_threshold is not None else float("inf"))):
            best_accuracy = accuracy
            best_threshold = threshold
    return best_accuracy, best_threshold


def compute_uncertainty_diagnostics(
    records: Sequence[dict[str, Any]],
    *,
    uncertainty_aggregation: str = "native",
) -> dict[str, Any]:
    uncertainties: list[float] = []
    correct_uncertainties: list[float] = []
    incorrect_uncertainties: list[float] = []

    for record in records:
        normalize_multi_trajectory_record(record)
        uncertainty = resolve_record_uncertainty(record, uncertainty_aggregation=uncertainty_aggregation)
        if uncertainty is None:
            continue
        uncertainties.append(uncertainty)
        correctness = _coerce_correctness(record.get("is_correct"))
        if correctness is True:
            correct_uncertainties.append(uncertainty)
        elif correctness is False:
            incorrect_uncertainties.append(uncertainty)

    mean_uncertainty = (sum(uncertainties) / len(uncertainties)) if uncertainties else None
    uncertainty_std = None
    if uncertainties:
        uncertainty_std = math.sqrt(sum((value - mean_uncertainty) ** 2 for value in uncertainties) / len(uncertainties))

    return {
        "mean_uncertainty": mean_uncertainty,
        "uncertainty_std": uncertainty_std,
        "correct_mean_uncertainty": (sum(correct_uncertainties) / len(correct_uncertainties)) if correct_uncertainties else None,
        "incorrect_mean_uncertainty": (sum(incorrect_uncertainties) / len(incorrect_uncertainties)) if incorrect_uncertainties else None,
    }


def compute_uq_success_metrics(
    records: Sequence[dict[str, Any]],
    *,
    uncertainty_aggregation: str = "native",
) -> dict[str, Any]:
    success_labels: list[int] = []
    failure_labels: list[int] = []
    uncertainties: list[float] = []
    for record in records:
        normalize_multi_trajectory_record(record)
        uncertainty = resolve_record_uncertainty(record, uncertainty_aggregation=uncertainty_aggregation)
        if uncertainty is None:
            continue
        # For UQ metrics, only an explicit True is a success. False and
        # missing/unknown correctness are both failures.
        is_success = _coerce_correctness(record.get("is_correct")) is True
        success_labels.append(1 if is_success else 0)
        failure_labels.append(0 if is_success else 1)
        uncertainties.append(uncertainty)

    uq_accuracy, uq_best_threshold = _compute_best_threshold_accuracy(success_labels, uncertainties)
    return {
        "uq_eval_samples": len(success_labels),
        "uq_auroc": _compute_auroc(failure_labels, uncertainties),
        "uq_aupr": _compute_average_precision(failure_labels, uncertainties),
        "uq_accuracy": uq_accuracy,
        "uq_best_threshold": uq_best_threshold,
    }


def compute_logprob_availability_metrics(
    records: Sequence[dict[str, Any]],
    *,
    default_method: str | None = None,
) -> dict[str, Any]:
    method = records[0].get("method", default_method) if records else default_method
    requires_logprobs = logprobs_required_for_method(method)
    available_records = 0
    unavailable_records = 0

    for record in records:
        normalize_multi_trajectory_record(record)
        estimate = record.get("estimate")
        metadata = estimate.get("metadata") if isinstance(estimate, dict) else None

        record_requires_logprobs = bool(isinstance(metadata, dict) and metadata.get("requires_logprobs")) or logprobs_required_for_method(
            record.get("method", method)
        )
        if record_requires_logprobs:
            requires_logprobs = True

        if _record_has_token_logprobs(record):
            available_records += 1
            continue

        if _record_has_explicit_unavailable_logprobs(record, requires_logprobs=record_requires_logprobs):
            unavailable_records += 1

    if unavailable_records > 0:
        return {
            "logprobs_required": requires_logprobs,
            "logprobs_available": False,
            "logprobs_status": "unavailable",
            "logprob_unavailable_records": unavailable_records,
        }
    if available_records > 0:
        return {
            "logprobs_required": requires_logprobs,
            "logprobs_available": True,
            "logprobs_status": "available",
            "logprob_unavailable_records": 0,
        }
    if not requires_logprobs:
        return {
            "logprobs_required": False,
            "logprobs_available": None,
            "logprobs_status": "not-recorded",
            "logprob_unavailable_records": 0,
        }
    return {
        "logprobs_required": True,
        "logprobs_available": None,
        "logprobs_status": "unknown",
        "logprob_unavailable_records": 0,
    }


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    default_method: str | None = None,
    empty_mean_uncertainty: float | None = 0.0,
    uncertainty_aggregation: str = "native",
) -> dict[str, Any]:
    total = len(records)
    method = records[0].get("method", default_method) if records else default_method
    if total == 0:
        return {
            "total": 0,
            "mean_uncertainty": empty_mean_uncertainty,
            "uncertainty_std": None,
            "correct_mean_uncertainty": None,
            "incorrect_mean_uncertainty": None,
            "success_rate": None,
            "accuracy": None,
            "method": method,
            "uncertainty_aggregation": uncertainty_aggregation,
            "uq_eval_samples": 0,
            "uq_auroc": None,
            "uq_aupr": None,
            "uq_accuracy": None,
            "uq_best_threshold": None,
            "logprob_requirement": method_logprob_requirement(method),
            "logprobs_required": logprobs_required_for_method(method),
            "logprobs_available": None,
            "logprobs_status": "unknown" if logprobs_required_for_method(method) else "not-recorded",
            "logprob_unavailable_records": 0,
        }

    diagnostics = compute_uncertainty_diagnostics(
        records,
        uncertainty_aggregation=uncertainty_aggregation,
    )
    correctness: list[bool] = []
    for record in records:
        normalize_multi_trajectory_record(record)
        normalized = _coerce_correctness(record.get("is_correct"))
        if normalized is not None:
            correctness.append(normalized)
    success_rate = (sum(1 for value in correctness if value) / len(correctness)) if correctness else None
    summary = {
        "total": total,
        "mean_uncertainty": diagnostics.get("mean_uncertainty"),
        "uncertainty_std": diagnostics.get("uncertainty_std"),
        "correct_mean_uncertainty": diagnostics.get("correct_mean_uncertainty"),
        "incorrect_mean_uncertainty": diagnostics.get("incorrect_mean_uncertainty"),
        "success_rate": success_rate,
        "accuracy": success_rate,
        "method": method,
        "uncertainty_aggregation": uncertainty_aggregation,
        "logprob_requirement": method_logprob_requirement(method),
    }
    summary.update(
        compute_uq_success_metrics(
            records,
            uncertainty_aggregation=uncertainty_aggregation,
        )
    )
    summary.update(compute_logprob_availability_metrics(records, default_method=default_method))
    if any("status" in record for record in records):
        summary["completed"] = sum(1 for record in records if record.get("status") == "completed")
    return summary
