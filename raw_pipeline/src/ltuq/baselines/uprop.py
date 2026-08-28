from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, cast
import math

import numpy as np
from thefuzz import fuzz

from ..trajectory import TDPCounterfactualRecord, TrajectoryDependentDecisionProcess


_MIN_POSITIVE_UNCERTAINTY = 1e-12
_DEFAULT_RATIO_EPSILON = 1e-6
_DEFAULT_RATIO_CAP = 10.0
_DEFAULT_INTRINSIC_TRANSFORM = "none"
_FORCED_FINALIZATION_TEXT = "reached the maximum rollout length"
_OFFICIAL_CHECKER_TEXT = "official environment checker"
_COUNTERFACTUALS_KEY = "counterfactuals"
_REALIZED_DECISIONS_KEY = "realized_decisions"
_SAMPLED_DECISIONS_KEY = "sampled_decisions"
@dataclass(frozen=True)
class UPropTDPDiagnostics:
    raw_length: int
    effective_length: int
    excluded_forced_terminal_step: bool
    intrinsic_trace: list[float]
    extrinsic_trace: list[float]
    final_step_pmi_u: float


def _is_nested_sequence(values: Any) -> bool:
    if isinstance(values, np.ndarray):
        return values.ndim > 1
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    if len(values) == 0:
        return False
    first = values[0]
    return isinstance(first, (Sequence, np.ndarray)) and not np.isscalar(first)


def k_tau(values: Sequence[float] | np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Unit-bounded Gaussian kernel used by the Equation 8 neighborhood estimator."""
    array = np.asarray(values, dtype=float)
    return np.exp(-0.5 * tau * np.square(array))


def default_decision_distance(anchor: str, candidate: str) -> float:
    """Paper-style string fuzzy-matching distance for short agent decisions."""
    normalized_anchor = anchor.strip().lower()
    normalized_candidate = candidate.strip().lower()
    similarity = fuzz.ratio(normalized_anchor, normalized_candidate) / 100.0
    return float(max(0.0, 1.0 - similarity))


def _is_forced_terminal_metadata(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get("forced_terminal_finish")) or isinstance(metadata.get("forced_terminal_finish_reason"), str)


def _status_indicates_task_limit(status: Any) -> bool:
    return isinstance(status, str) and "task" in status.lower() and "limit" in status.lower()


def _result_indicates_task_limit(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    status = result.get("status") or result.get("state") or result.get("error")
    reason = result.get("reason") or result.get("message")
    return _status_indicates_task_limit(status) or _status_indicates_task_limit(reason)


def _is_synthetic_terminal_metadata(metadata: Mapping[str, Any] | None, tdp_metadata: Mapping[str, Any] | None = None) -> bool:
    if _is_forced_terminal_metadata(metadata):
        return True
    if not isinstance(metadata, Mapping):
        metadata = {}
    if isinstance(tdp_metadata, Mapping) and (
        _status_indicates_task_limit(tdp_metadata.get("status"))
        or _result_indicates_task_limit(tdp_metadata.get("result"))
    ):
        marker = (
            metadata.get("synthetic_terminal_step")
            or metadata.get("forced_terminal_step")
            or metadata.get("task_limit_terminal_step")
        )
        return bool(marker)
    return bool(metadata.get("synthetic_terminal_step") or metadata.get("task_limit_terminal_step"))


def _is_forced_terminal_decision(decision: str) -> bool:
    normalized = decision.lower()
    return (
        "finish_action" in normalized
        and _FORCED_FINALIZATION_TEXT in normalized
        and _OFFICIAL_CHECKER_TEXT in normalized
    )


def _is_synthetic_terminal_decision(decision: str, tdp_metadata: Mapping[str, Any] | None = None) -> bool:
    if _is_forced_terminal_decision(decision):
        return True
    if not isinstance(tdp_metadata, Mapping):
        return False
    normalized = decision.lower()
    return (
        _status_indicates_task_limit(tdp_metadata.get("status"))
        and "finish_action" in normalized
        and (
            "task limit" in normalized
            or "maximum rollout" in normalized
            or "max step" in normalized
            or "max_steps" in normalized
        )
    )


def _effective_trace_length(
    uncertainty_trace: Sequence[float],
    realized_decisions: Sequence[str],
    step_metadata: Sequence[Mapping[str, Any]] | None = None,
    tdp_metadata: Mapping[str, Any] | None = None,
) -> int:
    """Drop one synthetic terminal action from the scoring horizon."""
    trace_length = len(uncertainty_trace)
    if trace_length <= 1:
        return trace_length
    last_index = trace_length - 1
    last_metadata = (
        step_metadata[last_index]
        if step_metadata is not None and len(step_metadata) == trace_length
        else None
    )
    if _is_synthetic_terminal_metadata(last_metadata, tdp_metadata) or _is_synthetic_terminal_decision(str(realized_decisions[-1]), tdp_metadata):
        return trace_length - 1
    return trace_length


def transform_intrinsic_uncertainty(
    uncertainty: float,
    *,
    cap: float | None = None,
    transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> float:
    value = max(0.0, float(uncertainty))
    if transform == "none":
        transformed = value
    elif transform == "log1p":
        transformed = math.log1p(value)
    else:
        raise ValueError("intrinsic_transform must be 'none' or 'log1p'.")
    if cap is not None:
        if cap <= 0.0:
            raise ValueError("intrinsic_cap must be positive when provided.")
        transformed = min(transformed, float(cap))
    return float(transformed)


def _resolve_intrinsic_trace(
    uncertainty_trace: Sequence[float],
    *,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> list[float]:
    return [
        transform_intrinsic_uncertainty(
            float(uncertainty),
            cap=intrinsic_cap,
            transform=intrinsic_transform,
        )
        for uncertainty in uncertainty_trace
    ]


def build_tdp_uncertainty_trace(
    tdp: TrajectoryDependentDecisionProcess,
    uncertainty_key: str = "pe",
) -> list[float]:
    """Extract one intrinsic-uncertainty trace from a typed TDP."""
    uncertainty_trace: list[float] = []
    for step in tdp.steps:
        if uncertainty_key not in step.uncertainty_measurements:
            raise ValueError(
                f"TDP step {step.index} is missing the '{uncertainty_key}' uncertainty measurement."
            )
        uncertainty_trace.append(float(step.uncertainty_measurements[uncertainty_key]))
    return uncertainty_trace


def build_tdp_realized_decisions(tdp: TrajectoryDependentDecisionProcess) -> list[str]:
    return [step.realized_decision for step in tdp.steps]


def build_tdp_sampled_decisions(tdp: TrajectoryDependentDecisionProcess) -> list[list[str]]:
    return [list(step.sampled_decisions) for step in tdp.steps]


def build_tdp_counterfactual_records(
    tdp: TrajectoryDependentDecisionProcess,
) -> list[list[TDPCounterfactualRecord]]:
    """Extract legacy counterfactual records without requiring replay-populated TDPs."""
    stepwise_records: list[list[TDPCounterfactualRecord]] = []
    for step in tdp.steps:
        records = list(step.counterfactual_records)
        if not records:
            stepwise_records.append([])
            continue
        expected_source_indices = set(range(1, step.index))
        actual_source_indices = {record.source_step_index for record in records}
        if actual_source_indices != expected_source_indices:
            raise ValueError(
                f"TDP step {step.index} is missing faithful counterfactual records for source steps "
                f"{sorted(expected_source_indices - actual_source_indices)}."
            )
        stepwise_records.append(records)
    return stepwise_records


def build_tdp_measurements(
    tdp: TrajectoryDependentDecisionProcess,
    uncertainty_key: str = "pe",
) -> dict[str, list[Any]]:
    """Convert one typed TDP into the measurement layout consumed by UProp."""
    uncertainty_trace = build_tdp_uncertainty_trace(tdp, uncertainty_key=uncertainty_key)
    return {
        uncertainty_key: uncertainty_trace,
        _REALIZED_DECISIONS_KEY: build_tdp_realized_decisions(tdp),
        _SAMPLED_DECISIONS_KEY: build_tdp_sampled_decisions(tdp),
    }


def build_tdps_measurements(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    uncertainty_key: str = "pe",
) -> dict[str, list[Any]]:
    """Convert multiple typed TDPs into the multi-TDP layout consumed by UProp."""
    return {
        uncertainty_key: [
            build_tdp_uncertainty_trace(tdp, uncertainty_key=uncertainty_key) for tdp in tdps
        ],
        _REALIZED_DECISIONS_KEY: [build_tdp_realized_decisions(tdp) for tdp in tdps],
        _SAMPLED_DECISIONS_KEY: [build_tdp_sampled_decisions(tdp) for tdp in tdps],
    }


def pmi_hat(
    realized_source_decision: str,
    sampled_source_decisions: Sequence[str],
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
) -> float:
    """Approximate one propagated PMI term from direct TDP source-step samples."""
    if not sampled_source_decisions:
        return 0.0

    distances = np.asarray(
        [distance_fn(realized_source_decision, sampled_decision) for sampled_decision in sampled_source_decisions],
        dtype=float,
    )
    neighborhood_mass = float(np.mean(k_tau(distances, tau=tau)))
    clipped_mass = min(1.0, max(_MIN_POSITIVE_UNCERTAINTY, neighborhood_mass))
    return float(-math.log(clipped_mass))


def build_stepwise_extrinsic_uncertainty(
    realized_decisions: Sequence[str],
    sampled_decisions: Sequence[Sequence[str]],
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
) -> list[float]:
    """Resolve Equation 9 per-step extrinsic uncertainty from direct TDP samples."""
    if len(realized_decisions) != len(sampled_decisions):
        raise ValueError("UProp expects one sampled-decision collection per realized decision.")

    source_pmis = [
        pmi_hat(
            realized_decision,
            source_sampled_decisions,
            tau=tau,
            distance_fn=distance_fn,
        )
        for realized_decision, source_sampled_decisions in zip(realized_decisions, sampled_decisions, strict=True)
    ]

    stepwise_extrinsic: list[float] = []
    running_extrinsic = 0.0
    for target_index in range(len(realized_decisions)):
        stepwise_extrinsic.append(float(running_extrinsic))
        running_extrinsic += source_pmis[target_index]
    return stepwise_extrinsic


def compute_tdp_uprop(
    uncertainty_trace: Sequence[float],
    realized_decisions: Sequence[str],
    sampled_decisions: Sequence[Sequence[str]],
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    ratio_epsilon: float = _DEFAULT_RATIO_EPSILON,
    ratio_cap: float | None = _DEFAULT_RATIO_CAP,
    step_metadata: Sequence[Mapping[str, Any]] | None = None,
    tdp_metadata: Mapping[str, Any] | None = None,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> float:
    """Compute the paper's normalized UProp score for one TDP.

    This matches Equation 9 when intrinsic uncertainty is provided as a per-step
    trace and extrinsic uncertainty is derived from direct TDP step samples.
    """
    if len(uncertainty_trace) != len(realized_decisions) or len(uncertainty_trace) != len(sampled_decisions):
        raise ValueError("UProp expects aligned uncertainty, decision, and sampled-decision traces.")
    if ratio_epsilon <= 0.0:
        raise ValueError("ratio_epsilon must be positive.")
    if ratio_cap is not None and ratio_cap <= 0.0:
        raise ValueError("ratio_cap must be positive when provided.")
    if not uncertainty_trace:
        return 0.0
    diagnostics = compute_tdp_uprop_diagnostics(
        uncertainty_trace,
        realized_decisions,
        sampled_decisions,
        tau=tau,
        distance_fn=distance_fn,
        step_metadata=step_metadata,
        tdp_metadata=tdp_metadata,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    )
    numerator = 0.0
    lambda_z = 0.0

    for uncertainty, extrinsic in zip(diagnostics.intrinsic_trace, diagnostics.extrinsic_trace, strict=True):
        resolved_uncertainty = max(float(uncertainty), _MIN_POSITIVE_UNCERTAINTY)
        numerator += resolved_uncertainty + extrinsic
        extrinsic_ratio = extrinsic / max(resolved_uncertainty, ratio_epsilon)
        if ratio_cap is not None:
            extrinsic_ratio = min(extrinsic_ratio, ratio_cap)
        lambda_z += 1.0 + extrinsic_ratio

    return float(numerator / lambda_z)


def compute_tdp_uprop_diagnostics(
    uncertainty_trace: Sequence[float],
    realized_decisions: Sequence[str],
    sampled_decisions: Sequence[Sequence[str]],
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    step_metadata: Sequence[Mapping[str, Any]] | None = None,
    tdp_metadata: Mapping[str, Any] | None = None,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> UPropTDPDiagnostics:
    """Return effective traces and the submitted final-step UProp score."""
    if len(uncertainty_trace) != len(realized_decisions) or len(uncertainty_trace) != len(sampled_decisions):
        raise ValueError("UProp expects aligned uncertainty, decision, and sampled-decision traces.")
    if not uncertainty_trace:
        return UPropTDPDiagnostics(
            raw_length=0,
            effective_length=0,
            excluded_forced_terminal_step=False,
            intrinsic_trace=[],
            extrinsic_trace=[],
            final_step_pmi_u=0.0,
        )

    effective_length = _effective_trace_length(
        uncertainty_trace,
        realized_decisions,
        step_metadata=step_metadata,
        tdp_metadata=tdp_metadata,
    )
    if effective_length == 0:
        intrinsic_trace: list[float] = []
        stepwise_extrinsic: list[float] = []
        outcomes: list[float] = []
    else:
        intrinsic_trace = _resolve_intrinsic_trace(
            uncertainty_trace[:effective_length],
            intrinsic_cap=intrinsic_cap,
            intrinsic_transform=intrinsic_transform,
        )
        effective_realized_decisions = realized_decisions[:effective_length]
        effective_sampled_decisions = sampled_decisions[:effective_length]
        stepwise_extrinsic = build_stepwise_extrinsic_uncertainty(
            effective_realized_decisions,
            effective_sampled_decisions,
            tau=tau,
            distance_fn=distance_fn,
        )
        outcomes = [intrinsic + extrinsic for intrinsic, extrinsic in zip(intrinsic_trace, stepwise_extrinsic, strict=True)]

    return UPropTDPDiagnostics(
        raw_length=len(uncertainty_trace),
        effective_length=effective_length,
        excluded_forced_terminal_step=effective_length < len(uncertainty_trace),
        intrinsic_trace=intrinsic_trace,
        extrinsic_trace=[float(value) for value in stepwise_extrinsic],
        final_step_pmi_u=float(outcomes[-1]) if outcomes else 0.0,
    )


def compute_tdp_final_step_pmi_u(
    uncertainty_trace: Sequence[float],
    realized_decisions: Sequence[str],
    sampled_decisions: Sequence[Sequence[str]],
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    step_metadata: Sequence[Mapping[str, Any]] | None = None,
    tdp_metadata: Mapping[str, Any] | None = None,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> float:
    """Compute the final-step PMI uncertainty for one TDP.

    This is H_T(TDP) = IU_T + EU_T, where T is the final realized step,
    IU_T is the final-step intrinsic uncertainty, and EU_T is the accumulated
    propagated uncertainty from preceding steps.
    """
    return compute_tdp_uprop_diagnostics(
        uncertainty_trace,
        realized_decisions,
        sampled_decisions,
        tau=tau,
        distance_fn=distance_fn,
        step_metadata=step_metadata,
        tdp_metadata=tdp_metadata,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    ).final_step_pmi_u


def compute_uprop(
    uncertainty_trace: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    realized_decisions: Sequence[str] | Sequence[Sequence[str]],
    sampled_decisions: Sequence[Any],
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    ratio_epsilon: float = _DEFAULT_RATIO_EPSILON,
    ratio_cap: float | None = _DEFAULT_RATIO_CAP,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> float:
    """Compute UProp for either one TDP or an average across multiple TDPs."""
    if len(uncertainty_trace) == 0:
        return 0.0

    if _is_nested_sequence(uncertainty_trace):
        multi_trace = cast(Sequence[Sequence[float]], uncertainty_trace)
        multi_decisions = cast(Sequence[Sequence[str]], realized_decisions)
        if len(uncertainty_trace) != len(sampled_decisions) or len(uncertainty_trace) != len(multi_decisions):
            raise ValueError("UProp expects one realized-decision and sampled-decision collection per TDP trace.")
        tdp_scores = [
            compute_tdp_uprop(
                single_trace,
                single_decisions,
                single_sampled_decisions,
                tau=tau,
                distance_fn=distance_fn,
                ratio_epsilon=ratio_epsilon,
                ratio_cap=ratio_cap,
                intrinsic_cap=intrinsic_cap,
                intrinsic_transform=intrinsic_transform,
            )
            for single_trace, single_decisions, single_sampled_decisions in zip(
                multi_trace, multi_decisions, sampled_decisions, strict=True
            )
        ]
        return float(np.mean(tdp_scores)) if tdp_scores else 0.0

    single_trace = cast(Sequence[float], uncertainty_trace)
    single_decisions = cast(Sequence[str], realized_decisions)
    return compute_tdp_uprop(
        single_trace,
        single_decisions,
        sampled_decisions,
        tau=tau,
        distance_fn=distance_fn,
        ratio_epsilon=ratio_epsilon,
        ratio_cap=ratio_cap,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    )


def compute_final_step_pmi_u(
    uncertainty_trace: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    realized_decisions: Sequence[str] | Sequence[Sequence[str]],
    sampled_decisions: Sequence[Any],
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> float:
    """Compute final_step_pmi_u for either one TDP or an average across TDPs."""
    if len(uncertainty_trace) == 0:
        return 0.0

    if _is_nested_sequence(uncertainty_trace):
        multi_trace = cast(Sequence[Sequence[float]], uncertainty_trace)
        multi_decisions = cast(Sequence[Sequence[str]], realized_decisions)
        if len(uncertainty_trace) != len(sampled_decisions) or len(uncertainty_trace) != len(multi_decisions):
            raise ValueError("UProp expects one realized-decision and sampled-decision collection per TDP trace.")
        tdp_scores = [
            compute_tdp_final_step_pmi_u(
                single_trace,
                single_decisions,
                single_sampled_decisions,
                tau=tau,
                distance_fn=distance_fn,
                intrinsic_cap=intrinsic_cap,
                intrinsic_transform=intrinsic_transform,
            )
            for single_trace, single_decisions, single_sampled_decisions in zip(
                multi_trace, multi_decisions, sampled_decisions, strict=True
            )
        ]
        return float(np.mean(tdp_scores)) if tdp_scores else 0.0

    single_trace = cast(Sequence[float], uncertainty_trace)
    single_decisions = cast(Sequence[str], realized_decisions)
    return compute_tdp_final_step_pmi_u(
        single_trace,
        single_decisions,
        sampled_decisions,
        tau=tau,
        distance_fn=distance_fn,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    )


def compute_tdp_uprop_from_tdp(
    tdp: TrajectoryDependentDecisionProcess,
    uncertainty_key: str = "pe",
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    ratio_epsilon: float = _DEFAULT_RATIO_EPSILON,
    ratio_cap: float | None = _DEFAULT_RATIO_CAP,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> float:
    measurements = build_tdp_measurements(tdp, uncertainty_key=uncertainty_key)
    step_metadata = [step.metadata for step in tdp.steps]
    return compute_tdp_uprop(
        measurements[uncertainty_key],
        measurements[_REALIZED_DECISIONS_KEY],
        measurements[_SAMPLED_DECISIONS_KEY],
        tau=tau,
        distance_fn=distance_fn,
        ratio_epsilon=ratio_epsilon,
        ratio_cap=ratio_cap,
        step_metadata=step_metadata,
        tdp_metadata=tdp.metadata,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    )


def compute_tdp_final_step_pmi_u_from_tdp(
    tdp: TrajectoryDependentDecisionProcess,
    uncertainty_key: str = "pe",
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> float:
    measurements = build_tdp_measurements(tdp, uncertainty_key=uncertainty_key)
    step_metadata = [step.metadata for step in tdp.steps]
    return compute_tdp_final_step_pmi_u(
        measurements[uncertainty_key],
        measurements[_REALIZED_DECISIONS_KEY],
        measurements[_SAMPLED_DECISIONS_KEY],
        tau=tau,
        distance_fn=distance_fn,
        step_metadata=step_metadata,
        tdp_metadata=tdp.metadata,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    )


def compute_tdp_uprop_diagnostics_from_tdp(
    tdp: TrajectoryDependentDecisionProcess,
    uncertainty_key: str = "pe",
    tau: float = 1.0,
    distance_fn: Callable[[str, str], float] = default_decision_distance,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = _DEFAULT_INTRINSIC_TRANSFORM,
) -> UPropTDPDiagnostics:
    measurements = build_tdp_measurements(tdp, uncertainty_key=uncertainty_key)
    return compute_tdp_uprop_diagnostics(
        measurements[uncertainty_key],
        measurements[_REALIZED_DECISIONS_KEY],
        measurements[_SAMPLED_DECISIONS_KEY],
        tau=tau,
        distance_fn=distance_fn,
        step_metadata=[step.metadata for step in tdp.steps],
        tdp_metadata=tdp.metadata,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    )


K_tau = k_tau
PMI_hat = pmi_hat
