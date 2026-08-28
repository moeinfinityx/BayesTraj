from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from ..baselines.multistep import compute_multitdp_baseline, compute_outcome_entropy
from ..estimators.uprop import UPropEstimator
from ..logprobs import compute_predictive_entropy_from_metadata
from ..trajectory import TDPStepRecord, TrajectoryDependentDecisionProcess, UPropEstimate


class InvalidPosthocCandidateBudgetError(ValueError):
    """Raised when a realized trajectory is reused at a smaller candidate budget."""


def tdp_from_mapping(value: Mapping[str, Any]) -> TrajectoryDependentDecisionProcess:
    """Deserialize the persisted, counterfactual-free TDP representation."""
    steps = []
    for raw_step in value.get("steps", []):
        if not isinstance(raw_step, Mapping):
            continue
        steps.append(
            TDPStepRecord(
                index=int(raw_step.get("index", len(steps))),
                realized_decision=str(raw_step.get("realized_decision", "")),
                sampled_decisions=[
                    str(item) for item in raw_step.get("sampled_decisions", [])
                ],
                uncertainty_measurements={
                    str(key): float(item)
                    for key, item in raw_step.get(
                        "uncertainty_measurements", {}
                    ).items()
                    if isinstance(item, (int, float)) and not isinstance(item, bool)
                },
                metadata=copy.deepcopy(dict(raw_step.get("metadata", {}))),
            )
        )
    return TrajectoryDependentDecisionProcess(
        sample_id=str(value.get("sample_id", "")),
        prompt=str(value.get("prompt", "")),
        steps=steps,
        final_answer=(
            str(value["final_answer"])
            if value.get("final_answer") is not None
            else None
        ),
        metadata=copy.deepcopy(dict(value.get("metadata", {}))),
    )


def tdps_from_record(record: Mapping[str, Any]) -> list[TrajectoryDependentDecisionProcess]:
    estimate = record.get("estimate")
    if not isinstance(estimate, Mapping):
        return []
    values = estimate.get("tdps")
    if not isinstance(values, list):
        return []
    return [tdp_from_mapping(value) for value in values if isinstance(value, Mapping)]


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _permutation(length: int, *, seed: int | None, parts: tuple[object, ...]) -> list[int]:
    order = list(range(length))
    if seed is not None:
        random.Random(_stable_seed(seed, *parts)).shuffle(order)
    return order


def nested_candidate_indices(
    source_n: int,
    *,
    n: int,
    permutation_seed: int | None,
    tdp_index: int,
    step_index: int,
    required_index: int | None = None,
) -> list[int]:
    """Return the candidate indices selected by the historical nested policy."""
    if source_n < 0 or n <= 0:
        raise ValueError("source_n must be non-negative and n must be positive")
    if required_index is not None and not 0 <= required_index < source_n:
        raise ValueError("required_index must identify a source candidate")
    order = _permutation(
        source_n,
        seed=permutation_seed,
        parts=("candidate", tdp_index, step_index),
    )
    if (
        required_index is not None
        and required_index not in order[: min(n, source_n)]
    ):
        order.remove(required_index)
        order.insert(0, required_index)
    return order[: min(n, source_n)]


def _subset_aligned_metadata(
    metadata: dict[str, Any],
    selected_order: list[int],
    source_count: int,
) -> dict[str, Any]:
    result = copy.deepcopy(metadata)
    for key in ("sampled_messages", "sampled_output_metadata"):
        values = result.get(key)
        if isinstance(values, list) and len(values) == source_count:
            result[key] = [
                copy.deepcopy(values[index])
                for index in selected_order
            ]
    chosen_index = result.get("chosen_output_index")
    if isinstance(chosen_index, int):
        if chosen_index in selected_order:
            result["chosen_output_index"] = selected_order.index(chosen_index)
        else:
            result.pop("chosen_output_index", None)
    result["nested_subset_candidate_indices"] = selected_order
    return result


def nested_tdp_subset(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    *,
    z: int,
    n: int,
    permutation_seed: int | None = None,
    allow_fixed_trajectory_candidate_subsampling: bool = False,
) -> list[TrajectoryDependentDecisionProcess]:
    """Return a deterministic nested trajectory prefix at the source N budget.

    Reducing ``Z`` is valid because each TDP is an independently realized complete
    rollout. Reducing ``N`` after execution is not: the retained candidates may
    omit the action that produced the downstream trajectory. Callers must provide
    trajectories generated at the requested ``n``.
    """
    if z <= 0 or n <= 0:
        raise ValueError("z and n must be positive")
    if z > len(tdps):
        raise ValueError(f"Requested z={z}, but only {len(tdps)} trajectories exist")

    trajectory_order = _permutation(
        len(tdps),
        seed=permutation_seed,
        parts=("trajectory",),
    )
    subset = []
    for original_tdp_index in trajectory_order[:z]:
        tdp = tdps[original_tdp_index]
        steps = []
        for step in tdp.steps:
            source_n = len(step.sampled_decisions)
            if source_n > n and not allow_fixed_trajectory_candidate_subsampling:
                raise InvalidPosthocCandidateBudgetError(
                    "Cannot evaluate a realized trajectory at a smaller candidate "
                    f"budget: tdp={tdp.sample_id!r}, step={step.index}, "
                    f"source_n={source_n}, requested_n={n}. Generate a true "
                    "budget-specific rollout instead."
                )
            if n == source_n:
                selected_order = list(range(source_n))
            else:
                chosen_index = step.metadata.get("chosen_output_index")
                if (
                    allow_fixed_trajectory_candidate_subsampling
                    and (
                        not isinstance(chosen_index, int)
                        or not 0 <= chosen_index < source_n
                    )
                ):
                    raise InvalidPosthocCandidateBudgetError(
                        "Fixed-trajectory candidate subsampling requires a valid "
                        f"chosen_output_index: tdp={tdp.sample_id!r}, "
                        f"step={step.index}, source_n={source_n}"
                    )
                selected_order = nested_candidate_indices(
                    source_n,
                    n=n,
                    permutation_seed=permutation_seed,
                    tdp_index=original_tdp_index,
                    step_index=step.index,
                    required_index=(
                        chosen_index
                        if allow_fixed_trajectory_candidate_subsampling
                        else None
                    ),
                )
            subset_metadata = _subset_aligned_metadata(
                step.metadata,
                selected_order,
                source_n,
            )
            subset_measurements = dict(step.uncertainty_measurements)
            if (
                allow_fixed_trajectory_candidate_subsampling
                and source_n > n
            ):
                sampled_output_metadata = subset_metadata.get(
                    "sampled_output_metadata"
                )
                if isinstance(sampled_output_metadata, list):
                    entropy, _ = compute_predictive_entropy_from_metadata(
                        sampled_output_metadata
                    )
                    if entropy is not None:
                        subset_measurements["pe"] = float(entropy)
            steps.append(
                replace(
                    step,
                    sampled_decisions=[
                        step.sampled_decisions[index]
                        for index in selected_order
                    ],
                    uncertainty_measurements=subset_measurements,
                    metadata=subset_metadata,
                )
            )
        metadata = copy.deepcopy(tdp.metadata)
        metadata.update(
            {
                "nested_subset_source_trajectory_index": original_tdp_index,
                "nested_subset_n": n,
                "nested_subset_permutation_seed": permutation_seed,
                "fixed_trajectory_candidate_subsampling": (
                    allow_fixed_trajectory_candidate_subsampling
                ),
            }
        )
        subset.append(replace(tdp, steps=steps, metadata=metadata))
    return subset


def recompute_declared_method_scores(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    *,
    z: int,
    n: int,
    permutation_seed: int | None = None,
    tau: float = 1.0,
    ratio_epsilon: float = 1e-6,
    ratio_cap: float | None = 10.0,
    intrinsic_cap: float | None = None,
    intrinsic_transform: str = "none",
    outcome_bucket_map: Mapping[str, str] | None = None,
    allow_fixed_trajectory_candidate_subsampling: bool = False,
) -> dict[str, float | None]:
    subset = nested_tdp_subset(
        tdps,
        z=z,
        n=n,
        permutation_seed=permutation_seed,
        allow_fixed_trajectory_candidate_subsampling=(
            allow_fixed_trajectory_candidate_subsampling
        ),
    )
    outcome_bucket_fn = _outcome_bucket_fn(outcome_bucket_map)
    propagated = UPropEstimator(
        trajectory_samples=z,
        per_step_samples=n,
        tau=tau,
        ratio_epsilon=ratio_epsilon,
        ratio_cap=ratio_cap,
        intrinsic_cap=intrinsic_cap,
        intrinsic_transform=intrinsic_transform,
    ).estimate_from_tdps(
        subset,
        executor_name="nested_max_budget_postprocess",
        metadata={
            "nested_subset_z": z,
            "nested_subset_n": n,
            "nested_subset_permutation_seed": permutation_seed,
            "postprocess_only": True,
        },
    )
    outcome_entropy, _ = compute_outcome_entropy(
        subset,
        **(
            {"outcome_bucket_fn": outcome_bucket_fn}
            if outcome_bucket_fn is not None
            else {}
        ),
    )
    scores: dict[str, float | None] = {
        "UProp": propagated.final_step_pmi_u,
        "OutcomeEntropy": outcome_entropy,
    }
    for display_name, baseline_name in (
        ("PE", "pe"),
        ("SentSAR", "sentsar"),
        ("DEG", "deg"),
        ("LS", "ls"),
        ("PPL", "ppl"),
        ("SE", "se"),
        ("SD", "sd"),
    ):
        scores[display_name] = compute_multitdp_baseline(
            subset,
            baseline_name,
            fallback_strategy=None,
        )
    return scores


def _outcome_bucket_fn(
    outcome_bucket_map: Mapping[str, str] | None,
) -> Callable[[TrajectoryDependentDecisionProcess], str] | None:
    if outcome_bucket_map is None:
        return None
    resolved = {str(key): str(value) for key, value in outcome_bucket_map.items()}

    def outcome_bucket(tdp: TrajectoryDependentDecisionProcess) -> str:
        try:
            return resolved[tdp.sample_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing aligned outcome bucket for TDP {tdp.sample_id!r}."
            ) from exc

    return outcome_bucket


def pool_coverage(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    *,
    required_z: int,
    required_n: int,
) -> dict[str, Any]:
    candidate_counts = [
        len(step.sampled_decisions)
        for tdp in tdps
        for step in tdp.steps
        if not step.metadata.get("forced_terminal_finish", False)
    ]
    excluded_forced_terminal_steps = sum(
        1
        for tdp in tdps
        for step in tdp.steps
        if step.metadata.get("forced_terminal_finish", False)
    )
    return {
        "trajectory_count": len(tdps),
        "required_z": required_z,
        "trajectory_complete": len(tdps) >= required_z,
        "step_count": len(candidate_counts),
        "excluded_forced_terminal_steps": excluded_forced_terminal_steps,
        "required_n": required_n,
        "minimum_candidates_per_step": min(candidate_counts) if candidate_counts else 0,
        "maximum_candidates_per_step": max(candidate_counts) if candidate_counts else 0,
        "candidate_complete": bool(candidate_counts)
        and all(count >= required_n for count in candidate_counts),
    }
