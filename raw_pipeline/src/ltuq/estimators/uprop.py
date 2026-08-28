from __future__ import annotations

from typing import Any, Callable

from ..baselines.uprop import (
    compute_tdp_final_step_pmi_u_from_tdp,
    compute_tdp_uprop_diagnostics_from_tdp,
    compute_tdp_uprop_from_tdp,
    default_decision_distance,
)
from ..executors.base import BranchingExecutor
from ..trajectory import UPropEstimate


class UPropEstimator:
    """Estimate UProp from executor-produced TDP samples."""

    def __init__(
        self,
        trajectory_samples: int = 4,
        per_step_samples: int = 4,
        tau: float = 1.0,
        uncertainty_key: str = "pe",
        distance_fn: Callable[[str, str], float] = default_decision_distance,
        ratio_epsilon: float = 1e-6,
        ratio_cap: float | None = 10.0,
        intrinsic_cap: float | None = None,
        intrinsic_transform: str = "none",
        track_tdps: bool = True,
    ) -> None:
        if trajectory_samples <= 0:
            raise ValueError("trajectory_samples must be positive")
        if per_step_samples <= 0:
            raise ValueError("per_step_samples must be positive")
        if ratio_epsilon <= 0.0:
            raise ValueError("ratio_epsilon must be positive")
        if ratio_cap is not None and ratio_cap <= 0.0:
            raise ValueError("ratio_cap must be positive when provided")
        if intrinsic_cap is not None and intrinsic_cap <= 0.0:
            raise ValueError("intrinsic_cap must be positive when provided")
        if intrinsic_transform not in {"none", "log1p"}:
            raise ValueError("intrinsic_transform must be 'none' or 'log1p'")
        self.trajectory_samples = trajectory_samples
        self.per_step_samples = per_step_samples
        self.tau = tau
        self.uncertainty_key = uncertainty_key
        self.distance_fn = distance_fn
        self.ratio_epsilon = ratio_epsilon
        self.ratio_cap = ratio_cap
        self.intrinsic_cap = intrinsic_cap
        self.intrinsic_transform = intrinsic_transform
        self.track_tdps = track_tdps

    async def estimate(
        self,
        sample: Any,
        executor: BranchingExecutor,
    ) -> UPropEstimate:
        tdps = await executor.sample_tdps(
            sample=sample,
            num_trajectories=self.trajectory_samples,
            per_step_samples=self.per_step_samples,
            include_counterfactuals=False,
        )
        if not tdps:
            raise ValueError("UProp estimation requires at least one sampled TDP.")

        return self.estimate_from_tdps(tdps, executor_name=executor.name)

    def estimate_from_tdps(
        self,
        tdps: list[Any],
        *,
        executor_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> UPropEstimate:
        if not tdps:
            raise ValueError("UProp estimation requires at least one sampled TDP.")
        observed_per_step_samples = max(
            (
                len(step.sampled_decisions)
                for tdp in tdps
                for step in tdp.steps
            ),
            default=0,
        )
        tdp_scores = []
        final_step_pmi_u_tdp_scores = []
        raw_tdp_lengths = []
        effective_tdp_lengths = []
        excluded_forced_terminal_steps = 0
        for tdp in tdps:
            tdp_scores.append(
                compute_tdp_uprop_from_tdp(
                    tdp,
                    uncertainty_key=self.uncertainty_key,
                    tau=self.tau,
                    distance_fn=self.distance_fn,
                    ratio_epsilon=self.ratio_epsilon,
                    ratio_cap=self.ratio_cap,
                    intrinsic_cap=self.intrinsic_cap,
                    intrinsic_transform=self.intrinsic_transform,
                )
            )
            diagnostics = compute_tdp_uprop_diagnostics_from_tdp(
                tdp,
                uncertainty_key=self.uncertainty_key,
                tau=self.tau,
                distance_fn=self.distance_fn,
                intrinsic_cap=self.intrinsic_cap,
                intrinsic_transform=self.intrinsic_transform,
            )
            final_step_pmi_u_tdp_scores.append(
                compute_tdp_final_step_pmi_u_from_tdp(
                    tdp,
                    uncertainty_key=self.uncertainty_key,
                    tau=self.tau,
                    distance_fn=self.distance_fn,
                    intrinsic_cap=self.intrinsic_cap,
                    intrinsic_transform=self.intrinsic_transform,
                )
            )
            raw_tdp_lengths.append(diagnostics.raw_length)
            effective_tdp_lengths.append(diagnostics.effective_length)
            excluded_forced_terminal_steps += int(diagnostics.excluded_forced_terminal_step)

        final_step_pmi_u = sum(final_step_pmi_u_tdp_scores) / len(final_step_pmi_u_tdp_scores)
        resolved_metadata = dict(metadata or {})
        resolved_metadata.update(
            {
                "executor": executor_name,
                "num_tdps": len(tdps),
                "per_step_samples": observed_per_step_samples or self.per_step_samples,
                "requested_per_step_samples": self.per_step_samples,
                "observed_per_step_samples": observed_per_step_samples,
                "tau": self.tau,
                "ratio_epsilon": self.ratio_epsilon,
                "ratio_cap": self.ratio_cap,
                "intrinsic_cap": self.intrinsic_cap,
                "intrinsic_transform": self.intrinsic_transform,
                "final_step_pmi_u": final_step_pmi_u,
                "final_step_pmi_u_tdp_scores": final_step_pmi_u_tdp_scores,
                "raw_tdp_lengths": raw_tdp_lengths,
                "effective_tdp_lengths": effective_tdp_lengths,
                "excluded_forced_terminal_steps": excluded_forced_terminal_steps,
                "used_no_tool_call_samples": any(bool(tdp.metadata.get("used_no_tool_call_samples")) for tdp in tdps),
                "no_tool_call_sample_count": sum(
                    [count for tdp in tdps if isinstance((count := tdp.metadata.get("no_tool_call_sample_count")), int)]
                ),
                "hard_requirement_failure": any(bool(tdp.metadata.get("hard_requirement_failure")) for tdp in tdps),
            }
        )

        return UPropEstimate(
            total_uncertainty=sum(tdp_scores) / len(tdp_scores),
            uncertainty_key=self.uncertainty_key,
            final_step_pmi_u=final_step_pmi_u,
            tdp_scores=tdp_scores,
            final_step_pmi_u_tdp_scores=final_step_pmi_u_tdp_scores,
            raw_tdp_lengths=raw_tdp_lengths,
            effective_tdp_lengths=effective_tdp_lengths,
            excluded_forced_terminal_steps=excluded_forced_terminal_steps,
            tdps=list(tdps) if self.track_tdps else [],
            metadata=resolved_metadata,
        )
