from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any

from ..trajectory import BackboneTrajectory, LocalBranchHistory, TrajectoryDependentDecisionProcess


class BranchingExecutor(ABC):
    """Executor boundary for applying Algorithm 1 to a concrete task family."""

    name: str

    @abstractmethod
    async def sample_backbone(self, sample: Any) -> BackboneTrajectory:
        """Run or load the main trajectory that local methods branch from."""
        raise NotImplementedError

    @abstractmethod
    async def backbone_step_entropies(
        self, sample: Any, backbone: BackboneTrajectory
    ) -> list[float]:
        """Return one uncertainty score for each realized backbone step."""
        raise NotImplementedError

    @abstractmethod
    async def sample_local_history(
        self,
        sample: Any,
        backbone: BackboneTrajectory,
        step_index: int,
        window: int,
        branch_index: int,
    ) -> LocalBranchHistory:
        """Recreate task state before a target step with a local branch applied."""
        raise NotImplementedError

    @abstractmethod
    async def estimate_next_step_entropy(
        self,
        sample: Any,
        history: LocalBranchHistory,
        step_index: int,
    ) -> float:
        """Estimate action uncertainty from the state captured in a local history."""
        raise NotImplementedError

    async def rollout_local_history_to_tdp(
        self,
        sample: Any,
        history: LocalBranchHistory,
        *,
        branch_index: int,
        rollout_horizon: int,
        per_step_samples: int = 1,
    ) -> TrajectoryDependentDecisionProcess:
        """Continue a local branch for a short horizon and return a TDP.

        Adaptive local-branch outcome methods use this hook after
        ``sample_local_history`` has restored/resampled the branch point. The
        returned TDP should include the branch history plus any newly executed
        rollout steps so outcome bucketing can inspect the full local future.
        Executors that do not support environment restoration can leave the
        default implementation in place.
        """
        del sample, history, branch_index, rollout_horizon, per_step_samples
        raise NotImplementedError("rollout_local_history_to_tdp is not implemented for this executor.")

    @abstractmethod
    async def sample_tdp(
        self,
        sample: Any,
        trajectory_index: int,
        per_step_samples: int,
        *,
        include_counterfactuals: bool = False,
    ) -> TrajectoryDependentDecisionProcess:
        """Sample one trajectory-dependent decision process for baseline estimators."""
        raise NotImplementedError

    async def sample_tdps(
        self,
        sample: Any,
        num_trajectories: int,
        per_step_samples: int,
        *,
        include_counterfactuals: bool = False,
    ) -> list[TrajectoryDependentDecisionProcess]:
        """Sample several TDPs by repeatedly calling the task-specific TDP sampler."""
        if os.getenv("LTUQ_PARALLEL_TDPS") == "1" and num_trajectories > 1:
            return list(
                await asyncio.gather(
                    *[
                        self.sample_tdp(
                            sample=sample,
                            trajectory_index=trajectory_index,
                            per_step_samples=per_step_samples,
                            include_counterfactuals=include_counterfactuals,
                        )
                        for trajectory_index in range(num_trajectories)
                    ]
                )
            )

        tdps: list[TrajectoryDependentDecisionProcess] = []
        for trajectory_index in range(num_trajectories):
            tdps.append(
                await self.sample_tdp(
                    sample=sample,
                    trajectory_index=trajectory_index,
                    per_step_samples=per_step_samples,
                    include_counterfactuals=include_counterfactuals,
                )
            )
        return tdps

    async def hard_finalize_tdp(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> str | None:
        """Extract a best-effort final answer for an unfinished trajectory.

        Paper scoring uses this hook when a trajectory reaches its step
        budget without an explicit final answer. The default implementation is
        conservative and returns ``None``; task-family executors can override it
        with a prompt that is appropriate for their environment.
        """
        del sample, tdp
        return None

    def should_hard_finalize_tdp(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> bool:
        """Return whether the generator should ask for a final answer.

        The default only finalizes trajectories that have no submitted answer.
        Task-family executors can override this when they know an existing
        answer has the wrong format for the official evaluator.
        """
        del sample
        return not (isinstance(tdp.final_answer, str) and tdp.final_answer.strip())
