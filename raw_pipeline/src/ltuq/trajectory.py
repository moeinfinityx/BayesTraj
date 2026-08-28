from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    index: int
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackboneTrajectory:
    sample_id: str
    prompt: str
    steps: list[StepRecord] = field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalBranchHistory:
    step_index: int
    window_start: int
    prefix_steps: list[StepRecord] = field(default_factory=list)
    branched_steps: list[StepRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TDPCounterfactualBranch:
    source_decision: str
    target_sampled_decisions: list[str] = field(default_factory=list)
    target_sampled_output_metadata: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TDPCounterfactualRecord:
    source_step_index: int
    realized_source_decision: str
    branches: list[TDPCounterfactualBranch] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TDPStepRecord:
    index: int
    realized_decision: str
    sampled_decisions: list[str] = field(default_factory=list)
    uncertainty_measurements: dict[str, float] = field(default_factory=dict)
    counterfactual_records: list[TDPCounterfactualRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryDependentDecisionProcess:
    sample_id: str
    prompt: str
    steps: list[TDPStepRecord] = field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UPropEstimate:
    total_uncertainty: float
    uncertainty_key: str
    final_step_pmi_u: float | None = None
    tdp_scores: list[float] = field(default_factory=list)
    final_step_pmi_u_tdp_scores: list[float] = field(default_factory=list)
    raw_tdp_lengths: list[int] = field(default_factory=list)
    effective_tdp_lengths: list[int] = field(default_factory=list)
    excluded_forced_terminal_steps: int = 0
    tdps: list[TrajectoryDependentDecisionProcess] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiStepBaselineEstimate:
    total_uncertainty: float | None
    baseline_name: str
    aggregation: str
    tdp_scores: list[float | None] = field(default_factory=list)
    tdp_step_scores: list[list[float | None]] = field(default_factory=list)
    tdps: list[TrajectoryDependentDecisionProcess] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
