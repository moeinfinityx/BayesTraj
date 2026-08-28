"""Label-free SAUP trajectory-uncertainty baseline reported in BayesTraj.

This module implements the distance surrogate specified in Algorithm 1 of
Zhao et al. (ACL 2025). It intentionally excludes unreported SAUP variants.

Paper: https://aclanthology.org/2025.acl-long.302/
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np

from ..trajectory import TDPStepRecord, TrajectoryDependentDecisionProcess


SAUP_BASELINE = "saup"
SAUP_FORMULA_VERSION = "ltuq-saup-v1"
AGENT_TRACER_GENERIC_DISTANCE_SAUP_VERSION = (
    "agent-tracer-dba3de264c886aa98af1fa2779dd49f7aed8cf92-lines-1048-1156"
)
SAUP_NORMALIZED_ENTROPY_KEY = "saup-normalized-entropy"
SAUP_ROBERTA_MODEL = "deepset/roberta-base-squad2"
SAUP_ROBERTA_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
SAUP_ROBERTA_POOLING = "attention-mask-mean-last-hidden-state"

TextDistance = Callable[[str, str], float]


class RobertaSquadSemanticDistance:
    """Lazy, cached semantic-distance operationalization for SAUP.

    The paper identifies a RoBERTa model fine-tuned on SQuAD v2, but does not
    identify a checkpoint, pooling rule, or distance normalization.  We pin a
    public checkpoint, mean-pool its last hidden state, and use clipped cosine
    distance.  These choices are provenance, not claims about unpublished
    author code.
    """

    model_name = SAUP_ROBERTA_MODEL
    model_revision = SAUP_ROBERTA_REVISION
    pooling = SAUP_ROBERTA_POOLING

    def __init__(self, *, device: str = "cpu", max_length: int = 512) -> None:
        if max_length <= 0:
            raise ValueError("SAUP semantic-distance max_length must be positive")
        self.device = device
        self.max_length = int(max_length)
        self._tokenizer: Any = None
        self._model: Any = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.model_revision,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                revision=self.model_revision,
            ).to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    def _embedding(self, text: str) -> np.ndarray:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        import torch

        tokenizer, model = self._load()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vector = pooled[0].detach().float().cpu().numpy()
        norm = float(np.linalg.norm(vector))
        resolved = vector / norm if norm > 0.0 else vector
        self._embedding_cache[text] = resolved
        return resolved

    def precompute(self, texts: Sequence[str], *, batch_size: int = 16) -> None:
        """Batch-encode uncached texts for efficient offline artifact scoring."""
        if batch_size <= 0:
            raise ValueError("SAUP semantic-distance batch_size must be positive")
        pending = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        if not pending:
            return

        import torch

        tokenizer, model = self._load()
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            vectors = pooled.detach().float().cpu().numpy()
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
            self._embedding_cache.update(zip(batch, vectors, strict=True))

    def __call__(self, left: str, right: str) -> float:
        left_vector = self._embedding(left)
        right_vector = self._embedding(right)
        similarity = float(np.dot(left_vector, right_vector))
        return float(np.clip(1.0 - similarity, 0.0, 2.0))

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": type(self).__name__,
            "model": self.model_name,
            "revision": self.model_revision,
            "pooling": self.pooling,
            "distance": "clipped-cosine-distance",
            "device": self.device,
            "max_length": self.max_length,
            "paper_checkpoint_and_pooling_specified": False,
        }


@dataclass(frozen=True)
class SAUPStepDiagnostics:
    """All inputs and intermediate values for one SAUP trajectory step."""

    step_index: int
    uncertainty: float | None
    inquiry_drift: float
    inference_gap: float
    position_weight: float
    distance_weight: float
    situational_weight: float
    weighted_uncertainty: float | None
    thought_action: str
    observation: str
    cumulative_state: str


@dataclass(frozen=True)
class SAUPDiagnostics:
    """Auditable result of one SAUP baseline calculation."""

    baseline_name: str
    score: float | None
    uncertainty_key: str
    position_mix: float | None
    steps: tuple[SAUPStepDiagnostics, ...]
    missing_uncertainty_steps: tuple[int, ...]
    missing_observation_steps: tuple[int, ...]
    formula_version: str = SAUP_FORMULA_VERSION
    pd_specification_status: str | None = None


def validate_saup_baseline_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized == SAUP_BASELINE:
        return SAUP_BASELINE
    raise ValueError(f"Unsupported SAUP baseline '{name}'. Expected '{SAUP_BASELINE}'.")


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def _step_thought_action(step: TDPStepRecord) -> str:
    thought = _coerce_text(step.metadata.get("thought"))
    action = _coerce_text(step.metadata.get("action")) or step.realized_decision.strip()
    if thought and action and thought not in action:
        return f"Thought: {thought}\nAction: {action}"
    return action or thought


def _step_observation(step: TDPStepRecord) -> str:
    observation = _coerce_text(step.metadata.get("observation"))
    if observation:
        return observation
    return _coerce_text(step.metadata.get("raw_observation_messages"))


def _step_uncertainty(step: TDPStepRecord, uncertainty_key: str) -> float | None:
    if uncertainty_key == SAUP_NORMALIZED_ENTROPY_KEY:
        metadata = step.metadata.get("chosen_output_metadata")
        if not isinstance(metadata, Mapping):
            return None
        components: list[float] = []
        for prefix in ("reasoning", "decision"):
            logprob_sum = metadata.get(f"{prefix}_token_logprob_sum")
            token_count = metadata.get(f"{prefix}_token_count")
            if (
                isinstance(logprob_sum, (int, float))
                and not isinstance(logprob_sum, bool)
                and isinstance(token_count, int)
                and not isinstance(token_count, bool)
                and token_count > 0
                and math.isfinite(float(logprob_sum))
            ):
                components.append(-float(logprob_sum) / float(token_count))
            else:
                return None
        return float(sum(components))
    value = step.uncertainty_measurements.get(uncertainty_key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        resolved = float(value)
        if math.isfinite(resolved) and resolved >= 0.0:
            return resolved
    return None


def _checked_distance(distance_fn: TextDistance, left: str, right: str) -> float:
    value = float(distance_fn(left, right))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("SAUP semantic distances must be finite and non-negative")
    return value


def _trajectory_components(
    tdp: TrajectoryDependentDecisionProcess,
    *,
    distance_fn: TextDistance,
    uncertainty_key: str,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    cumulative_entries: list[str] = []
    components: list[dict[str, Any]] = []
    missing_uncertainties: list[int] = []
    missing_observations: list[int] = []

    for position, step in enumerate(tdp.steps, start=1):
        thought_action = _step_thought_action(step)
        observation = _step_observation(step)
        entry = f"Step {position}\n{thought_action}"
        if observation:
            entry += f"\nObservation: {observation}"
        cumulative_entries.append(entry)
        cumulative_state = "\n\n".join(cumulative_entries)

        uncertainty = _step_uncertainty(step, uncertainty_key)
        if uncertainty is None:
            missing_uncertainties.append(step.index)
        inquiry_drift = _checked_distance(distance_fn, cumulative_state, tdp.prompt)
        if observation:
            inference_gap = _checked_distance(distance_fn, thought_action, observation)
        else:
            # Some environments have a terminal action with no returned
            # observation.  It contributes no local action-observation gap,
            # while still contributing inquiry drift and step uncertainty.
            inference_gap = 0.0
            missing_observations.append(step.index)

        components.append(
            {
                "step": step,
                "uncertainty": uncertainty,
                "inquiry_drift": inquiry_drift,
                "inference_gap": inference_gap,
                "distance_weight": inquiry_drift + inference_gap,
                "thought_action": thought_action,
                "observation": observation,
                "cumulative_state": cumulative_state,
            }
        )
    return components, missing_uncertainties, missing_observations


def compute_saup_diagnostics(
    tdp: TrajectoryDependentDecisionProcess,
    baseline_name: str,
    *,
    distance_fn: TextDistance,
    uncertainty_key: str = SAUP_NORMALIZED_ENTROPY_KEY,
) -> SAUPDiagnostics:
    """Compute the pinned, paper-reported SAUP implementation.

    The agent-tracer generic distance implementation uses::

        W_i = 1 + alpha * D_a,i + beta * D_o-agent,i + gamma * D_o-user,i
        U(tau) = sqrt(mean((W_i * U_i) ** 2))

    Here alpha=beta=gamma=1, the stored action-observation inference gap is
    mapped to ``D_o-agent``, and ``D_o-user`` is zero because LTUQ's DBBench
    step records do not contain a distinct user-response coherence channel.

    A score is unavailable when any step lacks the requested uncertainty;
    silently dropping such steps would change both the trajectory and RMS
    denominator.  Empty trajectories likewise return ``None``.
    """

    resolved_name = validate_saup_baseline_name(baseline_name)
    components, missing_uncertainties, missing_observations = _trajectory_components(
        tdp,
        distance_fn=distance_fn,
        uncertainty_key=uncertainty_key,
                )
    total_steps = len(components)
    step_diagnostics: list[SAUPStepDiagnostics] = []
    weighted_uncertainties: list[float] = []
    for position, component in enumerate(components, start=1):
        position_weight = float(position / total_steps) if total_steps else 0.0
        distance_weight = float(component["distance_weight"])
        # Exact upstream generic aggregation with alpha=beta=gamma=1.
        # DBBench has no distinct do_user channel, so its contribution is 0.
        situational_weight = 1.0 + distance_weight

        uncertainty = component["uncertainty"]
        weighted_uncertainty = (
            situational_weight * float(uncertainty)
            if uncertainty is not None
            else None
        )
        if weighted_uncertainty is not None:
            weighted_uncertainties.append(weighted_uncertainty)
        step = component["step"]
        step_diagnostics.append(
            SAUPStepDiagnostics(
                step_index=step.index,
                uncertainty=uncertainty,
                inquiry_drift=float(component["inquiry_drift"]),
                inference_gap=float(component["inference_gap"]),
                position_weight=position_weight,
                distance_weight=distance_weight,
                situational_weight=float(situational_weight),
                weighted_uncertainty=weighted_uncertainty,
                thought_action=str(component["thought_action"]),
                observation=str(component["observation"]),
                cumulative_state=str(component["cumulative_state"]),
            )
        )

    score: float | None = None
    if total_steps > 0 and not missing_uncertainties:
        score = float(math.sqrt(np.mean(np.square(weighted_uncertainties))))

    return SAUPDiagnostics(
        baseline_name=resolved_name,
        score=score,
        uncertainty_key=uncertainty_key,
        position_mix=None,
        steps=tuple(step_diagnostics),
        missing_uncertainty_steps=tuple(missing_uncertainties),
        missing_observation_steps=tuple(missing_observations),
        formula_version=AGENT_TRACER_GENERIC_DISTANCE_SAUP_VERSION,
        pd_specification_status=None,
    )
