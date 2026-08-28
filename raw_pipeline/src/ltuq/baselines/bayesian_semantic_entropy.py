"""Faithful CPU port of Ciosek et al.'s Bayesian semantic-entropy estimator.

The implementation follows ``notebooks/estimation_final.ipynb`` from
spotify-research/bayesian-semantic-entropy at commit
``d4f1be324aee7ed0ff5ac787c7af11e8546db6d1``.  It generalizes the notebook's
semantic IDs to arbitrary canonical categorical outcomes while preserving its
Dirichlet prior, learned support-size distribution, probability lower bounds,
Monte Carlo importance sampler, and posterior mean/variance aggregation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np
from scipy import stats


RELEASED_IMPLEMENTATION_COMMIT = "d4f1be324aee7ed0ff5ac787c7af11e8546db6d1"


@dataclass(frozen=True)
class BayesianEntropyEstimate:
    """Posterior moments returned by the released estimator."""

    mean: float
    variance: float


def train_support_distribution(
    outcome_rows: Iterable[Sequence[Hashable]],
    *,
    truncation: int = 7,
) -> tuple[tuple[int, float], ...]:
    """Fit the released empirical prior over categorical support size.

    Each training row is expected to be the largest available outcome pool for
    one prompt/task.  As in the authors' notebook, observed support sizes are
    capped at seven by default.
    """

    support_sizes = [min(len(set(row)), truncation) for row in outcome_rows]
    if not support_sizes:
        raise ValueError("At least one training outcome row is required")
    if any(size < 1 for size in support_sizes):
        raise ValueError("Every training outcome row must be non-empty")
    counts = Counter(support_sizes)
    total = len(support_sizes)
    return tuple(
        (size, counts[size] / total)
        for size in range(1, max(support_sizes) + 1)
        if counts[size] > 0
    )


def condition_support_distribution(
    support_prior: Sequence[tuple[int, float]],
    observed_support: int,
) -> tuple[tuple[int, float], ...]:
    """Condition the empirical support prior as in the released notebook."""

    if observed_support < 1:
        raise ValueError("observed_support must be positive")
    if not support_prior:
        raise ValueError("support_prior must be non-empty")
    ordered = sorted((int(size), float(probability)) for size, probability in support_prior)
    maximum = ordered[-1][0]
    if observed_support >= maximum:
        return ((observed_support, 1.0),)
    eligible = [(size, probability) for size, probability in ordered if size >= observed_support]
    normalizer = sum(probability for _, probability in eligible)
    if normalizer <= 0.0:
        return ((observed_support, 1.0),)
    return tuple((size, probability / normalizer) for size, probability in eligible)


def _entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, None)
    return -np.where(probabilities > 1e-6, clipped * np.log(clipped), 0.0).sum(axis=1)


def _mean_variance_truncated_dirichlet(
    alpha: np.ndarray,
    lower_bounds: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> BayesianEntropyEstimate:
    """Mirror the released uniform-simplex importance sampler."""

    bounds = np.clip(np.asarray(lower_bounds, dtype=float), 0.0, 1.0)
    if bounds.sum() >= 0.99:
        bounds = bounds * (0.99 / bounds.sum())
    residual = 1.0 - float(bounds.sum())
    simplex = rng.dirichlet(np.ones_like(bounds), size=samples)
    draws = simplex * residual + bounds
    log_weights = stats.dirichlet.logpdf(draws.T, np.asarray(alpha, dtype=float))
    log_weights = np.asarray(log_weights, dtype=float)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    normalizer = float(weights.sum())
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("Invalid importance weights in Bayesian entropy estimator")
    entropies = _entropy_rows(draws)
    mean = float(np.sum(weights * entropies) / normalizer)
    second_moment = float(np.sum(weights * entropies**2) / normalizer)
    return BayesianEntropyEstimate(mean=mean, variance=max(0.0, second_moment - mean**2))


def estimate_bayesian_entropy(
    outcomes: Sequence[Hashable],
    *,
    support_prior: Sequence[tuple[int, float]],
    sequence_ids: Sequence[Hashable] | None = None,
    sequence_probabilities: Sequence[float] | None = None,
    alpha: float = 0.5,
    monte_carlo_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> BayesianEntropyEstimate:
    """Estimate categorical entropy with the released hierarchical model.

    ``sequence_ids`` and ``sequence_probabilities`` implement the paper's
    lower-bound constraint: probabilities of distinct sampled sequences are
    summed within each observed outcome class.  Omitting both reproduces the
    released estimator's ``use_probs=False`` ablation.
    """

    if not outcomes:
        raise ValueError("At least one outcome is required")
    if alpha <= 0.0 or monte_carlo_samples < 1:
        raise ValueError("alpha and monte_carlo_samples must be positive")
    use_probabilities = sequence_ids is not None or sequence_probabilities is not None
    if use_probabilities:
        if sequence_ids is None or sequence_probabilities is None:
            raise ValueError("sequence_ids and sequence_probabilities must be supplied together")
        if len(sequence_ids) != len(outcomes) or len(sequence_probabilities) != len(outcomes):
            raise ValueError("Sequence evidence must align with outcomes")
    generator = rng or np.random.default_rng()
    counts = Counter(outcomes)
    observed = list(counts)
    conditioned_support = condition_support_distribution(support_prior, len(observed))
    total_mean = 0.0
    total_second_moment = 0.0
    for support_size, support_probability in conditioned_support:
        lower_bounds = np.zeros(support_size, dtype=float)
        if use_probabilities:
            distinct: set[Hashable] = set()
            for outcome, sequence_id, probability in zip(
                outcomes, sequence_ids, sequence_probabilities, strict=True
            ):
                if sequence_id in distinct:
                    continue
                distinct.add(sequence_id)
                index = observed.index(outcome)
                lower_bounds[index] += max(0.0, float(probability))
        parameters = np.full(support_size, alpha, dtype=float)
        parameters[: len(observed)] += np.asarray([counts[value] for value in observed], dtype=float)
        estimate = _mean_variance_truncated_dirichlet(
            parameters,
            lower_bounds,
            samples=monte_carlo_samples,
            rng=generator,
        )
        total_mean += support_probability * estimate.mean
        total_second_moment += support_probability * (estimate.variance + estimate.mean**2)
    return BayesianEntropyEstimate(
        mean=float(total_mean),
        variance=max(0.0, float(total_second_moment - total_mean**2)),
    )
