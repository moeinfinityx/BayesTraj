"""Minimal, label-free implementation of the BayesTraj estimator.

The paper-facing reproduction uses frozen held-out metrics. This module makes
the statistical method itself independently inspectable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import LedoitWolf


EPS = np.finfo(float).tiny


def entropy(probabilities: ArrayLike) -> float:
    """Natural-log Shannon entropy, with the usual 0 log 0 convention."""
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def entropy_grid(reference_pool_size: int = 16, points: int = 257) -> NDArray[np.float64]:
    """Endpoint-inclusive grid from Eq. (8) of the submission."""
    if reference_pool_size < 1 or points < 2:
        raise ValueError("reference_pool_size >= 1 and points >= 2 are required")
    return np.linspace(0.0, np.log(reference_pool_size + 1), points)


def count_entropy_prior(
    counts: Sequence[int],
    *,
    draws: int = 8192,
    seed: int = 0,
) -> tuple[float, float]:
    """Moment-match the Jeffreys-smoothed entropy pushforward in Eqs. (4--5).

    Observed buckets receive count + 1/2 and one additional component reserves
    1/2 unit of prior mass for unseen outcomes.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1 or np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("counts must be a finite nonnegative vector")
    alpha = np.concatenate([c + 0.5, [0.5]])
    samples = np.random.default_rng(seed).dirichlet(alpha, size=draws)
    entropies = -np.sum(samples * np.log(np.maximum(samples, EPS)), axis=1)
    return float(entropies.mean()), float(entropies.var(ddof=1))


@dataclass(frozen=True)
class LinearGaussianModel:
    """Per-prefix model Z | H ~ N(a + bH, Sigma), fitted label-free."""

    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    intercept: NDArray[np.float64]
    slope: NDArray[np.float64]
    covariance: NDArray[np.float64]

    @classmethod
    def fit(cls, features: ArrayLike, targets: ArrayLike) -> "LinearGaussianModel":
        x = np.asarray(features, dtype=float)
        h = np.asarray(targets, dtype=float)
        if x.ndim != 2 or h.ndim != 1 or len(x) != len(h):
            raise ValueError("features must be Nxd and targets must be length N")
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        scale = np.where(scale > 0, scale, 1.0)
        z = (x - mean) / scale
        design = np.column_stack([np.ones(len(h)), h])
        coefficients = np.linalg.lstsq(design, z, rcond=None)[0]
        fitted = design @ coefficients
        covariance = LedoitWolf().fit(z - fitted).covariance_
        return cls(mean, scale, coefficients[0], coefficients[1], covariance)

    def standardize(self, feature: ArrayLike) -> NDArray[np.float64]:
        return (np.asarray(feature, dtype=float) - self.feature_mean) / self.feature_scale

    def log_likelihood(self, feature: ArrayLike, grid: ArrayLike) -> NDArray[np.float64]:
        z = self.standardize(feature)
        h = np.asarray(grid, dtype=float)
        residual = z[None, :] - self.intercept[None, :] - h[:, None] * self.slope[None, :]
        precision = np.linalg.pinv(self.covariance)
        return -0.5 * np.einsum("gi,ij,gj->g", residual, precision, residual)


def fuse_on_grid(
    prior_mean: float,
    prior_variance: float,
    feature: ArrayLike,
    model: LinearGaussianModel,
    grid: ArrayLike | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Normalize the fused posterior from Eq. (7) and return its moments."""
    h = entropy_grid() if grid is None else np.asarray(grid, dtype=float)
    variance = max(float(prior_variance), 1e-12)
    log_prior = -0.5 * (h - float(prior_mean)) ** 2 / variance
    log_weight = log_prior + model.log_likelihood(feature, h)
    log_weight -= np.max(log_weight)
    probability = np.exp(log_weight)
    probability /= probability.sum()
    mean = float(probability @ h)
    posterior_variance = float(probability @ ((h - mean) ** 2))
    return mean, posterior_variance, probability


def lcb_score(mean: float, variance: float, kappa: float = 1.96) -> float:
    """Uncertainty-aware failure score from Eq. (9)."""
    return float(mean - kappa * np.sqrt(max(float(variance), 0.0)))


def early_window(budget: int, width: int = 4) -> range:
    """Early-only search window W_B^- from Eq. (11)."""
    if budget < 2 or width < 1:
        raise ValueError("budget >= 2 and width >= 1 are required")
    return range(max(2, budget - width), budget)


def select_stopping_time(
    variances: Mapping[int, float], budget: int, threshold: float, width: int = 4
) -> int:
    """First certified prefix, with full-budget fallback (Eq. 12)."""
    for prefix in early_window(budget, width):
        if prefix in variances and float(variances[prefix]) <= threshold:
            return prefix
    return budget


def calibrate_variance_threshold(
    validation_variances: Iterable[Mapping[int, float]],
    budget: int,
    *,
    rho: float = 0.80,
    width: int = 4,
    candidates: ArrayLike | None = None,
) -> float:
    """Label-free threshold calibration from Eq. (13).

    Ties favor lower stopping-time variance and then the smaller threshold.
    """
    rows = list(validation_variances)
    if not rows:
        raise ValueError("at least one validation task is required")
    if not 0 < rho <= 1:
        raise ValueError("rho must lie in (0, 1]")
    if candidates is None:
        observed = sorted({float(v[n]) for v in rows for n in early_window(budget, width) if n in v})
        candidates_array = np.asarray(observed, dtype=float)
    else:
        candidates_array = np.sort(np.asarray(candidates, dtype=float))
    if candidates_array.size == 0:
        raise ValueError("no threshold candidates are available")
    target = rho * budget
    ranked: list[tuple[float, float, float, float]] = []
    for threshold in candidates_array:
        stops = np.asarray([
            select_stopping_time(row, budget, float(threshold), width) for row in rows
        ], dtype=float)
        ranked.append((abs(float(stops.mean()) - target), float(stops.var()), float(threshold), float(threshold)))
    return min(ranked)[-1]

