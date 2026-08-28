"""BayesTraj empirical-Bayes trajectory-feature posterior for outcome entropy.

The model treats a counts-only entropy posterior as a Gaussian prior over the
latent outcome entropy H and trajectory summary features z as a multivariate
Gaussian observation:

    z | H ~ Normal(a + b H, Sigma).

The trajectory-feature likelihood is fitted without correctness labels on separate tasks.
Conditioning the count prior on z is normalized on the submitted 257-point entropy grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class EntropyPosterior:
    mean: float
    variance: float
    mark_precision: float


@dataclass(frozen=True)
class LinearGaussianTrajectoryLikelihood:
    """Linear-Gaussian trajectory-feature likelihood evaluated on an entropy grid."""

    feature_imputation: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    covariance_inverse: np.ndarray
    active: np.ndarray


def _as_feature_matrix(features: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("features must contain at least two rows and one column")
    return matrix


def fit_linear_gaussian_trajectory_likelihood(
    features: Sequence[Sequence[float]],
    target_entropies: Sequence[float],
    *,
    covariance_structure: str = "full",
) -> LinearGaussianTrajectoryLikelihood:
    """Fit the submitted linear-Gaussian ``z | H`` likelihood."""

    if covariance_structure not in ("full", "diagonal"):
        raise ValueError("covariance_structure must be full or diagonal")
    matrix = _as_feature_matrix(features)
    targets = np.asarray(target_entropies, dtype=float)
    if targets.shape != (matrix.shape[0],) or not np.all(np.isfinite(targets)):
        raise ValueError("target_entropies must be finite and align with features")
    imputation = np.asarray(
        [
            float(np.median(column[np.isfinite(column)]))
            if np.any(np.isfinite(column))
            else 0.0
            for column in matrix.T
        ]
    )
    clean = np.where(np.isfinite(matrix), matrix, imputation)
    raw_scale = np.std(clean, axis=0, ddof=1)
    active = np.isfinite(raw_scale) & (raw_scale > 1e-8)
    scale = np.where(active, raw_scale, 1.0)
    standardized = (clean - imputation) / scale
    design = np.column_stack([np.ones_like(targets), targets])
    coefficients = np.linalg.lstsq(design, standardized, rcond=None)[0]
    residual = standardized - design @ coefficients
    inverse = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    if np.any(active):
        active_residual = residual[:, active]
        if covariance_structure == "full":
            covariance = np.atleast_2d(LedoitWolf().fit(active_residual).covariance_)
        elif covariance_structure == "diagonal":
            variances = np.var(active_residual, axis=0, ddof=1)
            covariance = np.diag(np.maximum(variances, 1e-8))
        covariance += np.eye(covariance.shape[0]) * 1e-8
        inverse[np.ix_(active, active)] = np.linalg.pinv(covariance, hermitian=True)
    return LinearGaussianTrajectoryLikelihood(
        feature_imputation=imputation,
        feature_scale=scale,
        coefficients=coefficients,
        covariance_inverse=inverse,
        active=active,
    )


def update_entropy_on_grid(
    prior_mean: float,
    prior_variance: float,
    features: Sequence[float],
    likelihood: LinearGaussianTrajectoryLikelihood,
    *,
    support_maximum: float = float(np.log(17.0)),
    grid_size: int = 257,
    use_count_prior: bool = True,
) -> EntropyPosterior:
    """Numerically normalize the one-dimensional trajectory-informed entropy posterior."""

    if grid_size < 33:
        raise ValueError("grid_size must be at least 33")
    variance = max(float(prior_variance), 1e-8)
    vector = np.asarray(features, dtype=float)
    if vector.shape != likelihood.feature_imputation.shape:
        raise ValueError("features do not match the fitted likelihood")
    clean = np.where(np.isfinite(vector), vector, likelihood.feature_imputation)
    standardized = (clean - likelihood.feature_imputation) / likelihood.feature_scale
    grid = np.linspace(0.0, float(support_maximum), grid_size)
    design = np.column_stack([np.ones_like(grid), grid])
    expected_marks = design @ likelihood.coefficients
    delta = standardized[None, :] - expected_marks
    mahalanobis = np.einsum(
        "gi,ij,gj->g", delta, likelihood.covariance_inverse, delta
    )
    log_likelihood = -0.5 * mahalanobis
    log_prior = -0.5 * (grid - float(prior_mean)) ** 2 / variance if use_count_prior else 0.0
    log_weights = log_prior + log_likelihood
    weights = np.exp(log_weights - float(np.max(log_weights)))
    weights /= float(np.sum(weights))
    mean = float(weights @ grid)
    posterior_variance = float(weights @ ((grid - mean) ** 2))
    return EntropyPosterior(
        mean=mean,
        variance=posterior_variance,
        mark_precision=(
            max(0.0, 1.0 / max(posterior_variance, 1e-12) - 1.0 / variance)
            if use_count_prior else 1.0 / max(posterior_variance, 1e-12)
        ),
    )
