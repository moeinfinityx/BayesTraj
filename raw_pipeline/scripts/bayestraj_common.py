"""Shared, paper-scoped utilities for BayesTraj evaluation.

This module contains only the linear-Gaussian BayesTraj data contract,
cross-fitting helpers, fixed/adaptive scoring, and output serialization used
by the submitted experiments.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

RAW_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(RAW_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(RAW_PIPELINE_ROOT))

from src.ltuq.estimators.bayestraj_posterior import (
    fit_linear_gaussian_trajectory_likelihood,
    update_entropy_on_grid,
)
from webshop_outcomes import posterior_moments, target

BUDGETS = (3, 4, 6, 8, 12, 16)
PREFIXES = tuple(range(2, 17))
FOLDS = 5
Z95 = 1.959963984540054
EXPECTED = {"dbbench": 300, "hotpotqa": 1000, "webshop": 200, "strategyqa": 687}
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
SEEDS = (101, 202, 303)
FEATURE_SETS = {
    "full": tuple(range(10)),
    "no_level": (1, 3, 4, 6, 8, 9),
    "no_dynamics": (0, 2, 4, 5, 7, 9),
    "no_structure": (0, 1, 2, 3, 5, 6, 7, 8),
    "no_dispersion": (0, 1, 2, 3, 4),
}


@dataclass(frozen=True)
class Cell:
    dataset: str
    backbone: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.dataset}-{self.backbone}-seed{self.seed}"


@dataclass
class CompactRows:
    ids: list[str]
    labels: np.ndarray
    folds: np.ndarray
    targets: np.ndarray
    count_means: np.ndarray
    count_variances: np.ndarray
    features: np.ndarray


@dataclass
class Posterior:
    mean: np.ndarray
    variance: np.ndarray
    precision: np.ndarray


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checkpoint_path(evaluation_root: Path, cell: Cell) -> Path:
    campaign = (
        "bayestraj_4datasets_3backbones_seeds101_202_z16"
        if cell.seed in (101, 202)
        else f"bayestraj_seed303_{cell.backbone}_z16"
    )
    return evaluation_root / campaign / "checkpoints" / f"{cell.name}.jsonl"


def parse_cell(value: str) -> Cell:
    dataset, backbone, seed_text = value.split("-")
    cell = Cell(dataset, backbone, int(seed_text.removeprefix("seed")))
    if dataset not in EXPECTED or backbone not in BACKBONES or cell.seed not in SEEDS:
        raise ValueError(f"unsupported paper cell: {value}")
    return cell


def compact(rows: list[dict[str, Any]], *, webshop_outcomes: bool, draws: int) -> CompactRows:
    size = len(rows)
    targets = np.empty(size, dtype=float)
    count_means = np.full((size, 17), np.nan)
    count_variances = np.full((size, 17), np.nan)
    features = np.full((size, 17, 10), np.nan)
    for index, row in enumerate(rows):
        buckets = list(map(str, row["buckets"]))
        targets[index] = target(buckets) if webshop_outcomes else float(row["target_oe16"])
        for prefix in range(1, 17):
            source = row["prefixes"][str(prefix)]
            if not webshop_outcomes:
                mean, variance = float(source["counts_prior_mean"]), float(source["counts_prior_variance"])
            else:
                mean, variance = posterior_moments(buckets[:prefix], str(row["sample_id"]), prefix, draws)
            count_means[index, prefix] = mean
            count_variances[index, prefix] = variance
            feature = np.asarray(source["features"], dtype=float)
            if feature.shape != (10,):
                raise RuntimeError(f"{row['sample_id']}/prefix{prefix}: feature shape {feature.shape}")
            features[index, prefix] = feature
    return CompactRows(
        [str(row["sample_id"]) for row in rows],
        np.asarray([int(row["label_failure"]) for row in rows]),
        np.asarray([int(row["fold"]) for row in rows]),
        targets, count_means, count_variances, features,
    )


def fit_split_posterior(
    data: CompactRows,
    training: np.ndarray,
    *,
    use_trajectory_likelihood: bool,
    feature_indices: Sequence[int],
) -> Posterior:
    shape = (len(data.ids), 17)
    means, variances, precisions = np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape)
    if not use_trajectory_likelihood:
        means[:, 1:], variances[:, 1:] = data.count_means[:, 1:], data.count_variances[:, 1:]
        return Posterior(means, variances, precisions)
    columns = np.asarray(feature_indices, dtype=int)
    for prefix in PREFIXES:
        likelihood = fit_linear_gaussian_trajectory_likelihood(
            data.features[training, prefix][:, columns], data.targets[training],
        )
        for index in range(len(data.ids)):
            value = update_entropy_on_grid(
                data.count_means[index, prefix], data.count_variances[index, prefix],
                data.features[index, prefix, columns], likelihood,
            )
            means[index, prefix], variances[index, prefix], precisions[index, prefix] = value.mean, value.variance, value.mark_precision
    return Posterior(means, variances, precisions)


def score(posterior: Posterior, stops: np.ndarray, *, kind: str) -> np.ndarray:
    indexes = np.arange(len(stops))
    means = posterior.mean[indexes, stops]
    if kind == "mean":
        return means
    if kind == "lcb95":
        return means - Z95 * np.sqrt(np.maximum(posterior.variance[indexes, stops], 1e-12))
    raise ValueError(kind)


def stop_at_boundary(values: np.ndarray, boundary: float, lower: int, upper: int) -> np.ndarray:
    if lower >= upper:
        return np.full(values.shape[0], upper, dtype=int)
    crossed = values[:, lower - 2:upper - 2] <= float(boundary)
    has_crossed = np.any(crossed, axis=1)
    return np.where(has_crossed, np.argmax(crossed, axis=1) + lower, upper).astype(int)


def calibrate_boundary(values: np.ndarray, target_cost: float, lower: int, upper: int) -> tuple[float, float]:
    eligible = values[:, lower - 2:max(lower - 2, upper - 2)]
    finite = eligible[np.isfinite(eligible)]
    if finite.size == 0:
        return -math.inf, float(upper)
    choices = np.unique(np.quantile(finite, np.linspace(0, 1, min(401, finite.size))))
    span = max(float(np.ptp(finite)), 1.0)
    choices = np.concatenate(([float(np.min(finite) - span)], choices, [float(np.max(finite) + span)]))
    candidates = []
    for boundary in choices:
        stops = stop_at_boundary(values, float(boundary), lower, upper)
        mean = float(np.mean(stops))
        candidates.append(((abs(mean - target_cost), float(np.var(stops)), float(boundary)), float(boundary), mean))
    _, boundary, mean = min(candidates)
    return boundary, mean


def add_variant(variants: dict[str, tuple[str, str, str, str]], name: str, posterior: str, allocation: str, score_kind: str, family: str) -> None:
    value = (posterior, allocation, score_kind, family)
    if name in variants and variants[name] != value:
        raise RuntimeError(f"conflicting variant: {name}")
    variants[name] = value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def score_outputs(*, cell: Cell, data_by_name: dict[str, CompactRows], posterior_by_name: dict[str, Posterior], allocation_by_name: dict[str, dict[str, dict[int, np.ndarray]]], variants: dict[str, tuple[str, str, str, str]], output: Path) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    temporary = (output / "task_scores.jsonl.gz").with_suffix(".gz.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for variant, (posterior_name, allocation, score_kind, family) in variants.items():
            data, posterior = data_by_name[posterior_name], posterior_by_name[posterior_name]
            for budget in BUDGETS:
                stops = allocation_by_name[posterior_name][allocation][budget]
                scores, labels = score(posterior, stops, kind=score_kind), data.labels
                metrics.append({
                    "dataset": cell.dataset, "backbone": cell.backbone, "seed": cell.seed,
                    "cell": cell.name, "variant": variant, "family": family, "budget": budget,
                    "tasks": len(labels), "failures": int(np.sum(labels)), "failure_rate": float(np.mean(labels)),
                    "auroc": float(roc_auc_score(labels, scores)), "aupr": float(average_precision_score(labels, scores)),
                    "mean_trajectories": float(np.mean(stops)), "trajectory_saving": float(1 - np.mean(stops) / budget),
                    "variance_trajectories": float(np.var(stops)),
                    "mean_absolute_budget_deviation": float(np.mean(np.abs(stops - budget))),
                    "fraction_above_budget": float(np.mean(stops > budget)),
                    "minimum_trajectories": int(np.min(stops)), "maximum_trajectories": int(np.max(stops)),
                })
                for sample_id, fold, label, value, used in zip(data.ids, data.folds, labels, scores, stops, strict=True):
                    handle.write(json.dumps({
                        "dataset": cell.dataset, "backbone": cell.backbone, "seed": cell.seed,
                        "cell": cell.name, "sample_id": sample_id, "fold": int(fold),
                        "label_failure": int(label), "budget": budget, "variant": variant,
                        "family": family, "score": float(value), "trajectories_used": int(used),
                    }, sort_keys=True) + "\n")
    temporary.replace(output / "task_scores.jsonl.gz")
    return metrics
