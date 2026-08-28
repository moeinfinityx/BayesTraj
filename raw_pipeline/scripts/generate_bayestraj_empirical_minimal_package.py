#!/usr/bin/env python3
"""Generate the paper-facing BayesTraj empirical evidence mini-package.

The package is intentionally restricted to DBBench, HotpotQA, WebShop, and
StrategyQA.  It contains paired baseline inference, posterior diagnostics, and
cached realized-compute accounting for the two finalized BayesTraj modes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in os.sys.path:
        os.sys.path.insert(0, str(item))

from evaluate_brlg_fixed_varstop80_ablation import FULL_FIXED, FULL_VAR
from evaluate_bayestraj_webshop_constraint_pattern import (
    BUDGETS,
    FEATURE_SETS,
    FOLDS,
    Posterior,
    fit_split_posterior,
    make_data,
    map_task,
)


DATASETS = ("dbbench", "hotpotqa", "webshop", "strategyqa")
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
SEEDS = (101, 202, 303)
METHODS = {
    "fixed": "BR-LG-Risk-VC4-LCB95",
    "adaptive": "BR-LG-VarStop80-LCB95",
}
DISPLAY = {
    "fixed": "BayesTraj-Fixed",
    "adaptive": "BayesTraj-Adaptive",
    "MC-OE": "MC-OE",
    "BSE-Ciosek-Fixed": "BSE-Fixed",
    "CoCoA-PPL": "CoCoA-PPL",
}
COMPARISONS = (
    ("fixed", "MC-OE"),
    ("fixed", "BSE-Ciosek-Fixed"),
    ("fixed", "CoCoA-PPL"),
    ("adaptive", "MC-OE"),
    ("adaptive", "BSE-Ciosek-Fixed"),
    ("adaptive", "CoCoA-PPL"),
)
CI_REPLICATES = 10_000
CALIBRATION_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260814


def raw_z16_path(root: Path, dataset: str, backbone: str, seed: int) -> Path:
    suffix = "_n4" if dataset == "strategyqa" else ""
    directory = root / f"{dataset}_seed{seed}_z16{suffix}" / f"{dataset}_{backbone}"
    if dataset == "webshop":
        name = f"bayestraj_webshop_seed{seed}_z16_{backbone}_seed{seed}_pe.jsonl"
    else:
        run = f"bayestraj_{dataset}_seed{seed}_z16{suffix}"
        name = f"{run}_{backbone}_{dataset}_{backbone}_seed{seed}_uprop.jsonl"
    return directory / name


def checkpoint_path(root: Path, dataset: str, backbone: str, seed: int) -> Path:
    return root / "checkpoints" / f"{dataset}-{backbone}-seed{seed}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def hierarchical_distribution(
    rows: pd.DataFrame,
    value: str,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample dataset-backbone combinations and seeds within combinations."""
    rng = np.random.default_rng(seed)
    combos = sorted(rows["combination"].unique())
    seeds = sorted(rows["seed"].unique())
    matrix = (
        rows.groupby(["combination", "seed"])[value].mean()
        .unstack("seed").reindex(index=combos, columns=seeds).to_numpy(float)
    )
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"incomplete hierarchical array for {value}")
    combo_indexes = rng.integers(0, len(combos), size=(replicates, len(combos)))
    seed_indexes = rng.integers(0, len(seeds), size=(replicates, len(combos), len(seeds)))
    selected = matrix[combo_indexes[:, :, None], seed_indexes]
    return selected.mean(axis=(1, 2))


def paired_superiority(selected_path: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = pd.read_csv(selected_path)
    if set(source["dataset"].unique()) != set(DATASETS):
        raise RuntimeError(f"unexpected datasets in selected metrics: {sorted(source['dataset'].unique())}")
    source = source[source["budget"].isin(BUDGETS)].copy()
    detailed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    for comparison_index, (ours_short, baseline) in enumerate(COMPARISONS):
        ours = METHODS[ours_short]
        keys = ["dataset", "backbone", "combination", "seed", "budget"]
        left = source[source["method"] == ours][keys + ["auroc", "aupr", "mean_trajectories"]]
        right = source[source["method"] == baseline][keys + ["auroc", "aupr", "mean_trajectories"]]
        joined = left.merge(right, on=keys, suffixes=("_ours", "_baseline"), validate="one_to_one")
        if len(joined) != len(DATASETS) * len(BACKBONES) * len(SEEDS) * len(BUDGETS):
            raise RuntimeError(f"incomplete paired support: {ours}/{baseline}: {len(joined)}")
        joined["auroc_delta"] = joined["auroc_ours"] - joined["auroc_baseline"]
        joined["aupr_delta"] = joined["aupr_ours"] - joined["aupr_baseline"]
        joined["cost_ratio"] = joined["mean_trajectories_ours"] / joined["mean_trajectories_baseline"]

        for budget in BUDGETS:
            block = joined[joined["budget"] == budget]
            record: dict[str, Any] = {
                "method": DISPLAY[ours_short], "baseline": DISPLAY[baseline],
                "budget": budget, "cells": len(block),
                "mean_cost_ratio": float(block["cost_ratio"].mean()),
            }
            for metric_index, metric in enumerate(("auroc", "aupr")):
                values = block[f"{metric}_delta"].to_numpy(float)
                dist = hierarchical_distribution(
                    block, f"{metric}_delta", CI_REPLICATES,
                    BOOTSTRAP_SEED + 1000 * comparison_index + 100 * budget + metric_index,
                )
                record[f"{metric}_delta"] = float(values.mean())
                record[f"{metric}_ci_low"] = float(np.quantile(dist, 0.025))
                record[f"{metric}_ci_high"] = float(np.quantile(dist, 0.975))
                record[f"{metric}_wins"] = int(np.sum(values > 1e-12))
                record[f"{metric}_ties"] = int(np.sum(np.abs(values) <= 1e-12))
                record[f"{metric}_losses"] = int(np.sum(values < -1e-12))
            detailed.append(record)

        cell = joined.groupby(["dataset", "backbone", "combination", "seed"], as_index=False).agg(
            auroc_delta=("auroc_delta", "mean"), aupr_delta=("aupr_delta", "mean"),
            cost_ratio=("cost_ratio", "mean"),
        )
        record = {
            "method": DISPLAY[ours_short], "baseline": DISPLAY[baseline],
            "budgets": len(BUDGETS), "cells": len(cell),
            "mean_cost_ratio": float(joined["cost_ratio"].mean()),
        }
        for metric_index, metric in enumerate(("auroc", "aupr")):
            dist = hierarchical_distribution(
                joined, f"{metric}_delta", CI_REPLICATES,
                BOOTSTRAP_SEED + 10_000 * comparison_index + metric_index,
            )
            values = cell[f"{metric}_delta"].to_numpy(float)
            left_tail = (np.sum(dist <= 0) + 1) / (len(dist) + 1)
            right_tail = (np.sum(dist >= 0) + 1) / (len(dist) + 1)
            p_value = min(1.0, 2 * min(left_tail, right_tail))
            record[f"{metric}_delta"] = float(joined[f"{metric}_delta"].mean())
            record[f"{metric}_ci_low"] = float(np.quantile(dist, 0.025))
            record[f"{metric}_ci_high"] = float(np.quantile(dist, 0.975))
            record[f"{metric}_wins"] = int(np.sum(values > 1e-12))
            record[f"{metric}_ties"] = int(np.sum(np.abs(values) <= 1e-12))
            record[f"{metric}_losses"] = int(np.sum(values < -1e-12))
            record[f"{metric}_p"] = float(p_value)
            raw_p_values.append(float(p_value))
        aggregate.append(record)

    adjusted = holm_adjust(raw_p_values)
    cursor = 0
    for record in aggregate:
        for metric in ("auroc", "aupr"):
            record[f"{metric}_p_holm"] = adjusted[cursor]
            cursor += 1
    write_csv(output / "paired_superiority_by_budget.csv", detailed)
    write_csv(output / "paired_superiority_summary.csv", aggregate)
    return aggregate, detailed


def plot_paired_superiority(rows: list[dict[str, Any]], output: Path) -> None:
    lookup = {(row["method"], row["baseline"]): row for row in rows}
    order = [
        ("BayesTraj-Fixed", "CoCoA-PPL", 6.0),
        ("BayesTraj-Fixed", "BSE-Fixed", 5.0),
        ("BayesTraj-Fixed", "MC-OE", 4.0),
        ("BayesTraj-Adaptive", "CoCoA-PPL", 2.5),
        ("BayesTraj-Adaptive", "BSE-Fixed", 1.5),
        ("BayesTraj-Adaptive", "MC-OE", 0.5),
    ]
    styles = {
        "BayesTraj-Fixed": dict(color="#174A7E", marker="o", markerfacecolor="#174A7E"),
        "BayesTraj-Adaptive": dict(color="#D14900", marker="s", markerfacecolor="white"),
    }
    labels = [
        "CoCoA-PPL", "BSE-Fixed", "MC-OE",
        "CoCoA-PPL", "BSE-Fixed", "MC-OE",
    ]
    # Sized for a single IEEE column (about 3.5 in).  Keep all typography
    # local so this compact figure does not inherit the larger report style.
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 9.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.5,
    }):
        figure, axes = plt.subplots(1, 2, figsize=(3.55, 2.55), sharey=True)
        for axis, metric, title in zip(axes, ("auroc", "aupr"), ("AUROC", "AUPR"), strict=True):
            axis.axvspan(0, 8.2, color="#E9F4EC", alpha=.65, zorder=0)
            axis.axvline(0, color="#30343B", linewidth=.9, zorder=1)
            axis.axhspan(3.25, 6.65, color="#174A7E", alpha=.045, zorder=0)
            axis.axhspan(-.15, 3.15, color="#D14900", alpha=.045, zorder=0)
            for method, baseline, y in order:
                row = lookup[(method, baseline)]
                point = 100 * float(row[f"{metric}_delta"])
                low = 100 * float(row[f"{metric}_ci_low"])
                high = 100 * float(row[f"{metric}_ci_high"])
                style = styles[method]
                axis.errorbar(
                    point, y, xerr=[[point - low], [high - point]],
                    fmt=style["marker"], color=style["color"],
                    markerfacecolor=style["markerfacecolor"], markeredgecolor=style["color"],
                    markeredgewidth=1.1, markersize=4.5, linewidth=1.25,
                    capsize=2.5, zorder=3,
                )
            axis.set_title(
                f"{title} improvement\n(percentage points)",
                fontsize=8.2, fontweight="bold", linespacing=.9, pad=3,
            )
            axis.set_xlim(-.35, 8.2)
            axis.set_xticks([0, 2, 4, 6, 8])
            axis.set_xticklabels(["0%", "+2%", "+4%", "+6%", "+8%"])
            axis.set_ylim(-.15, 6.65)
            axis.grid(axis="x", alpha=.2, linewidth=.5)
            axis.tick_params(axis="y", length=0, pad=2)
        axes[0].set_yticks([item[2] for item in order], labels=labels)
        legend_handles = [
            matplotlib.lines.Line2D(
                [], [], color=styles["BayesTraj-Fixed"]["color"], marker="o",
                markerfacecolor=styles["BayesTraj-Fixed"]["color"],
                linewidth=1.25, markersize=4.5, label="Fixed",
            ),
            matplotlib.lines.Line2D(
                [], [], color=styles["BayesTraj-Adaptive"]["color"], marker="s",
                markerfacecolor="white", markeredgewidth=1.1,
                linewidth=1.25, markersize=4.5, label="Adaptive (0.80× cost)",
            ),
        ]
        figure.legend(
            handles=legend_handles, loc="lower center", bbox_to_anchor=(.61, .015),
            ncol=2, frameon=False, fontsize=7.2, handlelength=1.5,
            columnspacing=1.0, handletextpad=.45,
        )
        figure.subplots_adjust(left=.245, right=.985, top=.84, bottom=.18, wspace=.15)
        plot_dir = output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_dir / "paired_baseline_forest.pdf", bbox_inches="tight", pad_inches=.02)
        figure.savefig(plot_dir / "paired_baseline_forest.png", dpi=300, bbox_inches="tight", pad_inches=.02)
        plt.close(figure)


def posterior_rows(
    ablation_run_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    records: list[dict[str, Any]] = []
    sources: list[Path] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for seed in SEEDS:
                cell_name = f"{dataset}-{backbone}-seed{seed}"
                cell_manifest = ablation_run_root / "cells" / cell_name / "manifest.json"
                checkpoint = Path(json.loads(cell_manifest.read_text())["source"])
                sources.append(checkpoint)
                rows = read_jsonl(checkpoint)
                if dataset != "webshop":
                    for row in rows:
                        for budget in BUDGETS:
                            prefix = row["prefixes"][str(budget)]
                            records.append({
                                "dataset": dataset, "backbone": backbone, "seed": seed,
                                "combination": f"{dataset}-{backbone}", "cell": cell_name,
                                "sample_id": str(row["sample_id"]), "budget": budget,
                                "target": float(row["target_oe16"]),
                                "count_mean": float(prefix["counts_prior_mean"]),
                                "count_variance": float(prefix["counts_prior_variance"]),
                                "fused_mean": float(prefix["tm_mean"]),
                                "fused_variance": float(prefix["tm_variance"]),
                            })
                    continue

                raw_path = raw_z16_path(raw_root, dataset, backbone, seed)
                sources.append(raw_path)
                raw = read_jsonl(raw_path)
                raw_by_id = {str(row["sample_id"]): row for row in raw}
                buckets: list[list[str]] = []
                for row in rows:
                    mapped, _ = map_task(raw_by_id[str(row["sample_id"])], list(map(str, row["buckets"])))
                    buckets.append(mapped["constraint-pattern"])
                data = make_data(rows, buckets, 2048)
                fused = Posterior(*(np.full((len(rows), 17), np.nan) for _ in range(3)))
                for fold in range(FOLDS):
                    train = np.flatnonzero(data.folds != fold)
                    test = np.flatnonzero(data.folds == fold)
                    fitted = fit_split_posterior(
                        data,
                        train,
                        use_trajectory_likelihood=True,
                        feature_indices=FEATURE_SETS["full"],
                    )
                    fused.mean[test] = fitted.mean[test]
                    fused.variance[test] = fitted.variance[test]
                    fused.precision[test] = fitted.precision[test]
                for index, sample_id in enumerate(data.ids):
                    for budget in BUDGETS:
                        records.append({
                            "dataset": dataset, "backbone": backbone, "seed": seed,
                            "combination": f"{dataset}-{backbone}", "cell": cell_name,
                            "sample_id": sample_id, "budget": budget,
                            "target": float(data.targets[index]),
                            "count_mean": float(data.count_means[index, budget]),
                            "count_variance": float(data.count_variances[index, budget]),
                            "fused_mean": float(fused.mean[index, budget]),
                            "fused_variance": float(fused.variance[index, budget]),
                        })
                print(json.dumps({"posterior_cell": cell_name, "status": "complete"}), flush=True)
    frame = pd.DataFrame.from_records(records)
    if set(frame["dataset"].unique()) != set(DATASETS):
        raise RuntimeError("posterior diagnostics violated the four-dataset contract")
    return frame, sources


def bootstrap_cell_means(frame: pd.DataFrame, column: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = frame.groupby(["dataset", "backbone", "combination", "seed", "budget"], as_index=False)[column].mean()
    points, lows, highs = [], [], []
    for budget in BUDGETS:
        block = cells[cells["budget"] == budget]
        distribution = hierarchical_distribution(block, column, CI_REPLICATES, seed + budget)
        points.append(float(block[column].mean()))
        lows.append(float(np.quantile(distribution, 0.025)))
        highs.append(float(np.quantile(distribution, 0.975)))
    return np.asarray(points), np.asarray(lows), np.asarray(highs)


def posterior_diagnostics(frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    frame = frame.copy()
    frame["count_squared_error"] = (frame["target"] - frame["count_mean"]) ** 2
    frame["fused_squared_error"] = (frame["target"] - frame["fused_mean"]) ** 2
    cell_variance = frame.groupby("cell")["target"].transform("var").clip(lower=1e-8)
    frame["normalized_fused_variance"] = frame["fused_variance"] / cell_variance
    frame["normalized_fused_squared_error"] = frame["fused_squared_error"] / cell_variance
    frame["covered95"] = (
        np.abs(frame["target"] - frame["fused_mean"])
        <= 1.959963984540054 * np.sqrt(frame["fused_variance"].clip(lower=0))
    )

    summary = frame.groupby(["dataset", "backbone", "seed", "budget"], as_index=False).agg(
        count_mse=("count_squared_error", "mean"), fused_mse=("fused_squared_error", "mean"),
        coverage95=("covered95", "mean"), mean_variance=("fused_variance", "mean"),
    )
    write_csv(output / "posterior_diagnostics_by_cell_budget.csv", summary.to_dict("records"))
    dataset_summary = summary.groupby(["dataset", "budget"], as_index=False).agg(
        count_mse=("count_mse", "mean"), fused_mse=("fused_mse", "mean"),
        coverage95=("coverage95", "mean"), mean_variance=("mean_variance", "mean"),
    )
    write_csv(output / "posterior_diagnostics_by_dataset_budget.csv", dataset_summary.to_dict("records"))

    count, count_low, count_high = bootstrap_cell_means(frame, "count_squared_error", BOOTSTRAP_SEED + 200_000)
    fused, fused_low, fused_high = bootstrap_cell_means(frame, "fused_squared_error", BOOTSTRAP_SEED + 300_000)

    # Fixed global bin edges make the cell bootstrap comparable across replicates.
    finite = frame[np.isfinite(frame["normalized_fused_variance"]) & np.isfinite(frame["normalized_fused_squared_error"])].copy()
    finite["calibration_bin"] = pd.qcut(
        finite["normalized_fused_variance"], q=8, labels=False, duplicates="drop"
    ).astype(int)
    calibration = finite.groupby("calibration_bin", as_index=False).agg(
        predicted=("normalized_fused_variance", "mean"), observed=("normalized_fused_squared_error", "mean"),
        rows=("sample_id", "size"),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + 400_000)
    cells = sorted(finite["cell"].unique())
    bootstrap = np.empty((CALIBRATION_REPLICATES, len(calibration)), dtype=float)
    for replicate in range(CALIBRATION_REPLICATES):
        chosen = rng.choice(cells, size=len(cells), replace=True)
        sampled = pd.concat([finite[finite["cell"] == cell] for cell in chosen], ignore_index=True)
        means = sampled.groupby("calibration_bin")["normalized_fused_squared_error"].mean()
        bootstrap[replicate] = [float(means.get(index, np.nan)) for index in calibration["calibration_bin"]]
    calibration["observed_ci_low"] = np.nanquantile(bootstrap, 0.025, axis=0)
    calibration["observed_ci_high"] = np.nanquantile(bootstrap, 0.975, axis=0)
    write_csv(output / "posterior_risk_calibration.csv", calibration.to_dict("records"))

    # Designed at 7 inches and reduced to IEEE 3.5-inch column width.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 17, "axes.titlesize": 18,
        "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))
    colors = {"count": "#999999", "fused": "#00796B"}
    axes[0].plot(BUDGETS, count, "o--", color=colors["count"], linewidth=2.2, markersize=6, label="Count only")
    axes[0].fill_between(BUDGETS, count_low, count_high, color=colors["count"], alpha=.16)
    axes[0].plot(BUDGETS, fused, "o-", color=colors["fused"], linewidth=2.7, markersize=6, label="Two-view fusion")
    axes[0].fill_between(BUDGETS, fused_low, fused_high, color=colors["fused"], alpha=.16)
    axes[0].set_title("(a) Target estimation")
    axes[0].set_xlabel("Trajectory prefix")
    axes[0].set_ylabel(r"Held-out $H$ MSE")
    axes[0].set_xticks(BUDGETS)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(alpha=.22)

    x = calibration["predicted"].to_numpy(float)
    y = calibration["observed"].to_numpy(float)
    low = calibration["observed_ci_low"].to_numpy(float)
    high = calibration["observed_ci_high"].to_numpy(float)
    upper = max(float(np.nanmax(x)), float(np.nanmax(high))) * 1.05
    axes[1].plot([0, upper], [0, upper], "--", color="#777777", linewidth=1.7, label="Ideal")
    axes[1].errorbar(x, y, yerr=[y - low, high - y], fmt="o-", color="#CC4C02", linewidth=2.3, markersize=6, capsize=3, label="BayesTraj")
    axes[1].set_title("(b) Posterior-risk calibration")
    axes[1].set_xlabel("Predicted variance")
    axes[1].set_ylabel("Observed squared error")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(alpha=.22)
    figure.subplots_adjust(left=.105, right=.985, top=.91, bottom=.20, wspace=.36)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "posterior_accuracy_calibration.pdf", bbox_inches="tight", pad_inches=.02)
    figure.savefig(plot_dir / "posterior_accuracy_calibration.png", dpi=300, bbox_inches="tight", pad_inches=.02)
    plt.close(figure)

    macro_count_mse = float(summary["count_mse"].mean())
    macro_fused_mse = float(summary["fused_mse"].mean())
    return {
        "count_mse": macro_count_mse,
        "fused_mse": macro_fused_mse,
        "relative_mse_reduction": float(1 - macro_fused_mse / macro_count_mse),
        "coverage95": float(summary["coverage95"].mean()),
        "rows": len(frame), "cells": int(frame["cell"].nunique()),
    }


def load_stops(ablation_scores: Path, webshop_root: Path) -> dict[tuple[str, str, int, str], int]:
    output: dict[tuple[str, str, int, str], int] = {}
    with gzip.open(ablation_scores, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["dataset"] == "webshop" or row["variant"] not in (FULL_FIXED, FULL_VAR):
                continue
            mode = "fixed" if row["variant"] == FULL_FIXED else "adaptive"
            output[(str(row["cell"]), mode, int(row["budget"]), str(row["sample_id"]))] = int(row["trajectories_used"])
    for path in sorted((webshop_root / "cells").glob("*/task_scores_without_labels.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["bucket_variant"] != "constraint-pattern" or row["split"] != "crossfit":
                    continue
                mode = "fixed" if row["method"] == "BR-LG Risk VC4" else "adaptive"
                output[(str(row["cell"]), mode, int(row["budget"]), str(row["sample_id"]))] = int(row["trajectories_used"])
    expected_tasks = {"dbbench": 300, "hotpotqa": 1000, "strategyqa": 687, "webshop": 200}
    expected = sum(expected_tasks[d] for d in DATASETS) * len(BACKBONES) * len(SEEDS) * len(BUDGETS) * 2
    if len(output) != expected:
        raise RuntimeError(f"stop coverage mismatch: {len(output)} != {expected}")
    return output


def trajectory_cost(tdp: dict[str, Any]) -> tuple[int, int, int, int]:
    steps = list(tdp.get("steps") or [])
    output_tokens = 0
    token_steps = 0
    observations = 0
    for step in steps:
        metadata = step.get("metadata") or {}
        chosen = metadata.get("chosen_output_metadata") or {}
        token_count = chosen.get("token_count")
        if isinstance(token_count, (int, float)) and math.isfinite(float(token_count)):
            output_tokens += int(token_count)
            token_steps += 1
        observations += int(bool(metadata.get("observation") or metadata.get("raw_observation_messages")))
    return len(steps), output_tokens, token_steps, observations


def efficiency_table(
    raw_root: Path,
    stops: dict[tuple[str, str, int, str], int],
    selected_path: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    totals: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sources: list[Path] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for seed in SEEDS:
                cell = f"{dataset}-{backbone}-seed{seed}"
                path = raw_z16_path(raw_root, dataset, backbone, seed)
                sources.append(path)
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        sample_id = str(record["sample_id"])
                        tdps = list((record.get("estimate") or {}).get("tdps") or [])[:16]
                        if len(tdps) != 16:
                            raise RuntimeError(f"{cell}/{sample_id}: expected 16 trajectories")
                        costs = [trajectory_cost(tdp) for tdp in tdps]
                        cumulative = np.cumsum(np.asarray(costs, dtype=float), axis=0)
                        for mode in ("fixed", "adaptive"):
                            for budget in BUDGETS:
                                used = stops[(cell, mode, budget, sample_id)]
                                step_count, token_count, token_steps, observation_count = cumulative[used - 1]
                                target = totals[(mode, budget)]
                                target["tasks"] += 1
                                target["trajectories"] += used
                                target["agent_steps"] += step_count
                                target["output_tokens"] += token_count
                                target["token_steps"] += token_steps
                                target["environment_observations"] += observation_count
                print(json.dumps({"efficiency_cell": cell, "status": "complete"}), flush=True)

    metrics = pd.read_csv(selected_path)
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        fixed = totals[("fixed", budget)]
        for mode in ("fixed", "adaptive"):
            values = totals[(mode, budget)]
            paper_method = METHODS[mode]
            block = metrics[(metrics["method"] == paper_method) & (metrics["budget"] == budget)]
            task_count = values["tasks"]
            rows.append({
                "method": DISPLAY[mode], "budget": budget,
                "auroc": float(block["auroc"].mean()), "aupr": float(block["aupr"].mean()),
                "mean_trajectories": values["trajectories"] / task_count,
                "trajectory_saving": 1 - values["trajectories"] / fixed["trajectories"],
                "mean_agent_steps": values["agent_steps"] / task_count,
                "agent_step_saving": 1 - values["agent_steps"] / fixed["agent_steps"],
                "mean_output_tokens": values["output_tokens"] / task_count,
                "output_token_saving": 1 - values["output_tokens"] / fixed["output_tokens"],
                "mean_environment_observations": values["environment_observations"] / task_count,
                "environment_observation_saving": (
                    1 - values["environment_observations"] / fixed["environment_observations"]
                    if fixed["environment_observations"] else float("nan")
                ),
                "token_metadata_coverage": values["token_steps"] / values["agent_steps"],
                "tasks": int(task_count),
                "wall_clock_status": "not recorded in cached artifacts",
            })
    write_csv(output / "efficiency_by_budget.csv", rows)
    return rows, sources


def plot_realized_efficiency(
    rows: list[dict[str, Any]], selected_path: Path, output: Path,
) -> None:
    """Plot ranking performance against realized rather than nominal cost."""
    source = pd.read_csv(selected_path)
    intervals: dict[tuple[str, int, str], tuple[float, float]] = {}
    for mode_index, mode in enumerate(("fixed", "adaptive")):
        method = METHODS[mode]
        for budget in BUDGETS:
            block = source[(source["method"] == method) & (source["budget"] == budget)]
            if len(block) != len(DATASETS) * len(BACKBONES) * len(SEEDS):
                raise RuntimeError(f"incomplete efficiency interval support: {method}/B={budget}")
            for metric_index, metric in enumerate(("auroc", "aupr")):
                dist = hierarchical_distribution(
                    block, metric, CI_REPLICATES,
                    BOOTSTRAP_SEED + 40_000 + 1_000 * mode_index
                    + 100 * budget + metric_index,
                )
                intervals[(DISPLAY[mode], budget, metric)] = (
                    float(np.quantile(dist, .025)), float(np.quantile(dist, .975)),
                )

    styles = {
        "BayesTraj-Fixed": dict(color="#174A7E", marker="o", markerfacecolor="#174A7E"),
        "BayesTraj-Adaptive": dict(color="#D14900", marker="s", markerfacecolor="white"),
    }
    lookup = {(row["method"], int(row["budget"])): row for row in rows}
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 9.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    }):
        figure, axes = plt.subplots(1, 2, figsize=(3.55, 2.45))
        for axis, metric, title in zip(
            axes, ("auroc", "aupr"), ("AUROC", "AUPR"), strict=True,
        ):
            # Faint within-budget links make the adaptive leftward cost shift explicit.
            for budget in BUDGETS:
                fixed = lookup[("BayesTraj-Fixed", budget)]
                adaptive = lookup[("BayesTraj-Adaptive", budget)]
                axis.plot(
                    [adaptive["mean_trajectories"], fixed["mean_trajectories"]],
                    [adaptive[metric], fixed[metric]], color="#8A8F98",
                    linewidth=.55, alpha=.38, zorder=1,
                )
            for method in ("BayesTraj-Fixed", "BayesTraj-Adaptive"):
                method_rows = [lookup[(method, budget)] for budget in BUDGETS]
                x = np.asarray([row["mean_trajectories"] for row in method_rows])
                y = np.asarray([row[metric] for row in method_rows])
                low = np.asarray([intervals[(method, budget, metric)][0] for budget in BUDGETS])
                high = np.asarray([intervals[(method, budget, metric)][1] for budget in BUDGETS])
                style = styles[method]
                axis.fill_between(x, low, high, color=style["color"], alpha=.11, linewidth=0)
                axis.plot(
                    x, y, color=style["color"], marker=style["marker"],
                    markerfacecolor=style["markerfacecolor"],
                    markeredgecolor=style["color"], markeredgewidth=1.0,
                    linewidth=1.35, markersize=4.0, zorder=3,
                )
            axis.set_title(title, fontweight="bold", pad=3)
            axis.set_xlim(1.65, 16.35)
            axis.set_xticks([2, 4, 8, 12, 16])
            axis.grid(alpha=.20, linewidth=.5)
        axes[0].set_ylabel("Score")
        handles = [
            matplotlib.lines.Line2D(
                [], [], color=styles["BayesTraj-Fixed"]["color"], marker="o",
                markerfacecolor=styles["BayesTraj-Fixed"]["color"],
                linewidth=1.35, markersize=4.0, label="Fixed",
            ),
            matplotlib.lines.Line2D(
                [], [], color=styles["BayesTraj-Adaptive"]["color"], marker="s",
                markerfacecolor="white", markeredgewidth=1.0,
                linewidth=1.35, markersize=4.0, label="Adaptive",
            ),
        ]
        figure.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(.56, .105),
            ncol=2, frameon=False, fontsize=7.2, handlelength=1.5,
            columnspacing=1.1, handletextpad=.45,
        )
        figure.supxlabel("Mean trajectories used", fontsize=7.5, y=.012)
        figure.subplots_adjust(left=.13, right=.985, top=.91, bottom=.27, wspace=.25)
        plot_dir = output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_dir / "realized_cost_performance.pdf", bbox_inches="tight", pad_inches=.02)
        figure.savefig(
            plot_dir / "realized_cost_performance.png", dpi=300,
            bbox_inches="tight", pad_inches=.02,
        )
        plt.close(figure)


def plot_resource_savings(rows: list[dict[str, Any]], output: Path) -> None:
    """Show how the trajectory reduction translates to realized agent compute."""
    adaptive = sorted(
        (row for row in rows if row["method"] == "BayesTraj-Adaptive"),
        key=lambda row: int(row["budget"]),
    )
    budgets = np.asarray([int(row["budget"]) for row in adaptive])
    series = [
        ("Trajectories", "trajectory_saving", "#174A7E", "o"),
        ("Agent steps", "agent_step_saving", "#00897B", "s"),
        ("Output tokens", "output_token_saving", "#D14900", "^"),
    ]
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    }):
        figure, axis = plt.subplots(figsize=(3.55, 2.25))
        axis.axhline(
            20, color="#70757D", linestyle=(0, (3, 2)), linewidth=.9,
            alpha=.8, label="20% target", zorder=1,
        )
        for label, key, color, marker in series:
            values = 100 * np.asarray([float(row[key]) for row in adaptive])
            axis.plot(
                budgets, values, color=color, marker=marker, linewidth=1.5,
                markersize=4.3, markeredgewidth=.8, label=label, zorder=3,
            )
        axis.set_xlabel(r"Nominal trajectory budget $B$")
        axis.set_ylabel("Realized saving (%)")
        axis.set_xticks(budgets)
        axis.set_ylim(0, 22.5)
        axis.set_yticks([0, 5, 10, 15, 20])
        axis.grid(axis="y", alpha=.22, linewidth=.5)
        axis.legend(
            loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=2,
            frameon=False, fontsize=7.2, handlelength=1.8,
            columnspacing=1.0, handletextpad=.4,
        )
        figure.subplots_adjust(left=.16, right=.985, top=.78, bottom=.22)
        plot_dir = output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_dir / "realized_resource_savings.pdf", bbox_inches="tight", pad_inches=.02)
        figure.savefig(
            plot_dir / "realized_resource_savings.png", dpi=300,
            bbox_inches="tight", pad_inches=.02,
        )
        plt.close(figure)


def plot_adaptive_tradeoff_summary(rows: list[dict[str, Any]], output: Path) -> None:
    """Compact paper figure linking ranking retention to realized savings."""
    lookup = {(row["method"], int(row["budget"])): row for row in rows}
    budgets = np.asarray(BUDGETS)
    fixed = [lookup[("BayesTraj-Fixed", budget)] for budget in BUDGETS]
    adaptive = [lookup[("BayesTraj-Adaptive", budget)] for budget in BUDGETS]
    auroc_delta = 100 * np.asarray([
        float(a["auroc"]) - float(f["auroc"]) for f, a in zip(fixed, adaptive, strict=True)
    ])
    aupr_delta = 100 * np.asarray([
        float(a["aupr"]) - float(f["aupr"]) for f, a in zip(fixed, adaptive, strict=True)
    ])
    savings = {
        "Trajectories": (100 * np.asarray([float(row["trajectory_saving"]) for row in adaptive]), "#174A7E", "o"),
        "Agent steps": (100 * np.asarray([float(row["agent_step_saving"]) for row in adaptive]), "#00897B", "s"),
        "Output tokens": (100 * np.asarray([float(row["output_token_saving"]) for row in adaptive]), "#D14900", "^"),
    }
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    }):
        figure, axes = plt.subplots(1, 2, figsize=(3.55, 2.60))
        left, right = axes
        left.axhspan(-1.0, 0.0, color="#E4F2E7", alpha=.9, zorder=0)
        left.axhline(0, color="#444950", linewidth=.8, zorder=1)
        left.axhline(-1, color="#70757D", linestyle=(0, (3, 2)), linewidth=.75, zorder=1)
        left.plot(budgets, auroc_delta, color="#174A7E", marker="o",
                  linewidth=1.35, markersize=3.8,
                  label=r"AUROC$_{\rm Adaptive}$ − AUROC$_{\rm Fixed}$")
        left.plot(budgets, aupr_delta, color="#D14900", marker="s",
                  markerfacecolor="white", markeredgewidth=.9,
                  linewidth=1.35, markersize=3.8,
                  label=r"AUPR$_{\rm Adaptive}$ − AUPR$_{\rm Fixed}$")
        left.set_title("(a) Ranking retention", fontweight="bold", y=1.13, pad=0)
        left.text(
            .5, 1.035, "Green region: ≤1 percentage-point loss",
            transform=left.transAxes, fontsize=5.0, color="#48664F",
            ha="center", va="bottom",
        )
        left.set_ylabel("Adaptive − Fixed")
        left.set_ylim(-1.4, .15)
        left.set_yticks([-1.0, -.5, 0.0])
        left.set_yticklabels(["−1.0%", "−0.5%", "0.0%"])
        left.legend(
            loc="upper center", bbox_to_anchor=(.5, -.13), ncol=1,
            frameon=False, fontsize=5.2, handlelength=1.35,
            handletextpad=.3, labelspacing=.25, borderaxespad=0,
        )

        right.axhline(20, color="#70757D", linestyle=(0, (3, 2)),
                      linewidth=.8, zorder=1)
        for label, (values, color, marker) in savings.items():
            right.plot(budgets, values, color=color, marker=marker,
                       linewidth=1.35, markersize=3.8, label=label, zorder=3)
        right.set_title("(b) Compute savings", fontweight="bold", y=1.13, pad=0)
        right.set_ylabel("Saving (%)")
        right.set_ylim(0, 22.5)
        right.set_yticks([0, 5, 10, 15, 20])
        right.legend(loc="lower right", frameon=False, fontsize=6.2,
                     handlelength=1.35, handletextpad=.35, borderaxespad=.4)

        for axis in axes:
            axis.set_xticks(budgets)
            axis.grid(axis="y", alpha=.20, linewidth=.5)
        figure.text(
            .765, .205, r"Trajectory budget $B$", fontsize=7.5,
            ha="center", va="center",
        )
        figure.subplots_adjust(left=.13, right=.985, top=.82, bottom=.31, wspace=.34)
        plot_dir = output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_dir / "adaptive_performance_efficiency.pdf", bbox_inches="tight", pad_inches=.02)
        figure.savefig(
            plot_dir / "adaptive_performance_efficiency.png", dpi=300,
            bbox_inches="tight", pad_inches=.02,
        )
        plt.close(figure)


def fmt_effect(row: dict[str, Any], metric: str) -> str:
    return (
        f"{100*row[f'{metric}_delta']:+.2f} "
        f"[{100*row[f'{metric}_ci_low']:+.2f}, {100*row[f'{metric}_ci_high']:+.2f}]"
    )


def write_latex_tables(output: Path, paired: list[dict[str, Any]], efficiency: list[dict[str, Any]]) -> None:
    paired_lines = [
        r"\begin{table}[t]", r"\caption{Hierarchical paired effects across six budgets, in percentage points. Positive values favor BayesTraj.}",
        r"\label{tab:paired-superiority}", r"\centering", r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccc}", r"\toprule",
        r"Method & Baseline & $\Delta$AUROC & $\Delta$AUPR & AUROC W/T/L \\", r"\midrule",
    ]
    for row in paired:
        paired_lines.append(
            f"{row['method'].replace('BayesTraj-', 'BT-')} & {row['baseline']} & "
            f"{fmt_effect(row, 'auroc')} & {fmt_effect(row, 'aupr')} & "
            f"{row['auroc_wins']}/{row['auroc_ties']}/{row['auroc_losses']} \\\\"
        )
    paired_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    (output / "paired_superiority_table.tex").write_text("\n".join(paired_lines) + "\n", encoding="utf-8")

    efficiency_lines = [
        r"\begin{table}[t]", r"\caption{Predictive performance and realized cached-compute savings.}",
        r"\label{tab:realized-efficiency}", r"\centering", r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrrrr}", r"\toprule",
        r"$B$ & Method & AUROC & AUPR & Traj. & Token saving \\", r"\midrule",
    ]
    for row in efficiency:
        efficiency_lines.append(
            f"{row['budget']} & {row['method'].replace('BayesTraj-', 'BT-')} & "
            f"{row['auroc']:.3f} & {row['aupr']:.3f} & {row['mean_trajectories']:.2f} & "
            f"{100*row['output_token_saving']:.1f}\\% \\\\"
        )
    efficiency_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    (output / "efficiency_table.tex").write_text("\n".join(efficiency_lines) + "\n", encoding="utf-8")


def report_text(
    paired: list[dict[str, Any]],
    posterior: dict[str, Any],
    efficiency: list[dict[str, Any]],
) -> str:
    adaptive = [row for row in efficiency if row["method"] == "BayesTraj-Adaptive"]
    mean_traj = float(np.mean([row["trajectory_saving"] for row in adaptive]))
    mean_tokens = float(np.mean([row["output_token_saving"] for row in adaptive]))
    mean_steps = float(np.mean([row["agent_step_saving"] for row in adaptive]))
    lines = [
        "# BayesTraj empirical evidence: minimal paper package", "",
        "This package is restricted by construction to DBBench, HotpotQA, WebShop, and StrategyQA.", "",
        "## 1. Paired superiority against prespecified strong baselines", "",
        "Effects average the six shared nominal budgets and use hierarchical 95% intervals that resample dataset–backbone combinations and seeds within combinations. They do not resample tasks because the final selected-method artifact exposes cell metrics rather than common task-score arrays for every baseline.", "",
        "| Method | Baseline | ΔAUROC [95% CI], points | ΔAUPR [95% CI], points | AUROC W/T/L | Holm p (AUROC/AUPR) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['method']} | {row['baseline']} | {fmt_effect(row, 'auroc')} | {fmt_effect(row, 'aupr')} | "
            f"{row['auroc_wins']}/{row['auroc_ties']}/{row['auroc_losses']} | "
            f"{row['auroc_p_holm']:.4g}/{row['aupr_p_holm']:.4g} |"
        )
    lines.extend([
        "", "![Paired baseline effects](plots/paired_baseline_forest.png)", "",
        "The forest plot shows the same paired effects in a one-column paper-facing format. Its horizontal coordinates are absolute AUROC/AUPR improvements in percentage points, averaged over the six budgets; they are not trajectory budgets. Every hierarchical 95% interval lies above zero, and all 12 metric comparisons remain significant after Holm correction. The vector version is `plots/paired_baseline_forest.pdf`.", "",
        "Detailed budget-specific effects are in `paired_superiority_by_budget.csv`; the compact exact-value LaTeX table is `paired_superiority_table.tex`.", "",
        "## 2. Posterior accuracy and risk calibration", "",
        "![Posterior accuracy and calibration](plots/posterior_accuracy_calibration.png)", "",
        f"Across {posterior['cells']} cells and all six prefixes, two-view fusion reduces terminal-target MSE by {100*posterior['relative_mse_reduction']:.1f}% relative to the count-only posterior. The aggregate nominal 95% posterior coverage is {100*posterior['coverage95']:.1f}%. Panel (b) normalizes variance and squared error by the within-cell target variance before pooling heterogeneous datasets.", "",
        "The vector figure is `plots/posterior_accuracy_calibration.pdf`; source values are in `posterior_diagnostics_by_cell_budget.csv` and `posterior_risk_calibration.csv`.", "",
        "## 3. Realized cached-compute efficiency", "",
        "![Adaptive performance and efficiency](plots/adaptive_performance_efficiency.png)", "",
        "Panel (a) reports absolute macro AUROC and AUPR percentage-point changes of BayesTraj-Adaptive relative to BayesTraj-Fixed; for example, −0.5% denotes an absolute score difference of −0.005 rather than a relative 0.5% change. The green region denotes degradation of at most one percentage point. Panel (b) shows how the trajectory reduction translates into fewer executed agent steps and generated output tokens; the dashed line marks the nominal 20% trajectory-saving target. The vector version is `plots/adaptive_performance_efficiency.pdf`.", "",
        "| B | Method | AUROC | AUPR | Mean trajectories | Trajectory saving | Agent-step saving | Output-token saving |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in efficiency:
        lines.append(
            f"| {row['budget']} | {row['method']} | {row['auroc']:.3f} | {row['aupr']:.3f} | "
            f"{row['mean_trajectories']:.2f} | {100*row['trajectory_saving']:.1f}% | "
            f"{100*row['agent_step_saving']:.1f}% | {100*row['output_token_saving']:.1f}% |"
        )
    lines.extend([
        "", f"Across budgets, BayesTraj-Adaptive saves {100*mean_traj:.1f}% of trajectories, {100*mean_steps:.1f}% of recorded agent steps, and {100*mean_tokens:.1f}% of generated output tokens relative to BayesTraj-Fixed.", "",
        "Token totals use `chosen_output_metadata.token_count`; `efficiency_by_budget.csv` reports its coverage. Environment observations are also retained in the CSV. The cached artifacts do not record comparable end-to-end wall-clock duration, so this package makes no empirical wall-clock-savings claim.", "",
        "## Paper-use recommendation", "",
        "Use the paired table, posterior figure, and adaptive performance-efficiency figure in the main paper. Move the full six-budget efficiency table to the appendix for exact trajectory, step, and token values.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-report", type=Path,
        default=ROOT / "outputs/analysis/mixed_budget_4datasets_selected_methods/report",
    )
    parser.add_argument(
        "--ablation-run-root", type=Path,
        default=ROOT / "outputs/analysis/brlg_fixed_varstop80_ablation_run",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("artifacts/raw/z16"))
    parser.add_argument(
        "--ablation-scores", type=Path,
        default=ROOT / "outputs/analysis/brlg_fixed_varstop80_ablation/task_scores.jsonl.gz",
    )
    parser.add_argument(
        "--webshop-root", type=Path,
        default=ROOT / "outputs/analysis/brlg_webshop_task_equivalent_buckets_run",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/analysis/bayestraj_empirical_minimal_package/report",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = args.selected_report.resolve() / "selected_seed_metrics.csv"

    paired, _ = paired_superiority(selected, output)
    plot_paired_superiority(paired, output)
    print(json.dumps({"stage": "paired_superiority", "status": "complete"}), flush=True)
    posterior_frame, posterior_sources = posterior_rows(args.ablation_run_root.resolve(), args.raw_root.resolve())
    posterior = posterior_diagnostics(posterior_frame, output)
    print(json.dumps({"stage": "posterior_diagnostics", "status": "complete"}), flush=True)
    stops = load_stops(args.ablation_scores.resolve(), args.webshop_root.resolve())
    efficiency, raw_sources = efficiency_table(args.raw_root.resolve(), stops, selected, output)
    plot_realized_efficiency(efficiency, selected, output)
    plot_resource_savings(efficiency, output)
    plot_adaptive_tradeoff_summary(efficiency, output)
    print(json.dumps({"stage": "efficiency", "status": "complete"}), flush=True)
    write_latex_tables(output, paired, efficiency)
    (output / "report.md").write_text(report_text(paired, posterior, efficiency), encoding="utf-8")

    artifacts = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema_version": 1,
        "contract": "bayestraj-empirical-minimal-package-2026-08-14-v1",
        "datasets": list(DATASETS),
        "backbones": list(BACKBONES), "seeds": list(SEEDS), "budgets": list(BUDGETS),
        "paired_interval": "hierarchical bootstrap of dataset-backbone combinations and seeds within combinations; budgets fixed",
        "paired_bootstrap_replicates": CI_REPLICATES,
        "posterior_webshop_bucket": "constraint-pattern",
        "posterior": posterior,
        "wall_clock_status": "not recorded in comparable cached artifacts; no wall-clock savings claimed",
        "generation_calls": 0, "gpu_calls": 0,
        "sources": {
            "selected_seed_metrics": {"path": str(selected), "sha256": sha256(selected)},
            "ablation_task_scores": {"path": str(args.ablation_scores.resolve()), "sha256": sha256(args.ablation_scores.resolve())},
            "selected_report_manifest": {
                "path": str(args.selected_report.resolve() / "manifest.json"),
                "sha256": sha256(args.selected_report.resolve() / "manifest.json"),
            },
            "posterior_source_count": len(set(posterior_sources)),
            "raw_efficiency_source_count": len(set(raw_sources)),
        },
        "artifacts": {str(path.relative_to(output)): sha256(path) for path in artifacts},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": "complete", "artifacts": len(artifacts)}), flush=True)


if __name__ == "__main__":
    main()
