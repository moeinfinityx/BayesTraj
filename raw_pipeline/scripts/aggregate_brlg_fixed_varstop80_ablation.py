#!/usr/bin/env python3
"""Aggregate and report the joint BR-LG-Fixed/VarStop80 ablation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import itertools
import json
import math
import random
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from paper_ablation_common import expected_cells, holm_adjust, sha256, write_csv
from evaluate_brlg_fixed_varstop80_ablation import (
    CONTRACT_VERSION, FULL_FIXED, FULL_VAR,
)
from bayestraj_common import BUDGETS, EXPECTED

REPORT_DATASETS = ("dbbench", "strategyqa", "hotpotqa", "webshop")


def report_cells() -> list[str]:
    return [name for name in expected_cells() if name.split("-", 1)[0] in REPORT_DATASETS]


CONTRASTS: dict[str, dict[str, Any]] = {
    "trajectory_update": {"full": FULL_VAR, "ablation": "BR-Count-VarStop80-LCB95", "group": "mechanism", "component": "Gaussian trajectory-feature update", "alternative": "No trajectory update"},
    "var_vs_fixed": {"full": FULL_VAR, "ablation": FULL_FIXED, "group": "stopping", "component": "VarStop80 allocation", "alternative": "Strict fixed B"},
    "var_vs_nonadaptive": {"full": FULL_VAR, "ablation": "BR-LG-NonAdaptive80-LCB95", "group": "stopping", "component": "Task-adaptive variance stop", "alternative": "Nonadaptive 80% allocation"},
    "count_prior_fusion": {"full": FULL_VAR, "ablation": "BR-LG-VarStop80-LikelihoodOnly-LCB95", "group": "mechanism", "component": "Count-prior fusion", "alternative": "Trajectory likelihood only"},
    "full_covariance": {"full": FULL_VAR, "ablation": "BR-LG-VarStop80-DiagonalCov-LCB95", "group": "mechanism", "component": "Full multivariate covariance", "alternative": "Diagonal covariance"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_campaign(cells_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in report_cells():
        root = cells_root / name
        if not (root / "COMPLETE").is_file():
            errors.append(f"{name}: incomplete")
            continue
        manifest = json.loads((root / "manifest.json").read_text())
        manifests.append(manifest)
        dataset = name.split("-", 1)[0]
        checks = {
            "contract": manifest.get("contract_version") == CONTRACT_VERSION,
            "tasks": int(manifest.get("tasks", -1)) == EXPECTED[dataset],
            "variants": int(manifest.get("variant_count", -1)) == 6,
            "generation": int(manifest.get("new_generation_calls", -1)) == 0,
            "gpu": int(manifest.get("gpu_calls", -1)) == 0,
            "reproduction": int(manifest.get("reproduction", {}).get("stop_mismatches", -1)) == 0
                and float(manifest.get("reproduction", {}).get("maximum_score_error", 1)) <= 1e-9,
        }
        errors.extend(f"{name}: {key}" for key, passed in checks.items() if not passed)
        selected = read_csv(root / "cell_metrics.csv")
        if len(selected) != 6 * len(BUDGETS):
            errors.append(f"{name}: metric rows {len(selected)}")
        for row in selected:
            for key in ("seed", "budget", "tasks", "failures", "minimum_trajectories", "maximum_trajectories"):
                row[key] = int(row[key])
            for key in ("auroc", "aupr", "mean_trajectories", "trajectory_saving", "variance_trajectories", "fraction_above_budget"):
                row[key] = float(row[key])
            records.append(row)
    if errors:
        raise RuntimeError("campaign validation failed:\n" + "\n".join(errors))
    frame = pd.DataFrame.from_records(records)
    keys = frame[["cell", "variant", "budget"]].astype(str).agg("|".join, axis=1)
    if keys.duplicated().any():
        raise RuntimeError("duplicate metric keys")
    return frame, manifests


def load_groups(path: Path) -> tuple[np.ndarray, dict[tuple[str, int], np.ndarray]]:
    wanted = {spec[key] for spec in CONTRASTS.values() for key in ("full", "ablation")}
    groups: dict[tuple[str, int], np.ndarray] = {}
    labels: np.ndarray | None = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        parsed = (json.loads(line) for line in handle if line.strip())
        for key, iterator in itertools.groupby(parsed, key=lambda row: (row["variant"], int(row["budget"]))):
            if key[0] not in wanted:
                continue
            block = list(iterator)
            block_labels = np.asarray([int(row["label_failure"]) for row in block])
            if labels is None:
                labels = block_labels
            elif not np.array_equal(labels, block_labels):
                raise RuntimeError(f"label ordering mismatch: {path}/{key}")
            groups[key] = np.asarray([float(row["score"]) for row in block])
    if labels is None:
        raise RuntimeError(f"no task scores: {path}")
    return labels, groups


def metric_pair(labels: np.ndarray, full: np.ndarray, alternative: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, full) - roc_auc_score(labels, alternative)),
        float(average_precision_score(labels, full) - average_precision_score(labels, alternative)),
    )


def analyze_cell(payload: tuple[str, str, int]) -> dict[str, Any]:
    path_text, name, replicates = payload
    labels, groups = load_groups(Path(path_text))
    dataset, backbone, seed_text = name.split("-")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little"))
    contrasts: list[dict[str, Any]] = []
    for contrast, spec in CONTRASTS.items():
        point = np.mean([
            metric_pair(labels, groups[(spec["full"], budget)], groups[(spec["ablation"], budget)])
            for budget in BUDGETS
        ], axis=0)
        boot = np.empty((replicates, 2))
        completed = 0
        while completed < replicates:
            indexes = rng.integers(0, len(labels), len(labels))
            chosen = labels[indexes]
            if len(np.unique(chosen)) < 2:
                continue
            boot[completed] = np.mean([
                metric_pair(chosen, groups[(spec["full"], budget)][indexes], groups[(spec["ablation"], budget)][indexes])
                for budget in BUDGETS
            ], axis=0)
            completed += 1
        contrasts.append({
            "contrast": contrast, "dataset": dataset, "backbone": backbone,
            "seed": int(seed_text.removeprefix("seed")), "cell": name,
            "auroc_delta": float(point[0]), "aupr_delta": float(point[1]),
            "auroc_bootstrap": boot[:, 0].tolist(), "aupr_bootstrap": boot[:, 1].tolist(),
        })
    top: list[dict[str, Any]] = []
    for variant in (FULL_FIXED, FULL_VAR):
        for budget in BUDGETS:
            scores = groups[(variant, budget)]
            count = max(1, math.ceil(0.1 * len(scores)))
            chosen = np.argsort(-scores, kind="stable")[:count]
            top.append({"cell": name, "dataset": dataset, "backbone": backbone, "seed": int(seed_text.removeprefix("seed")), "variant": variant, "budget": budget, "top_10pct_failure_precision": float(labels[chosen].mean())})
    return {"contrasts": contrasts, "top": top}


def hierarchical(cell_results: list[dict[str, Any]], replicates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flat = [row for item in cell_results for row in item["contrasts"]]
    cell_rows = [{k: v for k, v in row.items() if not k.endswith("_bootstrap")} for row in flat]
    rng = random.Random(20260811)
    output: list[dict[str, Any]] = []
    for contrast, spec in CONTRASTS.items():
        chosen = [row for row in flat if row["contrast"] == contrast]
        combos: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in chosen:
            combos.setdefault((row["dataset"], row["backbone"]), []).append(row)
        combo_keys = sorted(combos)
        distributions = {metric: np.empty(replicates) for metric in ("auroc", "aupr")}
        for index in range(replicates):
            values = {metric: [] for metric in distributions}
            for combo in rng.choices(combo_keys, k=len(combo_keys)):
                seed_rows = combos[combo]
                for row in rng.choices(seed_rows, k=len(seed_rows)):
                    for metric in distributions:
                        boot = row[f"{metric}_bootstrap"]
                        values[metric].append(float(boot[rng.randrange(len(boot))]))
            for metric in distributions:
                distributions[metric][index] = statistics.fmean(values[metric])
        result: dict[str, Any] = {"contrast": contrast, **spec, "cells": len(chosen)}
        for metric in ("auroc", "aupr"):
            points = [float(row[f"{metric}_delta"]) for row in chosen]
            dist = distributions[metric]
            result[f"{metric}_delta"] = statistics.fmean(points)
            result[f"{metric}_ci_low"] = float(np.quantile(dist, 0.025))
            result[f"{metric}_ci_high"] = float(np.quantile(dist, 0.975))
            result[f"{metric}_wins"] = sum(value > 0 for value in points)
            result[f"{metric}_ties"] = sum(value == 0 for value in points)
            result[f"{metric}_losses"] = sum(value < 0 for value in points)
            left = (int(np.sum(dist <= 0)) + 1) / (replicates + 1)
            right = (int(np.sum(dist >= 0)) + 1) / (replicates + 1)
            result[f"{metric}_p_two_sided"] = min(1.0, 2 * min(left, right))
        output.append(result)
    holm_adjust(output, "auroc")
    holm_adjust(output, "aupr")
    return cell_rows, output


def add_cost(metrics: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        full = metrics[metrics.variant == row["full"]].set_index(["cell", "budget"])
        alt = metrics[metrics.variant == row["ablation"]].set_index(["cell", "budget"])
        joined = full[["mean_trajectories", "trajectory_saving"]].join(
            alt[["mean_trajectories", "trajectory_saving"]], lsuffix="_full", rsuffix="_alternative"
        )
        row["cost_delta"] = float((joined.mean_trajectories_full - joined.mean_trajectories_alternative).mean())
        row["saving_delta"] = float((joined.trajectory_saving_full - joined.trajectory_saving_alternative).mean())


def averages(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant, group in metrics.groupby("variant", sort=False):
        output.append({
            "variant": variant, "rows": len(group), "auroc": float(group.auroc.mean()),
            "aupr": float(group.aupr.mean()), "mean_trajectories": float(group.mean_trajectories.mean()),
            "saving": float(group.trajectory_saving.mean()),
        })
    return output


CORE_MECHANISMS = (
    ("trajectory_update", "trajectory_feature_update", "Gaussian trajectory-\nfeature update", "No trajectory update"),
    ("count_prior_fusion", "count_prior_fusion", "Count-prior fusion", "Trajectory likelihood only"),
    ("full_covariance", "full_covariance", "Full multivariate\ncovariance", "Diagonal covariance"),
    ("var_vs_fixed", "adaptive_stopping", "Adaptive variance\nstopping", "Fixed budget"),
    ("var_vs_nonadaptive", "task_adaptive", "Task-adaptive\nallocation", "Cost-matched nonadaptive"),
)


def core_mechanism_forest(rows: list[dict[str, Any]], output: Path) -> None:
    lookup = {str(row["contrast"]): row for row in rows}
    chosen = [lookup[key] for key, _, _, _ in CORE_MECHANISMS]
    labels = [f"{title.replace(chr(10), ' ')}\nvs. {alternative}" for _, _, title, alternative in CORE_MECHANISMS]
    positions = np.arange(len(chosen))[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), sharey=True)
    for axis, metric in zip(axes, ("auroc", "aupr"), strict=True):
        for y, row in zip(positions, chosen, strict=True):
            point, low, high = (float(row[f"{metric}_{key}"]) for key in ("delta", "ci_low", "ci_high"))
            significant = float(row[f"{metric}_p_holm"]) < 0.05
            color = "#087f78" if significant else "#526074"
            axis.errorbar(point, y, xerr=[[max(0, point-low)], [max(0, high-point)]], fmt="o", markersize=7, color=color, markerfacecolor=color if significant else "white", markeredgewidth=2, capsize=4, linewidth=2.2)
        axis.axvline(0, color="#344054", linewidth=1.2)
        axis.grid(axis="x", alpha=0.2)
        axis.set_xlabel(f"Full minus ablation: Δ{metric.upper()}", fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(positions, labels, fontsize=11)
    figure.suptitle("Paired effects for core BR-LG-VarStop80 mechanisms", fontsize=18, fontweight="bold")
    figure.text(0.5, 0.925, "Hierarchical 95% confidence intervals; teal points remain significant after Holm correction.", ha="center", fontsize=10, color="#667085")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def core_mechanism_heatmap(cell_rows: pd.DataFrame, output: Path, source_output: Path) -> None:
    dataset_order = REPORT_DATASETS
    dataset_labels = {"dbbench": "AgentBench-DB", "strategyqa": "StrategyQA", "hotpotqa": "HotpotQA", "webshop": "WebShop"}
    backbone_order = ("qwen35", "gemma3", "gptoss20b")
    backbone_labels = {"qwen35": "Qwen", "gemma3": "Gemma", "gptoss20b": "GPT-OSS"}
    indexed = cell_rows.groupby(["dataset", "backbone", "contrast"]).aupr_delta.mean()
    matrix: list[list[float]] = []
    labels: list[str] = []
    source: list[dict[str, Any]] = []
    for dataset in dataset_order:
        for backbone in backbone_order:
            values = [float(indexed.loc[(dataset, backbone, key)]) for key, _, _, _ in CORE_MECHANISMS]
            matrix.append(values)
            label = f"{dataset_labels[dataset]} / {backbone_labels[backbone]}"
            labels.append(label)
            source.append({"dataset": dataset, "backbone": backbone, "row": label, **{source_key: value for (_, source_key, _, _), value in zip(CORE_MECHANISMS, values, strict=True)}})
    array = np.asarray(matrix)
    macro = array.mean(axis=0)
    array = np.vstack([array, macro])
    labels.append("Macro average")
    source.append({"dataset": "macro", "backbone": "macro", "row": "Macro average", **{source_key: value for (_, source_key, _, _), value in zip(CORE_MECHANISMS, macro, strict=True)}})
    write_csv(source_output, source)
    maximum = max(0.01, float(np.max(np.abs(array))))
    figure, axis = plt.subplots(figsize=(11.8, 10.2))
    image = axis.imshow(array, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum), aspect="auto")
    axis.set_xticks(np.arange(len(CORE_MECHANISMS)), [title for _, _, title, _ in CORE_MECHANISMS], fontsize=10.5, fontweight="bold")
    axis.xaxis.tick_top()
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=10.5)
    axis.tick_params(length=0)
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[row, column]
            axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=10, fontweight="bold" if row == len(labels)-1 else "normal", color="white" if abs(value) > 0.58 * maximum else "#111827")
    for boundary in (2.5, 5.5, 8.5, 11.5):
        axis.axhline(boundary, color="#475467", linewidth=1.5)
    axis.set_xticks(np.arange(-0.5, len(CORE_MECHANISMS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title("ΔAUPR = full BR-LG-VarStop80 − mechanism ablation", fontsize=20, fontweight="bold", pad=30)
    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", fraction=0.05, pad=0.08, aspect=35)
    colorbar.set_label("Ablation better  ←  ΔAUPR  →  Full method better", fontsize=12)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def core_mechanism_tradeoff(rows: list[dict[str, Any]], output: Path, source_output: Path) -> None:
    lookup = {str(row["contrast"]): row for row in rows}
    records: list[dict[str, Any]] = []
    for key, source_key, title, alternative in CORE_MECHANISMS:
        row = lookup[key]
        records.append({
            "contrast": key, "source_key": source_key, "component": title.replace("\n", " "), "alternative": alternative,
            "delta_auroc": float(row["auroc_delta"]), "delta_aupr": float(row["aupr_delta"]),
            "saving_delta": float(row["saving_delta"]),
        })
    write_csv(source_output, records)
    labels = [
        "Trajectory update", "Count-prior fusion", "Full covariance",
        "Adaptive stopping", "Task-adaptive allocation",
    ]
    positions = np.arange(len(records))[::-1]
    figure, (performance, saving) = plt.subplots(
        2, 1, figsize=(3.5, 3.0), sharey=True,
        gridspec_kw={"height_ratios": (1.45, 1.0), "hspace": .30},
    )

    # Put both predictive metrics in one readable panel.
    auroc = 100 * np.asarray([row["delta_auroc"] for row in records])
    aupr = 100 * np.asarray([row["delta_aupr"] for row in records])
    height = .28
    performance.barh(positions + height/2, auroc, height=height, color="#0072B2", label="ΔAUROC")
    performance.barh(positions - height/2, aupr, height=height, color="#D55E00", label="ΔAUPR")
    performance.axvline(0, color="#344054", linewidth=.65)
    performance.set_xlabel("Full − ablation (percentage points)", fontsize=6.2, labelpad=2)
    performance.set_title("(a) Predictive performance", fontsize=7.2, fontweight="bold", pad=2)
    performance.legend(loc="lower right", frameon=False, fontsize=5.2, ncol=2,
                       handlelength=1.3, columnspacing=.7, handletextpad=.3)
    for y, x1, x2 in zip(positions, auroc, aupr, strict=True):
        x1_text, x1_align = (x1 + .08, "left") if x1 >= 0 else (.08, "left")
        x2_text, x2_align = (x2 + .08, "left") if x2 >= 0 else (.08, "left")
        performance.text(x1_text, y + height/2, f"{x1:+.2f}",
                         va="center", ha=x1_align, fontsize=4.7)
        performance.text(x2_text, y - height/2, f"{x2:+.2f}",
                         va="center", ha=x2_align, fontsize=4.7)

    savings = 100 * np.asarray([row["saving_delta"] for row in records])
    colors = ["#087F78" if value >= 0 else "#C44E52" for value in savings]
    saving.barh(positions, savings, height=.48, color=colors, alpha=.9)
    saving.axvline(0, color="#344054", linewidth=.65)
    saving.set_xlabel("Δ trajectory saving (percentage points)", fontsize=6.2, labelpad=2)
    saving.set_title("(b) Sampling efficiency", fontsize=7.2, fontweight="bold", pad=2)
    for y, value in zip(positions, savings, strict=True):
        x_text, align = (value + .15, "left") if value >= 0 else (.16, "left")
        saving.text(x_text, y, f"{value:+.2f}",
                    va="center", ha=align, fontsize=4.8)

    for axis in (performance, saving):
        axis.set_yticks(positions, labels, fontsize=5.4)
        axis.tick_params(axis="x", labelsize=5.2, length=2.2, width=.5, pad=1.2)
        axis.tick_params(axis="y", length=0, pad=2)
        axis.grid(axis="x", alpha=.18, linewidth=.35)
        axis.spines[["top", "right", "left"]].set_visible(False)
    performance.set_xlim(-1.0, 6.3)
    saving.set_xlim(-1.2, 20.8)
    figure.subplots_adjust(left=.39, right=.985, top=.965, bottom=.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight", pad_inches=.01)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.01)
    plt.close(figure)


def curves(metrics: pd.DataFrame, output: Path) -> None:
    variants = (FULL_FIXED, FULL_VAR, "BR-LG-NonAdaptive80-LCB95")
    labels = {FULL_FIXED: "BR-LG-Fixed", FULL_VAR: "BR-LG-VarStop80", "BR-LG-NonAdaptive80-LCB95": "Nonadaptive80"}
    colors = {FULL_FIXED: "#222222", FULL_VAR: "#0072B2", "BR-LG-NonAdaptive80-LCB95": "#D55E00"}
    macro = metrics[metrics.variant.isin(variants)].groupby(["variant", "budget"], as_index=False).agg(auroc=("auroc", "mean"), aupr=("aupr", "mean"), cost=("mean_trajectories", "mean"))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, metric in zip(axes, ("auroc", "aupr"), strict=True):
        for variant in variants:
            block = macro[macro.variant == variant].sort_values("cost")
            axis.plot(block.cost, block[metric], marker="o", linewidth=2, color=colors[variant], label=labels[variant])
        axis.set_xlabel("Realized mean trajectories")
        axis.set_ylabel(metric.upper())
        axis.grid(alpha=0.2)
    handles, labels_ = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Fixed and adaptive BR-LG at realized trajectory cost", fontweight="bold")
    figure.subplots_adjust(bottom=0.18, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def fmt(value: Any) -> str:
    return f"{float(value):+.3f}"


def report_text(
    summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    dataset: pd.DataFrame,
    budget_summary: pd.DataFrame,
    top_rows: list[dict[str, Any]],
) -> str:
    del top_rows
    lines = [
        "# BayesTraj component ablation",
        "",
        "This paper-scoped report evaluates only the five mechanisms shown in the submission.",
        "",
        "![Core mechanism forest](plots/core_mechanism_forest.png)",
        "",
        "![Core mechanism heatmap](plots/core_mechanism_aupr_heatmap.png)",
        "",
        "![Performance and savings](plots/core_mechanism_tradeoff.png)",
        "",
        "| Mechanism | Comparator | ΔAUROC [95% CI] | ΔAUPR [95% CI] |",
        "|---|---|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['component']} | {row['alternative']} | "
            f"{float(row['auroc_delta']):+.3f} [{float(row['auroc_ci_low']):+.3f}, {float(row['auroc_ci_high']):+.3f}] | "
            f"{float(row['aupr_delta']):+.3f} [{float(row['aupr_ci_low']):+.3f}, {float(row['aupr_ci_high']):+.3f}] |"
        )
    lines += ["", "## Submitted variants", "", "| Variant | AUROC | AUPR | Mean trajectories | Saving |", "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| `{row['variant']}` | {float(row['auroc']):.3f} | {float(row['aupr']):.3f} | "
            f"{float(row['mean_trajectories']):.3f} | {100*float(row['saving']):.1f}% |"
        )
    lines += ["", "## Dataset summary", "", dataset.to_markdown(index=False), "", "## Budget summary", "", budget_summary.to_markdown(index=False), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-bootstrap-replicates", type=int, default=500)
    parser.add_argument("--hierarchical-bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    cells_root = args.run_root.resolve() / "cells"
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics, manifests = load_campaign(cells_root)
    metrics.to_csv(output / "cell_budget_metrics.csv", index=False)
    summary = averages(metrics)
    write_csv(output / "all_budget_summary.csv", summary)
    dataset = metrics[metrics.variant.isin((FULL_FIXED, FULL_VAR))].groupby(["dataset", "variant"], as_index=False).agg(auroc=("auroc", "mean"), aupr=("aupr", "mean"), mean_trajectories=("mean_trajectories", "mean"), saving=("trajectory_saving", "mean"))
    dataset.to_csv(output / "dataset_summary.csv", index=False)
    budget_summary = metrics[metrics.variant.isin((FULL_FIXED, FULL_VAR))].groupby(["budget", "variant"], as_index=False).agg(auroc=("auroc", "mean"), aupr=("aupr", "mean"), mean_trajectories=("mean_trajectories", "mean"), saving=("trajectory_saving", "mean"))
    budget_summary.to_csv(output / "budget_summary.csv", index=False)
    payloads = [(str(cells_root / name / "task_scores.jsonl.gz"), name, args.task_bootstrap_replicates) for name in report_cells()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cell_results = list(pool.map(analyze_cell, payloads))
    cell_rows, inference = hierarchical(cell_results, args.hierarchical_bootstrap_replicates)
    add_cost(metrics, inference)
    write_csv(output / "paired_contrasts_by_cell.csv", cell_rows)
    write_csv(output / "hierarchical_paired_contrasts.csv", inference)
    top_rows = [row for item in cell_results for row in item["top"]]
    write_csv(output / "top10_precision.csv", top_rows)
    core_mechanism_forest(inference, output / "report/plots/core_mechanism_forest.png")
    core_mechanism_heatmap(
        pd.DataFrame.from_records(cell_rows),
        output / "report/plots/core_mechanism_aupr_heatmap.png",
        output / "core_mechanism_aupr_heatmap.csv",
    )
    core_mechanism_tradeoff(
        inference,
        output / "report/plots/core_mechanism_tradeoff.png",
        output / "core_mechanism_tradeoff.csv",
    )
    curves(metrics, output / "report/plots/realized_cost_curves.png")
    report = report_text(summary, inference, dataset, budget_summary, top_rows)
    (output / "report/report.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "report/report.md").write_text(report, encoding="utf-8")
    temporary = output / "task_scores.jsonl.gz.tmp"
    with temporary.open("wb") as destination:
        for name in report_cells():
            with (cells_root / name / "task_scores.jsonl.gz").open("rb") as source:
                shutil.copyfileobj(source, destination)
    temporary.replace(output / "task_scores.jsonl.gz")
    protocol = Path(__file__).resolve().parents[1] / "docs/brlg_fixed_varstop80_ablation_protocol.md"
    validation = {
        "schema_version": 1, "contract_version": CONTRACT_VERSION, "cells": len(manifests),
        "datasets": list(REPORT_DATASETS),
        "variants": 25, "contrasts": len(CONTRASTS), "new_generation_calls": 0, "gpu_calls": 0,
        "all_reproduction_checks_passed": True,
        "task_bootstrap_replicates": args.task_bootstrap_replicates,
        "hierarchical_bootstrap_replicates": args.hierarchical_bootstrap_replicates,
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "artifacts": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in ("validation_manifest.json", "audit.json", "AUDIT_PASSED")
        },
    }
    (output / "validation_manifest.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "cells": len(manifests), "contrasts": len(inference)}, indent=2))


if __name__ == "__main__":
    main()
