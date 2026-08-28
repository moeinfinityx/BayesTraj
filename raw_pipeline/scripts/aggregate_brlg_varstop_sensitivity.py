#!/usr/bin/env python3
"""Aggregate BR-LG rho/window sensitivity and produce paper candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from paper_ablation_common import expected_cells, write_csv
from evaluate_brlg_varstop_sensitivity import CONTRACT_VERSION, RATIOS, WINDOWS


DATASETS = ("dbbench", "webshop", "strategyqa", "hotpotqa")
DATASET_LABELS = {
    "dbbench": "AgentBench-DB", "webshop": "WebShop",
    "strategyqa": "StrategyQA", "hotpotqa": "HotpotQA",
}
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
BACKBONE_LABELS = {"qwen35": "Qwen", "gemma3": "Gemma", "gptoss20b": "GPT-OSS"}
PERCENT_TICK = FuncFormatter(lambda value, _: f"{value:.2f}%")
SAVING_TICK = FuncFormatter(lambda value, _: f"{value:.0f}%")


def cells() -> list[str]:
    return [name for name in expected_cells() if name.split("-", 1)[0] in DATASETS]


def load(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for name in cells():
        cell_root = root / name
        try:
            manifest = json.loads((cell_root / "manifest.json").read_text())
            if manifest.get("contract_version") != CONTRACT_VERSION or not (cell_root / "COMPLETE").is_file():
                problems.append(f"{name}: contract or completion")
                continue
            with (cell_root / "cell_metrics.csv").open(newline="", encoding="utf-8") as handle:
                records.extend(csv.DictReader(handle))
        except Exception as error:
            problems.append(f"{name}: {error}")
    if problems:
        raise RuntimeError("campaign validation failed:\n" + "\n".join(problems))
    frame = pd.DataFrame.from_records(records)
    integer = ("seed", "budget", "tasks", "minimum_trajectories", "maximum_trajectories")
    numeric = ("rho", "auroc", "aupr", "mean_trajectories", "trajectory_saving", "fraction_early")
    for column in integer:
        frame[column] = frame[column].astype(int)
    for column in numeric:
        frame[column] = frame[column].astype(float)
    return frame


def paired(metrics: pd.DataFrame) -> pd.DataFrame:
    fixed = metrics[metrics.setting == "fixed"].set_index(["cell", "budget"])
    adaptive = metrics[metrics.setting != "fixed"].copy()
    joined = adaptive.join(fixed[["auroc", "aupr"]], on=["cell", "budget"], rsuffix="_fixed")
    joined["delta_auroc"] = joined.auroc - joined.auroc_fixed
    joined["delta_aupr"] = joined.aupr - joined.aupr_fixed
    return joined


def hierarchical_intervals(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    # Budget-average first, retaining the 36 paired cell units.
    cell = frame.groupby(
        ["setting", "rho", "window", "dataset", "backbone", "seed", "cell"], as_index=False
    ).agg(
        delta_auroc=("delta_auroc", "mean"), delta_aupr=("delta_aupr", "mean"),
        saving=("trajectory_saving", "mean"), mean_trajectories=("mean_trajectories", "mean"),
    )
    rng = np.random.default_rng(20260811)
    combos = [(dataset, backbone) for dataset in DATASETS for backbone in BACKBONES]
    output: list[dict[str, Any]] = []
    for setting, group in cell.groupby("setting", sort=False):
        arrays: dict[str, np.ndarray] = {}
        for metric in ("delta_auroc", "delta_aupr"):
            arrays[metric] = np.asarray([
                group[(group.dataset == dataset) & (group.backbone == backbone)]
                .sort_values("seed")[metric].to_numpy(dtype=float)
                for dataset, backbone in combos
            ])
        if any(values.shape != (len(combos), 3) for values in arrays.values()):
            raise RuntimeError(f"{setting}: incomplete dataset-backbone-seed hierarchy")
        combo_indexes = rng.integers(0, len(combos), size=(replicates, len(combos)))
        seed_indexes = rng.integers(0, 3, size=(replicates, len(combos), 3))
        distributions = {
            metric: values[combo_indexes[:, :, None], seed_indexes].mean(axis=(1, 2))
            for metric, values in arrays.items()
        }
        record: dict[str, Any] = {
            "setting": setting, "rho": float(group.rho.iloc[0]), "window": str(group.window.iloc[0]),
            "delta_auroc": float(group.delta_auroc.mean()), "delta_aupr": float(group.delta_aupr.mean()),
            "saving": float(group.saving.mean()), "mean_trajectories": float(group.mean_trajectories.mean()),
        }
        for metric, distribution in distributions.items():
            record[f"{metric}_ci_low"] = float(np.quantile(distribution, 0.025))
            record[f"{metric}_ci_high"] = float(np.quantile(distribution, 0.975))
        output.append(record)
    return pd.DataFrame.from_records(output)


def frontier(summary: pd.DataFrame, cell: pd.DataFrame, output: Path) -> None:
    colors = {"2": "#0072B2", "4": "#D55E00", "6": "#009E73", "all": "#6A3D9A"}
    markers = {0.70: "v", 0.75: "D", 0.80: "o", 0.85: "s", 0.90: "^"}
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True)
    for axis, metric in zip(axes, ("delta_auroc", "delta_aupr"), strict=True):
        axis.axhspan(-1.0, 0.0, color="#E8F4EC", alpha=0.9, zorder=0)
        axis.axhline(0, color="#303642", linewidth=1.1)
        axis.axhline(-1, color="#73808C", linestyle="--", linewidth=1)
        for window in ("2", "4", "6", "all"):
            block = summary[summary.window.astype(str) == window].sort_values("rho")
            x = 100 * block.saving.to_numpy()
            y = 100 * block[metric].to_numpy()
            axis.plot(x, y, color=colors[window], linewidth=1.8, alpha=0.9, label=f"w={window}")
            for _, row in block.iterrows():
                low, high = 100 * row[f"{metric}_ci_low"], 100 * row[f"{metric}_ci_high"]
                axis.errorbar(
                    100 * row.saving, 100 * row[metric],
                    yerr=[[100 * row[metric] - low], [high - 100 * row[metric]]],
                    fmt=markers[float(row.rho)], color=colors[window], markersize=6,
                    markeredgecolor="white", markeredgewidth=0.6, capsize=2.5, linewidth=1,
                )
        default = summary[(summary.window.astype(str) == "4") & np.isclose(summary.rho, 0.8)].iloc[0]
        axis.scatter(100 * default.saving, 100 * default[metric], marker="*", s=230,
                     facecolor="#FFD23F", edgecolor="#222", linewidth=1.1, zorder=8)
        # Dataset-specific default summaries are compact hollow diamonds.
        default_cells = cell[(cell.window.astype(str) == "4") & np.isclose(cell.rho, 0.8)]
        for dataset in DATASETS:
            point = default_cells[default_cells.dataset == dataset]
            axis.scatter(100 * point.saving.mean(), 100 * point[metric].mean(), marker="D", s=42,
                         facecolors="none", edgecolors="#333", linewidths=1, zorder=7)
        axis.set_xlabel("Realized trajectory saving (%)")
        axis.set_ylabel(f"Δ{metric.removeprefix('delta_').upper()} (percentage points)")
        axis.xaxis.set_major_formatter(SAVING_TICK)
        axis.yaxis.set_major_formatter(PERCENT_TICK)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles[:4], labels[:4], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.94))
    figure.suptitle("Sensitivity of BR-LG-VarStop to target cost ratio and search width", fontsize=17, fontweight="bold", y=1.0)
    figure.text(0.5, 0.02, "Points are 36-cell macro means; vertical bars are hierarchical 95% intervals. ★ default (ρ=.80,w=4); hollow diamonds: four dataset means.", ha="center", fontsize=9, color="#667085")
    figure.tight_layout(rect=(0, 0.055, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def one_column_auroc(summary: pd.DataFrame, cell: pd.DataFrame, output: Path, replicates: int) -> None:
    """Compact frontier plus default-setting dataset forest for one-column use."""
    colors = {"2": "#0072B2", "4": "#D55E00", "6": "#009E73", "all": "#6A3D9A"}
    figure, (front, forest) = plt.subplots(
        1, 2, figsize=(7.15, 3.15), gridspec_kw={"width_ratios": (1.28, 1.0)}
    )

    # Left: the complete cost–AUROC frontier. Ratios decrease from left to
    # right, so labels are needed only on the highlighted w=4 curve.
    front.axhspan(-1.0, 0.0, color="#E8F4EC", alpha=0.9, zorder=0)
    front.axhline(0, color="#303642", linewidth=0.9)
    front.axhline(-1, color="#73808C", linestyle="--", linewidth=0.8)
    for window in ("2", "4", "6", "all"):
        block = summary[summary.window.astype(str) == window].sort_values("saving")
        x = 100 * block.saving.to_numpy()
        y = 100 * block.delta_auroc.to_numpy()
        front.plot(x, y, "o-", color=colors[window], linewidth=1.25, markersize=3.5, label=f"w={window}")
        front.vlines(
            x, 100 * block.delta_auroc_ci_low.to_numpy(), 100 * block.delta_auroc_ci_high.to_numpy(),
            color=colors[window], alpha=0.48, linewidth=0.65,
        )
        if window == "4":
            for _, row in block.iterrows():
                if np.isclose(row.rho, (0.9, 0.8, 0.7)).any():
                    front.annotate(f"{row.rho:.1f}", (100*row.saving, 100*row.delta_auroc),
                                   xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.2,
                                   color=colors[window])
    default = summary[(summary.window.astype(str) == "4") & np.isclose(summary.rho, .8)].iloc[0]
    front.scatter(100*default.saving, 100*default.delta_auroc, marker="*", s=105,
                  facecolor="#FFD23F", edgecolor="#222", linewidth=0.7, zorder=7)
    front.set_xlabel("Trajectory saving (%)", fontsize=8)
    front.set_ylabel("ΔAUROC (percentage points)", fontsize=8)
    front.xaxis.set_major_formatter(SAVING_TICK)
    front.yaxis.set_major_formatter(PERCENT_TICK)
    front.set_title("(a) Cost–performance frontier", fontsize=9.2, fontweight="bold")
    front.tick_params(labelsize=7)
    front.grid(alpha=0.16)
    front.spines[["top", "right"]].set_visible(False)
    front.legend(loc="lower left", ncol=2, fontsize=6.2, frameon=False, handlelength=1.8,
                 columnspacing=0.8, borderaxespad=0.2)

    # Right: default-setting heterogeneity over four datasets plus the macro.
    default_cells = cell[(cell.window.astype(str) == "4") & np.isclose(cell.rho, .8)]
    rng = np.random.default_rng(20260812)
    rows: list[tuple[str, float, float, float]] = []
    for dataset in DATASETS:
        block = default_cells[default_cells.dataset == dataset]
        values = np.asarray([
            block[block.backbone == backbone].sort_values("seed").delta_auroc.to_numpy(dtype=float)
            for backbone in BACKBONES
        ])
        backbone_indexes = rng.integers(0, 3, size=(replicates, 3))
        seed_indexes = rng.integers(0, 3, size=(replicates, 3, 3))
        distribution = values[backbone_indexes[:, :, None], seed_indexes].mean(axis=(1, 2))
        rows.append((DATASET_LABELS[dataset], float(values.mean()),
                     float(np.quantile(distribution, .025)), float(np.quantile(distribution, .975))))
    rows.append(("Macro", float(default.delta_auroc),
                 float(default.delta_auroc_ci_low), float(default.delta_auroc_ci_high)))
    positions = np.arange(len(rows))[::-1]
    forest.axvspan(-1.0, 0.0, color="#E8F4EC", alpha=0.9, zorder=0)
    forest.axvline(0, color="#303642", linewidth=0.9)
    forest.axvline(-1, color="#73808C", linestyle="--", linewidth=0.8)
    for y, (label, point, low, high) in zip(positions, rows, strict=True):
        is_macro = label == "Macro"
        forest.errorbar(100*point, y, xerr=[[100*(point-low)], [100*(high-point)]], fmt="o",
                        markersize=5 if is_macro else 4, color="#087F78" if is_macro else "#526074",
                        markerfacecolor="#087F78" if is_macro else "white", markeredgewidth=1.2,
                        capsize=2.2, linewidth=1.15)
    forest.set_yticks(positions, [row[0] for row in rows], fontsize=7)
    forest.set_xlabel("ΔAUROC (percentage points)", fontsize=8)
    forest.xaxis.set_major_formatter(PERCENT_TICK)
    forest.set_title("(b) Default by dataset", fontsize=9.2, fontweight="bold")
    forest.tick_params(axis="x", labelsize=7)
    forest.grid(axis="x", alpha=0.16)
    forest.spines[["top", "right", "left"]].set_visible(False)
    forest.tick_params(axis="y", length=0)

    figure.suptitle("BR-LG-VarStop sensitivity", fontsize=10.5, fontweight="bold", y=1.01)
    figure.text(0.5, 0.005, "★ default (ρ=.80,w=4); bars are hierarchical 95% intervals; green denotes ≤1-point degradation.",
                ha="center", fontsize=6.4, color="#667085")
    figure.tight_layout(rect=(0, .045, 1, .97), w_pad=1.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def one_column_auroc_window_effect(
    summary: pd.DataFrame, cell: pd.DataFrame, output: Path, replicates: int
) -> None:
    """Compact nonredundant view: frontier plus paired window effect."""
    colors = {"2": "#0072B2", "4": "#D55E00", "6": "#009E73", "all": "#6A3D9A"}
    # Draw at final IEEE one-column width instead of shrinking a full-width
    # figure after export.
    figure, (front, effect) = plt.subplots(
        1, 2, figsize=(3.5, 2.05), gridspec_kw={"width_ratios": (1.08, 1.0)}
    )

    # Panel (a): actual cost-performance tradeoff for all operating points.
    front.axhspan(-1.0, 0.0, color="#E8F4EC", alpha=0.9, zorder=0)
    front.axhline(0, color="#303642", linewidth=0.9)
    front.axhline(-1, color="#73808C", linestyle="--", linewidth=0.8)
    for window in ("2", "4", "6", "all"):
        block = summary[summary.window.astype(str) == window].sort_values("saving")
        x, y = 100 * block.saving.to_numpy(), 100 * block.delta_auroc.to_numpy()
        front.plot(x, y, "o-", color=colors[window], linewidth=.9, markersize=2.4, label=f"w={window}")
        front.vlines(x, 100*block.delta_auroc_ci_low, 100*block.delta_auroc_ci_high,
                     color=colors[window], alpha=.38, linewidth=.42)
        if window == "4":
            for _, row in block.iterrows():
                if any(np.isclose(row.rho, value) for value in (.9, .8, .7)):
                    front.annotate(f"ρ={row.rho:.1f}", (100*row.saving, 100*row.delta_auroc),
                                   xytext=(0, 5), textcoords="offset points", ha="center",
                                   fontsize=4.4, color=colors[window])
    default = summary[(summary.window.astype(str) == "4") & np.isclose(summary.rho, .8)].iloc[0]
    front.scatter(100*default.saving, 100*default.delta_auroc, marker="*", s=46,
                  facecolor="#FFD23F", edgecolor="#222", linewidth=.5, zorder=7)
    front.set_xlabel("Realized saving (%)", fontsize=5.8, labelpad=1.5)
    front.set_ylabel("ΔAUROC vs. fixed (pp)", fontsize=5.8, labelpad=1.5)
    front.xaxis.set_major_formatter(SAVING_TICK)
    front.yaxis.set_major_formatter(PERCENT_TICK)
    front.set_title("(a) Cost–performance", fontsize=6.5, fontweight="bold", pad=2)
    front.tick_params(labelsize=5.2, length=2.2, width=.5, pad=1.2)
    front.grid(alpha=.16, linewidth=.35)
    front.spines[["top", "right"]].set_visible(False)
    front.legend(loc="lower left", ncol=2, fontsize=4.4, frameon=False,
                 handlelength=1.35, columnspacing=.45, handletextpad=.25,
                 labelspacing=.2, borderaxespad=.15)

    # Panel (b): paired within-cell AUROC effect of changing w while holding rho.
    # This removes the fixed-budget effect shared by both methods and directly
    # isolates sensitivity to the search support.
    rng = np.random.default_rng(20260813)
    windows = ("2", "6", "all")
    effect.axhline(0, color="#303642", linewidth=.9)
    for window in windows:
        points, lows, highs = [], [], []
        for ratio in RATIOS:
            candidate = cell[(cell.window.astype(str) == window) & np.isclose(cell.rho, ratio)]
            reference = cell[(cell.window.astype(str) == "4") & np.isclose(cell.rho, ratio)]
            joined = candidate.set_index("cell")[["dataset", "backbone", "seed", "delta_auroc"]].join(
                reference.set_index("cell")[["delta_auroc"]], rsuffix="_w4"
            )
            joined["effect"] = joined.delta_auroc - joined.delta_auroc_w4
            arrays = np.asarray([
                joined[(joined.dataset == dataset) & (joined.backbone == backbone)]
                .sort_values("seed").effect.to_numpy(dtype=float)
                for dataset in DATASETS for backbone in BACKBONES
            ])
            combo_indexes = rng.integers(0, 12, size=(replicates, 12))
            seed_indexes = rng.integers(0, 3, size=(replicates, 12, 3))
            distribution = arrays[combo_indexes[:, :, None], seed_indexes].mean(axis=(1, 2))
            points.append(float(arrays.mean()))
            lows.append(float(np.quantile(distribution, .025)))
            highs.append(float(np.quantile(distribution, .975)))
        x = np.asarray(RATIOS)
        y = 100 * np.asarray(points)
        effect.plot(x, y, "o-", color=colors[window], linewidth=.9, markersize=2.5, label=f"w={window}")
        effect.fill_between(
            x, 100*np.asarray(lows), 100*np.asarray(highs),
            color=colors[window], alpha=.075, linewidth=0,
        )
    effect.scatter(.8, 0, marker="*", s=46, facecolor="#FFD23F", edgecolor="#222", linewidth=.5, zorder=7)
    effect.set_xlabel("Target cost ratio ρ", fontsize=5.8, labelpad=1.5)
    effect.set_ylabel("ΔAUROC vs. w=4 (pp)", fontsize=5.8, labelpad=1.5)
    effect.yaxis.set_major_formatter(PERCENT_TICK)
    effect.set_title("(b) Window sensitivity", fontsize=6.5, fontweight="bold", pad=2)
    effect.set_xticks(RATIOS, [f"{value:.2f}" for value in RATIOS], fontsize=5.0)
    effect.tick_params(axis="y", labelsize=5.2, length=2.2, width=.5, pad=1.2)
    effect.grid(alpha=.16, linewidth=.35)
    effect.spines[["top", "right"]].set_visible(False)
    effect.legend(loc="best", fontsize=4.4, frameon=False, handlelength=1.35,
                  handletextpad=.25, labelspacing=.2)
    effect.text(
        .97, .035, "ribbons: hierarchical 95% CI",
        transform=effect.transAxes, ha="right", va="bottom",
        fontsize=3.9, color="#303642", fontweight="bold",
        bbox={"boxstyle": "round,pad=.16", "facecolor": "white", "edgecolor": "none", "alpha": .88},
        zorder=9,
    )

    figure.text(.5, .055, "★ default (ρ=.80, w=4); bars/ribbons: hierarchical 95% intervals",
                ha="center", fontsize=5.4, color="black", fontweight="bold")
    figure.subplots_adjust(left=.12, right=.995, top=.92, bottom=.24, wspace=.42)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight", pad_inches=.01)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.01)
    plt.close(figure)


def heatmap(cell: pd.DataFrame, output: Path, source: Path) -> None:
    order = [(str(window), ratio) for window in WINDOWS for ratio in RATIOS]
    rows: list[dict[str, Any]] = []
    matrices = {metric: [] for metric in ("delta_auroc", "delta_aupr")}
    labels: list[str] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            block = cell[(cell.dataset == dataset) & (cell.backbone == backbone)]
            label = f"{DATASET_LABELS[dataset]} / {BACKBONE_LABELS[backbone]}"
            labels.append(label)
            record: dict[str, Any] = {"dataset": dataset, "backbone": backbone, "row": label}
            for metric in matrices:
                values = []
                for window, ratio in order:
                    value = float(block[(block.window.astype(str) == window) & np.isclose(block.rho, ratio)][metric].mean())
                    values.append(value)
                    record[f"{metric}_w{window}_rho{int(100*ratio)}"] = value
                matrices[metric].append(values)
            rows.append(record)
    labels.append("Macro average")
    for metric in matrices:
        matrices[metric].append(np.mean(np.asarray(matrices[metric]), axis=0).tolist())
    macro: dict[str, Any] = {"dataset": "macro", "backbone": "macro", "row": "Macro average"}
    for metric in matrices:
        for (window, ratio), value in zip(order, matrices[metric][-1], strict=True):
            macro[f"{metric}_w{window}_rho{int(100*ratio)}"] = value
    rows.append(macro)
    write_csv(source, rows)

    maximum = max(abs(float(np.min(matrices["delta_auroc"]))), abs(float(np.max(matrices["delta_auroc"]))),
                  abs(float(np.min(matrices["delta_aupr"]))), abs(float(np.max(matrices["delta_aupr"]))))
    figure, axes = plt.subplots(1, 2, figsize=(22, 8.8), sharey=True)
    for axis, metric in zip(axes, ("delta_auroc", "delta_aupr"), strict=True):
        matrix = 100 * np.asarray(matrices[metric])
        image = axis.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-100*maximum, vcenter=0, vmax=100*maximum), aspect="auto")
        axis.set_xticks(np.arange(len(order)), [f".{int(100*r):02d}" for _, r in order], rotation=55, ha="left", fontsize=8)
        axis.xaxis.tick_top()
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=9)
        axis.tick_params(length=0)
        for row_index in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row_index, column]
                axis.text(column, row_index, f"{value:+.1f}%", ha="center", va="center", fontsize=6.0,
                          color="white" if abs(value) > 0.58 * 100 * maximum else "#111827",
                          fontweight="bold" if row_index == len(labels)-1 else "normal")
        for boundary in (4.5, 9.5, 14.5):
            axis.axvline(boundary, color="#344054", linewidth=1.6)
        for boundary in (2.5, 5.5, 8.5, 11.5):
            axis.axhline(boundary, color="#475467", linewidth=1.3)
        # Default rho=.80, w=4 column.
        default_column = order.index(("4", 0.80))
        axis.add_patch(plt.Rectangle((default_column-.5, -.5), 1, len(labels), fill=False, edgecolor="#FFD23F", linewidth=2.3))
        axis.set_title(f"Δ{metric.removeprefix('delta_').upper()} (percentage points)", fontsize=14, fontweight="bold", pad=60)
        for group_index, window in enumerate(("2", "4", "6", "all")):
            axis.text(group_index*5+2, -1.65, f"w={window}", ha="center", va="center", fontsize=10, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axes, orientation="horizontal", fraction=0.035, pad=0.09, aspect=50)
    colorbar.ax.xaxis.set_major_formatter(PERCENT_TICK)
    colorbar.set_label("Adaptive worse  ←  percentage-point difference from fixed  →  Adaptive better")
    figure.suptitle("BR-LG-VarStop sensitivity across datasets and backbones", fontsize=19, fontweight="bold", y=1.02)
    figure.text(0.5, 0.02, "Columns show target cost ratio ρ; each cell averages three seeds and six budgets. Yellow outline marks the default (ρ=.80,w=4).", ha="center", fontsize=9, color="#667085")
    figure.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.14, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def selection_table(summary: pd.DataFrame, cell: pd.DataFrame, output_csv: Path, output_md: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        block = cell[cell.setting == row.setting]
        dataset = block.groupby("dataset", as_index=False).agg(delta_auroc=("delta_auroc", "mean"), delta_aupr=("delta_aupr", "mean"))
        within = block[(block.delta_auroc >= -0.01) & (block.delta_aupr >= -0.01)]
        records.append({
            "window": row.window, "rho": row.rho, "realized_saving": row.saving,
            "delta_auroc": row.delta_auroc, "delta_aupr": row.delta_aupr,
            "worst_dataset_delta_auroc": float(dataset.delta_auroc.min()),
            "worst_dataset_delta_aupr": float(dataset.delta_aupr.min()),
            "cells_within_1pt": len(within), "cells_total": len(block),
            "default": str(row.window) == "4" and np.isclose(row.rho, .8),
        })
    table = pd.DataFrame.from_records(records)
    # A setting is Pareto-efficient if no other setting has at least as much
    # saving and no worse AUROC/AUPR, with one strict improvement.
    pareto = []
    for index, row in table.iterrows():
        dominated = False
        for other_index, other in table.iterrows():
            if index == other_index:
                continue
            weak = other.realized_saving >= row.realized_saving and other.delta_auroc >= row.delta_auroc and other.delta_aupr >= row.delta_aupr
            strict = other.realized_saving > row.realized_saving or other.delta_auroc > row.delta_auroc or other.delta_aupr > row.delta_aupr
            if weak and strict:
                dominated = True
                break
        pareto.append(not dominated)
    table["pareto"] = pareto
    table = table.sort_values(["window", "rho"], key=lambda series: series.astype(str) if series.name == "window" else series)
    table.to_csv(output_csv, index=False)
    lines = [
        "| w | ρ | Saving | ΔAUROC | ΔAUPR | Worst-dataset ΔAUROC | Worst-dataset ΔAUPR | Cells within 1 pt | Pareto |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in table.itertuples():
        label = f"**{row.window}**" if row.default else str(row.window)
        lines.append(
            f"| {label} | {row.rho:.2f} | {100*row.realized_saving:.1f}% | {100*row.delta_auroc:+.2f} | "
            f"{100*row.delta_aupr:+.2f} | {100*row.worst_dataset_delta_auroc:+.2f} | "
            f"{100*row.worst_dataset_delta_aupr:+.2f} | {row.cells_within_1pt}/{row.cells_total} | {'✓' if row.pareto else ''} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hierarchical-replicates", type=int, default=10000)
    args = parser.parse_args()
    run = args.run_root.resolve() / "cells"
    output = args.output_root.resolve()
    plots = output / "report/plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = load(run)
    metrics.to_csv(output / "cell_budget_metrics.csv", index=False)
    comparison = paired(metrics)
    comparison.to_csv(output / "paired_cell_budget_metrics.csv", index=False)
    cell = comparison.groupby(
        ["setting", "rho", "window", "dataset", "backbone", "seed", "cell"], as_index=False
    ).agg(delta_auroc=("delta_auroc", "mean"), delta_aupr=("delta_aupr", "mean"),
          saving=("trajectory_saving", "mean"), mean_trajectories=("mean_trajectories", "mean"))
    cell.to_csv(output / "paired_cell_summary.csv", index=False)
    summary = hierarchical_intervals(comparison, args.hierarchical_replicates)
    summary.to_csv(output / "sensitivity_summary.csv", index=False)
    frontier(summary, cell, plots / "rho_window_frontier.png")
    one_column_auroc(summary, cell, plots / "rho_window_auroc_one_column.png", args.hierarchical_replicates)
    one_column_auroc_window_effect(
        summary, cell, plots / "rho_window_auroc_window_effect_one_column.png",
        args.hierarchical_replicates,
    )
    heatmap(cell, plots / "rho_window_robustness_heatmap.png", output / "rho_window_robustness_heatmap.csv")
    table = selection_table(summary, cell, output / "sensitivity_selection_table.csv", output / "sensitivity_selection_table.md")
    default = summary[(summary.window.astype(str) == "4") & np.isclose(summary.rho, .80)].iloc[0]
    conservative = summary[(summary.window.astype(str) == "2") & np.isclose(summary.rho, .80)].iloc[0]
    aggressive = summary[(summary.window.astype(str) == "4") & np.isclose(summary.rho, .75)].iloc[0]
    markdown = [
        "# BR-LG-VarStop sensitivity to target cost ratio and search width", "",
        "All BR-LG estimator components are frozen. Only the target cost ratio ρ and early-search width w vary. Results cover AgentBench-DB, WebShop, StrategyQA, and HotpotQA; three backbones; three seeds; and six budgets.", "",
        "## Main observations", "",
        f"- The target cost ratio ρ controls the principal tradeoff. The default (ρ=.80,w=4) realizes {100*default.saving:.1f}% saving with ΔAUROC {100*default.delta_auroc:+.2f} and ΔAUPR {100*default.delta_aupr:+.2f} percentage points relative to fixed budget.",
        f"- Narrowing the window at the same ρ=.80 to w=2 is more conservative: {100*conservative.saving:.1f}% saving with ΔAUROC {100*conservative.delta_auroc:+.2f} and ΔAUPR {100*conservative.delta_aupr:+.2f} points.",
        f"- A more aggressive (ρ=.75,w=4) setting reaches {100*aggressive.saving:.1f}% saving while retaining macro losses below one point: ΔAUROC {100*aggressive.delta_auroc:+.2f} and ΔAUPR {100*aggressive.delta_aupr:+.2f}. Its worst-dataset losses are larger, as shown in the table.",
        "- Curves for w=4, w=6, and all-prefix search nearly overlap at a fixed ρ. This indicates substantially greater sensitivity to the requested cost ratio than to enlarging the search window beyond four prefixes.", "",
        "## Recommended one-column AUROC sensitivity figure", "", "![One-column AUROC frontier and window effect](plots/rho_window_auroc_window_effect_one_column.png)", "",
        "Panel (a) presents the realized cost–performance frontier. Panel (b) isolates search-window sensitivity by subtracting the w=4 AUROC at the same ρ, so it does not repeat the cost-ratio relationship from panel (a). Y-axis values are AUROC percentage-point changes: for example, −0.25% means a decrease of 0.25 percentage points, not a 25% relative decrease. The curve-following ribbons in panel (b) are hierarchical 95% confidence intervals.", "",
        "## One-column AUROC sensitivity figure", "", "![One-column AUROC sensitivity](plots/rho_window_auroc_one_column.png)", "",
        "Panel (a) presents all 20 operating points; labels on the highlighted w=4 curve identify selected ρ values. Panel (b) reports the four dataset effects and macro effect at the default (ρ=.80,w=4), averaging three backbones and three seeds within each dataset.", "",
        "## Compact macro frontier", "", "![Macro sensitivity frontier](plots/rho_window_frontier.png)", "",
        "## Dataset–backbone robustness heatmap", "", "![Robustness heatmap](plots/rho_window_robustness_heatmap.png)", "",
        "## Compact operating-point table", "", (output / "sensitivity_selection_table.md").read_text(), "",
        "The default BR-LG-VarStop80 operating point is ρ=.80 and w=4. Confidence intervals use hierarchical resampling of dataset–backbone combinations and seeds after averaging budgets within each cell. Correctness labels are used only for final AUROC/AUPR evaluation.", "",
    ]
    (output / "report/report.md").write_text("\n".join(markdown), encoding="utf-8")
    manifest = {
        "contract_version": CONTRACT_VERSION, "cells": 36, "datasets": list(DATASETS),
        "backbones": list(BACKBONES), "seeds": [101, 202, 303], "settings": 20,
        "ratios": list(RATIOS), "windows": list(WINDOWS), "hierarchical_replicates": args.hierarchical_replicates,
        "default": {"rho": .8, "window": 4}, "new_generation_calls": 0, "gpu_calls": 0,
        "rows_in_selection_table": len(table),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "cells": 36, "settings": 20}, indent=2))


if __name__ == "__main__":
    main()
