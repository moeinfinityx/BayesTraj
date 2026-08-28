#!/usr/bin/env python3
"""Build a separate four-dataset curve report for the selected paper methods."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/analysis/mixed_budget_5datasets_3backbones_3seeds/report"
OUTPUT = ROOT / "outputs/analysis/mixed_budget_4datasets_selected_methods/report"
WEBSHOP_BUCKET_RUN = ROOT / "outputs/analysis/brlg_webshop_task_equivalent_buckets_run/cells"
WEBSHOP_BUCKET_VARIANT = "constraint-pattern"
DATASETS = ("dbbench", "hotpotqa", "webshop", "strategyqa")
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
DATASET_DISPLAY = {
    "dbbench": "DBBench",
    "hotpotqa": "HotpotQA",
    "webshop": "WebShop",
    "strategyqa": "StrategyQA",
}
BACKBONE_DISPLAY = {
    "qwen35": "Qwen-3.5 9B",
    "gemma3": "Gemma-3 12B",
    "gptoss20b": "GPT-OSS 20B",
}
SEEDS = (101, 202, 303)
# Paper-facing comparisons use the six nominal budgets shared by BayesTraj
# and the baselines.  Baseline-only B=2 points are intentionally excluded.
BUDGETS = (3, 4, 6, 8, 12, 16)

OURS = ("BR-LG-Risk-VC4-LCB95", "BR-LG-VarStop80-LCB95")
BASELINES = (
    "SNNE", "MC-OE", "BSE-Ciosek-Fixed", "BSE-Ciosek-Adaptive", "EigV",
    "CoCoA-MaxProb", "CoCoA-PPL", "KLE", "SAUP", "PE", "SentSAR",
    "Degree", "UProp", "SE", "LS", "SD", "PPL",
)
METHODS = OURS + BASELINES
DISPLAY = {
    "BR-LG-Risk-VC4-LCB95": "BayesTraj-Fixed (ours)",
    "BR-LG-VarStop80-LCB95": "BayesTraj-Adaptive (ours)",
    "BSE-Ciosek-Fixed": "BSE-Fixed",
    "BSE-Ciosek-Adaptive": "BSE-Adaptive",
    "UProp": "UProp",
    "Degree": "Degree (N=4)",
}
COLORS = {
    "BR-LG-Risk-VC4-LCB95": "#004488", "BR-LG-VarStop80-LCB95": "#CC3311",
    "SNNE": "#56B4E9", "MC-OE": "#332288", "BSE-Ciosek-Fixed": "#009E73",
    "BSE-Ciosek-Adaptive": "#117733", "EigV": "#777777", "CoCoA-MaxProb": "#A6761D",
    "CoCoA-PPL": "#66A61E", "KLE": "#00A087", "SAUP": "#00A6D6", "PE": "#006D2C",
    "SentSAR": "#2B8CBE", "Degree": "#56B4E9", "UProp": "#E69F00", "SE": "#807D00",
    "LS": "#C51B7D", "SD": "#8C510A", "PPL": "#8C2D4A",
}
MARKERS = {
    method: marker for method, marker in zip(
        METHODS, ("p", "P", "h", ">", "^", ">", "x", "<", ">", "^", "*", "^", "X", "v", "P", "D", "8", "v", "o"), strict=True
    )
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(selected)


def aggregate(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result: list[dict[str, Any]] = []
    for values, selected in sorted(grouped.items()):
        item = dict(zip(keys, values, strict=True))
        item["cells"] = len(selected)
        item["coverage_mean"] = fmean(float(row["coverage"]) for row in selected)
        for metric in ("auroc", "aupr"):
            observed = [float(row[metric]) for row in selected if row.get(metric) not in (None, "")]
            item[f"{metric}_mean"] = fmean(observed)
            item[f"{metric}_std"] = statistics.stdev(observed) if len(observed) > 1 else 0.0
        costs = [
            float(row["mean_trajectories"]) if row.get("mean_trajectories") not in (None, "") else float(row["budget"])
            for row in selected
        ]
        item["mean_trajectories"] = fmean(costs)
        result.append(item)
    return result


def aggregate_seed_macro(seed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average within seed first, then use the three seed means for error bars."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["method"]), int(row["budget"]))].append(row)
    result: list[dict[str, Any]] = []
    for (method, budget), selected in sorted(grouped.items()):
        if len(selected) != len(SEEDS):
            raise RuntimeError(f"{method}/B{budget}: expected {len(SEEDS)} seed macro rows")
        item: dict[str, Any] = {
            "method": method, "budget": budget, "seeds": len(selected),
            "coverage_mean": fmean(float(row["coverage_mean"]) for row in selected),
            "mean_trajectories": fmean(float(row["mean_trajectories"]) for row in selected),
        }
        for metric in ("auroc", "aupr"):
            values = [float(row[f"{metric}_mean"]) for row in selected]
            item[f"{metric}_mean"] = fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values)
        result.append(item)
    return result


def bucket_webshop_rows() -> list[dict[str, Any]]:
    """Reconstruct seed metrics for the requested cached WebShop bucket design."""
    output: list[dict[str, Any]] = []
    method_names = {
        "BR-LG Risk VC4": "BR-LG-Risk-VC4-LCB95",
        "BR-LG-VarStop80": "BR-LG-VarStop80-LCB95",
    }
    for backbone in BACKBONES:
        for seed in SEEDS:
            cell = f"webshop-{backbone}-seed{seed}"
            directory = WEBSHOP_BUCKET_RUN / cell
            manifest = json.loads((directory / "manifest.json").read_text())
            labels = {
                str(row["sample_id"]): int(row["label_failure"])
                for row in (json.loads(line) for line in Path(manifest["checkpoint"]).open() if line.strip())
            }
            grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
            with gzip.open(directory / "task_scores_without_labels.jsonl.gz", "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["split"] == "crossfit" and row["bucket_variant"] == WEBSHOP_BUCKET_VARIANT:
                        grouped[(row["method"], int(row["budget"]))].append(row)
            for (method, budget), selected in sorted(grouped.items()):
                y = np.asarray([labels[str(row["sample_id"])] for row in selected], dtype=int)
                score_values = np.asarray([float(row["score"]) for row in selected])
                output.append({
                    "dataset": "webshop", "backbone": backbone,
                    "combination": f"webshop-{backbone}", "seed": seed,
                    "method": method_names[method], "budget": budget,
                    "trajectory_regime": "complete_z16_crossfit_constraint_pattern_bucket",
                    "tasks": len(labels), "used": len(selected), "coverage": len(selected) / len(labels),
                    "auroc": float(roc_auc_score(y, score_values)),
                    "aupr": float(average_precision_score(y, score_values)),
                    "mean_trajectories": float(np.mean([row["trajectories_used"] for row in selected])),
                    "webshop_bucket_variant": WEBSHOP_BUCKET_VARIANT,
                })
    expected = len(BACKBONES) * len(SEEDS) * 6 * len(OURS)
    if len(output) != expected:
        raise RuntimeError(f"expected {expected} WebShop replacement rows, found {len(output)}")
    return output


def draw(
    axis: Any,
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    uncertainty: str = "bars",
    compact_style: bool = False,
    one_column_style: bool = False,
) -> None:
    observed_y: list[float] = []
    for method in METHODS:
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["budget"]))
        if not selected:
            continue
        x = np.asarray([float(row["mean_trajectories"]) for row in selected])
        y = np.asarray([float(row[f"{metric}_mean"]) for row in selected])
        error = np.asarray([float(row[f"{metric}_std"]) for row in selected])
        observed_y.extend(y.tolist())
        endpoint = len(selected) == 1
        ours = method in OURS
        if one_column_style:
            linewidth = 1.35 if ours else 0.68
            markersize = 3.0 if ours else 1.85
            markeredgewidth = 0.5
            alpha = 0.98 if ours else 0.84
        elif compact_style:
            linewidth = 1.65 if ours else 0.82
            markersize = 3.5 if ours else 2.35
            markeredgewidth = 0.55
            alpha = 0.98 if ours else 0.82
        else:
            linewidth = 2.6 if ours else 1.45
            markersize = 5.7 if ours else 4.2
            markeredgewidth = 0.8
            alpha = 0.95
        common = {
            "label": DISPLAY.get(method, method), "color": COLORS[method], "marker": MARKERS[method],
            "linestyle": "None" if endpoint else "-" if method in OURS else ":",
            "linewidth": linewidth,
            "markersize": markersize,
            "markerfacecolor": "white" if method == "BR-LG-VarStop80-LCB95" else COLORS[method],
            "markeredgecolor": COLORS[method],
            "markeredgewidth": markeredgewidth,
            "alpha": alpha,
            "zorder": 5 if ours else 2,
        }
        if uncertainty == "bars":
            axis.errorbar(
                x, y, yerr=error,
                elinewidth=(0.48 if ours else 0.32) if one_column_style else ((0.52 if ours else 0.38) if compact_style else 0.75),
                capsize=1.0 if one_column_style else (1.25 if compact_style else 1.8),
                capthick=0.4 if one_column_style else (0.45 if compact_style else 0.7),
                **common,
            )
        elif uncertainty == "bands":
            axis.plot(x, y, **common)
            lower, upper = np.clip(y - error, 0.0, 1.0), np.clip(y + error, 0.0, 1.0)
            if endpoint:
                half_width = 0.13
                axis.fill_between(
                    [x[0] - half_width, x[0] + half_width],
                    [lower[0], lower[0]], [upper[0], upper[0]],
                    color=COLORS[method],
                    alpha=(0.10 if ours else 0.065) if one_column_style else ((0.075 if ours else 0.075) if compact_style else 0.16),
                    edgecolor=COLORS[method],
                    linewidth=0.22 if compact_style else 0.18,
                    zorder=1,
                )
            else:
                axis.fill_between(
                    x, lower, upper, color=COLORS[method],
                    alpha=(0.09 if ours else 0.060) if one_column_style else ((0.065 if ours else 0.070) if compact_style else (0.13 if ours else 0.070)),
                    edgecolor=COLORS[method],
                    linewidth=0.22 if compact_style else 0.18,
                    zorder=4 if method in OURS else 1,
                )
        else:
            raise ValueError(f"unknown uncertainty rendering: {uncertainty}")
    axis.set_xticks(BUDGETS)
    axis.set_xlabel("Mean trajectories used")
    axis.set_ylabel(metric.upper())
    axis.grid(alpha=0.18)
    if observed_y:
        low, high = min(observed_y), max(observed_y)
        pad = max(0.02, 0.08 * max(high - low, 0.05))
        axis.set_ylim(max(0.0, low - pad), min(1.0, high + pad))


def macro_plot(rows: Sequence[dict[str, Any]], path: Path, *, uncertainty: str = "bars") -> None:
    # Purpose-built for one IEEE column (approximately 3.5 inches).  Building
    # at final physical size avoids unreadable text produced by downscaling a
    # wide desktop figure.
    figure, axes = plt.subplots(1, 2, figsize=(3.5, 2.32))
    for axis, metric in zip(axes, ("auroc", "aupr"), strict=True):
        draw(axis, rows, metric, uncertainty=uncertainty, one_column_style=True)
        axis.set_title(metric.upper(), fontweight="bold", fontsize=7.2, pad=2.0)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks(BUDGETS)
        axis.tick_params(axis="both", labelsize=5.5, length=2.2, width=.55, pad=1.2)
        axis.grid(alpha=0.16, linewidth=.35)
        for spine in axis.spines.values():
            spine.set_linewidth(.55)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.255), fontsize=4.0,
        columnspacing=0.55, handlelength=1.55, handletextpad=0.25,
        borderaxespad=0.0, labelspacing=0.22,
    )
    figure.supxlabel("Mean trajectories used", fontsize=6.2, y=.292)
    figure.subplots_adjust(left=0.095, right=0.995, top=0.965, bottom=0.38, wspace=0.27)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=400, bbox_inches="tight", pad_inches=.01)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.01)
    plt.close(figure)


def combination_plot(
    rows: Sequence[dict[str, Any]], metric: str, path: Path, *, uncertainty: str = "bars"
) -> None:
    # IEEE two-column full-page width is approximately 7.16 inches.  Keep the
    # panel grid compact enough to fit on one page together with its caption.
    figure, axes = plt.subplots(4, 3, figsize=(7.16, 8.25))
    for panel, (axis, dataset, backbone) in enumerate(zip(
        axes.flat,
        (dataset for dataset in DATASETS for _ in BACKBONES),
        (backbone for _ in DATASETS for backbone in BACKBONES),
        strict=True,
    )):
        selected = [row for row in rows if row["dataset"] == dataset and row["backbone"] == backbone]
        draw(axis, selected, metric, uncertainty=uncertainty, compact_style=True)
        column = panel % 3
        row = panel // 3
        axis.set_title(
            f"{DATASET_DISPLAY[dataset]} / {BACKBONE_DISPLAY[backbone]}",
            fontweight="bold", fontsize=7.4, pad=3.0,
        )
        axis.set_ylabel(metric.upper() if column == 0 else "", fontsize=7.2, labelpad=2.0)
        axis.set_xlabel("Mean trajectories used" if row == 3 else "", fontsize=7.2, labelpad=2.0)
        axis.tick_params(axis="both", labelsize=6.6, pad=1.5)
        axis.grid(alpha=0.16, linewidth=0.45)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, 0.008), fontsize=5.7,
        columnspacing=0.75, handlelength=2.0, handletextpad=0.35,
        borderaxespad=0.0, labelspacing=0.35,
    )
    # No figure-level title: the twelve panel titles carry all identifying
    # information.  The small bottom reserve puts the legend close to row four.
    figure.subplots_adjust(
        left=0.075, right=0.995, top=0.975, bottom=0.105,
        hspace=0.30, wspace=0.16,
    )
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches=None)
    plt.close(figure)


def report(endpoint: Sequence[dict[str, Any]]) -> str:
    ranked = sorted(endpoint, key=lambda row: float(row["auroc_mean"]), reverse=True)
    lines = [
        "# Selected-Method Four-Dataset Comparison", "",
        "This is a separate paper-focused report; it does not replace the complete mixed-budget report. It covers DBBench, HotpotQA, WebShop, and StrategyQA; three backbones; and seeds 101, 202, and 303.", "",
        "## Plot contract", "",
        "- **Solid lines:** BayesTraj-Fixed (ours) and BayesTraj-Adaptive (ours).",
        "- **Dotted lines:** all baselines with multiple supported budgets.",
        "- **Markers only:** endpoint-only SAUP.",
        "- Error bars are standard deviations across the three seeds. In the macro panel, each seed is first macro-averaged over the 12 dataset–backbone combinations.",
        "- The x-axis is nominal cost for fixed methods and realized mean cost for adaptive methods.",
        "- BSE-Ciosek-Fixed and BSE-Ciosek-Adaptive are displayed as **BSE-Fixed** and **BSE-Adaptive**.", "",
        "For WebShop only, both BayesTraj curves use the submission's fixed, label-free **constraint-pattern** bucket design. It groups cached purchases by coarse request-token coverage, selected-option overlap, and price-threshold status. All baselines and all BayesTraj results on the other three datasets are unchanged.", "",
        "At the B=16 endpoint, BSE-Adaptive reaches the maximum budget for every task and is therefore exactly identical to BSE-Fixed. Their smaller-budget curve points remain distinct and show the adaptive behavior.", "",
        "## Cross-seed macro curves", "",
        "The macro figures are typeset at 3.5-inch IEEE single-column width; use the vector PDF directly rather than resizing the PNG preview.", "",
        "![Selected-method macro curves with seed error bars](plots/selected_macro_curves.png)", "",
        "[Vector PDF for the macro error-bar figure](plots/selected_macro_curves.pdf)", "",
        "## Curves by dataset and backbone", "",
        "![Selected-method AUROC curves with seed error bars](plots/selected_by_combination_auroc.png)", "",
        "[Vector PDF for the AUROC figure](plots/selected_by_combination_auroc.pdf)", "",
        "![Selected-method AUPR curves with seed error bars](plots/selected_by_combination_aupr.png)", "",
        "[Vector PDF for the AUPR figure](plots/selected_by_combination_aupr.pdf)", "",
        "## Alternative uncertainty-ribbon figures", "",
        "These versions display the same mean and standard deviation using translucent mean ± one-standard-deviation areas instead of capped error bars.", "",
        "![Selected-method macro curves with seed uncertainty ribbons](plots/selected_macro_curves_bands.png)", "",
        "[Vector PDF for the macro uncertainty-ribbon figure](plots/selected_macro_curves_bands.pdf)", "",
        "![Selected-method AUROC curves with seed uncertainty ribbons](plots/selected_by_combination_auroc_bands.png)", "",
        "[Vector PDF for the AUROC ribbon figure](plots/selected_by_combination_auroc_bands.pdf)", "",
        "![Selected-method AUPR curves with seed uncertainty ribbons](plots/selected_by_combination_aupr_bands.png)", "",
        "[Vector PDF for the AUPR ribbon figure](plots/selected_by_combination_aupr_bands.pdf)", "",
        "## Native maximum-budget summary", "",
        "Each method is summarized at its largest supported nominal budget. For UProp and Degree this is Z=12; all other complete curves use Z=16; SAUP is endpoint-only at Z=16.", "",
        "| Rank | Method | Budget | Mean cost | AUROC mean ± seed std | AUPR mean ± seed std |", "|---:|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, 1):
        lines.append(
            f"| {index} | `{DISPLAY.get(row['method'], row['method'])}` | {int(row['budget'])} | "
            f"{float(row['mean_trajectories']):.2f} | {float(row['auroc_mean']):.3f} ± {float(row['auroc_std']):.3f} | "
            f"{float(row['aupr_mean']):.3f} ± {float(row['aupr_std']):.3f} |"
        )
    lines += ["", "## Reproducibility artifacts", "",
              "- `selected_seed_metrics.csv`: filtered source cell metrics.",
              "- `selected_macro_metrics.csv`: seed-macro means and cross-seed error bars.",
              "- `selected_combination_metrics.csv`: dataset–backbone means and cross-seed error bars.",
              "- `webshop_constraint_pattern_seed_metrics.csv`: the 108 replacement rows for the two BR-LG methods.",
              "- `manifest.json`: method registry, source path, and plotting contract."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    source_rows = read_csv(source / "trajectory_curve_seed_metrics.csv")
    rows = [
        row for row in source_rows
        if row["dataset"] in DATASETS
        and row["method"] in METHODS
        and int(float(row["budget"])) in BUDGETS
    ]
    rows = [
        row for row in rows
        if not (row["dataset"] == "webshop" and row["method"] in OURS)
    ]
    webshop_bucket = bucket_webshop_rows()
    rows.extend(webshop_bucket)
    observed = {row["method"] for row in rows}
    if observed != set(METHODS):
        raise RuntimeError(f"method support mismatch: missing={set(METHODS)-observed}, extra={observed-set(METHODS)}")
    for row in rows:
        row["seed"] = int(row["seed"])
        row["budget"] = int(float(row["budget"]))
        if row.get("mean_trajectories") in (None, ""):
            row["mean_trajectories"] = float(row["budget"])

    seed_macro = aggregate(rows, ("seed", "method", "budget"))
    macro = aggregate_seed_macro(seed_macro)
    combinations = aggregate(rows, ("dataset", "backbone", "combination", "method", "budget"))
    endpoint = []
    for method in METHODS:
        selected = [row for row in macro if row["method"] == method]
        endpoint.append(max(selected, key=lambda row: int(row["budget"])))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "selected_seed_metrics.csv", rows)
    write_csv(output / "webshop_constraint_pattern_seed_metrics.csv", webshop_bucket)
    write_csv(output / "selected_macro_metrics.csv", macro)
    write_csv(output / "selected_combination_metrics.csv", combinations)
    macro_plot(macro, output / "plots/selected_macro_curves.png")
    combination_plot(combinations, "auroc", output / "plots/selected_by_combination_auroc.png")
    combination_plot(combinations, "aupr", output / "plots/selected_by_combination_aupr.png")
    macro_plot(macro, output / "plots/selected_macro_curves_bands.png", uncertainty="bands")
    combination_plot(
        combinations, "auroc", output / "plots/selected_by_combination_auroc_bands.png", uncertainty="bands"
    )
    combination_plot(
        combinations, "aupr", output / "plots/selected_by_combination_aupr_bands.png", uncertainty="bands"
    )
    (output / "report.md").write_text(report(endpoint), encoding="utf-8")
    manifest = {
        "source_report": str(source / "report.md"),
        "source_curve_metrics": str(source / "trajectory_curve_seed_metrics.csv"),
        "webshop_bucket_task_scores": str(WEBSHOP_BUCKET_RUN / "*/task_scores_without_labels.jsonl.gz"),
        "webshop_bucket_variant_for_our_methods": WEBSHOP_BUCKET_VARIANT,
        "webshop_replaced_methods": list(OURS),
        "webshop_bucket_choice_status": "retrospective performance-leading eligible design; not the preregistered label-free winner",
        "webshop_replacement_rows": len(webshop_bucket),
        "datasets": list(DATASETS), "backbones": list(BACKBONES), "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "budget_contract": "six shared nominal budgets; baseline-only B=2 results excluded",
        "methods": list(METHODS), "display_names": {method: DISPLAY.get(method, method) for method in METHODS},
        "line_contract": {"solid": list(OURS), "dotted": list(BASELINES), "endpoint_only": ["SAUP"]},
        "uncertainty": "sample standard deviation across seeds, rendered in both capped-error-bar and translucent-ribbon versions; macro first averages the 12 combinations within each seed",
        "generation_calls": 0,
        "seed_metric_rows": len(rows), "macro_rows": len(macro), "combination_rows": len(combinations),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "report.md"), "plots": 6, "methods": len(METHODS), "seed_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
