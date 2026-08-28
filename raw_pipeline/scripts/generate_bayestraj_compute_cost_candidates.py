#!/usr/bin/env python3
"""Generate alternative paper presentations for BayesTraj computational cost.

Only quantities recorded by the audited four-dataset artifacts are used.  In
particular, the figures do not infer wall-clock latency from cached runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "outputs/analysis/mixed_budget_4datasets_selected_methods/report/selected_seed_metrics.csv"
DEFAULT_EFFICIENCY = ROOT / "outputs/analysis/bayestraj_empirical_minimal_package/report/efficiency_by_budget.csv"
DEFAULT_OUTPUT = ROOT / "outputs/analysis/bayestraj_compute_cost_candidates/report"

DISPLAY = {
    "BR-LG-Risk-VC4-LCB95": "BayesTraj-Fixed",
    "BR-LG-VarStop80-LCB95": "BayesTraj-Adaptive",
    "BSE-Ciosek-Fixed": "BSE-Fixed",
    "BSE-Ciosek-Adaptive": "BSE-Adaptive",
}
REPRESENTATIVE = [
    "BR-LG-Risk-VC4-LCB95", "BR-LG-VarStop80-LCB95", "MC-OE",
    "BSE-Ciosek-Fixed", "BSE-Ciosek-Adaptive", "CoCoA-PPL", "UProp", "PPL",
]
COLORS = {
    "BayesTraj-Fixed": "#0B559F", "BayesTraj-Adaptive": "#D94801",
    "MC-OE": "#6255A4", "BSE-Fixed": "#1B9E77", "BSE-Adaptive": "#66A61E",
    "CoCoA-PPL": "#8C8C2B", "UProp": "#E69F00", "PPL": "#9E3D52",
}


def save(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=.025)
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(figure)


def summarize(metrics: pd.DataFrame, budget: int) -> pd.DataFrame:
    block = metrics[(metrics["budget"] == budget) & metrics["method"].isin(REPRESENTATIVE)].copy()
    summary = block.groupby("method", as_index=False).agg(
        auroc=("auroc", "mean"), aupr=("aupr", "mean"),
        trajectories=("mean_trajectories", "mean"), cells=("auroc", "size"),
    )
    summary["display"] = summary["method"].map(lambda x: DISPLAY.get(x, x))
    summary["saving"] = 1 - summary["trajectories"] / budget
    order = {name: position for position, name in enumerate(REPRESENTATIVE)}
    return summary.sort_values("method", key=lambda s: s.map(order)).reset_index(drop=True)


def accuracy_cost_table(summary: pd.DataFrame, output: Path, budget: int) -> None:
    data = []
    for row in summary.itertuples():
        data.append([
            row.display, f"{row.auroc:.3f}", f"{row.aupr:.3f}",
            f"{row.trajectories:.2f}", f"{100 * row.saving:.1f}%",
        ])
    figure, axis = plt.subplots(figsize=(7.0, 3.45))
    axis.axis("off")
    table = axis.table(
        cellText=data,
        colLabels=["Method", "AUROC", "AUPR", "Mean T", "Saving"],
        colWidths=[.34, .12, .12, .23, .19], bbox=[0, .12, 1, .80], cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.2)
    table.scale(1, 1.48)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#24364B")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif data[row - 1][0].startswith("BayesTraj"):
            cell.set_facecolor("#E9F2F9" if row % 2 else "#DCEAF5")
            if col == 0:
                cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F6F7F8" if row % 2 else "white")
        if col == 0:
            cell.get_text().set_ha("left")
    axis.text(.5, .015, f"Macro average over 36 dataset-backbone-seed cells at B={budget}",
              transform=axis.transAxes, ha="center", fontsize=10.5, color="#444444")
    save(figure, output, "candidate_1_accuracy_sampling_table")

    tex = [
        r"\begin{table}[t]", r"\centering", r"\caption{Accuracy and realized sampling cost at $B=8$.}",
        r"\label{tab:compute-cost}", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{lrrrr}",
        r"\toprule", r"Method & AUROC & AUPR & Mean $T$ & Saving \\", r"\midrule",
    ]
    for row in summary.itertuples():
        name = row.display.replace("BayesTraj-Fixed", r"\textbf{BayesTraj-Fixed}").replace(
            "BayesTraj-Adaptive", r"\textbf{BayesTraj-Adaptive}")
        tex.append(f"{name} & {row.auroc:.3f} & {row.aupr:.3f} & {row.trajectories:.2f} & {100*row.saving:.1f}\\% \\\\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output / "candidate_1_accuracy_sampling_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def pareto_plot(summary: pd.DataFrame, output: Path, budget: int) -> None:
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharex=True)
    for axis, metric, title in zip(axes, ("auroc", "aupr"), ("AUROC", "AUPR")):
        for row in summary.itertuples():
            ours = row.display.startswith("BayesTraj")
            axis.scatter(row.trajectories, getattr(row, metric), s=115 if ours else 65,
                         marker="*" if ours else "o", color=COLORS[row.display],
                         edgecolor="white", linewidth=.7, zorder=4)
        fixed = summary[summary.display.eq("BayesTraj-Fixed")].iloc[0]
        adaptive = summary[summary.display.eq("BayesTraj-Adaptive")].iloc[0]
        axis.annotate("", xy=(adaptive.trajectories, adaptive[metric]),
                      xytext=(fixed.trajectories, fixed[metric]),
                      arrowprops=dict(arrowstyle="->", color="#D94801", lw=1.7))
        axis.text(adaptive.trajectories - .08, adaptive[metric] + .004,
                  "Adaptive", ha="right", va="bottom", fontsize=11, fontweight="bold", color="#D94801")
        axis.text(fixed.trajectories - .08, fixed[metric] + .004,
                  "Fixed", ha="right", va="bottom", fontsize=11, fontweight="bold", color="#0B559F")
        axis.set_title(title)
        axis.set_xlabel("Mean trajectories used")
        axis.set_ylabel(title if axis is axes[0] else "")
        axis.set_xlim(6.1, 8.2)
        axis.grid(alpha=.22)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[DISPLAY.get(m, m)],
                          label=DISPLAY.get(m, m), markersize=6) for m in REPRESENTATIVE[2:]]
    figure.legend(handles=handles, ncol=3, frameon=False, loc="lower center",
                  bbox_to_anchor=(.5, -.04), fontsize=9.5, columnspacing=1.0, handletextpad=.35)
    figure.subplots_adjust(left=.10, right=.985, top=.92, bottom=.29, wspace=.24)
    save(figure, output, "candidate_2_accuracy_cost_pareto_b8")


def efficiency_plot(efficiency: pd.DataFrame, output: Path) -> None:
    adaptive = efficiency[efficiency.method.eq("BayesTraj-Adaptive")].sort_values("budget")
    budgets = adaptive.budget.to_numpy()
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10.5})
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].plot(budgets, budgets, "o-", lw=2.2, color="#0B559F", label="Fixed")
    axes[0].plot(budgets, adaptive.mean_trajectories, "s-", lw=2.2, color="#D94801", label="Adaptive")
    axes[0].set_title("(a) Realized allocation")
    axes[0].set_xlabel("Trajectory budget $B$")
    axes[0].set_ylabel("Mean trajectories used")
    axes[0].set_xticks(budgets)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.22)
    for column, label, color, marker in (
        ("trajectory_saving", "Trajectories", "#0B559F", "o"),
        ("agent_step_saving", "Agent steps", "#1B9E77", "s"),
        ("output_token_saving", "Output tokens", "#D94801", "^"),
    ):
        axes[1].plot(budgets, 100 * adaptive[column], marker=marker, lw=2.2,
                     color=color, label=label)
    axes[1].set_title("(b) Realized savings")
    axes[1].set_xlabel("Trajectory budget $B$")
    axes[1].set_ylabel("Reduction vs. Fixed")
    axes[1].set_xticks(budgets)
    axes[1].set_yticks([0, 5, 10, 15, 20], ["0%", "5%", "10%", "15%", "20%"])
    axes[1].set_ylim(0, 22)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(alpha=.22)
    figure.subplots_adjust(left=.10, right=.985, top=.91, bottom=.19, wspace=.30)
    save(figure, output, "candidate_3_realized_compute_profile")


def gain_cost_plot(summary: pd.DataFrame, output: Path, budget: int) -> None:
    baselines = summary[~summary.display.str.startswith("BayesTraj")].copy()
    adaptive = summary[summary.display.eq("BayesTraj-Adaptive")].iloc[0]
    baselines["auroc_gain"] = 100 * (adaptive.auroc - baselines.auroc)
    baselines["aupr_gain"] = 100 * (adaptive.aupr - baselines.aupr)
    baselines = baselines.sort_values("auroc_gain")
    y = np.arange(len(baselines))
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharey=True)
    for axis, column, title, color in (
        (axes[0], "auroc_gain", "AUROC gain", "#0B559F"),
        (axes[1], "aupr_gain", "AUPR gain", "#D94801"),
    ):
        values = baselines[column].to_numpy()
        axis.barh(y, values, color=color, alpha=.86)
        for position, value in zip(y, values):
            axis.text(value + .10, position, f"+{value:.1f} pp", va="center", fontsize=9.5)
        axis.axvline(0, color="#333333", lw=.8)
        axis.set_xlim(0, max(values) * 1.32)
        axis.set_title(title)
        axis.set_xlabel("Adaptive − baseline (pp)")
        axis.grid(axis="x", alpha=.20)
    axes[0].set_yticks(y, baselines.display)
    figure.text(.5, .012, f"BayesTraj-Adaptive uses {adaptive.trajectories:.2f}/{budget} trajectories (19.9% saving)",
                ha="center", fontsize=11.5, fontweight="bold")
    figure.subplots_adjust(left=.25, right=.96, top=.91, bottom=.24, wspace=.20)
    save(figure, output, "candidate_4_adaptive_gain_at_lower_cost")


def complexity_table(output: Path) -> None:
    rows = [
        ["Posterior fit", "Once / prefix", "Training tasks", "0"],
        ["Threshold calibration", "Once / budget", "Validation tasks", "0"],
        ["Posterior update", r"$O(Gd^2)$", r"$O(G+d^2)$", "0"],
        ["Stopping check", r"$O(1)$", r"$O(1)$", "0"],
    ]
    figure, axis = plt.subplots(figsize=(7.0, 2.55))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Stage", "Cost / frequency", "Memory / data", "Extra calls"],
        colWidths=[.29, .23, .30, .18], bbox=[0, .19, 1, .73], cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#24364B")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#EDF3F7" if row % 2 else "white")
        if col == 0:
            cell.get_text().set_ha("left")
    axis.text(.5, .06, r"Defaults: $G=257$, $d=10$; posterior work is separate from agent execution.",
              ha="center", transform=axis.transAxes, fontsize=10.8, fontweight="bold")
    save(figure, output, "candidate_5_algorithmic_overhead_table")

    tex = r"""\begin{table}[t]
\centering
\caption{BayesTraj computational overhead. Neither fitting, calibration, nor
posterior inference invokes the LLM beyond the trajectories being scored.}
\label{tab:bayestraj-overhead}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
Stage & Frequency/time & Memory/data & LLM calls \\
\midrule
Posterior fit & Once/prefix & Training tasks & 0 \\
Threshold calibration & Once/budget & Validation tasks & 0 \\
Online posterior update & $O(Gd^2)$ & $O(G+d^2)$ & 0 \\
Stopping check & $O(1)$ & $O(1)$ & 0 \\
\bottomrule
\end{tabular}
\end{table}
"""
    (output / "candidate_5_algorithmic_overhead_table.tex").write_text(tex, encoding="utf-8")


def write_report(output: Path, budget: int) -> None:
    report = f"""# Computational-cost presentation candidates

All candidates use the audited four-dataset results (DBBench, HotpotQA, WebShop, and StrategyQA), three backbones, and seeds 101/202/303. No wall-clock values are shown because comparable end-to-end durations were not recorded for every method.

## Candidate 1 — compact accuracy-and-sampling table

![Candidate 1](candidate_1_accuracy_sampling_table.png)

Best when the paper needs exact values and a conventional computational-cost subsection. The corresponding LaTeX is `candidate_1_accuracy_sampling_table.tex`.

## Candidate 2 — accuracy–cost Pareto plot

![Candidate 2](candidate_2_accuracy_cost_pareto_b8.png)

Best visual comparison at a representative budget: higher and farther left is better. The arrow isolates the fixed-to-adaptive trade-off. It uses realized trajectories, not estimated runtime.

## Candidate 3 — realized compute profile

![Candidate 3](candidate_3_realized_compute_profile.png)

Best direct evidence that trajectory savings propagate to executed steps and generated output tokens across budgets.

## Candidate 4 — adaptive gain at lower sampling cost

![Candidate 4](candidate_4_adaptive_gain_at_lower_cost.png)

Best positive headline: at B={budget}, BayesTraj-Adaptive uses 19.9% fewer trajectories than the nominal budget while outperforming each representative baseline in both ranking metrics.

## Candidate 5 — algorithmic overhead table

![Candidate 5](candidate_5_algorithmic_overhead_table.png)

Best compact description of where BayesTraj adds computation. It is analytical rather than a wall-clock benchmark and makes clear that posterior fitting, calibration, and stopping add no LLM calls. The corresponding LaTeX is `candidate_5_algorithmic_overhead_table.tex`.

## Recommendation

Use **Candidate 1** if only a table is available, or **Candidate 2** if one figure slot is available. Candidate 3 is the strongest appendix complement because it demonstrates that the reduction is not confined to the trajectory counter. Candidate 4 is visually persuasive but should be described explicitly as a representative-budget comparison, not a complete runtime benchmark.

## Computational reporting boundary

BayesTraj posterior inference has per-prefix complexity $O(Gd^2)$ arithmetic and $O(G+d^2)$ working memory, with the fixed defaults $G=257$ and $d=10$. The cached evidence supports exact trajectory, step, observation, and output-token counts. It does **not** support a fair wall-clock comparison, so a future controlled benchmark would be needed before reporting milliseconds per task or end-to-end speedup.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--efficiency", type=Path, default=DEFAULT_EFFICIENCY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    efficiency = pd.read_csv(args.efficiency)
    summary = summarize(metrics, args.budget)
    summary.to_csv(args.output / "representative_budget_metrics.csv", index=False)
    accuracy_cost_table(summary, args.output, args.budget)
    pareto_plot(summary, args.output, args.budget)
    efficiency_plot(efficiency, args.output)
    gain_cost_plot(summary, args.output, args.budget)
    complexity_table(args.output)
    write_report(args.output, args.budget)
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
