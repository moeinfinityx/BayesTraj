#!/usr/bin/env python3
"""Generate the paper-facing BayesTraj methodology diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis" / "bayestraj_method_diagram"

NAVY = "#17365D"
BLUE = "#DDEBF7"
BLUE_EDGE = "#5B9BD5"
GREEN = "#E2F0D9"
GREEN_EDGE = "#70AD47"
ORANGE = "#FCE4D6"
ORANGE_EDGE = "#ED7D31"
PURPLE = "#E4DFEC"
PURPLE_EDGE = "#8064A2"
GRAY = "#F2F2F2"
GRAY_EDGE = "#8A8A8A"
RED = "#F4CCCC"
TEXT = "#1F2937"


def box(ax, x, y, w, h, title, detail="", *, fill=GRAY, edge=GRAY_EDGE,
        title_size=8.2, detail_size=6.8, lw=1.1, radius=0.018):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    center = x + w / 2
    if detail:
        ax.text(center, y + h * 0.64, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=TEXT)
        ax.text(center, y + h * 0.31, detail, ha="center", va="center",
                fontsize=detail_size, color=TEXT, linespacing=1.12)
    else:
        ax.text(center, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=TEXT)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=NAVY, lw=1.15, style="-|>",
          label=None, label_y=None, dashed=False, mutation=9):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=mutation,
        linewidth=lw, color=color,
        linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=1.5, shrinkB=1.5,
    )
    ax.add_patch(patch)
    if label:
        ax.text((x1 + x2) / 2, label_y if label_y is not None else (y1 + y2) / 2 + 0.018,
                label, ha="center", va="center", fontsize=6.4, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    return patch


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(7.16, 4.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Lane backgrounds and labels.
    lane_top = FancyBboxPatch(
        (0.012, 0.690), 0.976, 0.285,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="#F7FAFC", edgecolor="#AAB7C4", linewidth=0.9,
    )
    lane_bottom = FancyBboxPatch(
        (0.012, 0.045), 0.976, 0.605,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="white", edgecolor="#AAB7C4", linewidth=0.9,
    )
    ax.add_patch(lane_top)
    ax.add_patch(lane_bottom)
    ax.text(0.027, 0.944, "A. Cross-fitted learning and calibration",
            fontsize=9.4, fontweight="bold", color=NAVY, va="center")
    ax.text(0.973, 0.944, "Four training folds; no correctness labels",
            fontsize=7.0, color="#52677B", ha="right", va="center")
    ax.text(0.027, 0.620, "B. Held-out task inference",
            fontsize=9.4, fontweight="bold", color=NAVY, va="center")
    ax.text(0.973, 0.620, "Same posterior and score; only the stopping rule changes",
            fontsize=7.0, color="#52677B", ha="right", va="center")

    # Cross-fitted training lane.
    box(ax, 0.035, 0.735, 0.165, 0.145,
        "Training pools", "16 trajectories per task",
        fill=GRAY, edge=GRAY_EDGE)
    box(ax, 0.258, 0.735, 0.185, 0.145,
        "Label-free target", r"$H_i=\mathrm{OE}_{16,i}$",
        fill=BLUE, edge=BLUE_EDGE)
    box(ax, 0.500, 0.735, 0.205, 0.145,
        "Linear-Gaussian fit", r"$Z_n\mid H\sim\mathcal{N}(a_n+b_nH,\Sigma_n)$",
        fill=PURPLE, edge=PURPLE_EDGE, detail_size=6.5)
    box(ax, 0.762, 0.735, 0.195, 0.145,
        "Adaptive calibration", r"fused $\sigma_n^2 \rightarrow \tau_{B,f}$ at $\rho=.80$",
        fill=ORANGE, edge=ORANGE_EDGE, detail_size=6.5)
    arrow(ax, 0.200, 0.807, 0.258, 0.807)
    arrow(ax, 0.443, 0.807, 0.500, 0.807)
    arrow(ax, 0.705, 0.807, 0.762, 0.807)

    # Held-out input and evidence channels.
    box(ax, 0.035, 0.345, 0.135, 0.150,
        "Trajectory prefix", r"$\tau_{i,1:n}$",
        fill=GRAY, edge=GRAY_EDGE)
    box(ax, 0.225, 0.485, 0.180, 0.115,
        "Outcome buckets", r"counts $c_{i,n}$",
        fill=BLUE, edge=BLUE_EDGE)
    box(ax, 0.225, 0.280, 0.180, 0.115,
        "Trajectory features", r"10 surprisal/length summaries",
        fill=PURPLE, edge=PURPLE_EDGE, detail_size=6.4)
    arrow(ax, 0.170, 0.435, 0.225, 0.542)
    arrow(ax, 0.170, 0.405, 0.225, 0.337)

    # Priors and likelihoods.
    box(ax, 0.455, 0.485, 0.170, 0.115,
        "Count prior", r"Dirichlet $\rightarrow p_0(H\mid c)$",
        fill=BLUE, edge=BLUE_EDGE, detail_size=6.5)
    box(ax, 0.455, 0.280, 0.170, 0.115,
        "Trajectory likelihood", r"$p(Z_n\mid H)$",
        fill=PURPLE, edge=PURPLE_EDGE)
    arrow(ax, 0.405, 0.542, 0.455, 0.542)
    arrow(ax, 0.405, 0.337, 0.455, 0.337)

    # Fusion and score.
    box(ax, 0.675, 0.365, 0.145, 0.155,
        "Posterior fusion", r"257-point grid" + "\n" + r"$p(H\mid c,Z)$",
        fill=GREEN, edge=GREEN_EDGE, detail_size=6.7, lw=1.35)
    arrow(ax, 0.625, 0.542, 0.675, 0.472)
    arrow(ax, 0.625, 0.337, 0.675, 0.408)
    box(ax, 0.855, 0.365, 0.115, 0.155,
        "Score + variance", r"$\mu_n,\sigma_n^2$" + "\n" + r"$S_n=\mu_n-1.96\sigma_n$",
        fill=GREEN, edge=GREEN_EDGE, title_size=7.8, detail_size=6.2, lw=1.35)
    arrow(ax, 0.820, 0.442, 0.855, 0.442)

    # Deployment variants.
    box(ax, 0.550, 0.095, 0.180, 0.105,
        "BayesTraj-Fixed", r"stop at $T_i=B$",
        fill=GREEN, edge=GREEN_EDGE)
    box(ax, 0.785, 0.095, 0.185, 0.105,
        "BayesTraj-Adaptive", r"first $\sigma_{i,n}^2\leq\tau_{B,f}$; else $B$",
        fill=ORANGE, edge=ORANGE_EDGE, detail_size=6.2)
    arrow(ax, 0.912, 0.365, 0.640, 0.200)
    arrow(ax, 0.920, 0.365, 0.877, 0.200)

    # Reviewer-facing safeguard.
    ax.text(
        0.035, 0.075,
        "Correctness labels are opened only after held-out scores and stopping times are frozen.",
        ha="left", va="center", fontsize=6.8, color="#7A1F1F", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=RED, edgecolor="#C0504D", linewidth=0.8),
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUT / f"bayestraj_method_overview.{suffix}",
                    bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
