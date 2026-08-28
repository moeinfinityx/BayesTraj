"""Paper-table loading, aggregation, and machine-checkable claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper_inputs"
CONFIG = json.loads((ROOT / "config" / "paper.json").read_text())
BUDGETS = tuple(CONFIG["budgets"])
FIXED = CONFIG["fixed_method"]
ADAPTIVE = CONFIG["adaptive_method"]
OURS = (FIXED, ADAPTIVE)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def seed_metrics() -> pd.DataFrame:
    frame = read("selected_seed_metrics.csv")
    frame["budget"] = frame["budget"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    return frame


def macro_metrics(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Macro-average 12 combinations inside each seed, then summarize seeds."""
    data = seed_metrics() if frame is None else frame
    per_seed = data.groupby(["seed", "method", "budget"], as_index=False).agg(
        auroc=("auroc", "mean"), aupr=("aupr", "mean"),
        mean_trajectories=("mean_trajectories", "mean"),
    )
    return per_seed.groupby(["method", "budget"], as_index=False).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        aupr_mean=("aupr", "mean"), aupr_std=("aupr", "std"),
        mean_trajectories=("mean_trajectories", "mean"), seeds=("seed", "size"),
    )


def combination_metrics(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    data = seed_metrics() if frame is None else frame
    return data.groupby(["dataset", "backbone", "method", "budget"], as_index=False).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        aupr_mean=("aupr", "mean"), aupr_std=("aupr", "std"),
        mean_trajectories=("mean_trajectories", "mean"), seeds=("seed", "size"),
    )


def compute_claims() -> dict[str, Any]:
    data = seed_metrics()
    macro = macro_metrics(data)
    gains: dict[str, list[float]] = {"auroc": [], "aupr": []}
    first = 0
    for budget in BUDGETS:
        block = macro[macro.budget.eq(budget)]
        for metric in ("auroc", "aupr"):
            column = f"{metric}_mean"
            fixed = float(block.loc[block.method.eq(FIXED), column].iloc[0])
            best = float(block.loc[~block.method.isin(OURS), column].max())
            first += int(fixed > best)
            gains[metric].append(100 * (fixed - best))

    paired_modes = macro[macro.method.isin(OURS)].pivot(index="budget", columns="method")
    adaptive_saving = 100 * (
        1 - macro.loc[macro.method.eq(ADAPTIVE), "mean_trajectories"].to_numpy() / np.asarray(BUDGETS)
    )

    combination = combination_metrics(data)
    wins = within = 0
    for _, block in combination.groupby(["dataset", "backbone", "budget"]):
        fixed = float(block.loc[block.method.eq(FIXED), "auroc_mean"].iloc[0])
        best = float(block.loc[~block.method.isin(OURS), "auroc_mean"].max())
        wins += int(fixed > best)
        within += int(fixed >= best - 0.01)

    b8 = read("representative_budget_metrics.csv")
    adaptive_b8 = b8[b8.display.eq("BayesTraj-Adaptive")].iloc[0]
    b8_baselines = b8[~b8.display.str.startswith("BayesTraj")]
    ablation = read("core_mechanism_tradeoff.csv").set_index("contrast")
    paired = read("paired_superiority_summary.csv")
    sensitivity = read("sensitivity_summary.csv")
    posterior = read("posterior_diagnostics_by_dataset_budget.csv")
    count_mse, fused_mse = posterior.count_mse.mean(), posterior.fused_mse.mean()

    result: dict[str, Any] = {
        "fixed_first_budget_metric_comparisons": first,
        "adaptive_mean_trajectory_saving_pct": float(adaptive_saving.mean()),
        "adaptive_mean_auroc_change_pp": float(100 * (
            paired_modes["auroc_mean"][ADAPTIVE] - paired_modes["auroc_mean"][FIXED]
        ).mean()),
        "adaptive_mean_aupr_change_pp": float(100 * (
            paired_modes["aupr_mean"][ADAPTIVE] - paired_modes["aupr_mean"][FIXED]
        ).mean()),
        "oracle_auroc_wins": wins,
        "oracle_auroc_best_or_within_1pp": within,
        "oracle_auroc_settings": int(len(CONFIG["datasets"]) * len(CONFIG["backbones"]) * len(BUDGETS)),
        "b8_adaptive_mean_trajectories": float(adaptive_b8.trajectories),
        "b8_adaptive_saving_pct": float(100 * adaptive_b8.saving),
        "posterior_relative_mse_reduction_pct": float(100 * (1 - fused_mse / count_mse)),
        "posterior_coverage95_pct": float(100 * posterior.coverage95.mean()),
    }
    for mode, prefix in (("BayesTraj-Fixed", "paired_fixed"), ("BayesTraj-Adaptive", "paired_adaptive")):
        block = paired[paired.method.eq(mode)]
        for metric in ("auroc", "aupr"):
            values = 100 * block[f"{metric}_delta"]
            result[f"{prefix}_{metric}_min_pp"] = float(values.min())
            result[f"{prefix}_{metric}_max_pp"] = float(values.max())
    result["paired_holm_significant_comparisons"] = int(
        (paired.auroc_p_holm.lt(.05)).sum() + (paired.aupr_p_holm.lt(.05)).sum()
    )
    for metric in ("auroc", "aupr"):
        values = 100 * (float(adaptive_b8[metric]) - b8_baselines[metric])
        result[f"b8_{metric}_gain_min_pp"] = float(values.min())
        result[f"b8_{metric}_gain_max_pp"] = float(values.max())
    for metric, values in gains.items():
        result[f"fixed_{metric}_gain_min_pp"] = float(min(values))
        result[f"fixed_{metric}_gain_max_pp"] = float(max(values))
        result[f"fixed_{metric}_gain_mean_pp"] = float(np.mean(values))
    for key, prefix in (
        ("var_features", "ablation_trajectory_update"),
        ("count_prior_fusion", "ablation_count_prior"),
        ("full_covariance", "ablation_full_covariance"),
    ):
        result[f"{prefix}_auroc_pp"] = float(100 * ablation.loc[key, "delta_auroc"])
        result[f"{prefix}_aupr_pp"] = float(100 * ablation.loc[key, "delta_aupr"])
    for key, prefix in (("var_vs_fixed", "ablation_adaptive_stopping"), ("var_vs_nonadaptive", "ablation_task_allocation")):
        result[f"{prefix}_auroc_pp"] = float(100 * ablation.loc[key, "delta_auroc"])
        result[f"{prefix}_aupr_pp"] = float(100 * ablation.loc[key, "delta_aupr"])
        result[f"{prefix}_saving_pp"] = float(100 * ablation.loc[key, "saving_delta"])
    for rho, window, prefix in ((.80, "4", "sensitivity_default"), (.75, "4", "sensitivity_rho75"), (.80, "2", "sensitivity_w2")):
        row = sensitivity[np.isclose(sensitivity.rho, rho) & sensitivity.window.astype(str).eq(window)].iloc[0]
        result[f"{prefix}_auroc_change_pp"] = float(100 * row.delta_auroc)
        result[f"{prefix}_saving_pct"] = float(100 * row.saving)
    return result


def results_markdown(claims: dict[str, Any]) -> str:
    return f"""# Reproduced BayesTraj results

- BayesTraj-Fixed ranks first in **{claims['fixed_first_budget_metric_comparisons']}/12** shared budget–metric comparisons.
- Its gain over the strongest budget-specific baseline is **{claims['fixed_auroc_gain_min_pp']:.2f}–{claims['fixed_auroc_gain_max_pp']:.2f} AUROC points** (mean {claims['fixed_auroc_gain_mean_pp']:.2f}) and **{claims['fixed_aupr_gain_min_pp']:.2f}–{claims['fixed_aupr_gain_max_pp']:.2f} AUPR points** (mean {claims['fixed_aupr_gain_mean_pp']:.2f}).
- BayesTraj-Adaptive saves **{claims['adaptive_mean_trajectory_saving_pct']:.1f}%** of trajectories; its mean changes from Fixed are {claims['adaptive_mean_auroc_change_pp']:.2f} AUROC and {claims['adaptive_mean_aupr_change_pp']:.2f} AUPR points.
- In the 72 dataset/backbone/budget AUROC settings, Fixed wins **{claims['oracle_auroc_wins']}** and is best or within one point in **{claims['oracle_auroc_best_or_within_1pp']}**.
- At B=8, Adaptive uses **{claims['b8_adaptive_mean_trajectories']:.2f}/8** trajectories ({claims['b8_adaptive_saving_pct']:.1f}% saving).
- Removing the trajectory update changes AUROC/AUPR by **{claims['ablation_trajectory_update_auroc_pp']:.2f}/{claims['ablation_trajectory_update_aupr_pp']:.2f} points**; removing count-prior fusion changes them by **{claims['ablation_count_prior_auroc_pp']:.2f}/{claims['ablation_count_prior_aupr_pp']:.2f} points**.
- Two-view fusion reduces held-out target MSE by **{claims['posterior_relative_mse_reduction_pct']:.1f}%**; empirical 95% posterior coverage is **{claims['posterior_coverage95_pct']:.1f}%**.
"""
