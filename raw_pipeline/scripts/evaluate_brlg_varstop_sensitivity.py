#!/usr/bin/env python3
"""Evaluate BR-LG variance-stop sensitivity for one held-out cell.

Only the target cost ratio and early-search width vary.  The BR-LG posterior,
LCB95 score, folds, trajectory pools, and OE16 target remain frozen.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_brlg_fixed_varstop80_ablation import fit_split_mechanism
from bayestraj_common import (
    BUDGETS, calibrate_boundary, stop_at_boundary, write_csv,
    EXPECTED, FOLDS, Posterior, atomic_text, checkpoint_path, compact,
    parse_cell, read_jsonl, sha256,
)


CONTRACT_VERSION = "brlg-varstop-sensitivity-2026-08-11-v1"
RATIOS = (0.70, 0.75, 0.80, 0.85, 0.90)
WINDOWS: tuple[int | str, ...] = (2, 4, 6, "all")
Z95 = 1.959963984540054


def setting_id(ratio: float, window: int | str) -> str:
    return f"rho{int(round(100 * ratio)):02d}_w{window}"


def lower_bound(budget: int, window: int | str) -> int:
    return 2 if window == "all" else max(2, budget - int(window))


def evaluate(args: argparse.Namespace) -> None:
    cell = parse_cell(args.cell)
    source = checkpoint_path(args.evaluation_root.resolve(), cell)
    output = args.output_root.resolve() / cell.name
    manifest_path = output / "manifest.json"
    if args.resume and (output / "COMPLETE").is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("contract_version") == CONTRACT_VERSION
            and manifest.get("source_sha256") == sha256(source)
        ):
            print(json.dumps({"cell": cell.name, "status": "reused"}), flush=True)
            return

    raw = read_jsonl(source)
    if len(raw) != EXPECTED[cell.dataset]:
        raise RuntimeError(f"{cell.name}: expected {EXPECTED[cell.dataset]} tasks, found {len(raw)}")
    data = compact(raw, webshop_outcomes=cell.dataset == "webshop", draws=args.posterior_draws)
    size = len(data.ids)
    shape = (size, 17)
    combined = Posterior(np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape))
    settings = [setting_id(ratio, window) for window in WINDOWS for ratio in RATIOS]
    stops = {
        name: {budget: np.empty(size, dtype=int) for budget in BUDGETS}
        for name in settings
    }
    diagnostics: dict[str, object] = {}

    for fold in range(FOLDS):
        training = np.flatnonzero(data.folds != fold)
        testing = np.flatnonzero(data.folds == fold)
        posterior = fit_split_mechanism(data, training)
        combined.mean[testing] = posterior.mean[testing]
        combined.variance[testing] = posterior.variance[testing]
        combined.precision[testing] = posterior.precision[testing]
        fold_detail: dict[str, object] = {}
        values = posterior.variance[:, 2:17]
        for budget in BUDGETS:
            budget_detail: dict[str, object] = {}
            for window in WINDOWS:
                lower = lower_bound(budget, window)
                for ratio in RATIOS:
                    name = setting_id(ratio, window)
                    boundary, train_cost = calibrate_boundary(
                        values[training], ratio * budget, lower, budget
                    )
                    stops[name][budget][testing] = stop_at_boundary(
                        values[testing], boundary, lower, budget
                    )
                    budget_detail[name] = {
                        "boundary": float(boundary), "training_mean_cost": float(train_cost),
                        "target_mean_cost": float(ratio * budget), "lower": lower, "upper": budget,
                    }
            fold_detail[str(budget)] = budget_detail
        diagnostics[str(fold)] = fold_detail

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    temporary = output / "task_scores.jsonl.gz.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for method in ("fixed", *settings):
            ratio = 1.0 if method == "fixed" else int(method[3:5]) / 100.0
            window_text = "fixed" if method == "fixed" else method.split("_w", 1)[1]
            for budget in BUDGETS:
                used = np.full(size, budget, dtype=int) if method == "fixed" else stops[method][budget]
                indexes = np.arange(size)
                means = combined.mean[indexes, used]
                variances = np.maximum(combined.variance[indexes, used], 1e-12)
                scores = means - Z95 * np.sqrt(variances)
                rows.append({
                    "cell": cell.name, "dataset": cell.dataset, "backbone": cell.backbone,
                    "seed": cell.seed, "setting": method, "rho": ratio, "window": window_text,
                    "budget": budget, "tasks": size,
                    "auroc": float(roc_auc_score(data.labels, scores)),
                    "aupr": float(average_precision_score(data.labels, scores)),
                    "mean_trajectories": float(np.mean(used)),
                    "trajectory_saving": float(1 - np.mean(used) / budget),
                    "fraction_early": float(np.mean(used < budget)),
                    "minimum_trajectories": int(np.min(used)),
                    "maximum_trajectories": int(np.max(used)),
                })
                for sid, fold, label, score_value, count in zip(
                    data.ids, data.folds, data.labels, scores, used, strict=True
                ):
                    handle.write(json.dumps({
                        "cell": cell.name, "dataset": cell.dataset, "backbone": cell.backbone,
                        "seed": cell.seed, "sample_id": sid, "fold": int(fold),
                        "label_failure": int(label), "setting": method, "rho": ratio,
                        "window": window_text, "budget": budget, "score": float(score_value),
                        "trajectories_used": int(count),
                    }, sort_keys=True) + "\n")
    temporary.replace(output / "task_scores.jsonl.gz")
    write_csv(output / "cell_metrics.csv", rows)
    atomic_text(output / "fold_diagnostics.json", json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    manifest = {
        "contract_version": CONTRACT_VERSION, "cell": cell.name, "dataset": cell.dataset,
        "backbone": cell.backbone, "seed": cell.seed, "tasks": size, "folds": FOLDS,
        "budgets": list(BUDGETS), "ratios": list(RATIOS), "windows": list(WINDOWS),
        "settings": len(settings), "posterior_draws": args.posterior_draws,
        "posterior": "frozen degree-1 Gaussian BR-LG with full covariance and count prior",
        "score": "mu - 1.959963984540054 sigma", "correctness_labels": "metrics only",
        "source": str(source), "source_sha256": sha256(source),
        "new_generation_calls": 0, "gpu_calls": 0,
    }
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_text(output / "COMPLETE", "")
    print(json.dumps({"cell": cell.name, "status": "complete", "settings": len(settings)}), flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluations"))
    parser.add_argument("--output-root", type=Path, default=root / "outputs/analysis/brlg_varstop_sensitivity_run/cells")
    parser.add_argument("--posterior-draws", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
