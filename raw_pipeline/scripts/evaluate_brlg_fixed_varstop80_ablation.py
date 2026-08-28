#!/usr/bin/env python3
"""Evaluate one cell of the BR-LG-Fixed/VarStop80 component ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from bayestraj_common import (
    BUDGETS, add_variant, calibrate_boundary, fit_split_posterior, score,
    score_outputs, stop_at_boundary, write_csv,
    EXPECTED, FEATURE_SETS, FOLDS, PREFIXES, Cell, CompactRows, Posterior, atomic_text,
    checkpoint_path, compact, parse_cell, read_jsonl, sha256,
)
from src.ltuq.estimators.bayestraj_posterior import (
    fit_linear_gaussian_trajectory_likelihood, update_entropy_on_grid,
)


CONTRACT_VERSION = "brlg-fixed-varstop80-ablation-2026-08-11-v2"
FULL_FIXED = "BR-LG-Risk-VC4-LCB95"
FULL_VAR = "BR-LG-VarStop80-LCB95"
FEATURE_BLOCKS = ("no_level", "no_dynamics", "no_structure", "no_dispersion")
HASH_SEED = "brlg-nonadaptive80-20260811"


def make_storage(size: int) -> tuple[Posterior, dict[str, dict[int, np.ndarray]]]:
    shape = (size, 17)
    posterior = Posterior(np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan))
    allocations = {
        name: {budget: np.empty(size, dtype=int) for budget in BUDGETS}
        for name in ("fixed", "var80", "nonadaptive80")
    }
    for budget in BUDGETS:
        allocations["fixed"][budget][:] = budget
    return posterior, allocations


def fit_split_mechanism(
    data: CompactRows,
    training: np.ndarray,
    *,
    covariance_structure: str = "full",
    use_count_prior: bool = True,
) -> Posterior:
    shape = (len(data.ids), 17)
    means = np.full(shape, np.nan)
    variances = np.full(shape, np.nan)
    precisions = np.zeros(shape)
    columns = np.asarray(FEATURE_SETS["full"], dtype=int)
    for prefix in PREFIXES:
        likelihood = fit_linear_gaussian_trajectory_likelihood(
            data.features[training, prefix][:, columns], data.targets[training],
            covariance_structure=covariance_structure,
        )
        for index in range(len(data.ids)):
            posterior = update_entropy_on_grid(
                data.count_means[index, prefix], data.count_variances[index, prefix],
                data.features[index, prefix, columns], likelihood,
                use_count_prior=use_count_prior,
            )
            means[index, prefix] = posterior.mean
            variances[index, prefix] = posterior.variance
            precisions[index, prefix] = posterior.mark_precision
    return Posterior(means, variances, precisions)


def variance_allocate(
    posterior: Posterior,
    training: np.ndarray,
    testing: np.ndarray,
    budget: int,
    *,
    lower: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    values = posterior.variance[:, 2:17]
    boundary, train_mean = calibrate_boundary(values[training], 0.8 * budget, lower, budget)
    stops = stop_at_boundary(values[testing], boundary, lower, budget)
    return stops, {
        "boundary": float(boundary), "training_mean_cost": float(train_mean),
        "target_mean_cost": 0.8 * budget, "lower": lower, "upper": budget,
    }


def nonadaptive_stops(ids: Sequence[str], indexes: np.ndarray, budget: int) -> np.ndarray:
    target = max(2.0, 0.8 * budget)
    low, high = int(math.floor(target)), int(math.ceil(target))
    values = np.full(len(indexes), low, dtype=int)
    if high == low:
        return values
    number_high = int(round((target - low) * len(indexes)))
    ordered = sorted(
        range(len(indexes)),
        key=lambda position: hashlib.sha256(
            f"{HASH_SEED}|{budget}|{ids[int(indexes[position])]}".encode()
        ).digest(),
    )
    values[np.asarray(ordered[:number_high], dtype=int)] = high
    return values


def reference_rows(path: Path) -> dict[tuple[str, int, str], tuple[float, int]]:
    wanted = {"fixed", "uncertainty__vc4e__cost80"}
    output: dict[tuple[str, int, str], tuple[float, int]] = {}
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["candidate"] in wanted:
                output[(str(row["candidate"]), int(row["budget"]), str(row["sample_id"]))] = (
                    float(row["score"]), int(row["trajectories_used"]),
                )
    return output


def evaluate(args: argparse.Namespace) -> None:
    cell: Cell = parse_cell(args.cell)
    source = checkpoint_path(args.evaluation_root.resolve(), cell)
    output = args.output_root.resolve() / cell.name
    manifest_path = output / "manifest.json"
    protocol = Path(__file__).resolve().parents[1] / "docs/brlg_fixed_varstop80_ablation_protocol.md"
    if args.resume and (output / "COMPLETE").is_file() and manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if (
            old.get("contract_version") == CONTRACT_VERSION
            and old.get("source_sha256") == sha256(source)
            and old.get("protocol_sha256") == sha256(protocol)
        ):
            print(json.dumps({"cell": cell.name, "status": "reused"}), flush=True)
            return

    raw = read_jsonl(source)
    if len(raw) != EXPECTED[cell.dataset]:
        raise RuntimeError(f"{cell.name}: expected {EXPECTED[cell.dataset]} tasks, found {len(raw)}")
    data = compact(raw, webshop_outcomes=cell.dataset == "webshop", draws=args.posterior_draws)
    specifications: dict[str, tuple[CompactRows, bool, Sequence[int]]] = {
        "lg_full": (data, True, FEATURE_SETS["full"]),
        "count_full": (data, False, ()),
    }
    mechanism_specs = {
        "lg_likelihood_only": {"covariance_structure": "full", "use_count_prior": False},
        "lg_diagonal": {"covariance_structure": "diagonal", "use_count_prior": True},
    }
    for name in mechanism_specs:
        specifications[name] = (data, True, FEATURE_SETS["full"])
    data_by_name = {name: item[0] for name, item in specifications.items()}
    posterior_by_name: dict[str, Posterior] = {}
    allocation_by_name: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    diagnostics: dict[str, object] = {}

    for name, (spec_data, use_trajectory_likelihood, features) in specifications.items():
        combined, allocations = make_storage(len(spec_data.ids))
        posterior_by_name[name] = combined
        allocation_by_name[name] = allocations
        diagnostics[name] = {}
        for fold in range(FOLDS):
            training = np.flatnonzero(spec_data.folds != fold)
            testing = np.flatnonzero(spec_data.folds == fold)
            if name in mechanism_specs:
                posterior = fit_split_mechanism(spec_data, training, **mechanism_specs[name])
            else:
                posterior = fit_split_posterior(
                    spec_data,
                    training,
                    use_trajectory_likelihood=use_trajectory_likelihood,
                    feature_indices=features,
                )
            combined.mean[testing] = posterior.mean[testing]
            combined.variance[testing] = posterior.variance[testing]
            combined.precision[testing] = posterior.precision[testing]
            fold_detail: dict[str, object] = {}
            for budget in BUDGETS:
                vc4e, detail = variance_allocate(
                    posterior, training, testing, budget, lower=max(2, budget - 4)
                )
                allocations["var80"][budget][testing] = vc4e
                allocations["nonadaptive80"][budget][testing] = nonadaptive_stops(
                    spec_data.ids, testing, budget
                )
                fold_detail[str(budget)] = {"var80": detail}
            diagnostics[name][str(fold)] = fold_detail
        print(json.dumps({"cell": cell.name, "posterior": name}), flush=True)

    variants: dict[str, tuple[str, str, str, str]] = {}
    add_variant(variants, FULL_FIXED, "lg_full", "fixed", "lcb95", "primary")
    add_variant(variants, FULL_VAR, "lg_full", "var80", "lcb95", "primary")
    add_variant(variants, "BR-Count-VarStop80-LCB95", "count_full", "var80", "lcb95", "mechanism")
    add_variant(variants, "BR-LG-NonAdaptive80-LCB95", "lg_full", "nonadaptive80", "lcb95", "stopping")
    add_variant(variants, "BR-LG-VarStop80-LikelihoodOnly-LCB95", "lg_likelihood_only", "var80", "lcb95", "mechanism")
    add_variant(variants, "BR-LG-VarStop80-DiagonalCov-LCB95", "lg_diagonal", "var80", "lcb95", "mechanism")

    output.mkdir(parents=True, exist_ok=True)
    metrics = score_outputs(
        cell=cell, data_by_name=data_by_name, posterior_by_name=posterior_by_name,
        allocation_by_name=allocation_by_name, variants=variants, output=output,
    )
    write_csv(output / "cell_metrics.csv", metrics)
    atomic_text(output / "fold_diagnostics.json", json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    maximum_score_error = 0.0
    stop_mismatches = 0
    reference_path = None
    if args.reference_root is not None:
        candidate_path = args.reference_root.resolve() / "cells" / cell.name / "task_scores.jsonl.gz"
        if candidate_path.is_file():
            reference_path = candidate_path
            expected = reference_rows(reference_path)
            for candidate, variant in (("fixed", FULL_FIXED), ("uncertainty__vc4e__cost80", FULL_VAR)):
                posterior_name, allocation, score_kind, _ = variants[variant]
                posterior = posterior_by_name[posterior_name]
                for budget in BUDGETS:
                    stops = allocation_by_name[posterior_name][allocation][budget]
                    scores = score(posterior, stops, kind=score_kind)
                    for sid, value, used in zip(data.ids, scores, stops, strict=True):
                        reference_score, reference_stop = expected[(candidate, budget, sid)]
                        maximum_score_error = max(maximum_score_error, abs(float(value) - reference_score))
                        stop_mismatches += int(int(used) != reference_stop)
            if maximum_score_error > 1e-9 or stop_mismatches:
                raise RuntimeError(f"reference reproduction failed: score={maximum_score_error}, stops={stop_mismatches}")

    manifest = {
        "schema_version": 1, "contract_version": CONTRACT_VERSION,
        "cell": cell.name, "dataset": cell.dataset, "backbone": cell.backbone,
        "seed": cell.seed, "source": str(source), "source_sha256": sha256(source),
        "reference": str(reference_path) if reference_path else None,
        "reference_sha256": sha256(reference_path) if reference_path else None,
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "tasks": len(data.ids), "folds": FOLDS, "budgets": list(BUDGETS),
        "posterior_draws": args.posterior_draws, "variants": list(variants),
        "variant_count": len(variants), "metric_rows": len(metrics),
        "correctness_labels": "metrics only", "posterior_target": "label-free OE-16",
        "new_generation_calls": 0, "gpu_calls": 0, "hash_seed": HASH_SEED,
        "reproduction": {"maximum_score_error": maximum_score_error, "stop_mismatches": stop_mismatches},
    }
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_text(output / "COMPLETE", "")
    print(json.dumps({"cell": cell.name, "status": "complete", "variants": len(variants)}), flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluations"))
    parser.add_argument("--reference-root", type=Path, default=None,
                        help="Optional historical reference outputs for an exact equivalence check")
    parser.add_argument("--output-root", type=Path, default=root / "outputs/analysis/brlg_fixed_varstop80_ablation_run/cells")
    parser.add_argument("--posterior-draws", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
