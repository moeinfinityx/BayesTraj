#!/usr/bin/env python3
"""Evaluate the paper's BSE-Fixed and BSE-Adaptive baselines on all cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_mixed_budget_core_multiseed import BACKBONES, BUDGETS, DATASETS, EXPECTED, SEEDS, read_jsonl, z16_path
from materialize_bayestraj_checkpoints import canonical_buckets
from src.ltuq.baselines.bayesian_semantic_entropy import estimate_bayesian_entropy, train_support_distribution
from src.ltuq.experiments.sampling_efficiency import tdps_from_record

FOLDS = 5
METHODS = ("BSE-Ciosek-Fixed", "BSE-Ciosek-Adaptive")


def seed(value: object) -> int:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return int(hashlib.sha1(payload.encode()).hexdigest()[:16], 16) % (2**32)


def evidence(tdp):
    identity = tuple([*(str(step.realized_decision) for step in tdp.steps), f"FINAL:{tdp.final_answer}"])
    log_probability = 0.0
    for step in tdp.steps:
        value = (step.metadata.get("chosen_output_metadata") or {}).get("token_logprob_sum")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return identity, 0.0
        log_probability += float(value)
    return identity, math.exp(log_probability) if log_probability > -745 else 0.0


def stopping(prefixes, threshold: float, cap: int):
    for prefix in range(1, cap + 1):
        if prefixes[prefix][1] <= threshold:
            return prefix, prefixes[prefix][0]
    return cap, prefixes[cap][0]


def calibrate(items, target: float, cap: int):
    values = np.asarray([items[index][prefix][1] for index in range(len(items)) for prefix in range(1, cap + 1)])
    choices = np.unique(np.concatenate(([0.0], np.percentile(values, np.arange(101)))))
    return min(choices, key=lambda threshold: (abs(statistics.fmean(stopping(item, threshold, cap)[0] for item in items) - target), threshold))


def evaluate(source: Path, dataset: str, backbone: str, run_seed: int, samples: int, replicates: int):
    records = read_jsonl(source)
    if len(records) != EXPECTED[dataset]:
        raise ValueError(f"{source}: incomplete cell")
    prepared = []
    for record in records:
        tdps = tdps_from_record(record)
        buckets = canonical_buckets(dataset, record, tdps)
        sequence = [evidence(tdp) for tdp in tdps]
        prepared.append((record, buckets, [item[0] for item in sequence], [item[1] for item in sequence]))
    priors = {fold: train_support_distribution(prepared[index][1] for index in range(len(prepared)) if index % FOLDS != fold) for fold in range(FOLDS)}
    posterior = []
    for index, (record, buckets, identities, probabilities) in enumerate(prepared):
        prefixes = {}
        for prefix in range(1, 17):
            estimates = [estimate_bayesian_entropy(
                buckets[:prefix], support_prior=priors[index % FOLDS], sequence_ids=identities[:prefix],
                sequence_probabilities=probabilities[:prefix], alpha=0.5, monte_carlo_samples=samples,
                rng=np.random.default_rng(seed((dataset, backbone, run_seed, record["sample_id"], prefix, replicate))),
            ) for replicate in range(replicates)]
            prefixes[prefix] = (statistics.fmean(item.mean for item in estimates), statistics.fmean(item.variance for item in estimates))
        posterior.append(prefixes)
    task_rows = []
    for budget in BUDGETS:
        thresholds = {}
        for fold in range(FOLDS):
            thresholds[fold] = calibrate([posterior[index] for index in range(len(posterior)) if index % FOLDS != fold], budget, 16)
        for index, (record, _, _, _) in enumerate(prepared):
            base = {"dataset": dataset, "backbone": backbone, "seed": run_seed,
                    "sample_id": str(record["sample_id"]), "task_index": record.get("task_index"),
                    "label_failure": int(record.get("is_correct") is not True), "budget": budget}
            task_rows.append({**base, "method": METHODS[0], "score": posterior[index][budget][0], "mean_trajectories": budget, "trajectories_used": budget})
            used, score_value = stopping(posterior[index], thresholds[index % FOLDS], 16)
            task_rows.append({**base, "method": METHODS[1], "score": score_value, "mean_trajectories": used, "trajectories_used": used})
    metrics = []
    for method in METHODS:
      for budget in BUDGETS:
        selected = [row for row in task_rows if row["method"] == method and row["budget"] == budget]
        labels, scores = [row["label_failure"] for row in selected], [row["score"] for row in selected]
        metrics.append({"dataset": dataset, "backbone": backbone, "seed": run_seed,
                        "combination": f"{dataset}-{backbone}", "method": method, "budget": budget,
                        "tasks": len(selected), "used": len(selected), "coverage": 1.0,
                        "mean_trajectories": statistics.fmean(row["trajectories_used"] for row in selected),
                        "auroc": roc_auc_score(labels, scores), "aupr": average_precision_score(labels, scores)})
    return task_rows, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monte-carlo-samples", type=int, default=1000); parser.add_argument("--integration-replicates", type=int, default=5)
    args = parser.parse_args()
    for run_seed in SEEDS:
      for dataset in DATASETS:
       for backbone in BACKBONES:
        name = f"{dataset}-{backbone}-seed{run_seed}"; output = args.output_dir / name; output.mkdir(parents=True, exist_ok=True)
        rows, metrics = evaluate(z16_path(args.pool_root, dataset, backbone, run_seed), dataset, backbone, run_seed, args.monte_carlo_samples, args.integration_replicates)
        (output / "task_scores.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        with (output / "seed_level_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
        print(json.dumps({"cell": name, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
