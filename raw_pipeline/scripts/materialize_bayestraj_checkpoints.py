#!/usr/bin/env python3
"""Create the paper's BayesTraj cross-fitting inputs from frozen raw pools.

This is a CPU-only transformation. It computes the label-free OE16 target,
the count-prior moments, and ten trajectory features. It does not fit or score
any additional uncertainty method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bayestraj_common import BACKBONES, EXPECTED, FOLDS, SEEDS, atomic_text
from evaluate_mixed_budget_core_multiseed import z16_path
from src.ltuq.experiments.sampling_efficiency import tdps_from_record
from src.ltuq.runners.agentbench import _dbbench_tdp_outcome_bucket
from src.ltuq.runners.agentbench_webshop import _webshop_prediction_outcome_bucket
from src.ltuq.runners.hotpotqa import _hotpotqa_tdp_outcome_bucket
from src.ltuq.runners.strategyqa import _strategyqa_tdp_outcome_bucket

DATASETS = tuple(EXPECTED)
PREFIXES = tuple(range(1, 17))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entropy(values: Sequence[str]) -> float:
    counts = Counter(values); total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def stable_seed(value: object) -> int:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return int(hashlib.sha1(payload.encode()).hexdigest()[:16], 16) % (2**32)


def posterior_moments(buckets: Sequence[str], sample_id: str, prefix: int, draws: int) -> tuple[float, float]:
    counts = Counter(map(str, buckets))
    parameters = np.asarray([counts[key] + 0.5 for key in sorted(counts)] + [0.5])
    probabilities = np.random.default_rng(stable_seed({
        "sample_id": sample_id, "prefix": prefix, "family": "dirichlet_other",
        "alpha_seen": 0.5, "alpha_other": 0.5, "theta": 0.5, "discount": 0.1,
    })).dirichlet(parameters, size=draws)
    values = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
    return float(np.mean(values)), float(np.var(values))


def trajectory_mark(tdp: Any) -> list[float]:
    rates: list[float] = []
    for step in tdp.steps:
        metadata = step.metadata.get("chosen_output_metadata") or {}
        log_probability, token_count = metadata.get("token_logprob_sum"), metadata.get("token_count")
        if isinstance(log_probability, (int, float)) and math.isfinite(float(log_probability)) and isinstance(token_count, (int, float)) and float(token_count) > 0:
            rates.append(-float(log_probability) / float(token_count))
    if not rates:
        return [math.nan, math.nan, math.nan, math.nan, math.log1p(len(tdp.steps))]
    return [statistics.fmean(rates), statistics.pstdev(rates) if len(rates) > 1 else 0.0,
            max(rates), rates[-1] - rates[0], math.log1p(len(tdp.steps))]


def aggregate_features(marks: Sequence[Sequence[float]]) -> list[float]:
    matrix = np.asarray(marks, dtype=float)
    means, deviations = [], []
    for column in range(matrix.shape[1]):
        finite = matrix[:, column][np.isfinite(matrix[:, column])]
        means.append(float(np.mean(finite)) if finite.size else math.nan)
        deviations.append(float(np.std(finite)) if finite.size else math.nan)
    return [*means, *deviations]


def canonical_buckets(dataset: str, record: dict[str, Any], tdps: Sequence[Any]) -> list[str]:
    bucket_fn = {
        "dbbench": _dbbench_tdp_outcome_bucket,
        "hotpotqa": _hotpotqa_tdp_outcome_bucket,
        "webshop": _webshop_prediction_outcome_bucket,
        "strategyqa": _strategyqa_tdp_outcome_bucket,
    }[dataset]
    return [str(bucket_fn(tdp)) for tdp in tdps]


def materialize(source: Path, dataset: str, backbone: str, seed: int, draws: int) -> list[dict[str, Any]]:
    records = read_jsonl(source)
    if len(records) != EXPECTED[dataset]:
        raise ValueError(f"{source}: expected {EXPECTED[dataset]} rows, found {len(records)}")
    if dataset in {"dbbench", "webshop"}:
        records.sort(key=lambda row: int(row["task_index"]))
    output = []
    for index, record in enumerate(records):
        tdps = tdps_from_record(record)
        if len(tdps) != 16:
            raise ValueError(f"{record.get('sample_id')}: expected 16 trajectories")
        buckets = canonical_buckets(dataset, record, tdps)
        marks = [trajectory_mark(tdp) for tdp in tdps]
        prefixes = {}
        for prefix in PREFIXES:
            mean, variance = posterior_moments(buckets[:prefix], str(record["sample_id"]), prefix, draws)
            prefixes[str(prefix)] = {"features": aggregate_features(marks[:prefix]),
                                     "counts_prior_mean": mean, "counts_prior_variance": variance}
        output.append({
            "dataset": dataset, "backbone": backbone, "seed": seed,
            "combination": f"{dataset}-{backbone}-seed{seed}",
            "sample_id": str(record["sample_id"]), "task_index": int(record.get("task_index", index)),
            "label_failure": int(record.get("is_correct") is not True), "fold": index % FOLDS,
            "target_oe16": entropy(buckets), "buckets": buckets, "prefixes": prefixes,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--posterior-draws", type=int, default=2048)
    args = parser.parse_args()
    for seed in args.seeds:
        campaign = "bayestraj_4datasets_3backbones_seeds101_202_z16" if seed in (101, 202) else None
        for dataset in args.datasets:
            for backbone in args.backbones:
                name = f"{dataset}-{backbone}-seed{seed}"
                resolved_campaign = campaign or f"bayestraj_seed303_{backbone}_z16"
                destination = args.output_dir / resolved_campaign / "checkpoints" / f"{name}.jsonl"
                rows = materialize(z16_path(args.pool_root, dataset, backbone, seed), dataset, backbone, seed, args.posterior_draws)
                atomic_text(destination, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                print(json.dumps({"cell": name, "tasks": len(rows), "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
