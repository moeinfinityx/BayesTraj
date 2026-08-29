#!/usr/bin/env python3
"""Evaluate the paper's cached core baselines on all 36 experiment cells.

UProp and Degree are recomputed from the authoritative strict
Z=12/N=4 pools.  Every baseline in ``Z16_BASELINES`` uses all sixteen ordered
trajectories from the corresponding frozen Z=16 pool.  The script performs no
trajectory or response generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltuq.baselines.multistep import compute_multitdp_baseline
from src.ltuq.experiments.sampling_efficiency import (
    recompute_declared_method_scores,
    tdps_from_record,
)
from src.ltuq.runners.agentbench import _dbbench_tdp_outcome_bucket
from src.ltuq.runners.agentbench_webshop import _webshop_prediction_outcome_bucket
from src.ltuq.runners.hotpotqa import _hotpotqa_tdp_outcome_bucket
from src.ltuq.runners.strategyqa import _strategyqa_tdp_outcome_bucket


DATASETS = ("dbbench", "hotpotqa", "webshop", "strategyqa")
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
SEEDS = (101, 202, 303)
EXPECTED = {"dbbench": 300, "hotpotqa": 1000, "webshop": 200, "strategyqa": 687}
STRICT_METHODS = ("UProp", "Degree")
BUDGETS = (3, 4, 6, 8, 12, 16)
Z16_BASELINES = {
    "PPL": "ppl",
    "LS": "ls",
    "PE": "pe",
    "SE": "se",
    "SD": "sd",
    "SentSAR": "sentsar",
}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def z16_path(root: Path, dataset: str, backbone: str, seed: int) -> Path:
    suffix = "_n4" if dataset == "strategyqa" else ""
    directory = root / f"{dataset}_seed{seed}_z16{suffix}" / f"{dataset}_{backbone}"
    if dataset == "webshop":
        name = f"bayestraj_webshop_seed{seed}_z16_{backbone}_seed{seed}_pe.jsonl"
    else:
        run = f"bayestraj_{dataset}_seed{seed}_z16{suffix}"
        name = f"{run}_{backbone}_{dataset}_{backbone}_seed{seed}_uprop.jsonl"
    return directory / name


def strict_path(root: Path, dataset: str, backbone: str, seed: int) -> Path:
    directory = root / f"{dataset}_seed{seed}_z12_n4" / f"{dataset}_{backbone}"
    if dataset == "webshop":
        name = f"bayestraj_webshop_seed{seed}_z12_n4_{backbone}_seed{seed}_pe.jsonl"
    else:
        name = (
            f"bayestraj_{dataset}_seed{seed}_z12_n4_{backbone}_"
            f"{dataset}_{backbone}_seed{seed}_uprop.jsonl"
        )
    return directory / name


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    resolved = float(value)
    return resolved if math.isfinite(resolved) else default


def settings(record: Mapping[str, Any]) -> dict[str, Any]:
    estimate = record.get("estimate") or {}
    metadata = estimate.get("metadata") or {}
    return {
        "tau": finite(metadata.get("tau"), 1.0),
        "ratio_epsilon": finite(metadata.get("ratio_epsilon"), 1e-6),
        "ratio_cap": finite(metadata.get("ratio_cap"), 10.0),
        "intrinsic_cap": finite(metadata.get("intrinsic_cap")),
        "intrinsic_transform": "none",
    }


def entropy(values: Sequence[str]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], int(row["seed"]), row["method"], int(row["budget"]))
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (dataset, backbone, seed, method, budget), selected in sorted(groups.items()):
        valid = [row for row in selected if finite(row.get("score")) is not None]
        labels = [int(row["label_failure"]) for row in valid]
        scores = [float(row["score"]) for row in valid]
        result.append({
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
            "combination": f"{dataset}-{backbone}",
            "method": method,
            "budget": budget,
            "mean_trajectories": statistics.fmean(float(row["mean_trajectories"]) for row in valid),
            "trajectory_regime": valid[0]["trajectory_regime"] if valid else selected[0]["trajectory_regime"],
            "trajectory_count": valid[0]["trajectory_count"] if valid else selected[0]["trajectory_count"],
            "candidate_n": valid[0]["candidate_n"] if valid else selected[0]["candidate_n"],
            "tasks": len(selected),
            "used": len(valid),
            "coverage": len(valid) / len(selected),
            "failure_rate": statistics.fmean(labels) if labels else None,
            "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "aupr": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
        })
    return result


def evaluate_cell(
    *, dataset: str, backbone: str, seed: int, z16_root: Path, strict_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    z16_source = z16_path(z16_root, dataset, backbone, seed)
    strict_source = strict_path(strict_root, dataset, backbone, seed)
    z16_records = read_jsonl(z16_source)
    strict_records = read_jsonl(strict_source)
    if len(z16_records) != EXPECTED[dataset] or len(strict_records) != EXPECTED[dataset]:
        raise ValueError(f"{dataset}-{backbone}-seed{seed}: incomplete source")
    z16_by_id = {str(row["sample_id"]): row for row in z16_records}
    strict_by_id = {str(row["sample_id"]): row for row in strict_records}
    if len(z16_by_id) != len(z16_records) or set(z16_by_id) != set(strict_by_id):
        raise ValueError(f"{dataset}-{backbone}-seed{seed}: sample-set mismatch")
    task_rows: list[dict[str, Any]] = []
    prefix_identity_matches = 0
    candidate_steps = 0
    for sample_id, z16_record in z16_by_id.items():
        strict_record = strict_by_id[sample_id]
        z16_tdps = tdps_from_record(z16_record)
        strict_tdps = tdps_from_record(strict_record)[:12]
        if len(z16_tdps) != 16 or len(strict_tdps) != 12:
            raise ValueError(f"{sample_id}: expected Z16 and strict Z12")
        ids16 = [tdp.sample_id for tdp in z16_tdps[:12]]
        ids12 = [tdp.sample_id for tdp in strict_tdps]
        if ids16 != ids12:
            raise ValueError(f"{sample_id}: strict prefix identity mismatch")
        prefix_identity_matches += 1
        for tdp in strict_tdps:
            for step in tdp.steps:
                if step.metadata.get("forced_terminal_finish", False):
                    continue
                if len(step.sampled_decisions) != 4:
                    raise ValueError(f"{sample_id}/{tdp.sample_id}/step{step.index}: N != 4")
                candidate_steps += 1
        label = int(z16_record.get("is_correct") is not True)
        bucket_fn = {
            "dbbench": _dbbench_tdp_outcome_bucket,
            "hotpotqa": _hotpotqa_tdp_outcome_bucket,
            "webshop": _webshop_prediction_outcome_bucket,
            "strategyqa": _strategyqa_tdp_outcome_bucket,
        }[dataset]
        buckets16 = [str(bucket_fn(tdp)) for tdp in z16_tdps]
        base = {
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
            "sample_id": sample_id,
            "task_index": z16_record.get("task_index"),
            "label_failure": label,
        }
        for budget in BUDGETS:
            if budget <= 12:
                selected = strict_tdps[:budget]
                bucket_map = {tdp.sample_id: bucket for tdp, bucket in zip(selected, buckets16[:budget], strict=True)}
                strict_scores = recompute_declared_method_scores(
                    selected, z=budget, n=4, outcome_bucket_map=bucket_map,
                    allow_fixed_trajectory_candidate_subsampling=False, **settings(z16_record),
                )
                for method, key in (("UProp", "UProp"), ("Degree", "DEG")):
                    task_rows.append({**base, "budget": budget, "method": method,
                                      "score": strict_scores[key], "trajectory_regime": "strict_z12_n4",
                                      "trajectory_count": budget, "mean_trajectories": budget, "candidate_n": 4})
            for method, baseline in Z16_BASELINES.items():
                task_rows.append({**base, "budget": budget, "method": method,
                                  "score": compute_multitdp_baseline(z16_tdps[:budget], baseline, fallback_strategy=None),
                                  "trajectory_regime": "ordered_z16_prefix", "trajectory_count": budget,
                                  "mean_trajectories": budget, "candidate_n": "stored"})
            task_rows.append({**base, "budget": budget, "method": "MC-OE",
                              "score": entropy(buckets16[:budget]), "trajectory_regime": "ordered_z16_prefix",
                              "trajectory_count": budget, "mean_trajectories": budget, "candidate_n": "not_applicable"})
    return task_rows, {
        "dataset": dataset,
        "backbone": backbone,
        "seed": seed,
        "z16_source": str(z16_source),
        "strict_source": str(strict_source),
        "tasks": len(z16_records),
        "strict_prefix_identity_matches": prefix_identity_matches,
        "strict_candidate_steps_n4": candidate_steps,
        "generation_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z16-root", type=Path, default=Path("artifacts/raw/z16"))
    parser.add_argument("--strict-root", type=Path, default=Path("artifacts/raw/z8n4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/mixed_budget_5datasets_3backbones_3seeds/core"))
    parser.add_argument("--only-cell", action="append", default=[])
    args = parser.parse_args()
    wanted = set(args.only_cell)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for backbone in BACKBONES:
                key = f"{dataset}-{backbone}-seed{seed}"
                if wanted and key not in wanted:
                    continue
                cell_rows, audit = evaluate_cell(
                    dataset=dataset,
                    backbone=backbone,
                    seed=seed,
                    z16_root=args.z16_root.resolve(),
                    strict_root=args.strict_root.resolve(),
                )
                rows.extend(cell_rows)
                audits.append(audit)
                print(json.dumps({"completed": key, "task_score_rows": len(cell_rows)}), flush=True)
    output = args.output_dir.resolve()
    write_jsonl(output / "task_scores.jsonl", rows)
    write_csv(output / "seed_level_metrics.csv", summarize(rows))
    atomic_text(output / "provenance.json", json.dumps({
        "protocol": "strict Z=12/N=4 for UProp and Degree; complete Z=16 for all other paper baselines",
        "strict_methods": list(STRICT_METHODS),
        "z16_methods": [*Z16_BASELINES, "MC-OE"],
        "generation_calls": 0,
        "audits": audits,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "cells": len(audits), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
