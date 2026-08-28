#!/usr/bin/env python3
"""Evaluate generic-distance SAUP from the complete cached Z=16 pools."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_mixed_budget_core_multiseed import (  # noqa: E402
    BACKBONES, BUDGETS, DATASETS, EXPECTED, SEEDS, read_jsonl, z16_path,
)
from src.ltuq.baselines.saup import (  # noqa: E402
    SAUP_BASELINE,
    AGENT_TRACER_GENERIC_DISTANCE_SAUP_VERSION,
    SAUP_ROBERTA_MODEL,
    SAUP_ROBERTA_REVISION,
    RobertaSquadSemanticDistance,
    compute_saup_diagnostics,
)
from src.ltuq.experiments.sampling_efficiency import tdps_from_record  # noqa: E402


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def evaluate(source: Path, dataset: str, backbone: str, seed: int, device: str, batch_size: int):
    records = read_jsonl(source)
    if len(records) != EXPECTED[dataset]:
        raise ValueError(f"{source}: expected {EXPECTED[dataset]}, found {len(records)}")
    prepared, texts = [], []
    for record in records:
        tdps = tdps_from_record(record)
        if len(tdps) != 16:
            raise ValueError(f"{record.get('sample_id')}: expected sixteen trajectories")
        for tdp in tdps:
            def collect(left: str, right: str) -> float:
                texts.extend((left, right))
                return 0.0
            compute_saup_diagnostics(
                tdp, SAUP_BASELINE,
                distance_fn=collect, uncertainty_key="pe",
            )
        prepared.append((record, tdps))
    distance = RobertaSquadSemanticDistance(device=device, max_length=512)
    distance.precompute(texts, batch_size=batch_size)
    rows = []
    for record, tdps in prepared:
        values = []
        for tdp in tdps:
            result = compute_saup_diagnostics(
                tdp, SAUP_BASELINE,
                distance_fn=distance, uncertainty_key="pe",
            )
            if result.score is not None and math.isfinite(result.score):
                values.append(float(result.score))
        if len(values) == 16:
          for budget in BUDGETS:
            rows.append({
                "dataset": dataset, "backbone": backbone, "seed": seed,
                "sample_id": str(record["sample_id"]), "task_index": record.get("task_index"),
                "label_failure": int(record.get("is_correct") is not True),
                "method": "SAUP", "budget": budget, "score": float(np.mean(values[:budget])),
                "mean_trajectories": budget,
                "trajectory_regime": "ordered_z16_prefix", "trajectory_count": budget,
                "candidate_n": "stored", "aggregation": "mean_of_16_trajectory_scores",
            })
    metrics = []
    for budget in BUDGETS:
        selected = [row for row in rows if row["budget"] == budget]
        labels, scores = [row["label_failure"] for row in selected], [row["score"] for row in selected]
        metrics.append({
            "dataset": dataset, "backbone": backbone, "seed": seed,
            "combination": f"{dataset}-{backbone}", "method": "SAUP", "budget": budget,
            "mean_trajectories": budget, "trajectory_regime": "ordered_z16_prefix", "trajectory_count": budget,
            "candidate_n": "stored", "tasks": EXPECTED[dataset], "used": len(selected),
            "coverage": len(selected) / EXPECTED[dataset],
            "failure_rate": float(np.mean(labels)) if labels else None,
            "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "aupr": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
        })
    return rows, metrics, distance.provenance, len(set(texts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--only-cell", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    wanted = set(args.only_cell)
    for seed in SEEDS:
        for dataset in DATASETS:
            for backbone in BACKBONES:
                key = f"{dataset}-{backbone}-seed{seed}"
                if wanted and key not in wanted:
                    continue
                output = args.output_dir.resolve() / key
                score_path, metric_path = output / "task_scores.jsonl", output / "seed_level_metrics.csv"
                if score_path.exists() and metric_path.exists():
                    print(json.dumps({"reused": key}), flush=True)
                    continue
                rows, metrics, provenance, unique_texts = evaluate(
                    z16_path(args.pool_root.resolve(), dataset, backbone, seed),
                    dataset, backbone, seed, args.device, args.batch_size,
                )
                atomic(score_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                temporary = metric_path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                    writer.writeheader()
                    writer.writerows(metrics)
                temporary.replace(metric_path)
                atomic(output / "provenance.json", json.dumps({
                    "cell": key, "generation_calls": 0,
                    "formula_version": AGENT_TRACER_GENERIC_DISTANCE_SAUP_VERSION,
                    "semantic_model": SAUP_ROBERTA_MODEL,
                    "semantic_revision": SAUP_ROBERTA_REVISION,
                    "semantic_distance": provenance, "unique_texts": unique_texts,
                }, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"completed": key, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
