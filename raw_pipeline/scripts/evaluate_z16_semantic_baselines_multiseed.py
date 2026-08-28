#!/usr/bin/env python3
"""Evaluate the paper's Z=16 semantic baselines on frozen cells.

KLE, EigV, and CoCoA use the official-style DeBERTa-v2-xlarge MNLI runner;
SNNE uses the authors' ROUGE-L path. The script never generates responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_mixed_budget_core_multiseed import BACKBONES, BUDGETS, DATASETS, EXPECTED, SEEDS, read_jsonl, z16_path
from run_uq_baselines import collect_nli_probabilities
from src.ltuq.baselines.semantic_nearest_neighbor_entropy import RougeLSimilarity, compute_snne
from src.ltuq.experiments.sampling_efficiency import tdps_from_record
from uq_baselines import cocoa_score, eigv_score, kle_heat_score, semantic_similarity, symmetrize


KLE_NLI = os.environ.get(
    "LTUQ_KLE_NLI_MODEL",
    "microsoft/deberta-v2-xlarge-mnli",
)
METHODS = ("SNNE", "KLE", "EigV", "CoCoA-MaxProb", "CoCoA-PPL")


def terminal_response(tdp: dict[str, Any]) -> dict[str, Any] | None:
    """Extract response text and its aligned terminal-decision likelihood."""
    steps = tdp.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict): return None
    step = steps[-1]; step_metadata = step.get("metadata")
    metadata = step_metadata.get("chosen_output_metadata") if isinstance(step_metadata, dict) else None
    if not isinstance(metadata, dict): return None
    logprob_sum, token_count = metadata.get("decision_token_logprob_sum"), metadata.get("decision_token_count")
    if isinstance(logprob_sum, bool) or not isinstance(logprob_sum, (int, float)) or not math.isfinite(float(logprob_sum)) or isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0 or float(logprob_sum) / token_count < -100: return None
    tdp_metadata = tdp.get("metadata"); hard = isinstance(tdp_metadata.get("hard_finalization") if isinstance(tdp_metadata, dict) else None, dict)
    final = tdp.get("final_answer")
    text = final.strip() if isinstance(final, str) and final.strip() and not hard else str(step.get("realized_decision") or "").strip()
    if not text: return None
    return {"text": text, "sequence_logprob": float(logprob_sum), "length_normalized_logprob": float(logprob_sum / token_count)}


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


def metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
      for budget in BUDGETS:
        selected = [row for row in rows if row["method"] == method and row["budget"] == budget and math.isfinite(float(row["score"]))]
        labels = [int(row["label_failure"]) for row in selected]
        scores = [float(row["score"]) for row in selected]
        output.append({
            "dataset": rows[0]["dataset"],
            "backbone": rows[0]["backbone"],
            "seed": rows[0]["seed"],
            "combination": f"{rows[0]['dataset']}-{rows[0]['backbone']}",
            "method": method,
            "budget": budget,
            "mean_trajectories": budget,
            "trajectory_regime": "ordered_z16_prefix",
            "trajectory_count": budget,
            "candidate_n": "not_applicable",
            "tasks": EXPECTED[rows[0]["dataset"]],
            "used": len(selected),
            "coverage": len(selected) / EXPECTED[rows[0]["dataset"]],
            "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "aupr": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
        })
    return output


def evaluate_cell(
    *,
    source: Path,
    dataset: str,
    backbone: str,
    seed: int,
    output: Path,
    device: str,
    batch_size: int,
    max_tasks: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(source)
    if len(records) != EXPECTED[dataset]:
        raise ValueError(f"{source}: expected {EXPECTED[dataset]}, found {len(records)}")
    if max_tasks is not None:
        records = records[:max_tasks]
    prepared: list[dict[str, Any]] = []
    scorer = RougeLSimilarity()
    for record in records:
        tdps = tdps_from_record(record)
        if len(tdps) != 16:
            raise ValueError(f"{record.get('sample_id')}: Z != 16")
        raw_tdps = (record.get("estimate") or {}).get("tdps") or []
        responses = [terminal_response(tdp) for tdp in raw_tdps]
        if any(response is None for response in responses):
            continue
        resolved = [response for response in responses if response is not None]
        prepared.append({
            "sample_id": str(record["sample_id"]),
            "task_index": record.get("task_index"),
            "label_failure": int(record.get("is_correct") is not True),
            "prompt": tdps[0].prompt,
            "tdps": tdps,
            "responses": resolved,
        })
    prompts = [row["prompt"] for row in prepared]
    primary = [row["responses"][0]["text"] for row in prepared]
    alternatives = [[item["text"] for item in row["responses"][1:]] for row in prepared]
    output.mkdir(parents=True, exist_ok=True)
    kle_nli = collect_nli_probabilities(
        questions=prompts,
        preds=primary,
        responses=alternatives,
        model_name=KLE_NLI,
        device=device,
        batch_size=batch_size,
        cache_path=output / "kle_pairwise_nli.npz",
        force=False,
    )
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(prepared):
        for budget in BUDGETS:
            response_records = item["responses"][:budget]
            task_nli = np.asarray(kle_nli[index], dtype=np.float64)[:budget, :budget]
            contradiction, neutral, entailment = task_nli[:, :, 0], task_nli[:, :, 1], task_nli[:, :, 2]
            similarity = semantic_similarity(entailment, neutral, contradiction)
            values = {
                "SNNE": compute_snne(item["tdps"][:budget], similarity_fn=scorer),
                "KLE": kle_heat_score(entailment, neutral, contradiction),
                "EigV": eigv_score(symmetrize(entailment)),
                "CoCoA-MaxProb": cocoa_score(float(response_records[0]["sequence_logprob"]), similarity[0, 1:], confidence="maxprob"),
                "CoCoA-PPL": cocoa_score(float(response_records[0]["length_normalized_logprob"]), similarity[0, 1:], confidence="ppl"),
            }
            for method, score in values.items():
              if score is None or not math.isfinite(float(score)):
                continue
              rows.append({
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "sample_id": item["sample_id"],
                "task_index": item["task_index"],
                "label_failure": item["label_failure"],
                "method": method,
                "budget": budget,
                "mean_trajectories": budget,
                "score": float(score),
                "trajectory_regime": "ordered_z16_prefix",
                "trajectory_count": budget,
                "candidate_n": "not_applicable",
              })
    return rows, metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, default=Path("artifacts/raw/z16"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-cell", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    wanted = set(args.only_cell)
    for seed in SEEDS:
        for dataset in DATASETS:
            for backbone in BACKBONES:
                key = f"{dataset}-{backbone}-seed{seed}"
                if wanted and key not in wanted:
                    continue
                cell_output = args.output_dir.resolve() / key
                score_path = cell_output / "task_scores.jsonl"
                metric_path = cell_output / "seed_level_metrics.csv"
                if score_path.exists() and metric_path.exists() and args.max_tasks is None:
                    print(json.dumps({"reused": key}), flush=True)
                    continue
                rows, summary = evaluate_cell(
                    source=z16_path(args.pool_root.resolve(), dataset, backbone, seed),
                    dataset=dataset,
                    backbone=backbone,
                    seed=seed,
                    output=cell_output,
                    device=args.device,
                    batch_size=args.batch_size,
                    max_tasks=args.max_tasks,
                )
                write_jsonl(score_path, rows)
                write_csv(metric_path, summary)
                atomic_text(cell_output / "provenance.json", json.dumps({
                    "cell": key,
                    "source": str(z16_path(args.pool_root.resolve(), dataset, backbone, seed)),
                    "trajectory_regime": "complete_z16",
                    "trajectory_count": 16,
                    "generation_calls": 0,
                    "kle_nli_model": KLE_NLI,
                    "snne_temperature": 1.0,
                }, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"completed": key, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
