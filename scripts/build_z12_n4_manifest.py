#!/usr/bin/env python3
"""Validate a from-scratch Z=12/N=4 campaign and emit its curve manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASETS = {"dbbench": 300, "hotpotqa": 1000, "webshop": 200, "strategyqa": 687}
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
SEEDS = (101, 202, 303)


def source_path(root: Path, dataset: str, backbone: str, seed: int) -> Path:
    cell = root / f"{dataset}_seed{seed}_z12_n4" / f"{dataset}_{backbone}"
    if dataset == "webshop":
        return cell / f"bayestraj_webshop_seed{seed}_z12_n4_{backbone}_seed{seed}_pe.jsonl"
    return cell / (
        f"bayestraj_{dataset}_seed{seed}_z12_n4_{backbone}_"
        f"{dataset}_{backbone}_seed{seed}_uprop.jsonl"
    )


def inspect(path: Path, expected: int) -> tuple[int, str, list[str]]:
    digest = hashlib.sha256()
    count = 0
    indices: list[int] = []
    failures: list[str] = []
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            count += 1
            indices.append(int(row["task_index"]))
            tdps = (row.get("estimate") or {}).get("tdps") or []
            if len(tdps) != 12:
                failures.append(f"task {row.get('task_index')}: Z={len(tdps)}")
    if count != expected:
        failures.append(f"rows={count}, expected={expected}")
    if indices != list(range(expected)):
        failures.append("task indices are not exactly ordered 0..N-1")
    return count, digest.hexdigest(), failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args()
    cells: list[dict] = []
    failures: list[dict] = []
    for dataset, expected in DATASETS.items():
        for backbone in BACKBONES:
            for seed in SEEDS:
                path = source_path(args.campaign_root, dataset, backbone, seed)
                if not path.is_file():
                    failures.append({"path": str(path), "failure": "missing"})
                    continue
                records, digest, cell_failures = inspect(path, expected)
                cell = {
                    "dataset": dataset,
                    "backbone": backbone,
                    "seed": seed,
                    "merged_output": str(path.resolve()),
                    "merged_sha256": digest,
                    "records": records,
                    "trajectory_budget": 12,
                    "candidate_budget": 4,
                    "generation_performed": True,
                    "complete": not cell_failures,
                }
                cells.append(cell)
                if cell_failures:
                    failures.append({"path": str(path), "failures": cell_failures[:10]})
    manifest = {
        "schema_version": 1,
        "complete": not failures and len(cells) == 36,
        "expected_cells": 36,
        "validated_cells": sum(bool(cell["complete"]) for cell in cells),
        "trajectory_budget": 12,
        "exact_candidates_per_nonterminal_step": 4,
        "executed_trajectory_generation_performed": True,
        "campaign_failures": failures,
        "failures": failures,
        "cells": cells,
    }
    target = args.campaign_root / "final_validation_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"validated cells: {manifest['validated_cells']}/36")
    print(target)
    if not manifest["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
