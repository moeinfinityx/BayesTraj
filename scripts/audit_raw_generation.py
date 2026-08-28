#!/usr/bin/env python3
"""Audit the complete 4-dataset/3-backbone/3-seed BayesTraj raw pools."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


DATASETS = {"dbbench": 300, "hotpotqa": 1000, "webshop": 200, "strategyqa": 687}
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
SEEDS = (101, 202, 303)


def pool_path(root: Path, dataset: str, backbone: str, seed: int) -> Path:
    suffix = "_n4" if dataset == "strategyqa" else ""
    base = root / f"{dataset}_seed{seed}_z16{suffix}" / f"{dataset}_{backbone}"
    if dataset == "webshop":
        return base / f"bayestraj_webshop_seed{seed}_z16_{backbone}_seed{seed}_pe.jsonl"
    return base / (
        f"bayestraj_{dataset}_seed{seed}_z16{suffix}_{backbone}_"
        f"{dataset}_{backbone}_seed{seed}_uprop.jsonl"
    )


def audit_file(path: Path, expected_rows: int) -> dict:
    digest = hashlib.sha256()
    indices: list[int] = []
    sample_ids: list[str] = []
    bad_z: list[dict] = []
    statuses: Counter[str] = Counter()
    inferred_task_indices = 0
    with path.open("rb") as raw:
        for line_no, line in enumerate(raw, 1):
            digest.update(line)
            row = json.loads(line)
            if row.get("task_index") is None:
                # Some Hugging Face runners preserve order and sample_id but
                # omit the redundant integer task_index field.
                indices.append(line_no - 1)
                inferred_task_indices += 1
            else:
                indices.append(int(row["task_index"]))
            sample_ids.append(str(row["sample_id"]))
            z = len(row.get("estimate", {}).get("tdps", []))
            if z != 16:
                bad_z.append({"line": line_no, "task_index": row.get("task_index"), "z": z})
            statuses[str(row.get("status", "not_recorded"))] += 1
    failures: list[str] = []
    if len(indices) != expected_rows:
        failures.append(f"row count {len(indices)} != {expected_rows}")
    if len(set(indices)) != len(indices):
        failures.append("duplicate task_index")
    if len(set(sample_ids)) != len(sample_ids):
        failures.append("duplicate sample_id")
    expected_indices = list(range(expected_rows))
    if sorted(indices) != expected_indices:
        failures.append("task indices are not exactly 0..expected_rows-1")
    if indices != sorted(indices):
        failures.append("rows are not in task-index order")
    if bad_z:
        failures.append(f"{len(bad_z)} rows do not contain exactly 16 ordered trajectories")
    return {
        "path": str(path.resolve()),
        "rows": len(indices),
        "sha256": digest.hexdigest(),
        "bad_z_examples": bad_z[:10],
        "task_indices_inferred_from_order": inferred_task_indices,
        "status_counts": dict(sorted(statuses.items())),
        "failures": failures,
        "valid": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("raw_generation_audit.json"))
    args = parser.parse_args()

    cells: list[dict] = []
    for dataset, expected_rows in DATASETS.items():
        for backbone in BACKBONES:
            for seed in SEEDS:
                path = pool_path(args.pool_root, dataset, backbone, seed)
                if not path.is_file():
                    cells.append({"path": str(path), "valid": False, "failures": ["missing"]})
                    continue
                result = audit_file(path, expected_rows)
                result.update(dataset=dataset, backbone=backbone, seed=seed)
                cells.append(result)

    summary = {
        "schema_version": 1,
        "pool_root": str(args.pool_root.resolve()),
        "expected_cells": 36,
        "valid_cells": sum(bool(cell["valid"]) for cell in cells),
        "expected_task_rows": sum(DATASETS.values()) * len(BACKBONES) * len(SEEDS),
        "observed_task_rows": sum(int(cell.get("rows", 0)) for cell in cells),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"valid cells: {summary['valid_cells']}/36")
    print(f"observed task rows: {summary['observed_task_rows']}/{summary['expected_task_rows']}")
    print(f"audit manifest: {args.output}")
    if summary["valid_cells"] != 36:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
