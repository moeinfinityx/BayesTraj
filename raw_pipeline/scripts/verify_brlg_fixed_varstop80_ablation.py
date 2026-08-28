#!/usr/bin/env python3
"""Independent audit of the BR-LG-Fixed/VarStop80 ablation report."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aggregate_brlg_fixed_varstop80_ablation import CONTRASTS, REPORT_DATASETS, report_cells
from paper_ablation_common import expected_cells
from evaluate_brlg_fixed_varstop80_ablation import CONTRACT_VERSION, FULL_FIXED, FULL_VAR
from bayestraj_common import BUDGETS, EXPECTED


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def audit_cell(root: Path, protocol_hash: str) -> tuple[int, list[dict[str, object]]]:
    name = root.name
    dataset = name.split("-", 1)[0]
    manifest = json.loads((root / "manifest.json").read_text())
    checks = {
        "complete": (root / "COMPLETE").is_file(),
        "contract": manifest.get("contract_version") == CONTRACT_VERSION,
        "protocol": manifest.get("protocol_sha256") == protocol_hash,
        "tasks": int(manifest.get("tasks", -1)) == EXPECTED[dataset],
        "variants": int(manifest.get("variant_count", -1)) == 6,
        "rows": int(manifest.get("metric_rows", -1)) == 6 * len(BUDGETS),
        "no generation": int(manifest.get("new_generation_calls", -1)) == 0,
        "no gpu": int(manifest.get("gpu_calls", -1)) == 0,
        "reproduction": int(manifest.get("reproduction", {}).get("stop_mismatches", -1)) == 0
            and float(manifest.get("reproduction", {}).get("maximum_score_error", 1)) <= 1e-9,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{name}: {', '.join(failed)}")
    if digest(Path(manifest["source"])) != manifest["source_sha256"] or digest(Path(manifest["reference"])) != manifest["reference_sha256"]:
        raise RuntimeError(f"{name}: source/reference hash mismatch")
    metrics = csv_rows(root / "cell_metrics.csv")
    metric_map = {(row["variant"], int(row["budget"])): row for row in metrics}
    if len(metric_map) != 6 * len(BUDGETS):
        raise RuntimeError(f"{name}: duplicate/missing metric rows")
    count = 0
    groups = 0
    with gzip.open(root / "task_scores.jsonl.gz", "rt", encoding="utf-8") as handle:
        parsed = (json.loads(line) for line in handle if line.strip())
        for key, iterator in itertools.groupby(parsed, key=lambda row: (row["variant"], int(row["budget"]))):
            block = list(iterator)
            count += len(block)
            groups += 1
            if len(block) != EXPECTED[dataset] or len({row["sample_id"] for row in block}) != len(block):
                raise RuntimeError(f"{name}/{key}: task identity/count")
            labels = np.asarray([int(row["label_failure"]) for row in block])
            scores = np.asarray([float(row["score"]) for row in block])
            stops = np.asarray([int(row["trajectories_used"]) for row in block])
            expected = metric_map[key]
            values = {
                "auroc": roc_auc_score(labels, scores), "aupr": average_precision_score(labels, scores),
                "mean_trajectories": stops.mean(), "trajectory_saving": 1 - stops.mean() / key[1],
                "variance_trajectories": stops.var(), "fraction_above_budget": np.mean(stops > key[1]),
            }
            for metric, value in values.items():
                if not close(float(value), float(expected[metric])):
                    raise RuntimeError(f"{name}/{key}: {metric}")
            if int(stops.min()) != int(expected["minimum_trajectories"]) or int(stops.max()) != int(expected["maximum_trajectories"]):
                raise RuntimeError(f"{name}/{key}: stop range")
            if key[0] == FULL_FIXED and not np.all(stops == key[1]):
                raise RuntimeError(f"{name}/{key}: fixed allocation violation")
            if key[0] == FULL_VAR and (np.any(stops > key[1]) or np.any(stops < max(2, key[1] - 4))):
                raise RuntimeError(f"{name}/{key}: VarStop80 support violation")
    if groups != 25 * len(BUDGETS) or count != EXPECTED[dataset] * 25 * len(BUDGETS):
        raise RuntimeError(f"{name}: task-score total")
    return count, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    output = args.output_root.resolve()
    protocol_hash = digest(args.protocol.resolve())
    problems: list[str] = []
    task_rows = 0
    all_metrics: list[dict[str, object]] = []
    for name in report_cells():
        try:
            count, metrics = audit_cell(run / "cells" / name, protocol_hash)
            task_rows += count
            all_metrics.extend(metrics)
        except Exception as error:
            problems.append(str(error))

    manifest = json.loads((output / "validation_manifest.json").read_text())
    checks = {
        "contract": manifest.get("contract_version") == CONTRACT_VERSION,
        "cells": int(manifest.get("cells", -1)) == 36,
        "datasets": tuple(manifest.get("datasets", ())) == REPORT_DATASETS,
        "variants": int(manifest.get("variants", -1)) == 25,
        "contrasts": int(manifest.get("contrasts", -1)) == 20,
        "protocol": manifest.get("protocol_sha256") == protocol_hash,
        "bootstrap": int(manifest.get("task_bootstrap_replicates", -1)) == 500 and int(manifest.get("hierarchical_bootstrap_replicates", -1)) == 10000,
        "reproduction": bool(manifest.get("all_reproduction_checks_passed")),
    }
    problems.extend(f"aggregate: {key}" for key, passed in checks.items() if not passed)
    for relative, expected_hash in manifest.get("artifacts", {}).items():
        path = output / relative
        if not path.is_file() or digest(path) != expected_hash:
            problems.append(f"artifact hash: {relative}")

    inference = {row["contrast"]: row for row in csv_rows(output / "hierarchical_paired_contrasts.csv")}
    if set(inference) != set(CONTRASTS):
        problems.append("contrast registry mismatch")
    frame = pd.DataFrame.from_records(all_metrics)
    if len(frame):
        for metric in ("auroc", "aupr"):
            frame[metric] = frame[metric].astype(float)
        frame["budget"] = frame.budget.astype(int)
        for name, spec in CONTRASTS.items():
            full = frame[frame.variant == spec["full"]].set_index(["cell", "budget"])
            alt = frame[frame.variant == spec["ablation"]].set_index(["cell", "budget"])
            joined = full[["auroc", "aupr"]].join(alt[["auroc", "aupr"]], lsuffix="_full", rsuffix="_alt")
            for metric in ("auroc", "aupr"):
                point = float((joined[f"{metric}_full"] - joined[f"{metric}_alt"]).groupby(level="cell").mean().mean())
                if name in inference and not close(point, float(inference[name][f"{metric}_delta"])):
                    problems.append(f"contrast point mismatch: {name}/{metric}")

    required = [
        output / "report/report.md", output / "report/plots/fixed_component_forest.png",
        output / "report/plots/varstop80_component_forest.png", output / "report/plots/stopping_forest.png",
        output / "report/plots/realized_cost_curves.png", output / "task_scores.jsonl.gz",
        output / "report/plots/core_mechanism_forest.png", output / "report/plots/core_mechanism_aupr_heatmap.png",
        output / "report/plots/core_mechanism_tradeoff.png", output / "core_mechanism_aupr_heatmap.csv",
        output / "core_mechanism_tradeoff.csv",
    ]
    problems.extend(f"missing: {path}" for path in required if not path.is_file())
    if (output / "task_scores.jsonl.gz").is_file():
        with gzip.open(output / "task_scores.jsonl.gz", "rt", encoding="utf-8") as handle:
            aggregate_datasets = {json.loads(line)["dataset"] for line in handle if line.strip()}
        if aggregate_datasets != set(REPORT_DATASETS):
            problems.append(f"aggregate task-score datasets: {sorted(aggregate_datasets)}")
    audit = {
        "status": "failed" if problems else "passed", "cells_audited": 36,
        "datasets_audited": list(REPORT_DATASETS),
        "task_rows_recomputed": task_rows, "metric_groups_recomputed": 36 * 25 * len(BUDGETS),
        "contrasts_recomputed": 20, "protocol_sha256": protocol_hash, "problems": problems,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if problems:
        raise RuntimeError("audit failed:\n" + "\n".join(problems))
    (output / "AUDIT_PASSED").write_text("independent task-level metric, contrast, provenance, and artifact audit passed\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
