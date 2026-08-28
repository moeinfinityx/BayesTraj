#!/usr/bin/env python3
"""Merge BayesTraj and exactly the 17 paper baselines into one curve table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bayestraj_common import BACKBONES, BUDGETS, EXPECTED, SEEDS

DATASETS = tuple(EXPECTED)
OURS = ("BR-LG-Risk-VC4-LCB95", "BR-LG-VarStop80-LCB95")
BASELINES = ("PPL", "PE", "SentSAR", "LS", "SE", "SNNE", "KLE", "Degree", "EigV", "SD",
             "CoCoA-MaxProb", "CoCoA-PPL", "MC-OE", "BSE-Ciosek-Fixed", "BSE-Ciosek-Adaptive", "SAUP", "UProp")


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def normalize(row, method=None):
    value = dict(row); value["method"] = method or value.get("method") or value.get("variant")
    value.setdefault("combination", f"{value['dataset']}-{value['backbone']}")
    value.setdefault("coverage", 1.0); value.setdefault("mean_trajectories", value["budget"])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--saup-root", type=Path, required=True)
    parser.add_argument("--bse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); rows = []
    core = read(args.core_root / "seed_level_metrics.csv")
    rows.extend(normalize(row) for row in core if row["method"] in BASELINES)
    for seed in SEEDS:
      for dataset in DATASETS:
       for backbone in BACKBONES:
        cell = f"{dataset}-{backbone}-seed{seed}"
        for source in (args.semantic_root, args.saup_root, args.bse_root):
            rows.extend(normalize(row) for row in read(source / cell / "seed_level_metrics.csv") if row["method"] in BASELINES)
        for row in read(args.ablation_root / "cells" / cell / "cell_metrics.csv"):
            if row["variant"] in OURS:
                rows.append(normalize(row, row["variant"]))
    allowed = set(OURS + BASELINES)
    if {row["method"] for row in rows} != allowed:
        raise RuntimeError("paper method registry is incomplete")
    keys = [(row["dataset"], row["backbone"], int(row["seed"]), row["method"], int(float(row["budget"]))) for row in rows]
    if len(keys) != len(set(keys)): raise RuntimeError("duplicate curve rows")
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    write(output / "trajectory_curve_seed_metrics.csv", rows)
    (output / "report.md").write_text("# Paper-scoped curve inputs\n\nExactly two BayesTraj variants and the 17 submitted baselines.\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"ours": OURS, "baselines": BASELINES, "rows": len(rows)}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "methods": len(allowed)}))


if __name__ == "__main__": main()
