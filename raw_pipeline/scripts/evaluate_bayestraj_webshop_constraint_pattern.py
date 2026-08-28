#!/usr/bin/env python3
"""Evaluate label-free WebShop bucket mappings for frozen BR-LG methods."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in os.sys.path:
        os.sys.path.insert(0, str(item))

from evaluate_brlg_fixed_varstop80_ablation import variance_allocate
from bayestraj_common import (
    BUDGETS, fit_split_posterior, score,
    EXPECTED, FEATURE_SETS, FOLDS, PREFIXES, Cell, CompactRows, Posterior,
    atomic_text, checkpoint_path, parse_cell, posterior_moments, read_jsonl,
    sha256, target,
)
from evaluate_mixed_budget_core_multiseed import z16_path


CONTRACT = "bayestraj-webshop-constraint-pattern-2026-08-28-v1"
VARIANTS = ("constraint-pattern",)
STOP = {
    "BR-LG Risk VC4": "fixed",
    "BR-LG-VarStop80": "varstop80",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
PRICE_RE = re.compile(r"price\s*:\s*\$\s*([0-9]+(?:\.[0-9]+)?)", re.I)
LIMIT_RE = re.compile(
    r"(?:under|below|less than|lower than|no more than|at most|max(?:imum)?)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
    re.I,
)
STOPWORDS = frozenset(
    "a an and are as at be buy can for from get i in is it looking me my need of on or please "
    "product than that the this to want which with would price dollar dollars under below lower less".split()
)


def action(step: dict[str, Any]) -> str:
    meta = step.get("metadata") if isinstance(step, dict) else None
    chosen = meta.get("chosen_output_metadata") if isinstance(meta, dict) else None
    value = chosen.get("webshop_action_text") if isinstance(chosen, dict) else None
    return str(value or "").strip().lower()


def instruction(record: dict[str, Any], tdp: dict[str, Any]) -> str:
    text = str(tdp.get("prompt") or record.get("question") or "")
    match = re.search(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*Search", text, re.I | re.S)
    return (match.group(1) if match else text).strip().lower()


def product_metadata(tdp: dict[str, Any], asin: str) -> tuple[str, float | None]:
    asin = asin.lower()
    for step in tdp.get("steps") or []:
        if action(step) != f"click[{asin}]":
            continue
        meta = step.get("metadata") or {}
        observation = str(meta.get("observation") or "")
        pieces = [part.strip() for part in observation.split("[SEP]")]
        for index, part in enumerate(pieces):
            found = PRICE_RE.search(part)
            if found:
                title = pieces[index - 1] if index else "unknown"
                return title.lower(), float(found.group(1))
    return "unknown", None


def parse_purchase(bucket: str) -> tuple[str, tuple[str, ...]] | None:
    if not bucket.startswith("purchase:"):
        return None
    head, _, tail = bucket.partition("|options:")
    asin = head.removeprefix("purchase:").strip().lower()
    options = () if tail in ("", "none") else tuple(sorted(x.strip().lower() for x in tail.split(",") if x.strip()))
    return asin, options


def tokens(text: str) -> set[str]:
    return {value for value in TOKEN_RE.findall(text.lower()) if len(value) >= 3 and value not in STOPWORDS}


def price_relation(request: str, price: float | None) -> str:
    match = LIMIT_RE.search(request)
    if not match or price is None:
        return "price-unknown"
    return "price-pass" if price <= float(match.group(1)) else "price-fail"


def signatures(request: str, title: str, price: float | None, options: tuple[str, ...]) -> tuple[str, str]:
    wanted = tokens(request)
    observed = tokens(title + " " + " ".join(options))
    overlap = sorted(wanted & observed)
    option_overlap = bool(tokens(" ".join(options)) & wanted)
    relation = price_relation(request, price)
    attribute = f"tok={','.join(overlap) or 'none'}|opt={int(option_overlap)}|{relation}"
    coverage = len(overlap) / max(1, len(wanted))
    coverage_bin = "zero" if coverage == 0 else "low" if coverage < .25 else "mid" if coverage < .5 else "high"
    pattern = f"coverage={coverage_bin}|opt={int(option_overlap)}|{relation}"
    return attribute, pattern


def map_task(record: dict[str, Any], current: list[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    tdps = list((record.get("estimate") or {}).get("tdps") or [])[:16]
    if len(tdps) != 16 or len(current) != 16:
        raise RuntimeError(f"{record.get('sample_id')}: expected sixteen trajectories")
    request = instruction(record, tdps[0])
    items: dict[str, tuple[str, float | None, tuple[str, ...], str, str]] = {}
    for tdp, bucket in zip(tdps, current, strict=True):
        parsed = parse_purchase(bucket)
        if parsed is None:
            continue
        asin, opts = parsed
        key = f"{asin}|{','.join(opts) or 'none'}"
        if key not in items:
            title, price = product_metadata(tdp, asin)
            attr, pattern = signatures(request, title, price, opts)
            items[key] = (title, price, opts, attr, pattern)
    keys = sorted(items)
    mapped = {"constraint-pattern": []}
    for bucket in current:
        parsed = parse_purchase(bucket)
        if parsed is None:
            mapped["constraint-pattern"].append(bucket)
            continue
        asin, opts = parsed
        key = f"{asin}|{','.join(opts) or 'none'}"
        _, _, _, _, pattern = items[key]
        mapped["constraint-pattern"].append(f"purchase:pattern:{pattern}|options:none")
    exact = len({x for x in current if x.startswith("purchase:")})
    coverage = sum(items[key][0] != "unknown" and items[key][1] is not None for key in keys)
    audit = {"purchase_leaves_exact": exact, "products": len(keys), "products_with_title_price": coverage}
    return mapped, audit


def make_data(rows: list[dict[str, Any]], bucket_rows: list[list[str]], draws: int) -> CompactRows:
    size = len(rows)
    targets = np.empty(size)
    count_mean = np.full((size, 17), np.nan)
    count_var = np.full((size, 17), np.nan)
    features = np.full((size, 17, 10), np.nan)
    for i, (row, buckets) in enumerate(zip(rows, bucket_rows, strict=True)):
        targets[i] = target(buckets)
        for prefix in range(1, 17):
            count_mean[i, prefix], count_var[i, prefix] = posterior_moments(
                buckets[:prefix], str(row["sample_id"]), prefix, draws
            )
            features[i, prefix] = np.asarray(row["prefixes"][str(prefix)]["features"], dtype=float)
    return CompactRows(
        ids=[str(row["sample_id"]) for row in rows], labels=np.zeros(size, dtype=int),
        folds=np.asarray([int(row["fold"]) for row in rows]), targets=targets,
        count_means=count_mean, count_variances=count_var, features=features,
    )


def write_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> None:
    cell: Cell = parse_cell(args.cell)
    if cell.dataset != "webshop":
        raise ValueError("this study supports WebShop cells only")
    checkpoint = checkpoint_path(args.evaluation_root.resolve(), cell)
    raw_path = z16_path(args.z16_root.resolve(), cell.dataset, cell.backbone, cell.seed)
    output = args.output_root.resolve() / cell.name
    manifest_path = output / "manifest.json"
    if args.resume and (output / "COMPLETE").is_file() and manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old.get("contract") == CONTRACT and old.get("checkpoint_sha256") == sha256(checkpoint) and old.get("raw_sha256") == sha256(raw_path):
            print(json.dumps({"cell": cell.name, "status": "reused"}), flush=True)
            return

    rows = read_jsonl(checkpoint)
    raw = read_jsonl(raw_path)
    if len(rows) != EXPECTED["webshop"] or len(raw) != len(rows):
        raise RuntimeError(f"{cell.name}: source cardinality mismatch")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    bucket_by_variant: dict[str, list[list[str]]] = {name: [] for name in VARIANTS}
    original_buckets: list[list[str]] = []
    task_audits: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["sample_id"])
        current = list(map(str, row["buckets"]))
        original_buckets.append(current)
        mapped, audit = map_task(raw_by_id[sid], current)
        for name in VARIANTS:
            bucket_by_variant[name].append(mapped[name])
        task_audits.append({"sample_id": sid, **audit})

    score_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for variant in VARIANTS:
        data = make_data(rows, bucket_by_variant[variant], args.posterior_draws)
        combined = Posterior(*(np.full((len(rows), 17), np.nan) for _ in range(3)))
        allocations = {budget: np.empty(len(rows), dtype=int) for budget in BUDGETS}
        for fold in range(FOLDS):
            train = np.flatnonzero(data.folds != fold)
            test = np.flatnonzero(data.folds == fold)
            fitted = fit_split_posterior(
                data,
                train,
                use_trajectory_likelihood=True,
                feature_indices=FEATURE_SETS["full"],
            )
            combined.mean[test] = fitted.mean[test]
            combined.variance[test] = fitted.variance[test]
            combined.precision[test] = fitted.precision[test]
            for budget in BUDGETS:
                allocations[budget][test], _ = variance_allocate(fitted, train, test, budget, lower=max(2, budget - 4))

        for budget in BUDGETS:
            for method, stop_kind in STOP.items():
                stops = np.full(len(rows), budget, dtype=int) if stop_kind == "fixed" else allocations[budget]
                selected_values = score(combined, stops, kind="lcb95")
                for index in range(len(rows)):
                        score_rows.append({
                            "cell": cell.name, "backbone": cell.backbone, "seed": cell.seed,
                            "sample_id": data.ids[int(index)], "bucket_variant": variant,
                            "method": method, "split": "crossfit", "budget": budget,
                            "score": float(selected_values[index]), "trajectories_used": int(stops[index]),
                        })

        error = combined.mean[:, 16] - data.targets
        standardizer = max(float(np.var(data.targets)), 1e-12)
        covered = np.abs(error) <= 1.959963984540054 * np.sqrt(combined.variance[:, 16])
        instability = np.mean(np.abs(np.diff(combined.mean[:, 2:17] - 1.959963984540054 * np.sqrt(combined.variance[:, 2:17]), axis=1)))
        exact_leaves = sum(len({x for x in values if x.startswith("purchase:")}) for values in original_buckets)
        mapped_leaves = sum(len({x for x in values if x.startswith("purchase:")}) for values in bucket_by_variant[variant])
        total_products = sum(item["products"] for item in task_audits)
        covered_products = sum(item["products_with_title_price"] for item in task_audits)
        compression = 0.0 if exact_leaves == 0 else 1.0 - mapped_leaves / exact_leaves
        audits.append({
            "cell": cell.name, "backbone": cell.backbone, "seed": cell.seed, "bucket_variant": variant,
            "tasks": len(rows),
            "metadata_coverage": covered_products / max(1, total_products),
            "target_std": float(np.std(target_dev)), "z16_mse": float(np.mean(error ** 2)),
            "standardized_z16_mse": float(np.mean(error ** 2) / standardizer),
            "posterior_coverage95": float(np.mean(covered)), "prefix_score_instability": float(instability),
            "purchase_leaf_compression": float(compression), "purchase_leaves_exact": exact_leaves,
            "purchase_leaves_mapped": mapped_leaves,
        })
        print(json.dumps({"cell": cell.name, "bucket_variant": variant}), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "label_free_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader(); writer.writerows(audits)
    atomic_text(output / "label_free_audit.json", json.dumps(audits, indent=2, sort_keys=True) + "\n")
    write_scores(output / "task_scores_without_labels.jsonl.gz", score_rows)
    manifest = {
        "contract": CONTRACT, "cell": cell.name, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "raw": str(raw_path), "raw_sha256": sha256(raw_path),
        "mapping": "constraint-pattern", "variants": list(VARIANTS),
        "tasks": len(rows), "score_rows": len(score_rows), "posterior_draws": args.posterior_draws,
        "correctness_labels_opened": False, "new_generation_calls": 0, "gpu_calls": 0,
    }
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_text(output / "COMPLETE", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/evaluations"))
    parser.add_argument("--z16-root", type=Path, default=Path("artifacts/raw/z16"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/analysis/brlg_webshop_task_equivalent_buckets_run/cells")
    parser.add_argument("--posterior-draws", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
