#!/usr/bin/env python3
"""Audit frozen inputs and compare recomputed claims with the manuscript."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayestraj.paper import CONFIG, compute_claims, seed_metrics  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    failures: list[str] = []
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"hash mismatch: {relative}")

    data = seed_metrics()
    if set(data.dataset) != set(CONFIG["datasets"]): failures.append("dataset registry mismatch")
    if set(data.backbone) != set(CONFIG["backbones"]): failures.append("backbone registry mismatch")
    if set(data.seed) != set(CONFIG["seeds"]): failures.append("seed registry mismatch")
    if data[["auroc", "aupr", "mean_trajectories"]].isna().any().any(): failures.append("missing core metrics")

    expected = json.loads((ROOT / "expected" / "claims.json").read_text())
    actual = compute_claims()
    for key, target in expected.items():
        observed = actual.get(key)
        if observed is None or not np.isclose(observed, target, atol=5e-4, rtol=0):
            failures.append(f"claim mismatch {key}: expected {target}, observed {observed}")

    size = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts)
    if size >= 20_000_000: failures.append(f"repository is {size} bytes (must be <20,000,000)")
    forbidden_suffixes = {".safetensors", ".pt", ".pth", ".ckpt", ".arrow", ".parquet"}
    forbidden = [
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden: failures.append("dataset/model artifacts present: " + ", ".join(forbidden))
    raw_non_python = [
        str(path.relative_to(ROOT)) for path in (ROOT / "raw_pipeline").rglob("*")
        if path.is_file() and path.suffix != ".py"
    ]
    if raw_non_python: failures.append("raw_pipeline contains non-Python artifacts: " + ", ".join(raw_non_python))
    if failures:
        raise SystemExit("AUDIT FAILED\n" + "\n".join(f"- {item}" for item in failures))
    print(json.dumps({
        "status": "AUDIT PASSED",
        "files_checked": len(manifest.read_text().splitlines()),
        "python_files": len(list(ROOT.rglob("*.py"))),
        "claims_checked": len(expected),
        "repository_bytes": size,
        "datasets_or_model_weights_included": False,
    }, indent=2))


if __name__ == "__main__":
    main()
