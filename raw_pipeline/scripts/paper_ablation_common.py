"""Small shared helpers for the paper ablation and sensitivity reports."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from bayestraj_common import BACKBONES, EXPECTED, SEEDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_cells() -> list[str]:
    return [f"{dataset}-{backbone}-seed{seed}" for dataset in EXPECTED for backbone in BACKBONES for seed in SEEDS]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def holm_adjust(rows: list[dict[str, Any]], metric: str) -> None:
    p_field = f"{metric}_p_two_sided"
    adjusted_field = f"{metric}_p_holm"
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index][p_field]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(rows[index][p_field]))
        running = max(running, adjusted)
        rows[index][adjusted_field] = running
