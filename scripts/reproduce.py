#!/usr/bin/env python3
"""One-command reproduction of every empirical submission figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayestraj.figures import generate_all  # noqa: E402
from bayestraj.paper import compute_claims, results_markdown  # noqa: E402


def main() -> None:
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    figures = generate_all(output)
    claims = compute_claims()
    (output / "claims.json").write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n")
    (output / "results.md").write_text(results_markdown(claims))
    print(json.dumps({"figures": len(figures), "output": str(output), "claims": str(output / "claims.json")}, indent=2))


if __name__ == "__main__":
    main()

