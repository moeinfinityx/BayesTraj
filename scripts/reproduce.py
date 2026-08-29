#!/usr/bin/env python3
"""Single public entry point for every BayesTraj reproduction workflow.

Commands
--------
``paper`` (the default)
    Rebuild all nine submission figures and numerical claims from the bundled
    frozen tables, then run the integrity/claim audit.
``raw``
    Orchestrate the from-scratch trajectory-generation and analysis pipeline.
    Run ``python scripts/reproduce.py raw --help`` for its stages.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayestraj.figures import generate_all  # noqa: E402
from bayestraj.paper import compute_claims, results_markdown  # noqa: E402
from bayestraj.pipeline import main as raw_pipeline_main  # noqa: E402


def reproduce_paper() -> None:
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    figures = generate_all(output)
    claims = compute_claims()
    (output / "claims.json").write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n")
    (output / "results.md").write_text(results_markdown(claims))
    print(
        json.dumps(
            {"figures": len(figures), "output": str(output), "claims": str(output / "claims.json")},
            indent=2,
        ),
        flush=True,
    )
    subprocess.run([sys.executable, str(ROOT / "scripts/verify.py")], cwd=ROOT, check=True)


def usage() -> str:
    return """usage: python scripts/reproduce.py [paper | raw <stage> [options]]

paper          rebuild Figures 1--9 and verify all frozen claims (default)
raw doctor     validate the from-scratch environment
raw generate   generate Z=16 and Z=12/N=4 pools for one served backbone
raw analyze    audit pools and rebuild all methods, baselines, and reports

Use `python scripts/reproduce.py raw --help` for full raw-pipeline options.
"""


def main() -> None:
    arguments = sys.argv[1:]
    if not arguments or arguments == ["paper"]:
        reproduce_paper()
        return
    if arguments[0] in {"-h", "--help", "help"}:
        print(usage())
        return
    if arguments[0] == "raw":
        raw_pipeline_main(arguments[1:])
        return
    raise SystemExit(f"Unknown command: {arguments[0]!r}\n\n{usage()}")


if __name__ == "__main__":
    main()
