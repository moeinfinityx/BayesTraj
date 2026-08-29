#!/usr/bin/env python3
"""Small orchestration layer for the full BayesTraj reproduction pipeline.

This driver does not reimplement any estimator.  It calls the frozen raw
generation and evaluation programs in ``raw_pipeline/`` with the submitted
datasets, seeds, budgets, and filenames.  Use ``--dry-run`` to inspect every
underlying command without executing it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
DATASETS = ("dbbench", "hotpotqa", "webshop", "strategyqa")
BACKBONES = ("qwen35", "gemma3", "gptoss20b")
SEEDS = (101, 202, 303)


def run(command: Sequence[object], *, dry_run: bool = False) -> None:
    resolved = [str(value) for value in command]
    print("+ " + shlex.join(resolved), flush=True)
    if not dry_run:
        subprocess.run(resolved, cwd=ROOT, check=True)


def paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "raw": args.raw_root.expanduser().resolve(),
        "n4": args.n4_root.expanduser().resolve(),
        "evaluation": args.evaluation_root.expanduser().resolve(),
        "work": args.work_root.expanduser().resolve(),
    }


def endpoint_model(port: int) -> list[str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
        payload = json.load(response)
    return [str(item.get("id", "")) for item in payload.get("data", [])]


def doctor(args: argparse.Namespace) -> None:
    resolved = paths(args)
    failures: list[str] = []
    if sys.version_info < (3, 13) and not getattr(args, "allow_older", False):
        failures.append(f"Track B requires Python 3.13; found {sys.version.split()[0]}")
    for name, path in resolved.items():
        if name == "raw" or args.require_outputs:
            path.mkdir(parents=True, exist_ok=True)
        print(f"{name:10s} {path}")
    if args.model:
        for port in args.ports:
            try:
                models = endpoint_model(port)
                if args.model not in models:
                    failures.append(f"port {port} serves {models}, expected {args.model!r}")
            except Exception as exc:  # diagnostic boundary
                failures.append(f"port {port} is unavailable: {exc}")
    configuration = ROOT / "config/bayestraj_raw_generation_configuration.json"
    manifest = ROOT / "config/bayestraj_raw_generation_manifest.json"
    if not configuration.is_file() or not manifest.is_file():
        failures.append("frozen generation configuration is missing")
    if failures:
        raise SystemExit("DOCTOR FAILED\n" + "\n".join(f"- {item}" for item in failures))
    print("DOCTOR PASSED")


def raw_launcher(
    *, dataset: str, seed: int, backbone: str, model: str, ports: str,
    root: Path, z: int, n: int, dry_run: bool,
) -> None:
    suffix = f"z{z}" + ("_n4" if n == 4 else "")
    run_id = f"bayestraj_{dataset}_seed{seed}_{suffix}_{backbone}"
    command: list[object] = [
        PYTHON, "raw_pipeline/scripts/generate_raw_trajectory_pools.py",
        "--run-id", run_id, "--model", model, "--backbone", backbone,
        "--ports", ports, "--datasets", dataset, "--seeds", seed,
        "--trajectory-samples", z, "--per-step-samples", n,
        "--output-root", root, "--log-root", root / "logs",
        "--state", root / f"state_{backbone}.json",
    ]
    if dataset == "dbbench":
        command += ["--dbbench-tasks", 300]
    elif dataset == "hotpotqa":
        command += ["--hotpot-tasks", 1000, "--hotpot-chunk", 5]
    elif dataset == "strategyqa":
        command += ["--strategyqa-tasks", 687, "--strategyqa-chunk", 5]
    run(command, dry_run=dry_run)


def webshop(
    *, seed: int, backbone: str, model: str, ports: Sequence[int], root: Path,
    z: int, n: int, controller_url: str, max_tokens: int, dry_run: bool,
) -> None:
    cell = root / f"webshop_{backbone}"
    parts = cell / "parts"
    sampling = cell / "sampling"
    if not dry_run:
        parts.mkdir(parents=True, exist_ok=True)
        sampling.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[bytes], object]] = []
    part_paths: list[Path] = []
    shard_size = 200 // len(ports)
    if shard_size * len(ports) != 200:
        raise ValueError("the number of ports must divide the 200 WebShop tasks")
    for shard, port in enumerate(ports):
        offset = shard * shard_size
        part = parts / f"offset{offset}_limit{shard_size}.jsonl"
        log = parts / f"offset{offset}.log"
        part_paths.append(part)
        command: list[object] = [
            PYTHON, "raw_pipeline/main.py", "run-agentbench-webshop",
            "--provider", "vllm", "--model", model,
            "--base-url", f"http://127.0.0.1:{port}/v1", "--api-key", "vllm",
            "--method", "uprop", "--controller-url", controller_url,
            "--task-name", "webshop-std", "--offset", offset, "--limit", shard_size,
            "--tdp-samples", z, "--per-step-samples", n, "--backbone-samples", n,
            "--next-step-samples", n, "--temperature", 1.0, "--seed", seed,
            "--max-steps", 20, "--max-tokens", max_tokens, "--parallel-requests", 1,
            "--no-fair-trajectory-budget", "--emulate-tool-calls", "--disable-tracking",
            "--sampling-dir", sampling, "--restart", "--output", part,
        ]
        print("+ " + shlex.join(str(value) for value in command) + f" > {shlex.quote(str(log))} 2>&1", flush=True)
        if not dry_run:
            handle = log.open("wb")
            processes.append((subprocess.Popen([str(value) for value in command], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT), handle))
    for process, handle in processes:
        returncode = process.wait()
        handle.close()
        if returncode:
            raise subprocess.CalledProcessError(returncode, process.args)
    suffix = f"z{z}" + ("_n4" if n == 4 else "")
    merged = cell / f"bayestraj_webshop_seed{seed}_{suffix}_{backbone}_seed{seed}_pe.jsonl"
    run([PYTHON, "scripts/merge_jsonl_parts.py", "--expected-rows", 200,
         "--output", merged, *part_paths], dry_run=dry_run)


def generate(args: argparse.Namespace) -> None:
    resolved = paths(args)
    port_text = ",".join(str(port) for port in args.ports)
    doctor(argparse.Namespace(**{
        **vars(args),
        "require_outputs": not args.dry_run,
        "allow_older": args.dry_run,
        "model": None if args.dry_run else args.model,
        "ports": [] if args.dry_run else args.ports,
    }))
    for seed in SEEDS:
        if not args.skip_z16:
            for dataset in ("dbbench", "hotpotqa", "strategyqa"):
                # StrategyQA's historical directory includes the _n4 suffix.
                suffix = "z16_n4" if dataset == "strategyqa" else "z16"
                root = resolved["raw"] / f"{dataset}_seed{seed}_{suffix}"
                raw_launcher(dataset=dataset, seed=seed, backbone=args.backbone,
                             model=args.model, ports=port_text, root=root,
                             z=16, n=4 if dataset == "strategyqa" else 1,
                             dry_run=args.dry_run)
            webshop(seed=seed, backbone=args.backbone, model=args.model,
                    ports=args.ports, root=resolved["raw"] / f"webshop_seed{seed}_z16",
                    z=16, n=1, controller_url=args.controller_url,
                    max_tokens=args.max_tokens, dry_run=args.dry_run)
        if not args.skip_n4:
            for dataset in ("dbbench", "hotpotqa", "strategyqa"):
                root = resolved["n4"] / f"{dataset}_seed{seed}_z12_n4"
                raw_launcher(dataset=dataset, seed=seed, backbone=args.backbone,
                             model=args.model, ports=port_text, root=root,
                             z=12, n=4, dry_run=args.dry_run)
            webshop(seed=seed, backbone=args.backbone, model=args.model,
                    ports=args.ports, root=resolved["n4"] / f"webshop_seed{seed}_z12_n4",
                    z=12, n=4, controller_url=args.controller_url,
                    max_tokens=args.max_tokens, dry_run=args.dry_run)


def run_cells(script: str, option: str, root: Path, evaluation: Path, *,
              datasets: Sequence[str] = DATASETS, workers: int, draws: int,
              extra: Sequence[object] = (), dry_run: bool) -> None:
    commands = []
    for dataset in datasets:
        for backbone in BACKBONES:
            for seed in SEEDS:
                cell = f"{dataset}-{backbone}-seed{seed}"
                commands.append([PYTHON, script, "--cell", cell,
                                 "--evaluation-root", evaluation, option, root,
                                 "--posterior-draws", draws, "--resume", *extra])
    if dry_run or workers == 1:
        for command in commands:
            run(command, dry_run=dry_run)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda command: run(command), commands))


def analyze(args: argparse.Namespace) -> None:
    resolved = paths(args)
    raw, n4, evaluation, work = (resolved[key] for key in ("raw", "n4", "evaluation", "work"))
    ablation_run = work / "brlg_fixed_varstop80_ablation_run"
    ablation_report = work / "brlg_fixed_varstop80_ablation"
    sensitivity_run = work / "brlg_varstop_sensitivity_run"
    sensitivity_report = work / "brlg_varstop_sensitivity"
    webshop_run = work / "brlg_webshop_task_equivalent_buckets_run"
    baseline = work / "mixed_budget_sources"
    complete = work / "mixed_budget_complete/report"
    selected = work / "mixed_budget_4datasets_selected_methods/report"
    empirical = work / "bayestraj_empirical/report"
    compute_cost = work / "bayestraj_compute_cost"
    run([PYTHON, "scripts/audit_raw_generation.py", "--pool-root", raw,
         "--output", raw / "raw_generation_audit.json"], dry_run=args.dry_run)
    run([PYTHON, "scripts/build_z12_n4_manifest.py", "--campaign-root", n4], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/materialize_bayestraj_checkpoints.py",
         "--pool-root", raw, "--output-dir", evaluation, "--datasets", *DATASETS,
         "--backbones", *BACKBONES, "--seeds", *SEEDS,
         "--posterior-draws", args.posterior_draws], dry_run=args.dry_run)
    run_cells("raw_pipeline/scripts/evaluate_brlg_fixed_varstop80_ablation.py",
              "--output-root", ablation_run / "cells", evaluation,
              workers=args.workers, draws=args.posterior_draws, dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/aggregate_brlg_fixed_varstop80_ablation.py",
         "--run-root", ablation_run, "--output-root", ablation_report,
         "--task-bootstrap-replicates", 500, "--hierarchical-bootstrap-replicates", 10000,
         "--workers", args.workers], dry_run=args.dry_run)
    run_cells("raw_pipeline/scripts/evaluate_brlg_varstop_sensitivity.py",
              "--output-root", sensitivity_run / "cells", evaluation,
              workers=args.workers, draws=args.posterior_draws, dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/aggregate_brlg_varstop_sensitivity.py",
         "--run-root", sensitivity_run, "--output-root", sensitivity_report,
         "--hierarchical-replicates", 10000], dry_run=args.dry_run)
    run_cells("raw_pipeline/scripts/evaluate_bayestraj_webshop_constraint_pattern.py",
              "--output-root", webshop_run / "cells", evaluation,
              datasets=("webshop",), workers=args.workers, draws=args.posterior_draws,
              extra=("--z16-root", raw), dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/evaluate_mixed_budget_core_multiseed.py",
         "--z16-root", raw, "--strict-root", n4, "--output-dir", baseline / "core"], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/evaluate_z16_semantic_baselines_multiseed.py",
         "--pool-root", raw, "--output-dir", baseline / "semantic",
         "--device", args.device, "--batch-size", 48], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/evaluate_z16_saup_multiseed.py",
         "--pool-root", raw, "--output-dir", baseline / "saup",
         "--device", args.device, "--batch-size", 128], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/evaluate_bse_multiseed.py",
         "--pool-root", raw, "--output-dir", baseline / "bse",
         "--monte-carlo-samples", 1000, "--integration-replicates", 5], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/build_paper_curve_table.py",
         "--ablation-root", ablation_run, "--core-root", baseline / "core",
         "--semantic-root", baseline / "semantic", "--saup-root", baseline / "saup",
         "--bse-root", baseline / "bse", "--output", complete], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/generate_selected_methods_four_dataset_report.py",
         "--source", complete, "--webshop-root", webshop_run / "cells",
         "--output", selected], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/generate_bayestraj_empirical_minimal_package.py",
         "--selected-report", selected, "--ablation-run-root", ablation_run,
         "--raw-root", raw, "--ablation-scores", ablation_report / "task_scores.jsonl.gz",
         "--webshop-root", webshop_run, "--output", empirical], dry_run=args.dry_run)
    run([PYTHON, "raw_pipeline/scripts/generate_bayestraj_compute_cost_candidates.py",
         "--metrics", selected / "selected_seed_metrics.csv",
         "--efficiency", empirical / "efficiency_by_budget.csv",
         "--output", compute_cost, "--budget", 8], dry_run=args.dry_run)


def add_roots(parser: argparse.ArgumentParser) -> None:
    for option, variable in (
        ("raw-root", "BAYESTRAJ_RAW_ROOT"),
        ("n4-root", "BAYESTRAJ_N4_ROOT"),
        ("evaluation-root", "BAYESTRAJ_EVAL_ROOT"),
        ("work-root", "BAYESTRAJ_WORK"),
    ):
        default = os.environ.get(variable)
        parser.add_argument(
            f"--{option}",
            type=Path,
            default=Path(default) if default else None,
            required=default is None,
            help=f"Defaults to ${variable} when set.",
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("doctor", help="validate paths and optional model endpoints")
    add_roots(check)
    check.add_argument("--ports", type=lambda value: [int(item) for item in value.split(",")], default=[])
    check.add_argument("--model")
    check.add_argument("--require-outputs", action="store_true")
    check.set_defaults(function=doctor)
    generation = subparsers.add_parser("generate", help="generate Z=16 and Z=12/N=4 pools for one served backbone")
    add_roots(generation)
    generation.add_argument("--backbone", choices=BACKBONES, required=True)
    generation.add_argument("--model", required=True)
    generation.add_argument("--ports", type=lambda value: [int(item) for item in value.split(",")], default=list(range(8000, 8008)))
    generation.add_argument("--controller-url", default=os.environ.get("AGENTBENCH_CONTROLLER_URL", "http://127.0.0.1:5000/api"))
    generation.add_argument("--max-tokens", type=int, default=2048)
    generation.add_argument("--skip-z16", action="store_true")
    generation.add_argument("--skip-n4", action="store_true")
    generation.add_argument("--dry-run", action="store_true")
    generation.set_defaults(function=generate, require_outputs=True)
    analysis = subparsers.add_parser("analyze", help="audit pools and rebuild all submitted methods, baselines, and reports")
    add_roots(analysis)
    analysis.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    analysis.add_argument("--posterior-draws", type=int, default=2048)
    analysis.add_argument("--device", default="cuda:0")
    analysis.add_argument("--dry-run", action="store_true")
    analysis.set_defaults(function=analyze)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
