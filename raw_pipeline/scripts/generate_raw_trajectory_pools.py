#!/usr/bin/env python3
"""Generate exact-budget paper trajectory pools with a resumable queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ltuq.datasets import load_hotpotqa_split, load_strategyqa_split
from src.ltuq.experiments.sampling_efficiency import pool_coverage, tdps_from_record


DEFAULT_MANIFEST = ROOT.parent / "config/bayestraj_raw_generation_manifest.json"
DEFAULT_CONFIGURATION = (
    ROOT.parent / "config/bayestraj_raw_generation_configuration.json"
)
DEFAULT_OUTPUT = ROOT.parent / "raw_work/pools"
DEFAULT_STATE = ROOT.parent / "raw_work/generation_state.json"
CANARY_DIR = ROOT / "outputs/paper_aaai27/canary"
CANARY_RUN_ID = "bayestraj_raw_generation_canary"
DATASET_DEFAULTS = {
    "dbbench": {
        "tasks": 300,
        "chunk": 1,
        "max_steps": 64,
        "max_tokens": 2048,
    },
    "strategyqa": {
        "tasks": 687,
        "chunk": 5,
        "max_steps": 6,
        "max_tokens": 512,
    },
    "hotpotqa": {
        "tasks": 1000,
        "chunk": 5,
        "max_steps": 6,
        "max_tokens": 512,
    },
}
DATASET_REVISIONS = {
    "strategyqa": "705562638fe1d8ca6bb98c66fc8f94d45fda8c83",
    "hotpotqa": "1908d6afbbead072334abe2965f91bd2709910ab",
}


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int
    offset: int
    limit: int

    @property
    def key(self) -> str:
        return f"{self.dataset}:seed{self.seed}:offset{self.offset}:limit{self.limit}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_ids_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def canonical_rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path.

    Long-running trajectory pools may live on a larger external filesystem such
    as ``/datapool``.  Persisting their state must not assume that every output
    path is below the source checkout.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def validate_frozen_sample_ids(
    dataset: str,
    sample_ids: list[str],
    configuration: dict[str, Any],
) -> None:
    source = configuration.get("dataset_sources", {}).get(dataset)
    if not isinstance(source, dict):
        raise RuntimeError(
            f"Frozen dataset source is missing for {dataset}."
        )
    expected_count = int(source.get("selected_count", -1))
    expected_digest = str(source.get("selected_ids_sha256", ""))
    actual_digest = ordered_ids_sha256(sample_ids)
    if len(sample_ids) != expected_count:
        raise RuntimeError(
            f"{dataset} exposes {len(sample_ids)} samples, expected "
            f"frozen count {expected_count}."
        )
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"{dataset} ordered sample IDs do not match the frozen source: "
            f"{actual_digest} != {expected_digest}."
        )


def validate_frozen_sample_rows(
    dataset: str,
    rows: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> None:
    source = configuration.get("dataset_sources", {}).get(dataset)
    if not isinstance(source, dict):
        raise RuntimeError(
            f"Frozen dataset source is missing for {dataset}."
        )
    expected_digest = str(source.get("selected_rows_sha256", ""))
    actual_digest = canonical_rows_sha256(rows)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"{dataset} selected rows do not match the frozen source: "
            f"{actual_digest} != {expected_digest}."
        )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def upsert_state_job(
    state: dict[str, Any],
    job: dict[str, Any],
) -> list[dict[str, Any]]:
    jobs = [
        value
        for value in state.get("jobs", [])
        if isinstance(value, dict) and value.get("run_id") != job["run_id"]
    ]
    jobs.append(job)
    return sorted(jobs, key=lambda value: str(value.get("run_id", "")))


def server_ready(port: int, model: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=3,
        ) as handle:
            values = json.load(handle).get("data", [])
        model_ids = [str(value.get("id", "")).lower() for value in values]
        return model.lower() in model_ids
    except Exception:
        return False


def agentbench_stack_task_count(task_name: str) -> int:
    with urllib.request.urlopen(
        f"http://localhost:5020/api/get_indices?name={task_name}",
        timeout=10,
    ) as handle:
        value = json.load(handle)
    indices = value.get("indices", value) if isinstance(value, dict) else value
    return len(indices) if isinstance(indices, (list, dict)) else 0


def part_path(
    output_root: Path,
    run_id: str,
    job: Job,
    *,
    backbone: str = "gemma3",
) -> Path:
    return (
        output_root
        / f"{job.dataset}_{backbone}"
        / f"{run_id}_parts"
        / f"seed{job.seed}"
        / f"offset{job.offset}_limit{job.limit}.jsonl"
    )


def log_path(log_root: Path, run_id: str, job: Job, attempt: int, port: int) -> Path:
    return (
        log_root
        / run_id
        / job.dataset
        / f"seed{job.seed}_offset{job.offset}_limit{job.limit}_attempt{attempt}_port{port}.log"
    )


def frozen_samples(
    dataset: str,
    hotpot_sample_ids_file: Path | None = None,
) -> list[Any] | None:
    if dataset == "strategyqa":
        return load_strategyqa_split(
            split="test",
            dataset_name="ChilleD/StrategyQA",
            revision=DATASET_REVISIONS["strategyqa"],
        )
    if dataset == "hotpotqa":
        samples = load_hotpotqa_split(
            split="validation",
            dataset_name="hotpotqa/hotpot_qa",
            config_name="distractor",
            revision=DATASET_REVISIONS["hotpotqa"],
        )
        if hotpot_sample_ids_file is not None:
            requested = [
                line.strip()
                for line in hotpot_sample_ids_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if len(requested) != len(set(requested)):
                raise RuntimeError(
                    f"Duplicate IDs in {hotpot_sample_ids_file}"
                )
            by_identifier = {str(sample.sample_id): sample for sample in samples}
            missing = [identifier for identifier in requested if identifier not in by_identifier]
            if missing:
                raise RuntimeError(
                    f"{hotpot_sample_ids_file} contains {len(missing)} "
                    "IDs absent from the pinned HotpotQA revision"
                )
            return [by_identifier[identifier] for identifier in requested]
        return samples[: int(DATASET_DEFAULTS["hotpotqa"]["tasks"])]
    return None


def frozen_sample_records(
    dataset: str,
) -> list[tuple[str, str, Any]] | None:
    samples = frozen_samples(dataset)
    if samples is None:
        return None
    return [
        (sample.sample_id, sample.question, sample.answer)
        for sample in samples
    ]


def frozen_sample_ids(dataset: str) -> list[str] | None:
    records = frozen_sample_records(dataset)
    return [sample_id for sample_id, _, _ in records] if records else None


def has_required_budget(tdps: list[Any], *, z: int, n: int) -> bool:
    """Accept an exact trajectory count and a per-step candidate superset.

    Older reusable caches can contain more step candidates than a follow-up
    experiment requires.  Those candidates are useful preserved intermediates;
    rejecting them would force needless regeneration.  Downstream variants can
    deterministically select the first ``n`` candidates from the ordered pool.
    """
    if len(tdps) != z:
        return False
    for tdp in tdps:
        for step in tdp.steps:
            if step.metadata.get("forced_terminal_finish"):
                continue
            candidate_count = len(step.sampled_decisions)
            if candidate_count < n:
                return False
            chosen = step.metadata.get("chosen_output_index")
            if not isinstance(chosen, int) or not 0 <= chosen < candidate_count:
                return False
    return True


def has_exact_budget(tdps: list[Any], *, z: int, n: int) -> bool:
    """Compatibility helper for callers that require an exact candidate pool.

    Resumable production validation intentionally uses ``has_required_budget``
    so an ordered candidate superset can be reused.  This stricter predicate is
    retained for diagnostics and historical tests of exact-budget behavior.
    """
    if len(tdps) != z:
        return False
    for tdp in tdps:
        for step in tdp.steps:
            if step.metadata.get("forced_terminal_finish"):
                continue
            candidate_count = len(step.sampled_decisions)
            chosen = step.metadata.get("chosen_output_index")
            if candidate_count != n:
                return False
            if not isinstance(chosen, int) or not 0 <= chosen < candidate_count:
                return False
    return True


def valid_part(
    path: Path,
    job: Job,
    *,
    z: int,
    n: int,
    expected_sample_ids: list[str] | None = None,
) -> bool:
    try:
        rows = read_jsonl(path)
        if len(rows) != job.limit:
            return False
        identifiers: list[Any] = []
        for row in rows:
            # ``status`` is the realized task/environment outcome, not the
            # completion state of this generation job.  Incomplete outcomes
            # are retained and labeled as failures by the shared-main-label
            # evaluator; exact pool coverage below establishes job completion.
            tdps = tdps_from_record(row)
            coverage = pool_coverage(tdps, required_z=z, required_n=n)
            if not coverage["trajectory_complete"] or not coverage["candidate_complete"]:
                return False
            if not has_required_budget(tdps, z=z, n=n):
                return False
            if job.dataset == "dbbench":
                identifiers.append(row.get("task_index"))
            else:
                identifiers.append(str(row.get("sample_id", "")))
        if job.dataset == "dbbench":
            return identifiers == list(range(job.offset, job.offset + job.limit))
        if expected_sample_ids is None:
            return len(identifiers) == len(set(identifiers)) and all(identifiers)
        return identifiers == expected_sample_ids
    except Exception:
        return False


def command_for(
    job: Job,
    *,
    port: int,
    output: Path,
    log_file: Path,
    sampling_root: Path,
    model: str,
    z: int,
    n: int,
    max_steps: int,
    max_tokens: int,
    parallel_requests: int,
    relaxed_dbbench_replay: bool = False,
    hotpot_sample_ids_file: Path | None = None,
) -> list[str]:
    common = [
        sys.executable,
        str(ROOT / "main.py"),
    ]
    provider = [
        "--method",
        "uprop",
        "--model",
        model,
        "--provider",
        "vllm",
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--api-key",
        os.environ.get("VLLM_API_KEY", "vllm"),
        "--tdp-samples",
        str(z),
        "--per-step-samples",
        str(n),
        "--backbone-samples",
        str(n),
        "--next-step-samples",
        str(n),
        "--temperature",
        "1.0",
        "--seed",
        str(job.seed),
        "--parallel-requests",
        str(parallel_requests),
        "--no-fair-trajectory-budget",
        "--sampling-dir",
        str(sampling_root / job.dataset / f"seed{job.seed}"),
        "--output",
        str(output),
        "--log-file",
        str(log_file),
        "--disable-tracking",
        "--restart",
    ]
    if job.dataset == "dbbench":
        return [
            *common,
            "run-agentbench-dbbench",
            *provider,
            "--task-name",
            "dbbench-std",
            "--task-index",
            str(job.offset),
            "--max-steps",
            str(max_steps),
            "--max-tokens",
            str(max_tokens),
            "--emulate-tool-calls",
            *(["--relaxed-replay"] if relaxed_dbbench_replay else []),
        ]
    if job.dataset == "strategyqa":
        return [
            *common,
            "run-strategyqa",
            *provider,
            "--split",
            "test",
            "--dataset-name",
            "ChilleD/StrategyQA",
            "--dataset-revision",
            DATASET_REVISIONS["strategyqa"],
            "--offset",
            str(job.offset),
            "--limit",
            str(job.limit),
            "--max-steps",
            str(max_steps),
            "--max-tokens",
            str(max_tokens),
        ]
    if job.dataset == "hotpotqa":
        command = [
            *common,
            "run-hotpotqa",
            *provider,
            "--split",
            "validation",
            "--dataset-name",
            "hotpotqa/hotpot_qa",
            "--dataset-config",
            "distractor",
            "--dataset-revision",
            DATASET_REVISIONS["hotpotqa"],
            "--offset",
            str(job.offset),
            "--limit",
            str(job.limit),
            "--max-steps",
            str(max_steps),
            "--max-tokens",
            str(max_tokens),
        ]
        if hotpot_sample_ids_file is not None:
            command.extend(
                ["--sample-ids-file", str(hotpot_sample_ids_file)]
            )
        return command
    raise ValueError(f"Unsupported dataset: {job.dataset}")


def build_jobs(
    seeds: list[int],
    *,
    datasets: list[str],
    task_counts: dict[str, int],
    chunk_sizes: dict[str, int],
    first_chunk_sizes: dict[str, int] | None = None,
) -> list[Job]:
    first_chunk_sizes = first_chunk_sizes or {}
    jobs = []
    for dataset in datasets:
        tasks = task_counts[dataset]
        chunk = chunk_sizes[dataset]
        for seed in seeds:
            offset = 0
            first_chunk = int(first_chunk_sizes.get(dataset, chunk))
            if first_chunk <= 0 or first_chunk > chunk:
                raise ValueError(
                    f"Invalid first chunk size for {dataset}: {first_chunk}"
                )
            while offset < tasks:
                limit = min(
                    first_chunk if offset == 0 else chunk,
                    tasks - offset,
                )
                jobs.append(Job(dataset, seed, offset, limit))
                offset += limit
    random.Random(2701).shuffle(jobs)
    return jobs


def merge_outputs(
    jobs: list[Job],
    output_root: Path,
    run_id: str,
    seeds: list[int],
    datasets: list[str],
    backbone: str,
) -> list[dict[str, Any]]:
    merged = []
    for dataset in datasets:
        for seed in seeds:
            selected = sorted(
                (
                    job
                    for job in jobs
                    if job.dataset == dataset and job.seed == seed
                ),
                key=lambda job: job.offset,
            )
            output = (
                output_root
                / f"{dataset}_{backbone}"
                / f"{run_id}_{dataset}_{backbone}_seed{seed}_uprop.jsonl"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for job in selected:
                rows.extend(
                    read_jsonl(
                        part_path(
                            output_root,
                            run_id,
                            job,
                            backbone=backbone,
                        )
                    )
                )
            temporary = output.with_suffix(output.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(output)
            merged.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "path": portable_path(output),
                    "records": len(rows),
                }
            )
    return merged


def reuse_canaries(
    output_root: Path,
    run_id: str,
    z: int,
    n: int,
    *,
    backbone: str,
    datasets: list[str],
) -> list[str]:
    if backbone != "gemma3":
        return []
    reused = []
    sources = {
        Job("hotpotqa", 101, 0, 1): (
            CANARY_DIR / f"{CANARY_RUN_ID}_hotpotqa_gemma3_seed101_uprop.jsonl"
        ),
    }
    for job, source in sources.items():
        if job.dataset not in datasets:
            continue
        destination = part_path(
            output_root,
            run_id,
            job,
            backbone=backbone,
        )
        if source.exists() and valid_part(source, job, z=z, n=n):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not valid_part(destination, job, z=z, n=n):
                shutil.copy2(source, destination)
            reused.append(job.key)
    return reused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--configuration",
        type=Path,
        default=DEFAULT_CONFIGURATION,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ports", default="8000,8001,8002,8003,8004,8005,8006,8007")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--backbone", default="gemma3")
    parser.add_argument(
        "--seeds",
        default="",
        help=(
            "Optional comma-separated subset of the frozen manifest seeds. "
            "The default runs every frozen seed."
        ),
    )
    parser.add_argument(
        "--datasets",
        default="dbbench,hotpotqa,strategyqa",
        help="Comma-separated frozen datasets to run.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-root", type=Path, default=ROOT / "outputs/run_logs")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--stage",
        choices=(
            "maximum_pool_generation",
            "sampling_budget_generation",
        ),
        default="maximum_pool_generation",
    )
    parser.add_argument("--dbbench-tasks", type=int, default=300)
    parser.add_argument("--strategyqa-tasks", type=int, default=687)
    parser.add_argument("--hotpot-tasks", type=int, default=1000)
    parser.add_argument(
        "--hotpot-sample-ids-file",
        type=Path,
        help=(
            "Optional ordered HotpotQA sample-ID file. Offsets and limits are "
            "applied within this explicit selection."
        ),
    )
    parser.add_argument(
        "--task-indices",
        default="",
        help=(
            "Optional comma-separated task indices for a single dataset. "
            "Only singleton jobs at these offsets are executed."
        ),
    )
    parser.add_argument("--strategyqa-chunk", type=int, default=5)
    parser.add_argument("--hotpot-chunk", type=int, default=5)
    parser.add_argument(
        "--singleton-prefix-datasets",
        default="",
        help=(
            "Comma-separated datasets whose existing run partition starts "
            "with offset0_limit1 before regular chunks."
        ),
    )
    parser.add_argument(
        "--max-interactive-workers",
        "--max-os-workers",
        dest="max_interactive_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--workers-per-port",
        type=int,
        default=1,
        help=(
            "Independent job workers sharing each model endpoint. Increasing "
            "this can overlap environment/tool latency without increasing "
            "the number of model replicas."
        ),
    )
    parser.add_argument("--parallel-requests", type=int, default=4)
    parser.add_argument(
        "--relaxed-dbbench-replay",
        action="store_true",
        help=(
            "During DBBench cache extension, retain the frozen cached observation "
            "context when a fresh database session replays an equivalent prefix "
            "with environment-dependent output drift."
        ),
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        help="Exact trajectory budget Z. Defaults to the manifest maximum.",
    )
    parser.add_argument(
        "--per-step-samples",
        type=int,
        help="Exact per-step candidate budget N. Defaults to the manifest maximum.",
    )
    args = parser.parse_args()
    if args.workers_per_port <= 0:
        parser.error("--workers-per-port must be positive")
    if args.max_interactive_workers <= 0:
        parser.error("--max-interactive-workers must be positive")

    manifest_path = args.manifest.resolve()
    configuration = read_json(args.configuration.resolve())
    manifest = read_json(manifest_path)
    if configuration.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError(
            "Frozen method configuration does not match the manifest."
        )
    manifest_seeds = [int(seed) for seed in manifest["primary_seeds"]]
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else manifest_seeds
    )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a non-empty unique list.")
    undeclared_seeds = sorted(set(seeds) - set(manifest_seeds))
    if undeclared_seeds:
        raise ValueError(
            "Requested seeds are outside the frozen manifest: "
            + ", ".join(map(str, undeclared_seeds))
        )
    sampling = manifest["sampling_efficiency"]
    z = (
        int(args.trajectory_samples)
        if args.trajectory_samples is not None
        else int(sampling["max_pool_z"])
    )
    n = (
        int(args.per_step_samples)
        if args.per_step_samples is not None
        else int(sampling["max_pool_n"])
    )
    if z <= 0 or n <= 0:
        raise ValueError("trajectory-samples and per-step-samples must be positive")
    datasets = [
        value.strip()
        for value in args.datasets.split(",")
        if value.strip()
    ]
    unsupported = sorted(set(datasets) - set(DATASET_DEFAULTS))
    if not datasets or unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")
    declared = {
        (str(cell["dataset"]), str(cell["backbone"]))
        for cell in manifest["evidence_cells"]
    }
    undeclared = [
        f"{dataset}/{args.backbone}"
        for dataset in datasets
        if (dataset, args.backbone) not in declared
    ]
    if undeclared:
        raise ValueError(
            "Requested cells are outside the frozen manifest: "
            + ", ".join(undeclared)
        )
    ports = [int(value) for value in args.ports.split(",") if value.strip()]
    if not ports or any(not server_ready(port, args.model) for port in ports):
        raise RuntimeError("Every configured port must serve the declared model.")
    if (
        "dbbench" in datasets
        and agentbench_stack_task_count("dbbench-std") != args.dbbench_tasks
    ):
        raise RuntimeError(
            "AgentBench DBBench stack does not expose the frozen 300 tasks."
        )

    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()
    sampling_root = output_root / "shared_sampling" / args.run_id
    status_path = output_root / f"{args.run_id}_job_status.json"
    task_counts = {
        "dbbench": args.dbbench_tasks,
        "strategyqa": args.strategyqa_tasks,
        "hotpotqa": args.hotpot_tasks,
    }
    task_indices = sorted(
        {
            int(value)
            for value in args.task_indices.split(",")
            if value.strip()
        }
    )
    if task_indices:
        if len(datasets) != 1:
            raise ValueError("--task-indices requires exactly one dataset")
        task_count = task_counts[datasets[0]]
        if any(index < 0 or index >= task_count for index in task_indices):
            raise ValueError(
                f"Task indices must be in [0, {task_count - 1}]"
            )
    hotpot_sample_ids_file = (
        args.hotpot_sample_ids_file.resolve()
        if args.hotpot_sample_ids_file is not None
        else None
    )
    if hotpot_sample_ids_file is not None and "hotpotqa" not in datasets:
        raise ValueError(
            "--hotpot-sample-ids-file requires the hotpotqa dataset"
        )
    dataset_samples = {
        dataset: frozen_samples(
            dataset,
            hotpot_sample_ids_file=(
                hotpot_sample_ids_file if dataset == "hotpotqa" else None
            ),
        )
        for dataset in datasets
    }
    dataset_sample_ids = {
        dataset: (
            [str(sample.sample_id) for sample in samples]
            if samples is not None
            else None
        )
        for dataset, samples in dataset_samples.items()
    }
    for dataset, samples in dataset_samples.items():
        if samples is None:
            continue
        sample_ids = dataset_sample_ids[dataset]
        assert sample_ids is not None
        if dataset != "hotpotqa" or hotpot_sample_ids_file is None:
            validate_frozen_sample_ids(dataset, sample_ids, configuration)
            validate_frozen_sample_rows(
                dataset,
                [dict(sample.metadata) for sample in samples],
                configuration,
            )
        if len(sample_ids) != task_counts[dataset]:
            raise RuntimeError(
                f"{dataset} exposes {len(sample_ids)} samples, expected "
                f"{task_counts[dataset]}."
            )
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError(f"{dataset} contains duplicate sample IDs.")
    chunk_sizes = {
        "dbbench": 1,
        "strategyqa": args.strategyqa_chunk,
        "hotpotqa": args.hotpot_chunk,
    }
    singleton_prefix_datasets = {
        value.strip()
        for value in args.singleton_prefix_datasets.split(",")
        if value.strip()
    }
    invalid_singleton_prefix = sorted(singleton_prefix_datasets - set(datasets))
    if invalid_singleton_prefix:
        raise ValueError(
            "Singleton-prefix datasets must be requested datasets: "
            + ", ".join(invalid_singleton_prefix)
        )
    first_chunk_sizes = {
        dataset: 1 for dataset in singleton_prefix_datasets
    }
    jobs = build_jobs(
        seeds,
        datasets=datasets,
        task_counts=task_counts,
        chunk_sizes=chunk_sizes,
        first_chunk_sizes=first_chunk_sizes,
    )
    if task_indices:
        selected = set(task_indices)
        jobs = [
            job
            for job in jobs
            if job.offset in selected and job.limit == 1
        ]
        observed = {job.offset for job in jobs}
        if observed != selected:
            raise ValueError(
                "Task-index selection requires singleton jobs for every "
                f"requested offset; missing {sorted(selected - observed)}"
            )
    reused = reuse_canaries(
        output_root,
        args.run_id,
        z,
        n,
        backbone=args.backbone,
        datasets=datasets,
    )
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        expected_ids = dataset_sample_ids[job.dataset]
        if not valid_part(
            part_path(
                output_root,
                args.run_id,
                job,
                backbone=args.backbone,
            ),
            job,
            z=z,
            n=n,
            expected_sample_ids=(
                expected_ids[job.offset : job.offset + job.limit]
                if expected_ids is not None
                else None
            ),
        ):
            pending.put(job)

    lock = threading.Lock()
    interactive_slots = threading.BoundedSemaphore(
        args.max_interactive_workers
    )
    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "started_at": now(),
        "updated_at": now(),
        "status": "running",
        "configuration": {
            "datasets": datasets,
            "backbone": args.backbone,
            "model": args.model,
            "seeds": seeds,
            "z": z,
            "n": n,
            "ports": ports,
            "task_counts": task_counts,
            "task_indices": task_indices,
            "chunk_sizes": chunk_sizes,
            "first_chunk_sizes": first_chunk_sizes,
            "hotpot_chunk": args.hotpot_chunk,
            "max_interactive_workers": args.max_interactive_workers,
            "workers_per_port": args.workers_per_port,
            "parallel_requests": args.parallel_requests,
            "fair_trajectory_budget": False,
        },
        "total_jobs": len(jobs),
        "completed_jobs": len(jobs) - pending.qsize(),
        "pending_jobs": pending.qsize(),
        "failed_jobs": {},
        "active_jobs": {},
        "reused_canary_jobs": reused,
    }
    write_json_atomic(status_path, status)

    def persist() -> None:
        status["updated_at"] = now()
        status["pending_jobs"] = pending.qsize()
        write_json_atomic(status_path, status)

    def worker(worker_key: str, port: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            acquired_interactive = False
            if job.dataset == "dbbench":
                acquired_interactive = interactive_slots.acquire(
                    blocking=False
                )
                if not acquired_interactive:
                    pending.put(job)
                    pending.task_done()
                    time.sleep(0.2)
                    continue
            try:
                output = part_path(
                    output_root,
                    args.run_id,
                    job,
                    backbone=args.backbone,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                success = False
                last_error = ""
                started = time.monotonic()
                expected_ids = dataset_sample_ids[job.dataset]
                expected_part_ids = (
                    expected_ids[job.offset : job.offset + job.limit]
                    if expected_ids is not None
                    else None
                )
                with lock:
                    status["active_jobs"][worker_key] = job.key
                    persist()
                for attempt in range(1, args.retries + 1):
                    task_log = log_path(log_root, args.run_id, job, attempt, port)
                    task_log.parent.mkdir(parents=True, exist_ok=True)
                    command = command_for(
                        job,
                        port=port,
                        output=output,
                        log_file=task_log.with_suffix(".runner.log"),
                        sampling_root=sampling_root,
                        model=args.model,
                        z=z,
                        n=n,
                        max_steps=int(
                            DATASET_DEFAULTS[job.dataset]["max_steps"]
                        ),
                        max_tokens=int(
                            DATASET_DEFAULTS[job.dataset]["max_tokens"]
                        ),
                        parallel_requests=args.parallel_requests,
                        relaxed_dbbench_replay=args.relaxed_dbbench_replay,
                        hotpot_sample_ids_file=(
                            hotpot_sample_ids_file
                            if job.dataset == "hotpotqa"
                            else None
                        ),
                    )
                    with task_log.open("w", encoding="utf-8") as handle:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            check=False,
                            env={
                                **os.environ,
                                "VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "vllm"),
                                "LTUQ_DISABLE_HARD_FINALIZATION": "1",
                            },
                        )
                    if completed.returncode == 0 and valid_part(
                        output,
                        job,
                        z=z,
                        n=n,
                        expected_sample_ids=expected_part_ids,
                    ):
                        success = True
                        break
                    last_error = (
                        f"attempt={attempt}, returncode={completed.returncode}, "
                        "valid_output="
                        f"{valid_part(output, job, z=z, n=n, expected_sample_ids=expected_part_ids)}"
                    )
                    time.sleep(2)
                elapsed = time.monotonic() - started
                with lock:
                    status["active_jobs"].pop(worker_key, None)
                    if success:
                        status["completed_jobs"] += 1
                    else:
                        status["failed_jobs"][job.key] = {
                            "error": last_error,
                            "elapsed_seconds": elapsed,
                            "port": port,
                            "worker": worker_key,
                        }
                    persist()
            finally:
                if acquired_interactive:
                    interactive_slots.release()
                pending.task_done()

    worker_specs = [
        (
            str(port) if args.workers_per_port == 1 else f"{port}:{replica}",
            port,
            replica,
        )
        for port in ports
        for replica in range(args.workers_per_port)
    ]
    threads = [
        threading.Thread(
            target=worker,
            args=(worker_key, port),
            name=f"gpu-port-{port}-worker-{replica}",
        )
        for worker_key, port, replica in worker_specs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Reconcile parts that may have been rejected by an earlier validator while
    # this long-running process was active.
    for job in jobs:
        if job.key not in status["failed_jobs"]:
            continue
        expected_ids = dataset_sample_ids[job.dataset]
        if valid_part(
            part_path(
                output_root,
                args.run_id,
                job,
                backbone=args.backbone,
            ),
            job,
            z=z,
            n=n,
            expected_sample_ids=(
                expected_ids[job.offset : job.offset + job.limit]
                if expected_ids is not None
                else None
            ),
        ):
            status["failed_jobs"].pop(job.key, None)
            status["completed_jobs"] += 1
    persist()

    failures = status["failed_jobs"]
    status["status"] = "failed" if failures else "complete"
    status["ended_at"] = now()
    if not failures:
        status["merged_outputs"] = merge_outputs(
            jobs,
            output_root,
            args.run_id,
            seeds,
            datasets,
            args.backbone,
        )
    persist()

    state_path = args.state.resolve()
    state = read_json(state_path) if state_path.exists() else {}
    state_job = {
        "job_id": args.run_id,
        "run_id": args.run_id,
        "tmux_session": os.environ.get("TMUX_SESSION_NAME"),
        "configuration": status["configuration"],
        "status": status["status"],
        "status_path": portable_path(status_path),
        "retries": args.retries,
    }
    state.update(
        {
            "updated_at": now(),
            "stage": args.stage,
            "stage_status": status["status"],
            "next_action": (
                (
                    "Run deterministic Z/N sweeps and representative analyses."
                    if args.stage == "maximum_pool_generation"
                    else "Advance to the next exact sampling-budget run."
                )
                if not failures
                else "Repair only failed jobs listed in the job-status file."
            ),
            "jobs": upsert_state_job(state, state_job),
        }
    )
    write_json_atomic(state_path, state)
    print(
        json.dumps(
            {
                "status": status["status"],
                "completed_jobs": status["completed_jobs"],
                "failed_jobs": len(failures),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
