from __future__ import annotations

import argparse
import asyncio
import os
from dotenv import load_dotenv

from .runners import (
    AgentBenchRunnerConfig,
    AgentBenchWebShopRunnerConfig,
    HotpotQARunnerConfig,
    StrategyQARunnerConfig,
    run_agentbench_dbbench,
    run_agentbench_webshop,
    run_hotpotqa,
    run_strategyqa,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BayesTraj paper raw-trajectory generator")
    parser.add_argument(
        "command",
        choices=[
            "run-strategyqa",
            "run-hotpotqa",
            "run-agentbench-dbbench",
            "run-agentbench-webshop",
        ],
    )
    parser.add_argument("--model", default="gpt-5-mini", help="Backbone model identifier")
    parser.add_argument(
        "--provider",
        choices=["openai", "openai-compatible", "azure-openai", "ollama", "vllm"],
        default="openai",
        help="Chat backend provider",
    )
    parser.add_argument(
        "--method",
        choices=["uprop"],
        default="uprop",
        help="Raw-generation instrumentation used by the submitted campaign",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--dataset", dest="dataset", default=None, help="Optional dataset/task-family filter for plot export")
    parser.add_argument("--analysis-dataset", dest="dataset", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-name", default=None, help="Optional Hugging Face dataset name override")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config name override")
    parser.add_argument("--dataset-revision", default=None, help="Optional immutable Hugging Face dataset revision")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to run; omit to process the full dataset",
    )
    parser.add_argument("--offset", type=int, default=0, help="Sample offset within the split")
    parser.add_argument("--sample-ids-file", default=None, help="Optional newline-delimited sample ids to run")
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument(
        "--plot-output-dir",
        default="outputs/analysis_plots",
        help="Directory for exported dashboard plot files",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "html", "svg", "pdf"],
        default="png",
        help="File format for exported dashboard plots",
    )
    parser.add_argument("--controller-url", default="http://localhost:5020/api", help="AgentBench controller API base URL")
    parser.add_argument("--task-name", default="dbbench-std", help="AgentBench task name")
    parser.add_argument("--task-index", default=None, help="Optional explicit AgentBench sample index")
    parser.add_argument("--backbone-samples", type=int, default=4, help="Per-step samples for backbone rollouts")
    parser.add_argument("--next-step-samples", type=int, default=4, help="Monte Carlo samples for next-step entropy")
    parser.add_argument("--tdp-samples", type=int, default=4, help="Number of TDPs for UProp")
    parser.add_argument("--per-step-samples", type=int, default=4, help="Per-step samples inside each TDP")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum reasoning steps; omit to use the task-specific default",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum completion tokens")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional backend sampling seed for OpenAI-compatible providers that support seeded decoding",
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=8,
        help="Maximum in-flight API requests per model client when logical batches are split across multiple calls",
    )
    parser.add_argument("--tau", type=float, default=1.0, help="UProp Gaussian-kernel sharpness")
    parser.add_argument("--uprop-ratio-epsilon", type=float, default=1e-6, help="Minimum intrinsic uncertainty used in UProp EU/IU denominator ratios")
    parser.add_argument("--uprop-ratio-cap", type=float, default=10.0, help="Maximum UProp EU/IU denominator ratio")
    parser.add_argument("--uprop-intrinsic-cap", type=float, default=None, help="Optional cap applied to per-step intrinsic uncertainty before UProp scoring")
    parser.add_argument(
        "--uprop-intrinsic-transform",
        choices=["none", "log1p"],
        default="none",
        help="Optional transform applied to per-step intrinsic uncertainty before UProp scoring",
    )
    parser.add_argument(
        "--fair-trajectory-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep trajectory and per-step candidate budgets aligned with the paper protocol, "
            "including shared post-rollout finalization where applicable. Enabled by default; use "
            "--no-fair-trajectory-budget to disable."
        ),
    )
    parser.set_defaults(
        branches=4,
        window=3,
        adaptive=False,
        low_branches=1,
        low_window=1,
        high_branches=4,
        high_window=3,
        top_fraction=0.2,
        adaptive_lb_branch_points=2,
        adaptive_lb_branches=3,
        adaptive_lb_horizon=5,
        adaptive_lb_window=1,
    )
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible, vLLM, or Ollama base URL")
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint URL")
    parser.add_argument("--api-version", default=None, help="Azure OpenAI API version")
    parser.add_argument("--deployment-name", default=None, help="Azure OpenAI deployment name")
    parser.add_argument("--api-key", default=None, help="Optional API key override")
    parser.add_argument(
        "--emulate-tool-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use plain-text JSON tool-call emulation instead of provider-native tool calling. Enabled by default; use --no-emulate-tool-calls to disable.",
    )
    parser.add_argument(
        "--relaxed-replay",
        dest="strict_replay",
        action="store_false",
        default=True,
        help=(
            "Do not discard replayed AgentBench prefix branches when replayed prefix observations differ from cached observations. "
            "The mismatch is recorded in replay_unmatched diagnostics."
        ),
    )
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name override")
    parser.add_argument("--run-name", default=None, help="Optional tracking run name")
    parser.add_argument("--notes", default="", help="Free-form experiment notes stored with the run")
    parser.add_argument("--tracking-dir", default="outputs/experiments", help="Directory for tracked experiment runs")
    parser.add_argument(
        "--sampling-dir",
        default="outputs/shared_sampling",
        help="Directory for shared backbone, branch, entropy, and TDP sampling artifacts",
    )
    parser.add_argument("--disable-tracking", action="store_true", help="Skip experiment tracking artifacts")
    parser.add_argument("--tag", action="append", default=None, help="Experiment tag; repeat to add multiple tags")
    parser.add_argument("--log-level", default="INFO", help="Python log level for the run")
    parser.add_argument("--log-file", default=None, help="Optional explicit log file path")
    parser.add_argument("--progress", dest="show_progress", action="store_true", default=None, help="Force the terminal progress UI")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="Disable the terminal progress UI")
    parser.add_argument("--restart", action="store_true", help="Ignore any existing checkpoint and rerun the selected samples")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard server host")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard server port")
    parser.add_argument("--sample-id", default=None, help="Optional sample id filter for plot export")
    return parser


def _summary_uncertainty_text(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "None"


def _print_run_result(result: dict[str, object]) -> None:
    summary = result["summary"]
    assert isinstance(summary, dict)
    success_rate = summary.get("success_rate")
    if success_rate is None:
        success_rate = summary.get("accuracy")
    print(f"method={summary['method']}")
    print(f"samples={summary['total']}")
    print(f"mean_uncertainty={_summary_uncertainty_text(summary.get('mean_uncertainty'))}")
    print(f"success_rate={success_rate}")
    print(f"output={result['output_path']}")
    if result.get("tracking_path") is not None:
        print(f"tracking={result['tracking_path']}")


def _print_plot_export_result(result: dict[str, object]) -> None:
    print(f"groups={result['group_count']}")
    print(f"samples={result['sample_count']}")
    print(f"charts={result['charts_written']}")
    print(f"format={result['export_format']}")
    print(f"output_dir={result['output_dir']}")
    print(f"manifest={result['manifest_path']}")


def _print_refresh_summary_result(result: dict[str, object]) -> None:
    print(f"updated={result['updated']}")
    print(f"skipped={result['skipped']}")


def _resolve_max_steps(args_max_steps: int | None, runner_kwargs: dict[str, object], default_max_steps: int) -> int:
    if args_max_steps is not None:
        return args_max_steps
    configured_max_steps = runner_kwargs.get("max_steps")
    if isinstance(configured_max_steps, int):
        return configured_max_steps
    return default_max_steps


def _model_supports_emulated_tool_calls(model: str) -> bool:
    model_name = model.strip().lower()
    # Restrict emulated tool-calling to model families where we have tested the
    # plain-chat JSON-tool prompt path. GPT-OSS benefits from this because some
    # OpenAI-compatible backends can expose an unstable native tool parser for
    # its Harmony tool-call headers.
    return model_name.startswith(("gpt-4", "gpt-5", "gpt-oss", "gemma", "qwen"))


def _resolve_emulate_tool_calls(*, requested: bool, model: str) -> bool:
    if not requested:
        return False
    return _model_supports_emulated_tool_calls(model)


def _build_strategyqa_config(args: argparse.Namespace) -> StrategyQARunnerConfig:
    runner_kwargs: dict[str, object] = {}
    split = runner_kwargs.get("split", args.split)
    method = args.method
    tags = tuple(args.tag or ())
    max_steps = _resolve_max_steps(args.max_steps, runner_kwargs, StrategyQARunnerConfig.max_steps)
    return StrategyQARunnerConfig(
        method=method,
        model=args.model,
        provider=args.provider,
        split=split,
        dataset_name=args.dataset_name or runner_kwargs.get("dataset_name") or "ChilleD/StrategyQA",
        dataset_revision=args.dataset_revision or runner_kwargs.get("dataset_revision"),
        limit=args.limit,
        offset=args.offset,
        sample_ids_file=args.sample_ids_file or runner_kwargs.get("sample_ids_file"),
        output_path=args.output,
        backbone_per_step_samples=args.backbone_samples,
        next_step_entropy_samples=args.next_step_samples,
        tdp_samples=args.tdp_samples,
        per_step_samples=args.per_step_samples,
        max_steps=max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel_requests=args.parallel_requests,
        seed=args.seed,
        tau=args.tau,
        uprop_ratio_epsilon=args.uprop_ratio_epsilon,
        uprop_ratio_cap=args.uprop_ratio_cap,
        uprop_intrinsic_cap=args.uprop_intrinsic_cap,
        uprop_intrinsic_transform=args.uprop_intrinsic_transform,
        fair_trajectory_budget=bool(runner_kwargs.get("fair_trajectory_budget", args.fair_trajectory_budget)),
        base_url=args.base_url,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        deployment_name=args.deployment_name,
        api_key=args.api_key,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        notes=args.notes,
        tracking_dir=args.tracking_dir,
        track_experiment=not args.disable_tracking,
        tags=tags,
        shared_sampling_dir=args.sampling_dir,
        log_level=args.log_level,
        log_path=args.log_file,
        show_progress=args.show_progress,
        restart=args.restart,
    )


def _build_hotpotqa_config(args: argparse.Namespace) -> HotpotQARunnerConfig:
    runner_kwargs: dict[str, object] = {}
    split = runner_kwargs.get("split", args.split)
    allow_train_split = os.environ.get("LTUQ_ALLOW_HOTPOTQA_TRAIN_SPLIT", "").strip().lower()
    if split == "train" and allow_train_split not in {"1", "true", "yes", "on"}:
        split = "validation"
    method = args.method
    tags = tuple(args.tag or ())
    max_steps = _resolve_max_steps(args.max_steps, runner_kwargs, HotpotQARunnerConfig.max_steps)
    emulate_tool_calls = _resolve_emulate_tool_calls(
        requested=bool(runner_kwargs.get("emulate_tool_calls", args.emulate_tool_calls)),
        model=args.model,
    )
    return HotpotQARunnerConfig(
        method=method,
        model=args.model,
        provider=args.provider,
        emulate_tool_calls=emulate_tool_calls,
        split=split,
        dataset_name=args.dataset_name or runner_kwargs.get("dataset_name") or "hotpotqa/hotpot_qa",
        config_name=args.dataset_config or runner_kwargs.get("config_name") or "distractor",
        dataset_revision=args.dataset_revision or runner_kwargs.get("dataset_revision"),
        limit=args.limit,
        offset=args.offset,
        sample_ids_file=args.sample_ids_file or runner_kwargs.get("sample_ids_file"),
        output_path=args.output,
        backbone_per_step_samples=args.backbone_samples,
        next_step_entropy_samples=args.next_step_samples,
        tdp_samples=args.tdp_samples,
        per_step_samples=args.per_step_samples,
        max_steps=max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel_requests=args.parallel_requests,
        seed=args.seed,
        tau=args.tau,
        uprop_ratio_epsilon=args.uprop_ratio_epsilon,
        uprop_ratio_cap=args.uprop_ratio_cap,
        uprop_intrinsic_cap=args.uprop_intrinsic_cap,
        uprop_intrinsic_transform=args.uprop_intrinsic_transform,
        fair_trajectory_budget=bool(runner_kwargs.get("fair_trajectory_budget", args.fair_trajectory_budget)),
        base_url=args.base_url,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        deployment_name=args.deployment_name,
        api_key=args.api_key,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        notes=args.notes,
        tracking_dir=args.tracking_dir,
        track_experiment=not args.disable_tracking,
        tags=tags,
        shared_sampling_dir=args.sampling_dir,
        log_level=args.log_level,
        log_path=args.log_file,
        show_progress=args.show_progress,
        restart=args.restart,
    )


def _build_agentbench_dbbench_config(args: argparse.Namespace) -> AgentBenchRunnerConfig:
    runner_kwargs: dict[str, object] = {}
    emulate_tool_calls = _resolve_emulate_tool_calls(
        requested=bool(runner_kwargs.get("emulate_tool_calls", args.emulate_tool_calls)),
        model=args.model,
    )
    method = args.method
    tags = tuple(args.tag or ())
    max_steps = _resolve_max_steps(args.max_steps, runner_kwargs, AgentBenchRunnerConfig.max_steps)
    task_index: int | str | None = args.task_index
    if isinstance(task_index, str) and task_index.isdigit():
        task_index = int(task_index)
    return AgentBenchRunnerConfig(
        method=method,
        model=args.model,
        provider=args.provider,
        emulate_tool_calls=emulate_tool_calls,
        controller_url=args.controller_url,
        task_name=runner_kwargs.get("task_name", args.task_name),
        task_index=task_index,
        limit=args.limit,
        offset=args.offset,
        output_path=args.output,
        backbone_per_step_samples=args.backbone_samples,
        next_step_entropy_samples=args.next_step_samples,
        tdp_samples=args.tdp_samples,
        per_step_samples=args.per_step_samples,
        max_steps=max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        parallel_requests=args.parallel_requests,
        tau=args.tau,
        uprop_ratio_epsilon=args.uprop_ratio_epsilon,
        uprop_ratio_cap=args.uprop_ratio_cap,
        uprop_intrinsic_cap=args.uprop_intrinsic_cap,
        uprop_intrinsic_transform=args.uprop_intrinsic_transform,
        fair_trajectory_budget=bool(runner_kwargs.get("fair_trajectory_budget", args.fair_trajectory_budget)),
        base_url=args.base_url,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        deployment_name=args.deployment_name,
        api_key=args.api_key,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        notes=args.notes,
        tracking_dir=args.tracking_dir,
        track_experiment=not args.disable_tracking,
        tags=tags,
        shared_sampling_dir=args.sampling_dir,
        log_level=args.log_level,
        log_path=args.log_file,
        show_progress=args.show_progress,
        restart=args.restart,
        strict_replay=bool(
            args.strict_replay and runner_kwargs.get("strict_replay", True)
        ),
    )


def _build_agentbench_webshop_config(args: argparse.Namespace) -> AgentBenchWebShopRunnerConfig:
    runner_kwargs: dict[str, object] = {}
    emulate_tool_calls = _resolve_emulate_tool_calls(
        requested=bool(runner_kwargs.get("emulate_tool_calls", args.emulate_tool_calls)),
        model=args.model,
    )
    method = args.method
    tags = tuple(args.tag or ())
    max_steps = _resolve_max_steps(args.max_steps, runner_kwargs, AgentBenchWebShopRunnerConfig.max_steps)
    task_index: int | str | None = args.task_index
    if isinstance(task_index, str) and task_index.isdigit():
        task_index = int(task_index)
    resolved_task_name = runner_kwargs.get("task_name") or (
        args.task_name if args.task_name != "dbbench-std" else "webshop-std"
    )
    return AgentBenchWebShopRunnerConfig(
        method=method,
        model=args.model,
        provider=args.provider,
        emulate_tool_calls=emulate_tool_calls,
        controller_url=args.controller_url,
        task_name=resolved_task_name,
        task_index=task_index,
        limit=args.limit,
        offset=args.offset,
        output_path=args.output,
        backbone_per_step_samples=args.backbone_samples,
        next_step_entropy_samples=args.next_step_samples,
        tdp_samples=args.tdp_samples,
        per_step_samples=args.per_step_samples,
        max_steps=max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel_requests=args.parallel_requests,
        tau=args.tau,
        uprop_ratio_epsilon=args.uprop_ratio_epsilon,
        uprop_ratio_cap=args.uprop_ratio_cap,
        uprop_intrinsic_cap=args.uprop_intrinsic_cap,
        uprop_intrinsic_transform=args.uprop_intrinsic_transform,
        fair_trajectory_budget=bool(
            runner_kwargs.get("fair_trajectory_budget", args.fair_trajectory_budget)
        ),
        base_url=args.base_url,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        deployment_name=args.deployment_name,
        api_key=args.api_key,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        notes=args.notes,
        tracking_dir=args.tracking_dir,
        track_experiment=not args.disable_tracking,
        tags=tags,
        shared_sampling_dir=args.sampling_dir,
        log_level=args.log_level,
        log_path=args.log_file,
        show_progress=args.show_progress,
        restart=args.restart,
        strict_replay=bool(runner_kwargs.get("strict_replay", args.strict_replay)),
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    runners = {
        "run-strategyqa": lambda: run_strategyqa(_build_strategyqa_config(args)),
        "run-hotpotqa": lambda: run_hotpotqa(_build_hotpotqa_config(args)),
        "run-agentbench-dbbench": lambda: run_agentbench_dbbench(
            _build_agentbench_dbbench_config(args)
        ),
        "run-agentbench-webshop": lambda: run_agentbench_webshop(
            _build_agentbench_webshop_config(args)
        ),
    }
    result = asyncio.run(runners[args.command]())
    _print_run_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
