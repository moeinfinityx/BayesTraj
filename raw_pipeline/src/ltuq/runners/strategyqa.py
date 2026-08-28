from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Sequence

from ..cli_progress import RunProgressDisplay
from ..config import ExperimentConfig
from ..datasets import StrategyQASample, load_strategyqa_split
from ..estimators import UPropEstimator
from ..experiments import ExperimentTracker
from ..executors import StrategyQALocalBranchingExecutor
from ..logging_utils import configure_logging
from ..models.base import ModelProvider
from ..models import ChatModelConfig, ModelGenerationError, RecordedChatModelClient, close_chat_model, create_chat_model
from ..models.openai import OpenAIChatModelClient
from ..results_summary import resolve_record_uncertainty, summarize_records
from ..sampling import SharedSamplingStorage, reusable_model_signature


LOGGER = logging.getLogger(__name__)

_RESUME_IGNORED_CONFIG_KEYS = {
    "api_key",
    "experiment_name",
    "run_name",
    "notes",
    "tracking_dir",
    "track_experiment",
    "tags",
    "log_level",
    "log_path",
    "show_progress",
    "parallel_requests",
    "provider",
    "base_url",
    "azure_endpoint",
    "api_version",
    "deployment_name",
    "restart",
    "shared_sampling_dir",
}

def _payload_contains_token_logprobs(payload: Any) -> bool:
    if isinstance(payload, dict):
        logprob_sum = payload.get("token_logprob_sum")
        token_count = payload.get("token_count")
        if isinstance(logprob_sum, (int, float)) and not isinstance(logprob_sum, bool):
            if isinstance(token_count, int) and not isinstance(token_count, bool) and token_count > 0:
                return True
        return any(_payload_contains_token_logprobs(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_contains_token_logprobs(item) for item in payload)
    return False


def _payload_marks_logprobs_unavailable(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("logprobs_unavailable") is True or payload.get("token_logprob_floor_detected") is True:
            return True
        return any(_payload_marks_logprobs_unavailable(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_marks_logprobs_unavailable(item) for item in payload)
    return False


def _record_has_missing_logprob_state(record: dict[str, Any], *, method: str = "uprop") -> bool:
    if str(record.get("method", "")) != method:
        return False
    if record.get("status") == "generation_failed":
        return False
    if _payload_marks_logprobs_unavailable(record):
        return True
    if _payload_contains_token_logprobs(record):
        return False
    return True


async def _supports_structured_text_logprobs(
    config: "StrategyQARunnerConfig",
    *,
    model_client: Any,
) -> bool:
    current_support = getattr(model_client, "_text_logprobs_supported", None)
    if current_support is False:
        return False
    if current_support is True:
        return True

    detailed_callable = getattr(model_client, "sample_many_detailed", None)
    if detailed_callable is None:
        return False

    probe_messages = [
        {"role": "system", "content": "Respond with a single token if possible."},
        {"role": "user", "content": "Answer yes."},
    ]
    try:
        outputs = await detailed_callable(probe_messages, temperature=0.0, n=1)
    except Exception as exc:
        LOGGER.warning(
            "Could not probe text logprob support for StrategyQA rerun model=%s provider=%s: %s",
            config.model,
            config.provider,
            exc,
        )
        return False

    current_support = getattr(model_client, "_text_logprobs_supported", None)
    if current_support is False:
        return False
    if current_support is True:
        return True

    for output in outputs:
        if not isinstance(output, dict):
            continue
        metadata = output.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if _payload_contains_token_logprobs(metadata):
            return True
        if _payload_marks_logprobs_unavailable(metadata):
            return True
    return False


def _filter_stale_strategyqa_records(
    records: Sequence[dict[str, Any]],
    *,
    refresh_missing_logprob_records: bool,
    method: str = "uprop",
) -> tuple[list[dict[str, Any]], list[str]]:
    reusable_records: list[dict[str, Any]] = []
    dropped_sample_ids: list[str] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        record["uncertainty"] = resolve_record_uncertainty(record)
        if refresh_missing_logprob_records and _record_has_missing_logprob_state(record, method=method):
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str):
                dropped_sample_ids.append(sample_id)
            continue
        reusable_records.append(record)

    return reusable_records, dropped_sample_ids


@dataclass(frozen=True)
class StrategyQARunnerConfig:
    method: str = "uprop"
    model: str = "gpt-5.4"
    provider: ModelProvider = "openai"
    split: str = "train"
    dataset_name: str = "ChilleD/StrategyQA"
    dataset_revision: str | None = None
    limit: int | None = None
    offset: int = 0
    sample_ids_file: str | None = None
    output_path: str | None = None
    branches: int = 4
    window: int = 3
    adaptive: bool = False
    low_branches: int = 1
    low_window: int = 1
    high_branches: int = 4
    high_window: int = 3
    top_fraction: float = 0.2
    backbone_per_step_samples: int = 4
    next_step_entropy_samples: int = 4
    tdp_samples: int = 4
    per_step_samples: int = 4
    max_steps: int = 6
    temperature: float = 0.8
    max_tokens: int = 8192
    parallel_requests: int = 8
    seed: int | None = None
    step_output_format: str = "auto"
    step_retry_attempts: int = 0
    tau: float = 1.0
    uprop_ratio_epsilon: float = 1e-6
    uprop_ratio_cap: float | None = 10.0
    uprop_intrinsic_cap: float | None = None
    uprop_intrinsic_transform: str = "none"
    fair_trajectory_budget: bool = True
    base_url: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None
    api_key: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    notes: str = ""
    tracking_dir: str = "outputs/experiments"
    track_experiment: bool = True
    tags: tuple[str, ...] = ()
    shared_sampling_dir: str = "outputs/shared_sampling"
    log_level: str = "INFO"
    log_path: str | None = None
    show_progress: bool | None = None
    restart: bool = False

    def __post_init__(self) -> None:
        if self.method != "uprop":
            raise ValueError("the paper raw-generation runner supports only 'uprop'")
        if self.step_output_format not in {"auto", "text", "structured_json"}:
            raise ValueError("step_output_format must be 'auto', 'text', or 'structured_json'")
        if self.step_retry_attempts < 0:
            raise ValueError("step_retry_attempts must be non-negative")
        if self.uprop_ratio_epsilon <= 0.0:
            raise ValueError("uprop_ratio_epsilon must be positive")
        if self.uprop_ratio_cap is not None and self.uprop_ratio_cap <= 0.0:
            raise ValueError("uprop_ratio_cap must be positive when provided")
        if self.uprop_intrinsic_cap is not None and self.uprop_intrinsic_cap <= 0.0:
            raise ValueError("uprop_intrinsic_cap must be positive when provided")
        if self.uprop_intrinsic_transform not in {"none", "log1p"}:
            raise ValueError("uprop_intrinsic_transform must be 'none' or 'log1p'")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        ChatModelConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            parallel_requests=self.parallel_requests,
            seed=self.seed,
            base_url=self.base_url,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            deployment_name=self.deployment_name,
        )


def _normalize_strategyqa_answer(answer: Any) -> str | None:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if answer is None:
        return None
    normalized = str(answer).strip().lower()
    if normalized in {"yes", "no"}:
        return normalized
    return None


def _strategyqa_tdp_outcome_bucket(tdp: Any) -> str:
    normalized = _normalize_strategyqa_answer(getattr(tdp, "final_answer", None))
    if normalized is not None:
        return f"answer:{normalized}"
    metadata = getattr(tdp, "metadata", None)
    status = metadata.get("status") if isinstance(metadata, dict) else None
    return f"no-answer:{status.strip()}" if isinstance(status, str) and status.strip() else "no-answer:unknown"


def _is_qwen35_9b_model(model: str) -> bool:
    normalized = model.strip().lower().replace("_", "-")
    return "qwen" in normalized and "9b" in normalized and ("3.5" in normalized or "35" in normalized)


def _resolved_step_output_format(config: StrategyQARunnerConfig) -> str:
    if config.step_output_format != "auto":
        return config.step_output_format
    return "structured_json" if _is_qwen35_9b_model(config.model) else "text"


def _resolved_step_retry_attempts(config: StrategyQARunnerConfig) -> int:
    if config.step_retry_attempts > 0:
        return config.step_retry_attempts
    return 2 if _resolved_step_output_format(config) == "structured_json" and _is_qwen35_9b_model(config.model) else 0


def _majority_final_answer(estimates: Sequence[str | None]) -> str | None:
    normalized = [_normalize_strategyqa_answer(answer) for answer in estimates]
    filtered = [answer for answer in normalized if answer is not None]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _serialize_uprop_result(sample: StrategyQASample, estimate: Any) -> dict[str, Any]:
    predicted_answer = _majority_final_answer([tdp.final_answer for tdp in estimate.tdps])
    reference_answer = _normalize_strategyqa_answer(sample.answer)
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "method": "uprop",
        "reference_answer": reference_answer,
        "predicted_answer": predicted_answer,
        "is_correct": predicted_answer == reference_answer if predicted_answer is not None else None,
        "uncertainty": float(estimate.total_uncertainty),
        "estimate": asdict(estimate),
    }


def _serialize_generation_failure(
    sample: StrategyQASample,
    *,
    method: str,
    error: ModelGenerationError,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "method": method,
        "status": "generation_failed",
        "reference_answer": _normalize_strategyqa_answer(sample.answer),
        "predicted_answer": None,
        "is_correct": None,
        "uncertainty": None,
        "error": str(error),
        "failure": error.to_record_metadata(),
        "estimate": None,
    }


def _format_uncertainty_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "n/a"


async def run_strategyqa_samples(
    samples: Sequence[StrategyQASample],
    *,
    model_client: Any,
    config: StrategyQARunnerConfig,
    tracker: ExperimentTracker | None = None,
    sample_offset: int = 0,
    sample_total: int | None = None,
    record_callback: Callable[[StrategyQASample, dict[str, Any], int, int], None] | None = None,
) -> list[dict[str, Any]]:
    shared_sampling_storage = SharedSamplingStorage(config.shared_sampling_dir)
    executor = StrategyQALocalBranchingExecutor(
        model_client,
        max_steps=config.max_steps,
        sample_temperature=config.temperature,
        backbone_per_step_samples=config.backbone_per_step_samples,
        next_step_entropy_samples=config.next_step_entropy_samples,
        collect_sample_logprobs=True,
        shared_sampling_storage=shared_sampling_storage,
        model_signature=_strategyqa_model_signature(config),
        step_output_format=_resolved_step_output_format(config),
        step_retry_attempts=_resolved_step_retry_attempts(config),
    )

    records: list[dict[str, Any]] = []
    total_samples = sample_total or len(samples)
    estimator = UPropEstimator(
        trajectory_samples=config.tdp_samples,
        per_step_samples=config.per_step_samples,
        tau=config.tau,
        ratio_epsilon=config.uprop_ratio_epsilon,
        ratio_cap=config.uprop_ratio_cap,
        intrinsic_cap=config.uprop_intrinsic_cap,
        intrinsic_transform=config.uprop_intrinsic_transform,
        outcome_bucket_fn=_strategyqa_tdp_outcome_bucket,
    )

    for index, sample in enumerate(samples, start=sample_offset + 1):
        LOGGER.info(
            "Running sample %s/%s with %s: sample_id=%s",
            index,
            total_samples,
            config.method,
            sample.sample_id,
        )
        if tracker is not None:
            tracker.log_event(
                "sample_started",
                sample_id=sample.sample_id,
                position=index,
                total_samples=total_samples,
                method=config.method,
            )
        try:
            estimate = await estimator.estimate(sample=sample, executor=executor)
            record = _serialize_uprop_result(sample, estimate)
        except ModelGenerationError as exc:
            LOGGER.warning(
                "StrategyQA sample_id=%s failed to generate due to model error code=%s retryable=%s",
                sample.sample_id,
                exc.error_code,
                exc.retryable,
            )
            record = _serialize_generation_failure(sample, method=config.method, error=exc)
            if tracker is not None:
                tracker.log_event(
                    "sample_failed",
                    sample_id=sample.sample_id,
                    position=index,
                    method=config.method,
                    error=str(exc),
                    error_code=exc.error_code,
                    retryable=exc.retryable,
                )
        else:
            LOGGER.info(
                "Completed sample_id=%s uncertainty=%s correct=%s",
                sample.sample_id,
                _format_uncertainty_value(record["uncertainty"]),
                record["is_correct"],
            )
            if tracker is not None:
                tracker.log_event(
                    "sample_completed",
                    sample_id=sample.sample_id,
                    position=index,
                    uncertainty=record["uncertainty"],
                    is_correct=record["is_correct"],
                )
        records.append(record)
        if record_callback is not None:
            record_callback(sample, record, index, total_samples)
    return records


def summarize_strategyqa_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return summarize_records(records)


def write_strategyqa_results(output_path: str, records: Sequence[dict[str, Any]]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def _build_experiment_name(config: StrategyQARunnerConfig) -> str:
    if config.experiment_name:
        return config.experiment_name
    return f"strategyqa-{config.method}-{config.model.replace('/', '-')}-{config.split}"


def _strategyqa_resume_fingerprint(config: StrategyQARunnerConfig) -> str:
    payload = asdict(config)
    for key in _RESUME_IGNORED_CONFIG_KEYS:
        payload.pop(key, None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha1(serialized.encode("utf-8")).hexdigest()[:12]


def _default_run_name(config: StrategyQARunnerConfig) -> str:
    limit_label = "all" if config.limit is None else str(config.limit)
    return f"offset-{config.offset}-limit-{limit_label}-{_strategyqa_resume_fingerprint(config)}"


def _strategyqa_model_signature(config: StrategyQARunnerConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "step_output_format": _resolved_step_output_format(config),
        "step_retry_attempts": _resolved_step_retry_attempts(config),
    }
    if config.seed is not None:
        payload["seed"] = config.seed
    return reusable_model_signature(payload)


def _resolve_output_path(config: StrategyQARunnerConfig, tracker: ExperimentTracker | None) -> str:
    if config.output_path is not None:
        return config.output_path
    if tracker is not None:
        return str(tracker.artifacts_dir / "results.jsonl")
    return f"outputs/strategyqa-{config.method}-{config.model.replace('/', '-')}-{config.split}.jsonl"


def _checkpoint_path(output_path: str, tracker: ExperimentTracker | None) -> Path:
    if tracker is not None:
        return tracker.artifacts_dir / "checkpoint.json"
    return Path(f"{output_path}.checkpoint.json")


def _load_strategyqa_results(path: str | Path) -> list[dict[str, Any]]:
    result_path = Path(path)
    if not result_path.exists():
        return []
    with result_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        record["uncertainty"] = resolve_record_uncertainty(record)
    return records


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    config: StrategyQARunnerConfig,
    selected_samples: Sequence[StrategyQASample],
    output_path: str,
    records: Sequence[dict[str, Any]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_fingerprint": _strategyqa_resume_fingerprint(config),
        "selected_sample_ids": [sample.sample_id for sample in selected_samples],
        "output_path": output_path,
        "records": list(records),
    }
    temp_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(checkpoint_path)


def _delete_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def _load_checkpoint_records(
    checkpoint_path: Path,
    *,
    config: StrategyQARunnerConfig,
    selected_samples: Sequence[StrategyQASample],
) -> list[dict[str, Any]]:
    if config.restart or not checkpoint_path.exists():
        return []
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected_fingerprint = _strategyqa_resume_fingerprint(config)
    if payload.get("config_fingerprint") != expected_fingerprint:
        raise ValueError(
            "Existing checkpoint belongs to a different StrategyQA configuration. Use --restart or change the run name/output path."
        )
    expected_sample_ids = [sample.sample_id for sample in selected_samples]
    if payload.get("selected_sample_ids") != expected_sample_ids:
        raise ValueError(
            "Existing checkpoint does not match the selected StrategyQA samples. Use --restart or change the run name/output path."
        )
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("StrategyQA checkpoint is invalid: records must be a list.")
    for record in records:
        if isinstance(record, dict):
            record["uncertainty"] = resolve_record_uncertainty(record)
    return records


def _build_tracker(config: StrategyQARunnerConfig) -> ExperimentTracker | None:
    if not config.track_experiment:
        return None
    return ExperimentTracker(
        experiment=ExperimentConfig(
            name=_build_experiment_name(config),
            model=config.model,
            task_family="strategyqa",
            output_dir=config.tracking_dir,
            notes=config.notes,
        ),
        tracking_dir=config.tracking_dir,
        run_name=config.run_name or _default_run_name(config),
        tags=config.tags,
        metadata={"runner_config": asdict(config)},
        resume=not config.restart,
    )


def _wrap_model_client_for_tracking(
    model_client: Any,
    *,
    tracker: ExperimentTracker | None,
    provider: str,
) -> Any:
    if tracker is None:
        return model_client
    if not any(hasattr(model_client, attribute) for attribute in ("sample_many", "ainference", "inference")):
        return model_client
    return RecordedChatModelClient(model_client, tracker, provider=provider)


def _create_strategyqa_model_client(config: StrategyQARunnerConfig) -> Any:
    return create_chat_model(
        ChatModelConfig(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            parallel_requests=config.parallel_requests,
            seed=config.seed,
            base_url=config.base_url,
            azure_endpoint=config.azure_endpoint,
            api_version=config.api_version,
            deployment_name=config.deployment_name,
        )
    )


async def run_strategyqa(config: StrategyQARunnerConfig) -> dict[str, Any]:
    tracker = _build_tracker(config)
    output_path = _resolve_output_path(config, tracker)
    checkpoint_path = _checkpoint_path(output_path, tracker)
    progress_display = RunProgressDisplay(enabled=config.show_progress)
    configure_logging(
        config.log_level,
        config.log_path or (str(tracker.log_path) if tracker is not None else None),
        console_handler=progress_display.create_logging_handler(),
    )
    LOGGER.info(
        "Starting StrategyQA run: method=%s model=%s provider=%s split=%s limit=%s offset=%s",
        config.method,
        config.model,
        config.provider,
        config.split,
        config.limit,
        config.offset,
    )
    if tracker is not None:
        tracker.log_event("run_started", method=config.method, model=config.model, provider=config.provider)
        if tracker.resumed:
            LOGGER.info("Resuming tracked run in %s", tracker.run_dir)
            tracker.log_event("run_resumed", previous_status=tracker.previous_status)

    samples = load_strategyqa_split(
        split=config.split,
        dataset_name=config.dataset_name,
        revision=config.dataset_revision,
    )
    if config.sample_ids_file:
        requested_ids = [
            value.strip()
            for value in Path(config.sample_ids_file).read_text(encoding="utf-8").splitlines()
            if value.strip() and not value.lstrip().startswith("#")
        ]
        samples_by_id = {sample.sample_id: sample for sample in samples}
        missing_ids = [sample_id for sample_id in requested_ids if sample_id not in samples_by_id]
        if missing_ids:
            raise ValueError(
                f"StrategyQA sample ids file contains {len(missing_ids)} id(s) not present in the selected split; "
                f"first missing id: {missing_ids[0]}"
            )
        samples = [samples_by_id[sample_id] for sample_id in requested_ids]
    if config.limit is None:
        selected_samples = samples[config.offset :]
    else:
        selected_samples = samples[config.offset : config.offset + config.limit]
    if not selected_samples:
        raise ValueError("No StrategyQA samples were selected for the requested split/offset/limit.")
    LOGGER.info("Selected %s StrategyQA samples from %s", len(selected_samples), config.dataset_name)
    if tracker is not None:
        tracker.log_event(
            "samples_selected",
            count=len(selected_samples),
            sample_ids=[sample.sample_id for sample in selected_samples],
        )

    model_client: Any | None = None
    refresh_missing_logprob_records = False

    async def ensure_refresh_decision(records: Sequence[dict[str, Any]]) -> bool:
        nonlocal model_client, refresh_missing_logprob_records
        if refresh_missing_logprob_records:
            return True
        if not any(
            isinstance(record, dict) and _record_has_missing_logprob_state(record, method=config.method)
            for record in records
        ):
            return False
        if model_client is None:
            model_client = _create_strategyqa_model_client(config)
        refresh_missing_logprob_records = await _supports_structured_text_logprobs(
            config,
            model_client=model_client,
        )
        return refresh_missing_logprob_records

    checkpoint_records = _load_checkpoint_records(
        checkpoint_path,
        config=config,
        selected_samples=selected_samples,
    )
    dropped_checkpoint_sample_ids: list[str] = []
    if checkpoint_records:
        refresh_checkpoint_records = await ensure_refresh_decision(checkpoint_records)
        checkpoint_records, dropped_checkpoint_sample_ids = _filter_stale_strategyqa_records(
            checkpoint_records,
            refresh_missing_logprob_records=refresh_checkpoint_records,
            method=config.method,
        )
        if dropped_checkpoint_sample_ids:
            LOGGER.info(
                "Dropping %s stale StrategyQA checkpoint record(s) without token logprobs: %s",
                len(dropped_checkpoint_sample_ids),
                ", ".join(dropped_checkpoint_sample_ids[:10]),
            )
            _write_checkpoint(
                checkpoint_path,
                config=config,
                selected_samples=selected_samples,
                output_path=output_path,
                records=checkpoint_records,
            )
            if tracker is not None:
                tracker.log_event(
                    "stale_records_dropped",
                    source="checkpoint",
                    count=len(dropped_checkpoint_sample_ids),
                    sample_ids=dropped_checkpoint_sample_ids,
                )
    if checkpoint_records:
        LOGGER.info("Loaded %s checkpointed StrategyQA records from %s", len(checkpoint_records), checkpoint_path)
        if tracker is not None:
            tracker.log_event("checkpoint_loaded", records=len(checkpoint_records), checkpoint_path=str(checkpoint_path))

    if (
        tracker is not None
        and not config.restart
        and tracker.previous_status == "completed"
        and not checkpoint_path.exists()
        and Path(output_path).exists()
    ):
        existing_records = _load_strategyqa_results(output_path)
        existing_ids = [record.get("sample_id") for record in existing_records]
        selected_ids = [sample.sample_id for sample in selected_samples]
        if existing_records and existing_ids == selected_ids:
            refresh_existing_records = await ensure_refresh_decision(existing_records)
            reusable_existing_records, dropped_existing_sample_ids = _filter_stale_strategyqa_records(
                existing_records,
                refresh_missing_logprob_records=refresh_existing_records,
                method=config.method,
            )
            if not dropped_existing_sample_ids:
                progress_display.start(
                    total=len(selected_samples),
                    description=f"StrategyQA {config.method}",
                    completed=len(reusable_existing_records),
                )
                LOGGER.info("Existing completed run found at %s; reusing recorded results", output_path)
                summary = summarize_strategyqa_results(reusable_existing_records)
                tracker.finalize(
                    status="completed",
                    summary=summary,
                    artifacts={
                        "results": str(output_path),
                        "log": str(tracker.log_path),
                    },
                )
                await close_chat_model(model_client)
                progress_display.close(status="completed")
                return {
                    "summary": summary,
                    "output_path": str(output_path),
                    "records": reusable_existing_records,
                    "tracking_path": str(tracker.run_dir),
                }

            LOGGER.info(
                "Dropping %s stale StrategyQA result record(s) without token logprobs from %s",
                len(dropped_existing_sample_ids),
                output_path,
            )
            checkpoint_records = reusable_existing_records
            _write_checkpoint(
                checkpoint_path,
                config=config,
                selected_samples=selected_samples,
                output_path=output_path,
                records=checkpoint_records,
            )
            if tracker is not None:
                tracker.log_event(
                    "stale_records_dropped",
                    source="results",
                    count=len(dropped_existing_sample_ids),
                    sample_ids=dropped_existing_sample_ids,
                )

    completed_sample_ids = {record.get("sample_id") for record in checkpoint_records}
    pending_samples = [sample for sample in selected_samples if sample.sample_id not in completed_sample_ids]
    all_records = list(checkpoint_records)
    progress_display.start(
        total=len(selected_samples),
        description=f"StrategyQA {config.method}",
        completed=len(all_records),
    )
    LOGGER.info("StrategyQA resume state: %s completed, %s pending", len(all_records), len(pending_samples))
    if tracker is not None:
        tracker.log_event(
            "resume_state",
            completed_records=len(all_records),
            pending_records=len(pending_samples),
        )

    def persist_record(sample: StrategyQASample, record: dict[str, Any], index: int, total: int) -> None:
        del total
        all_records.append(record)
        progress_display.advance(
            position=index,
            sample_id=sample.sample_id,
            status=str(record.get("status") or "completed"),
            uncertainty=record.get("uncertainty"),
        )
        _write_checkpoint(
            checkpoint_path,
            config=config,
            selected_samples=selected_samples,
            output_path=output_path,
            records=all_records,
        )
        if tracker is not None:
            tracker.log_event("checkpoint_saved", records=len(all_records), checkpoint_path=str(checkpoint_path))

    try:
        if model_client is None:
            model_client = _create_strategyqa_model_client(config)
        model_client = _wrap_model_client_for_tracking(
            model_client,
            tracker=tracker,
            provider=config.provider,
        )
        if pending_samples:
            record_count_before_run = len(all_records)
            new_records = await run_strategyqa_samples(
                pending_samples,
                model_client=model_client,
                config=config,
                tracker=tracker,
                sample_offset=len(all_records),
                sample_total=len(selected_samples),
                record_callback=persist_record,
            )
            if len(all_records) == record_count_before_run and new_records:
                all_records.extend(new_records)
                _write_checkpoint(
                    checkpoint_path,
                    config=config,
                    selected_samples=selected_samples,
                    output_path=output_path,
                    records=all_records,
                )

        summary = summarize_strategyqa_results(all_records)
        written_path = write_strategyqa_results(output_path, all_records)
        _delete_checkpoint(checkpoint_path)
        LOGGER.info("Wrote %s StrategyQA records to %s", len(all_records), written_path)
        if tracker is not None:
            tracker.log_event("results_written", output_path=str(written_path))
            tracker.finalize(
                status="completed",
                summary=summary,
                artifacts={
                    "results": str(written_path),
                    "log": str(tracker.log_path),
                },
            )
            LOGGER.info("Tracked experiment run under %s", tracker.run_dir)
        progress_display.close(status="completed")
        return {
            "summary": summary,
            "output_path": str(written_path),
            "records": all_records,
            "tracking_path": str(tracker.run_dir) if tracker is not None else None,
        }
    except Exception as exc:
        LOGGER.exception("StrategyQA run failed")
        progress_display.close(status="failed")
        if tracker is not None:
            tracker.log_event("run_failed", error=str(exc))
            tracker.finalize(
                status="failed",
                summary=summarize_strategyqa_results(all_records),
                artifacts={
                    "results": str(output_path),
                    "log": str(tracker.log_path),
                    "checkpoint": str(checkpoint_path),
                },
            )
        raise
    finally:
        try:
            await close_chat_model(model_client)
        except Exception as exc:
            LOGGER.warning("Failed to close StrategyQA model client: %s", exc)
        progress_display.close()
