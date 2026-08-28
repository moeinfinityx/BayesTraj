from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..baselines import OUTCOME_ENTROPY_BASELINE
from ..cli_progress import RunProgressDisplay
from ..config import ExperimentConfig
from ..estimators import UPropEstimator
from ..executors import AgentBenchWebShopControllerClient, AgentBenchWebShopExecutor, AgentBenchWebShopSample
from ..executors.agentbench import AgentBenchControllerError
from ..experiments import ExperimentTracker
from ..logging_utils import configure_logging
from ..models import ChatModelConfig, ModelGenerationError, close_chat_model, create_chat_model
from ..results_summary import resolve_record_uncertainty
from ..sampling import SharedSamplingStorage
from ..trajectory import (
    MultiStepBaselineEstimate,
    TDPCounterfactualBranch,
    TDPCounterfactualRecord,
    TDPStepRecord,
    TrajectoryDependentDecisionProcess,
    UPropEstimate,
)
from .agentbench import (
    AgentBenchRunnerConfig,
    _RESUME_IGNORED_CONFIG_KEYS,
    _agentbench_model_signature,
    _agentbench_task_family,
    _agentbench_variant_name_suffix,
    _default_run_name,
    _DISABLE_HARD_FINALIZATION_ENV,
    _delete_checkpoint,
    _format_uncertainty_value,
    _load_agentbench_results,
    _load_checkpoint_records as _load_agentbench_checkpoint_records,
    _majority_bool,
    _majority_final_answer,
    _merge_record_info,
    _normalize_accuracy_flag,
    _serialize_generation_failure,
    _wrap_model_client_for_tracking,
    _write_checkpoint,
    summarize_agentbench_dbbench_results,
    write_agentbench_dbbench_results,
)
from .logprob_resume import (
    filter_stale_records,
    probe_structured_logprob_support,
    record_has_missing_logprob_state,
)


LOGGER = logging.getLogger(__name__)

_WEBSHOP_SHARED_TDP_CACHE_CATEGORY = "tdp_trajectories"
_WEBSHOP_SHARED_TDP_CACHE_VERSION = 1
_WEBSHOP_SUCCESS_REWARD_THRESHOLD = 1.0 - 1e-9
_WEBSHOP_PAGE_NAVIGATION_CLICKS = {
    "< prev",
    "back to search",
    "next >",
    "prev",
    "search",
}
_WEBSHOP_PRODUCT_DETAIL_CLICKS = {
    "description",
    "features",
    "reviews",
}
_WEBSHOP_NON_OPTION_CLICKS = {
    *_WEBSHOP_PAGE_NAVIGATION_CLICKS,
    *_WEBSHOP_PRODUCT_DETAIL_CLICKS,
    "buy now",
}


@dataclass(frozen=True)
class AgentBenchWebShopRunnerConfig(AgentBenchRunnerConfig):
    task_name: str = "webshop-std"
    max_steps: int = 20


def _resume_config_payload(config: AgentBenchWebShopRunnerConfig | Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config) if isinstance(config, Mapping) else asdict(config)
    for key in _RESUME_IGNORED_CONFIG_KEYS:
        payload.pop(key, None)
    payload.pop("restart", None)
    return payload


def _load_checkpoint_runner_config(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        runner_config = payload.get("runner_config")
        if isinstance(runner_config, dict):
            return runner_config

    run_path = checkpoint_path.parent.parent / "run.json"
    if not run_path.exists():
        return None
    try:
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    runner_config = metadata.get("runner_config")
    return runner_config if isinstance(runner_config, dict) else None


def _deserialize_counterfactual_branch(payload: Any) -> TDPCounterfactualBranch:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached AgentBench WebShop counterfactual branch.")
    metadata = payload.get("metadata")
    output_metadata = payload.get("target_sampled_output_metadata")
    return TDPCounterfactualBranch(
        source_decision=str(payload.get("source_decision", "")),
        target_sampled_decisions=[str(value) for value in payload.get("target_sampled_decisions", [])],
        target_sampled_output_metadata=[dict(value) for value in output_metadata if isinstance(value, Mapping)] if isinstance(output_metadata, list) else [],
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _deserialize_counterfactual_record(payload: Any) -> TDPCounterfactualRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached AgentBench WebShop counterfactual record.")
    metadata = payload.get("metadata")
    branches = payload.get("branches")
    return TDPCounterfactualRecord(
        source_step_index=int(payload.get("source_step_index", 0)),
        realized_source_decision=str(payload.get("realized_source_decision", "")),
        branches=[_deserialize_counterfactual_branch(branch) for branch in branches] if isinstance(branches, list) else [],
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _deserialize_tdp_step(payload: Any) -> TDPStepRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached AgentBench WebShop TDP step.")
    metadata = payload.get("metadata")
    measurements = payload.get("uncertainty_measurements")
    records = payload.get("counterfactual_records")
    return TDPStepRecord(
        index=int(payload.get("index", 0)),
        realized_decision=str(payload.get("realized_decision", "")),
        sampled_decisions=[str(value) for value in payload.get("sampled_decisions", [])],
        uncertainty_measurements={str(key): float(value) for key, value in measurements.items() if isinstance(value, (int, float))} if isinstance(measurements, Mapping) else {},
        counterfactual_records=[_deserialize_counterfactual_record(record) for record in records] if isinstance(records, list) else [],
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _deserialize_tdp(payload: Any) -> TrajectoryDependentDecisionProcess:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached AgentBench WebShop TDP.")
    metadata = payload.get("metadata")
    steps = payload.get("steps")
    return TrajectoryDependentDecisionProcess(
        sample_id=str(payload.get("sample_id", "")),
        prompt=str(payload.get("prompt", "")),
        steps=[_deserialize_tdp_step(step) for step in steps] if isinstance(steps, list) else [],
        final_answer=str(payload.get("final_answer")) if payload.get("final_answer") is not None else None,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _webshop_tdp_cache_enabled(config: AgentBenchWebShopRunnerConfig) -> bool:
    return config.method == "uprop"


def _webshop_hard_finalization_disabled() -> bool:
    return os.getenv(_DISABLE_HARD_FINALIZATION_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _webshop_shared_tdp_cache_key(
    config: AgentBenchWebShopRunnerConfig,
    sample: AgentBenchWebShopSample,
    *,
    emulate_tool_calls: bool | None = None,
) -> dict[str, Any]:
    resolved_emulate_tool_calls = config.emulate_tool_calls if emulate_tool_calls is None else bool(emulate_tool_calls)
    model_signature = dict(_agentbench_model_signature(config))
    model_signature["emulate_tool_calls"] = resolved_emulate_tool_calls
    return {
        "cache_version": _WEBSHOP_SHARED_TDP_CACHE_VERSION,
        "task_name": sample.task_name,
        "task_index": sample.task_index,
        "model": model_signature,
        "temperature": config.temperature,
        "max_steps": config.max_steps,
        "max_tokens": config.max_tokens,
        "tdp_samples": config.tdp_samples,
        "per_step_samples": 1,
        "include_counterfactuals": False,
        "fair_trajectory_budget": True,
        "hard_finalization_disabled": _webshop_hard_finalization_disabled(),
        "emulate_tool_calls": resolved_emulate_tool_calls,
        "executor": "agentbench_webshop",
    }


def _load_webshop_shared_tdp_cache(
    storage: SharedSamplingStorage,
    *,
    config: AgentBenchWebShopRunnerConfig,
    sample: AgentBenchWebShopSample,
) -> list[TrajectoryDependentDecisionProcess] | None:
    if not config.fair_trajectory_budget:
        return None
    if not _webshop_tdp_cache_enabled(config):
        return None
    payload = None
    cache_key_emulate_tool_calls = config.emulate_tool_calls
    for candidate_emulate_tool_calls in (
        [config.emulate_tool_calls, False]
        if config.emulate_tool_calls
        else [config.emulate_tool_calls]
    ):
        payload = storage.load(
            _WEBSHOP_SHARED_TDP_CACHE_CATEGORY,
            sample_id=sample.sample_id,
            key=_webshop_shared_tdp_cache_key(config, sample, emulate_tool_calls=candidate_emulate_tool_calls),
        )
        if isinstance(payload, Mapping):
            cache_key_emulate_tool_calls = bool(candidate_emulate_tool_calls)
            break
    if not isinstance(payload, Mapping):
        return None
    raw_tdps = payload.get("tdps")
    required_tdps = config.tdp_samples
    if not isinstance(raw_tdps, list) or len(raw_tdps) < required_tdps:
        return None
    try:
        tdps = [_deserialize_tdp(item) for item in raw_tdps[: required_tdps]]
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Ignoring invalid WebShop shared TDP cache for sample_id=%s: %s", sample.sample_id, exc)
        return None
    source_method = str(payload.get("source_method") or "unknown")
    for tdp in tdps:
        tdp.metadata["tdp_cache_status"] = "hit"
        tdp.metadata["tdp_cache_source_method"] = source_method
        tdp.metadata["tdp_cache_key_emulate_tool_calls"] = cache_key_emulate_tool_calls
        if cache_key_emulate_tool_calls != config.emulate_tool_calls:
            tdp.metadata["tdp_cache_status"] = "hit_legacy_emulate_false"
    return tdps


async def _extend_webshop_cached_tdps_for_degree(
    *,
    config: AgentBenchWebShopRunnerConfig,
    sample: AgentBenchWebShopSample,
    executor: AgentBenchWebShopExecutor,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
) -> tuple[list[TrajectoryDependentDecisionProcess], int, int]:
    per_step_samples = max(config.per_step_samples, _DEGREE_SHARED_CACHE_PER_STEP_SAMPLES)
    extended_tdps: list[TrajectoryDependentDecisionProcess] = []
    extended_count = 0
    for position, tdp in enumerate(tdps):
        cached_per_step_samples = tdp.metadata.get("per_step_samples")
        if isinstance(cached_per_step_samples, int) and cached_per_step_samples >= per_step_samples:
            extended_tdps.append(tdp)
            continue
        trajectory_index = tdp.metadata.get("trajectory_index")
        if not isinstance(trajectory_index, int) or trajectory_index < 0:
            trajectory_index = position
        extended_tdp = await executor._extend_cached_tdp(
            sample=sample,
            trajectory_index=trajectory_index,
            cached_tdp=tdp,
            per_step_samples=per_step_samples,
            include_counterfactuals=False,
        )
        extended_tdp.metadata["tdp_cache_status"] = "hit_extended"
        extended_tdp.metadata["tdp_cache_source_method"] = str(tdp.metadata.get("tdp_cache_source_method") or "pe")
        extended_tdp.metadata["tdp_cache_extended_from_per_step_samples"] = cached_per_step_samples
        extended_tdp.metadata["tdp_cache_extended_to_per_step_samples"] = per_step_samples
        extended_tdps.append(extended_tdp)
        extended_count += 1
    return extended_tdps, extended_count, per_step_samples


def _store_webshop_shared_tdp_cache(
    storage: SharedSamplingStorage,
    *,
    config: AgentBenchWebShopRunnerConfig,
    sample: AgentBenchWebShopSample,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
) -> bool:
    if config.method != "pe" or not _webshop_tdp_cache_enabled(config):
        return False
    if not config.fair_trajectory_budget:
        return False
    try:
        storage.store(
            _WEBSHOP_SHARED_TDP_CACHE_CATEGORY,
            sample_id=sample.sample_id,
            key=_webshop_shared_tdp_cache_key(config, sample),
            value={
                "source_method": "pe",
                "tdps": [asdict(tdp) for tdp in tdps],
            },
        )
    except OSError as exc:
        LOGGER.warning("Failed to write WebShop shared TDP cache for sample_id=%s: %s", sample.sample_id, exc)
        return False
    return True


def _truncate_counterfactual_branch(branch: TDPCounterfactualBranch, per_step_samples: int) -> TDPCounterfactualBranch:
    return TDPCounterfactualBranch(
        source_decision=branch.source_decision,
        target_sampled_decisions=list(branch.target_sampled_decisions[:per_step_samples]),
        target_sampled_output_metadata=[dict(value) for value in branch.target_sampled_output_metadata[:per_step_samples]],
        metadata=dict(branch.metadata),
    )


def _truncate_counterfactual_record(record: TDPCounterfactualRecord, per_step_samples: int) -> TDPCounterfactualRecord:
    return TDPCounterfactualRecord(
        source_step_index=record.source_step_index,
        realized_source_decision=record.realized_source_decision,
        branches=[_truncate_counterfactual_branch(branch, per_step_samples) for branch in record.branches],
        metadata=dict(record.metadata),
    )


def _truncate_tdp_step(step: TDPStepRecord, per_step_samples: int) -> TDPStepRecord:
    metadata = dict(step.metadata)
    sampled_output_metadata = metadata.get("sampled_output_metadata")
    if isinstance(sampled_output_metadata, list):
        metadata["sampled_output_metadata"] = [dict(value) for value in sampled_output_metadata[:per_step_samples] if isinstance(value, Mapping)]
    chosen_output_index = metadata.get("chosen_output_index")
    if isinstance(chosen_output_index, int) and per_step_samples > 0:
        metadata["chosen_output_index"] = min(chosen_output_index, per_step_samples - 1)
    return TDPStepRecord(
        index=step.index,
        realized_decision=step.realized_decision,
        sampled_decisions=list(step.sampled_decisions[:per_step_samples]),
        uncertainty_measurements=dict(step.uncertainty_measurements),
        counterfactual_records=[_truncate_counterfactual_record(record, per_step_samples) for record in step.counterfactual_records],
        metadata=metadata,
    )


def _truncate_tdp(tdp: TrajectoryDependentDecisionProcess, *, per_step_samples: int) -> TrajectoryDependentDecisionProcess:
    return TrajectoryDependentDecisionProcess(
        sample_id=tdp.sample_id,
        prompt=tdp.prompt,
        steps=[_truncate_tdp_step(step, per_step_samples) for step in tdp.steps],
        final_answer=tdp.final_answer,
        metadata=dict(tdp.metadata),
    )


def _recompute_uprop_record_for_budget(
    record: dict[str, Any],
    *,
    sample: AgentBenchWebShopSample,
    tdp_samples: int,
    per_step_samples: int,
) -> dict[str, Any]:
    """Recompute the reported UProp score from a nested cached prefix."""
    estimate_payload = record.get("estimate")
    if not isinstance(estimate_payload, Mapping):
        return dict(record)
    tdps_payload = estimate_payload.get("tdps")
    if not isinstance(tdps_payload, list) or len(tdps_payload) < tdp_samples:
        raise ValueError("Cached WebShop UProp record does not contain enough TDPs for the requested sampling budget.")
    metadata = estimate_payload.get("metadata")
    resolved_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    tdps = [
        _truncate_tdp(_deserialize_tdp(payload), per_step_samples=per_step_samples)
        for payload in tdps_payload[:tdp_samples]
    ]
    estimator = UPropEstimator(
        trajectory_samples=tdp_samples,
        per_step_samples=per_step_samples,
        tau=float(resolved_metadata.get("tau", 1.0)),
        ratio_epsilon=float(resolved_metadata.get("ratio_epsilon", 1e-6)),
        ratio_cap=resolved_metadata.get("ratio_cap", 10.0),
        intrinsic_cap=resolved_metadata.get("intrinsic_cap"),
        intrinsic_transform=str(resolved_metadata.get("intrinsic_transform") or "none"),
    )
    estimate = estimator.estimate_from_tdps(
        tdps,
        executor_name=str(resolved_metadata.get("executor") or "webshop_cached_prefix"),
        metadata=resolved_metadata,
    )
    return _serialize_uprop_result(sample, estimate)

def _load_checkpoint_records(
    checkpoint_path: Path,
    *,
    config: AgentBenchWebShopRunnerConfig,
    selected_samples: Sequence[AgentBenchWebShopSample],
) -> tuple[list[dict[str, Any]], bool]:
    try:
        return _load_agentbench_checkpoint_records(
            checkpoint_path,
            config=config,
            selected_samples=selected_samples,
        ), False
    except ValueError as exc:
        if config.restart or config.method != "uprop" or not checkpoint_path.exists():
            raise
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise
        expected_sample_ids = [sample.sample_id for sample in selected_samples]
        if payload.get("selected_sample_ids") != expected_sample_ids:
            raise

        previous_config = _load_checkpoint_runner_config(checkpoint_path)
        if previous_config is None:
            raise
        previous_payload = _resume_config_payload(previous_config)
        current_payload = _resume_config_payload(config)
        previous_tdp_samples = previous_payload.pop("tdp_samples", None)
        previous_per_step_samples = previous_payload.pop("per_step_samples", None)
        current_tdp_samples = current_payload.pop("tdp_samples", None)
        current_per_step_samples = current_payload.pop("per_step_samples", None)
        if previous_payload != current_payload:
            raise exc
        if not isinstance(previous_tdp_samples, int) or not isinstance(previous_per_step_samples, int):
            raise exc
        if not isinstance(current_tdp_samples, int) or not isinstance(current_per_step_samples, int):
            raise exc
        if previous_tdp_samples < current_tdp_samples or previous_per_step_samples < current_per_step_samples:
            raise exc
        if previous_tdp_samples == current_tdp_samples and previous_per_step_samples == current_per_step_samples:
            raise exc

        sample_map = {sample.sample_id: sample for sample in selected_samples}
        records = payload.get("records")
        if not isinstance(records, list):
            raise exc
        rewritten = [
            _recompute_uprop_record_for_budget(
                dict(record),
                sample=sample_map[str(record.get("sample_id"))],
                tdp_samples=current_tdp_samples,
                per_step_samples=current_per_step_samples,
            )
            if isinstance(record, dict) and record.get("sample_id") in sample_map
            else record
            for record in records
        ]
        return [dict(record) for record in rewritten if isinstance(record, dict)], True


def _extract_webshop_correctness(result: Any) -> bool | None:
    if not isinstance(result, dict):
        return None
    reward = _extract_webshop_reward(result)
    if reward is not None:
        return reward >= _WEBSHOP_SUCCESS_REWARD_THRESHOLD
    score = result.get("score")
    if isinstance(score, (int, float)):
        return float(score) >= _WEBSHOP_SUCCESS_REWARD_THRESHOLD
    is_correct = _normalize_accuracy_flag(result.get("is_correct"))
    if is_correct is not None:
        return is_correct
    raw_result = result.get("result")
    if isinstance(raw_result, bool):
        return raw_result
    if isinstance(raw_result, (int, float)):
        return float(raw_result) >= _WEBSHOP_SUCCESS_REWARD_THRESHOLD
    return None


def _extract_webshop_reward(result: Any) -> float | None:
    """Return the numeric WebShop reward/score when present."""
    if not isinstance(result, Mapping):
        return None
    for key in ("reward", "score"):
        value = result.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    nested = result.get("result")
    if isinstance(nested, Mapping):
        return _extract_webshop_reward(nested)
    if isinstance(nested, bool):
        return float(nested)
    if isinstance(nested, (int, float)):
        return float(nested)
    return None


def _normalize_webshop_status(status: Any) -> str | None:
    """Normalize AgentBench/WebShop status variants to stable bucket suffixes."""
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def _webshop_prediction_outcome_bucket(tdp: TrajectoryDependentDecisionProcess) -> str:
    """Return a prediction-only WebShop outcome bucket for fair OE scoring.

    This bucket intentionally ignores reward and official correctness. It only
    uses what the sampled trajectory produced: the purchased product/options,
    the last product reached without purchase, the last compact action, or the
    terminal status as a no-answer reason.
    """
    outcome = _webshop_prediction_outcome_state(tdp)
    if outcome:
        return outcome
    metadata = tdp.metadata if isinstance(tdp.metadata, Mapping) else {}
    status = _normalize_webshop_status(metadata.get("status"))
    final_answer = tdp.final_answer.strip() if isinstance(tdp.final_answer, str) else ""
    final_action = _normalize_webshop_action_text(final_answer) if final_answer else ""
    final_kind, final_value = _split_webshop_action(final_action)
    if final_kind == "click" and _is_webshop_buy_now_action(final_value):
        return "purchase:unknown_product|options:none"
    if final_answer:
        return f"answer:{final_action or final_answer}"
    if status:
        return f"no-answer:{status}"
    return "no-answer:unknown"


def _webshop_prediction_outcome_state(tdp: TrajectoryDependentDecisionProcess) -> str | None:
    """Infer the submitted WebShop product/options from prediction artifacts only."""
    current_product: str | None = None
    selected_options: list[str] = []
    last_action: str | None = None
    for step in tdp.steps:
        action = _webshop_step_action(step)
        if not action:
            continue
        last_action = action
        kind, value = _split_webshop_action(action)
        if kind == "search":
            current_product = None
            selected_options = []
            continue
        if kind == "click":
            normalized_value = value.strip()
            product_id = _normalize_webshop_product_id(normalized_value)
            if product_id is not None:
                current_product = product_id
                selected_options = []
                continue
            if _is_webshop_buy_now_action(normalized_value):
                if current_product is not None:
                    return _format_webshop_purchase_bucket(current_product, selected_options)
                return "purchase:unknown_product|options:none"
            if _is_webshop_page_navigation_click(normalized_value):
                current_product = None
                selected_options = []
                continue
            if current_product is not None and _is_webshop_option_click(normalized_value):
                option = _normalize_webshop_option(normalized_value)
                if option not in selected_options:
                    selected_options.append(option)
    if current_product is not None:
        return f"no-purchase:on_product:{current_product}"
    if last_action is not None:
        return f"no-purchase:last_action:{last_action}"
    return None


def _webshop_step_action(step: TDPStepRecord) -> str | None:
    """Return the normalized action chosen for a WebShop TDP step."""
    metadata = step.metadata if isinstance(step.metadata, Mapping) else {}
    chosen_metadata = metadata.get("chosen_output_metadata")
    if isinstance(chosen_metadata, Mapping):
        action = chosen_metadata.get("webshop_action_text")
        if isinstance(action, str) and action.strip():
            return _normalize_webshop_action_text(action)
    return _extract_webshop_action_text(step.realized_decision)


def _split_webshop_action(action: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*(search|click)\[([^\]]*)\]\s*", action, flags=re.IGNORECASE)
    if not match:
        return "", ""
    return match.group(1).lower(), match.group(2).strip()


def _normalize_webshop_product_id(value: str) -> str | None:
    normalized = value.strip().lower()
    if re.fullmatch(r"b[0-9a-z]{9}", normalized):
        return normalized
    return None


def _is_webshop_buy_now_action(value: str) -> bool:
    return value.strip().lower() == "buy now"


def _is_webshop_page_navigation_click(value: str) -> bool:
    return value.strip().lower() in _WEBSHOP_PAGE_NAVIGATION_CLICKS


def _is_webshop_option_click(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    if normalized in _WEBSHOP_NON_OPTION_CLICKS:
        return False
    if _normalize_webshop_product_id(normalized) is not None:
        return False
    return True


def _normalize_webshop_option(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _format_webshop_purchase_bucket(product_id: str, selected_options: Sequence[str]) -> str:
    options = ",".join(sorted(selected_options)) if selected_options else "none"
    return f"purchase:{product_id}|options:{options}"


def _extract_webshop_action_text(text: Any) -> str | None:
    """Extract a compact search[...] or click[...] action from model text."""
    if not isinstance(text, str):
        return None

    tool_matches = list(re.finditer(r"\b(search_action|click_action)\s*\((\{.*?\})\)", text, flags=re.DOTALL))
    for match in reversed(tool_matches):
        name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, Mapping):
            continue
        if name == "search_action":
            keywords = arguments.get("keywords")
            if isinstance(keywords, str) and keywords.strip():
                return _normalize_webshop_action_text(f"search[{keywords.strip()}]")
            continue
        value = arguments.get("value")
        if isinstance(value, str) and value.strip():
            return _normalize_webshop_action_text(f"click[{value.strip()}]")

    direct_matches = list(re.finditer(r"\b(search|click)\[([^\]]+)\]", text, flags=re.IGNORECASE))
    for match in reversed(direct_matches):
        return _normalize_webshop_action_text(f"{match.group(1).lower()}[{match.group(2).strip()}]")
    return None


def _normalize_webshop_action_text(action: str) -> str:
    kind, value = _split_webshop_action(action)
    if not kind:
        return action.strip()
    if kind == "click":
        product_id = _normalize_webshop_product_id(value)
        if product_id is not None:
            value = product_id
        elif _is_webshop_buy_now_action(value):
            value = "Buy Now"
        else:
            value = " ".join(value.strip().split())
    else:
        value = " ".join(value.strip().split()).lower()
    return f"{kind}[{value}]"


def _entropy_from_counts(counts: Mapping[str, Any]) -> float | None:
    total = float(sum(max(0.0, float(count)) for count in counts.values() if isinstance(count, (int, float))))
    if total <= 0.0:
        return None
    entropy = 0.0
    for count in counts.values():
        if not isinstance(count, (int, float)):
            continue
        probability = max(0.0, float(count)) / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return float(entropy)


def _estimate_webshop_outcome_entropy_from_tdps(
    *,
    config: AgentBenchWebShopRunnerConfig,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    cache_status: str,
) -> MultiStepBaselineEstimate:
    return MultiStepBaselineEstimate(
        total_uncertainty=None,
        baseline_name=OUTCOME_ENTROPY_BASELINE,
        aggregation="outcome",
        tdp_scores=[],
        tdp_step_scores=[],
        tdps=list(tdps),
        metadata={
            "executor": "agentbench_webshop",
            "num_tdps": len(tdps),
            "per_step_samples": 1,
            "requested_per_step_samples": config.per_step_samples,
            "uncertainty_key": "webshop_prediction_outcome_entropy",
            "requires_logprobs": False,
            "uncertainty_available": False,
            "hard_finalization_count": 0,
            "fair_trajectory_budget": config.fair_trajectory_budget,
            "tdp_cache_status": cache_status,
            "tdp_cache_source_method": "pe" if cache_status == "hit" else None,
            "used_no_tool_call_samples": any(bool(tdp.metadata.get("used_no_tool_call_samples")) for tdp in tdps),
            "no_tool_call_sample_count": sum(
                [count for tdp in tdps if isinstance((count := tdp.metadata.get("no_tool_call_sample_count")), int)]
            ),
            "hard_requirement_failure": any(bool(tdp.metadata.get("hard_requirement_failure")) for tdp in tdps),
        },
    )


def _webshop_judge_score_from_record(
    record: dict[str, Any],
    *,
    confidence_key: str,
) -> float | None:
    uncertainty = record.get("uncertainty")
    if isinstance(uncertainty, (int, float)) and not isinstance(uncertainty, bool) and math.isfinite(float(uncertainty)):
        return max(0.0, min(1.0, float(uncertainty)))
    confidence = record.get(confidence_key)
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(float(confidence)):
        score = 1.0 - max(0.0, min(1.0, float(confidence)))
        record["uncertainty"] = score
        return score
    return None


def _webshop_cached_tdp_metadata(
    *,
    config: AgentBenchWebShopRunnerConfig,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    cache_status: str,
    uncertainty_key: str,
    valid_tdp_scores: int,
) -> dict[str, Any]:
    return {
        "executor": "agentbench_webshop",
        "num_tdps": len(tdps),
        "valid_tdp_scores": valid_tdp_scores,
        "per_step_samples": 1,
        "requested_per_step_samples": config.per_step_samples,
        "uncertainty_key": uncertainty_key,
        "requires_logprobs": False,
        "uncertainty_available": valid_tdp_scores > 0,
        "hard_finalization_count": 0,
        "fair_trajectory_budget": config.fair_trajectory_budget,
        "tdp_cache_status": cache_status,
        "tdp_cache_source_method": "pe" if cache_status == "hit" else None,
        "used_no_tool_call_samples": any(bool(tdp.metadata.get("used_no_tool_call_samples")) for tdp in tdps),
        "no_tool_call_sample_count": sum(
            [count for tdp in tdps if isinstance((count := tdp.metadata.get("no_tool_call_sample_count")), int)]
        ),
        "hard_requirement_failure": any(bool(tdp.metadata.get("hard_requirement_failure")) for tdp in tdps),
    }


def _webshop_outcome_evaluation(estimate: Any) -> dict[str, Any]:
    """Compute prediction-only OE diagnostics for WebShop."""
    prediction_counts: dict[str, int] = {}
    tdp_results: list[dict[str, Any]] = []
    for tdp in estimate.tdps:
        prediction_bucket = _webshop_prediction_outcome_bucket(tdp)
        prediction_counts[prediction_bucket] = prediction_counts.get(prediction_bucket, 0) + 1
        metadata = tdp.metadata if isinstance(tdp.metadata, Mapping) else {}
        tdp_results.append(
            {
                "sample_id": tdp.sample_id,
                "final_answer": tdp.final_answer,
                "status": metadata.get("status"),
                "webshop_outcome": prediction_bucket,
            }
        )

    prediction_entropy = _entropy_from_counts(prediction_counts)
    return {
        "webshop_outcome_counts": prediction_counts,
        "webshop_outcome_entropy": prediction_entropy,
        "webshop_prediction_outcome_counts": prediction_counts,
        "webshop_prediction_outcome_entropy": prediction_entropy,
        "webshop_prediction_outcome_uncertainty": prediction_entropy,
        "webshop_outcome_uncertainty": prediction_entropy,
        "webshop_outcome_evaluation": {
            "uncertainty_key": "webshop_prediction_outcome_entropy",
            "outcome_counts": prediction_counts,
            "tdps": tdp_results,
        },
    }


def _apply_webshop_prediction_outcome_uncertainty(record: dict[str, Any]) -> None:
    """Prefer prediction-only WebShop outcome entropy for outcome-entropy runs."""
    if record.get("method") != OUTCOME_ENTROPY_BASELINE:
        return
    uncertainty = record.get("webshop_prediction_outcome_uncertainty")
    if not isinstance(uncertainty, (int, float)):
        return
    original_uncertainty = record.get("uncertainty")
    record["uncertainty"] = float(uncertainty)
    info = record.get("info")
    if not isinstance(info, dict):
        info = {}
        record["info"] = info
    info["original_baseline_uncertainty"] = original_uncertainty
    info["original_prompt_heuristic_uncertainty"] = original_uncertainty
    info["uncertainty_key"] = "webshop_prediction_outcome_entropy"
    info["webshop_outcome_counts"] = record.get("webshop_outcome_counts")
    info["webshop_outcome_entropy"] = record.get("webshop_outcome_entropy")

    estimate = record.get("estimate")
    if isinstance(estimate, dict):
        metadata = estimate.get("metadata")
        if isinstance(metadata, dict):
            metadata["original_baseline_uncertainty"] = original_uncertainty
            metadata["original_prompt_heuristic_uncertainty"] = original_uncertainty
            metadata["uncertainty_key"] = "webshop_prediction_outcome_entropy"
            metadata["webshop_outcome_counts"] = record.get("webshop_outcome_counts")
            metadata["webshop_outcome_entropy"] = record.get("webshop_outcome_entropy")


def _serialize_uprop_result(sample: AgentBenchWebShopSample, estimate: Any) -> dict[str, Any]:
    predicted_answer = _majority_final_answer([tdp.final_answer for tdp in estimate.tdps])
    is_correct = _majority_bool(
        [_extract_webshop_correctness(tdp.metadata.get("result")) for tdp in estimate.tdps]
    )
    status = _majority_final_answer(
        [str(tdp.metadata.get("status")) if tdp.metadata.get("status") is not None else None for tdp in estimate.tdps]
    )
    return {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "task_index": sample.task_index,
        "question": estimate.tdps[0].prompt if estimate.tdps else None,
        "method": "uprop",
        "status": status,
        "reference_answer": None,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "uncertainty": float(estimate.total_uncertainty),
        "error": None,
        "info": _merge_record_info(None, estimate.metadata, [tdp.metadata for tdp in estimate.tdps]),
        "estimate": asdict(estimate),
    }


def _serialize_baseline_result(sample: AgentBenchWebShopSample, estimate: Any) -> dict[str, Any]:
    predicted_answer = _majority_final_answer([tdp.final_answer for tdp in estimate.tdps])
    is_correct = _majority_bool(
        [_extract_webshop_correctness(tdp.metadata.get("result")) for tdp in estimate.tdps]
    )
    status = _majority_final_answer(
        [str(tdp.metadata.get("status")) if tdp.metadata.get("status") is not None else None for tdp in estimate.tdps]
    )
    record = {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "task_index": sample.task_index,
        "question": estimate.tdps[0].prompt if estimate.tdps else None,
        "method": estimate.baseline_name,
        "status": status,
        "reference_answer": None,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "uncertainty": float(estimate.total_uncertainty) if estimate.total_uncertainty is not None else None,
        "error": None,
        "info": _merge_record_info(None, estimate.metadata, [tdp.metadata for tdp in estimate.tdps]),
        "estimate": asdict(estimate),
    }
    if estimate.baseline_name in {OUTCOME_ENTROPY_BASELINE, "__unavailable_method"}:
        record.update(_webshop_outcome_evaluation(estimate))
        if estimate.baseline_name == OUTCOME_ENTROPY_BASELINE:
            _apply_webshop_prediction_outcome_uncertainty(record)
    return record


async def run_agentbench_webshop_samples(
    samples: Sequence[AgentBenchWebShopSample],
    *,
    model_client: Any,
    config: AgentBenchWebShopRunnerConfig,
    tracker: ExperimentTracker | None = None,
    sample_offset: int = 0,
    sample_total: int | None = None,
    record_callback: Callable[[AgentBenchWebShopSample, dict[str, Any], int, int], None] | None = None,
    controller_client: AgentBenchWebShopControllerClient | None = None,
) -> list[dict[str, Any]]:
    """Generate the paper's UProp trajectory pools for WebShop."""
    active_controller = controller_client or AgentBenchWebShopControllerClient(
        config.task_name,
        config.controller_url,
    )
    shared_sampling_storage = SharedSamplingStorage(config.shared_sampling_dir)
    executor = AgentBenchWebShopExecutor(
        active_controller,
        model_client,
        temperature=config.temperature,
        backbone_per_step_samples=config.backbone_per_step_samples,
        next_step_entropy_samples=config.next_step_entropy_samples,
        max_steps=config.max_steps,
        collect_sample_logprobs=True,
        shared_sampling_storage=shared_sampling_storage,
        model_signature=_agentbench_model_signature(config),
        strict_replay=config.strict_replay,
    )
    estimator = UPropEstimator(
        trajectory_samples=config.tdp_samples,
        per_step_samples=config.per_step_samples,
        tau=config.tau,
        ratio_epsilon=config.uprop_ratio_epsilon,
        ratio_cap=config.uprop_ratio_cap,
        intrinsic_cap=config.uprop_intrinsic_cap,
        intrinsic_transform=config.uprop_intrinsic_transform,
    )
    records: list[dict[str, Any]] = []
    total_samples = sample_total or len(samples)
    for index, sample in enumerate(samples, start=sample_offset + 1):
        LOGGER.info(
            "Running AgentBench WebShop sample %s/%s with UProp: task=%s index=%s",
            index,
            total_samples,
            sample.task_name,
            sample.task_index,
        )
        if tracker is not None:
            tracker.log_event(
                "sample_started",
                sample_id=sample.sample_id,
                position=index,
                total_samples=total_samples,
                method="uprop",
            )
        try:
            cached_tdps = _load_webshop_shared_tdp_cache(
                shared_sampling_storage,
                config=config,
                sample=sample,
            )
            if cached_tdps is None:
                estimate = await estimator.estimate(sample=sample, executor=executor)
                estimate.metadata.setdefault("tdp_cache_status", "miss")
            else:
                estimate = estimator.estimate_from_tdps(
                    list(cached_tdps),
                    executor_name=executor.name,
                    metadata={
                        "tdp_cache_status": "hit",
                        "tdp_cache_source_method": str(
                            cached_tdps[0].metadata.get("tdp_cache_source_method") or "uprop"
                        ),
                        "fair_trajectory_budget": config.fair_trajectory_budget,
                    },
                )
            record = _serialize_uprop_result(sample, estimate)
        except ModelGenerationError as exc:
            LOGGER.warning(
                "WebShop sample_id=%s failed due to model error code=%s retryable=%s",
                sample.sample_id,
                exc.error_code,
                exc.retryable,
            )
            record = _serialize_generation_failure(sample, method="uprop", error=exc)
            if tracker is not None:
                tracker.log_event(
                    "sample_failed",
                    sample_id=sample.sample_id,
                    position=index,
                    method="uprop",
                    error=str(exc),
                    error_code=exc.error_code,
                    retryable=exc.retryable,
                )
        else:
            if tracker is not None:
                tracker.log_event(
                    "sample_completed",
                    sample_id=sample.sample_id,
                    position=index,
                    uncertainty=record["uncertainty"],
                    is_correct=record["is_correct"],
                    status=record["status"],
                )
        records.append(record)
        if record_callback is not None:
            record_callback(sample, record, index, total_samples)
    return records


def summarize_agentbench_webshop_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return summarize_agentbench_dbbench_results(records)


def write_agentbench_webshop_results(output_path: str, records: Sequence[dict[str, Any]]) -> Path:
    return write_agentbench_dbbench_results(output_path, records)


def _build_experiment_name(config: AgentBenchWebShopRunnerConfig) -> str:
    if config.experiment_name:
        return config.experiment_name
    variant_suffix = _agentbench_variant_name_suffix(emulate_tool_calls=config.emulate_tool_calls)
    return f"agentbench-webshop-{config.method}-{config.model.replace('/', '-')}-{config.task_name}{variant_suffix}"


def _resolve_output_path(config: AgentBenchWebShopRunnerConfig, tracker: ExperimentTracker | None) -> str:
    if config.output_path is not None:
        return config.output_path
    if tracker is not None:
        return str(tracker.artifacts_dir / "results.jsonl")
    variant_suffix = _agentbench_variant_name_suffix(emulate_tool_calls=config.emulate_tool_calls)
    return f"outputs/agentbench-webshop-{config.method}-{config.model.replace('/', '-')}-{config.task_name}{variant_suffix}.jsonl"


def _checkpoint_path(output_path: str, tracker: ExperimentTracker | None) -> Path:
    if tracker is not None:
        return tracker.artifacts_dir / "checkpoint.json"
    return Path(f"{output_path}.checkpoint.json")


def _build_tracker(config: AgentBenchWebShopRunnerConfig) -> ExperimentTracker | None:
    if not config.track_experiment:
        return None
    return ExperimentTracker(
        experiment=ExperimentConfig(
            name=_build_experiment_name(config),
            model=config.model,
            task_family=_agentbench_task_family("agentbench_webshop", emulate_tool_calls=config.emulate_tool_calls),
            output_dir=config.tracking_dir,
            notes=config.notes,
        ),
        tracking_dir=config.tracking_dir,
        run_name=config.run_name or _default_run_name(config),
        tags=config.tags,
        metadata={"runner_config": asdict(config)},
        resume=not config.restart,
    )


async def run_agentbench_webshop(
    config: AgentBenchWebShopRunnerConfig,
    *,
    controller_client: AgentBenchWebShopControllerClient | None = None,
) -> dict[str, Any]:
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
        "Starting AgentBench webshop run: method=%s model=%s provider=%s task=%s task_index=%s limit=%s offset=%s",
        config.method,
        config.model,
        config.provider,
        config.task_name,
        config.task_index,
        config.limit,
        config.offset,
    )
    if tracker is not None:
        tracker.log_event(
            "run_started",
            method=config.method,
            model=config.model,
            provider=config.provider,
            task_name=config.task_name,
        )
        if tracker.resumed:
            LOGGER.info("Resuming tracked run in %s", tracker.run_dir)
            tracker.log_event("run_resumed", previous_status=tracker.previous_status)

    active_controller = controller_client or AgentBenchWebShopControllerClient(config.task_name, config.controller_url)
    if config.task_index is not None:
        selected_indices = [config.task_index]
    else:
        indices = await asyncio.to_thread(active_controller.get_indices)
        selected_indices = indices[config.offset :]
        if config.limit is not None:
            selected_indices = selected_indices[: config.limit]

    if not selected_indices:
        raise ValueError("No AgentBench webshop samples were selected for the requested task_index/offset/limit.")

    selected_samples = [
        AgentBenchWebShopSample(
            controller_url=config.controller_url,
            task_name=config.task_name,
            task_index=task_index,
        )
        for task_index in selected_indices
    ]
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
            isinstance(record, dict) and record_has_missing_logprob_state(record, method=config.method)
            for record in records
        ):
            return False
        if model_client is None:
            model_client = create_chat_model(
                ChatModelConfig(
                    provider=config.provider,
                    model=config.model,
                    api_key=config.api_key,
                    emulate_tool_calls=config.emulate_tool_calls,
                    max_tokens=config.max_tokens,
                    parallel_requests=config.parallel_requests,
                    base_url=config.base_url,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.api_version,
                    deployment_name=config.deployment_name,
                )
            )
        refresh_missing_logprob_records = await probe_structured_logprob_support(
            model_client,
            probe_kind="tool",
            logger=LOGGER,
            model=config.model,
            provider=config.provider,
        )
        return refresh_missing_logprob_records

    checkpoint_records, migrated_checkpoint_records = _load_checkpoint_records(
        checkpoint_path,
        config=config,
        selected_samples=selected_samples,
    )
    if migrated_checkpoint_records:
        _write_checkpoint(
            checkpoint_path,
            config=config,
            selected_samples=selected_samples,
            output_path=output_path,
            records=checkpoint_records,
        )
        LOGGER.info(
            "Recomputed %s AgentBench webshop checkpoint record(s) for a smaller UProp sampling budget without new LLM sampling",
            len(checkpoint_records),
        )
    dropped_checkpoint_sample_ids: list[str] = []
    if checkpoint_records:
        refresh_checkpoint_records = await ensure_refresh_decision(checkpoint_records)
        checkpoint_records, dropped_checkpoint_sample_ids = filter_stale_records(
            checkpoint_records,
            refresh_missing_logprob_records=refresh_checkpoint_records,
            resolve_uncertainty=resolve_record_uncertainty,
            method=config.method,
        )
        if dropped_checkpoint_sample_ids:
            LOGGER.info(
                "Dropping %s stale AgentBench webshop checkpoint record(s) without logprob state: %s",
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
        LOGGER.info(
            "Loaded %s checkpointed AgentBench webshop records from %s",
            len(checkpoint_records),
            checkpoint_path,
        )
        if tracker is not None:
            tracker.log_event("checkpoint_loaded", records=len(checkpoint_records), checkpoint_path=str(checkpoint_path))

    if (
        tracker is not None
        and not config.restart
        and tracker.previous_status == "completed"
        and not checkpoint_path.exists()
        and Path(output_path).exists()
    ):
        existing_records = _load_agentbench_results(output_path)
        existing_ids = [record.get("sample_id") for record in existing_records]
        selected_ids = [sample.sample_id for sample in selected_samples]
        if existing_records and existing_ids == selected_ids:
            refresh_existing_records = await ensure_refresh_decision(existing_records)
            reusable_existing_records, dropped_existing_sample_ids = filter_stale_records(
                existing_records,
                refresh_missing_logprob_records=refresh_existing_records,
                resolve_uncertainty=resolve_record_uncertainty,
                method=config.method,
            )
            if not dropped_existing_sample_ids:
                progress_display.start(
                    total=len(selected_samples),
                    description=f"AgentBench webshop {config.method}",
                    completed=len(reusable_existing_records),
                )
                LOGGER.info("Existing completed run found at %s; reusing recorded results", output_path)
                summary = summarize_agentbench_webshop_results(reusable_existing_records)
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
                "Dropping %s stale AgentBench webshop result record(s) without logprob state from %s",
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
        description=f"AgentBench webshop {config.method}",
        completed=len(all_records),
    )
    LOGGER.info("AgentBench webshop resume state: %s completed, %s pending", len(all_records), len(pending_samples))
    if tracker is not None:
        tracker.log_event(
            "resume_state",
            completed_records=len(all_records),
            pending_records=len(pending_samples),
        )

    def persist_record(sample: AgentBenchWebShopSample, record: dict[str, Any], index: int, total: int) -> None:
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
            model_client = create_chat_model(
                ChatModelConfig(
                    provider=config.provider,
                    model=config.model,
                    api_key=config.api_key,
                    emulate_tool_calls=config.emulate_tool_calls,
                    max_tokens=config.max_tokens,
                    parallel_requests=config.parallel_requests,
                    base_url=config.base_url,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.api_version,
                    deployment_name=config.deployment_name,
                )
            )
        model_client = _wrap_model_client_for_tracking(
            model_client,
            tracker=tracker,
            provider=config.provider,
        )
        if pending_samples:
            record_count_before_run = len(all_records)
            new_records = await run_agentbench_webshop_samples(
                pending_samples,
                model_client=model_client,
                config=config,
                tracker=tracker,
                sample_offset=len(all_records),
                sample_total=len(selected_samples),
                controller_client=active_controller,
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

        summary = summarize_agentbench_webshop_results(all_records)
        written_path = write_agentbench_webshop_results(output_path, all_records)
        _delete_checkpoint(checkpoint_path)
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

        progress_display.close(status="completed")
        return {
            "summary": summary,
            "output_path": str(written_path),
            "tracking_path": str(tracker.run_dir) if tracker is not None else None,
            "records": all_records,
        }
    except Exception as exc:
        LOGGER.exception("AgentBench webshop run failed")
        progress_display.close(status="failed")
        if tracker is not None:
            tracker.log_event("run_failed", error=str(exc))
            tracker.finalize(
                status="failed",
                summary=summarize_agentbench_webshop_results(all_records),
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
            LOGGER.warning("Failed to close AgentBench webshop model client: %s", exc)
        progress_display.close()
