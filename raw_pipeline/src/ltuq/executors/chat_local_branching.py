from __future__ import annotations

import inspect
import json
import logging
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from ..logprobs import compute_predictive_entropy_from_metadata
from ..sampling import SharedSamplingStorage, SharedStepSampler, StepSample, reusable_model_signature, sampling_fingerprint
from ..trajectory import (
    BackboneTrajectory,
    LocalBranchHistory,
    StepRecord,
    TDPCounterfactualBranch,
    TDPCounterfactualRecord,
    TDPStepRecord,
    TrajectoryDependentDecisionProcess,
)
from .base import BranchingExecutor


LOGGER = logging.getLogger(__name__)

_CHAT_LOCAL_BRANCHING_CACHE_VERSION = 5
_CHAT_LOCAL_BRANCHING_STEP_SAMPLING_VERSION = 7
_REQUIRE_FROZEN_TDP_CACHE_ENV_VAR = "LTUQ_REQUIRE_FROZEN_TDP_CACHE"


@dataclass(frozen=True)
class ParsedChatStep:
    thought: str | None
    action: str
    raw_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampledChatStep:
    index: int
    chosen_step: ParsedChatStep
    sampled_outputs: list[str]
    sampled_actions: list[str]
    sampled_output_metadata: list[dict[str, Any]]
    chosen_output_index: int
    entropy: float
    observation: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampledChatRollout:
    prompt: str
    steps: list[SampledChatStep]
    final_answer: str | None
    metadata: dict[str, Any]


class ChatLocalBranchingExecutor(BranchingExecutor):
    _ACTION_PATTERN = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
    _ACTION_LINE_PATTERN = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    _THOUGHT_LINE_PATTERN = re.compile(r"^\s*Thought\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    _THOUGHT_PATTERN = re.compile(
        r"Thought\s*:\s*(.+?)(?:\n\s*Action\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _BRACKET_ACTION_PATTERN = re.compile(r"(Search\[[^\]]*\]|Lookup\[[^\]]*\]|Finish\[[^\]]*\])", re.IGNORECASE)
    _PLACEHOLDER_ACTION_ARGUMENTS = {
        "",
        "entity",
        "<entity>",
        "keyword",
        "<keyword>",
        "yes/no",
        "yes or no",
        "yes / no",
        "short answer",
        "<short answer>",
        "answer",
        "<answer>",
    }

    def __init__(
        self,
        model_client: Any,
        prompt_template: Any,
        *,
        max_steps: int = 6,
        sample_temperature: float = 1.0,
        backbone_per_step_samples: int = 4,
        next_step_entropy_samples: int | None = None,
        collect_sample_logprobs: bool = False,
        shared_sampling_storage: SharedSamplingStorage | None = None,
        model_signature: dict[str, Any] | None = None,
        step_output_format: str = "text",
        step_retry_attempts: int = 0,
        step_stop_callback: Callable[..., Any] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if sample_temperature < 0.0:
            raise ValueError("sample_temperature must be non-negative")
        if backbone_per_step_samples <= 0:
            raise ValueError("backbone_per_step_samples must be positive")
        resolved_next_step_samples = backbone_per_step_samples if next_step_entropy_samples is None else next_step_entropy_samples
        if resolved_next_step_samples <= 0:
            raise ValueError("next_step_entropy_samples must be positive")
        if step_output_format not in {"text", "structured_json"}:
            raise ValueError("step_output_format must be 'text' or 'structured_json'")
        if step_retry_attempts < 0:
            raise ValueError("step_retry_attempts must be non-negative")

        self.model_client = model_client
        self.prompt_template = prompt_template
        self.max_steps = max_steps
        self.sample_temperature = sample_temperature
        self.backbone_per_step_samples = backbone_per_step_samples
        self.next_step_entropy_samples = resolved_next_step_samples
        self.collect_sample_logprobs = collect_sample_logprobs
        self.shared_sampling_storage = shared_sampling_storage
        self.shared_step_sampler = SharedStepSampler(shared_sampling_storage)
        self.model_signature = reusable_model_signature(model_signature or self._default_model_signature())
        self.step_output_format = step_output_format
        self.step_retry_attempts = step_retry_attempts
        self.step_stop_callback = step_stop_callback

    async def sample_backbone(self, sample: Any) -> BackboneTrajectory:
        cache_key = self._backbone_cache_key(sample)
        cached_backbone = self._load_cached_backbone(sample, cache_key)
        if cached_backbone is not None:
            return cached_backbone

        rollout = await self._sample_rollout(
            sample=sample,
            rollout_id="backbone",
            per_step_samples=self.backbone_per_step_samples,
        )
        backbone = self._build_backbone_from_rollout(sample, rollout)
        self._store_cached_backbone(sample, cache_key, backbone)
        return backbone

    async def backbone_step_entropies(self, sample: Any, backbone: BackboneTrajectory) -> list[float]:
        del sample
        cached_entropies = backbone.metadata.get("step_entropies")
        if isinstance(cached_entropies, list) and all(
            isinstance(value, (int, float)) for value in cached_entropies
        ):
            return [float(value) for value in cached_entropies]

        recovered_entropies: list[float] = []
        for step in backbone.steps:
            entropy = step.metadata.get("pe")
            if not isinstance(entropy, (int, float)):
                raise ValueError("Backbone trajectory is missing cached step entropies.")
            recovered_entropies.append(float(entropy))
        return recovered_entropies

    async def sample_local_history(
        self,
        sample: Any,
        backbone: BackboneTrajectory,
        step_index: int,
        window: int,
        branch_index: int,
    ) -> LocalBranchHistory:
        if step_index <= 0:
            raise ValueError("step_index must be positive")
        if window <= 0:
            raise ValueError("window must be positive")

        window_start = max(1, step_index - window)
        fixed_prefix_count = window_start - 1
        prefix_steps = list(backbone.steps[:fixed_prefix_count])
        branch_start_index = window_start
        branch_end_index = step_index - 1
        conversation = self._build_conversation_from_steps(sample, prefix_steps)
        initial_branch_cursor = self.backbone_per_step_samples + branch_index

        branched_steps: list[StepRecord] = []
        branch_rollout: SampledChatRollout | None = None
        if branch_end_index >= branch_start_index:
            branch_rollout = await self._sample_rollout(
                sample=sample,
                rollout_id=f"branch-{step_index}-{window}-{branch_index}",
                per_step_samples=1,
                initial_conversation=conversation,
                start_step_index=branch_start_index,
                stop_step_index=branch_end_index,
                sampling_cursors={branch_start_index: initial_branch_cursor},
            )
            branched_steps = [self._build_backbone_step_record(step) for step in branch_rollout.steps]

        return LocalBranchHistory(
            step_index=step_index,
            window_start=window_start,
            prefix_steps=prefix_steps,
            branched_steps=branched_steps,
            metadata={
                "executor": self.name,
                "branch_index": branch_index,
                "fixed_prefix_count": fixed_prefix_count,
                "branch_rollout_id": branch_rollout.metadata.get("rollout_id") if branch_rollout is not None else None,
                "branch_final_answer": branch_rollout.final_answer if branch_rollout is not None else None,
                "branch_sampling_cursor": initial_branch_cursor,
            },
        )

    async def estimate_next_step_entropy(
        self,
        sample: Any,
        history: LocalBranchHistory,
        step_index: int,
    ) -> float:
        if step_index <= 0:
            raise ValueError("step_index must be positive")

        realized_steps = history.prefix_steps + history.branched_steps
        if realized_steps:
            last_action = realized_steps[-1].action
            if isinstance(last_action, str) and self._is_finish_action(last_action):
                history.metadata["estimated_step_index"] = step_index
                history.metadata["estimated_entropy"] = 0.0
                history.metadata["estimated_sampled_actions"] = []
                history.metadata["estimated_sampled_raw_outputs"] = []
                history.metadata["estimated_sampled_output_metadata"] = []
                return 0.0

        conversation = self._build_conversation_from_steps(sample, realized_steps)
        messages = conversation + [self._step_instruction_message(step_index)]
        sampled_step_samples = await self._sample_step_samples(
            sample=sample,
            messages=messages,
            per_step_samples=self.next_step_entropy_samples,
            step_index=step_index,
        )
        prepared_step_samples = self._prepare_step_samples(
            sample_id=sample.sample_id,
            context_id="entropy-estimate",
            step_index=step_index,
            sampled_step_samples=sampled_step_samples,
        )
        sampled_outputs = [step_sample.output for step_sample in prepared_step_samples]
        parsed_steps = [
            self._parse_step(output)
            for output in sampled_outputs
        ]
        sampled_output_metadata = [
            self._merge_parser_metadata(step_sample.metadata, parsed_step)
            for step_sample, parsed_step in zip(prepared_step_samples, parsed_steps)
        ]
        sampled_actions = [parsed_step.action for parsed_step in parsed_steps]
        entropy, _ = compute_predictive_entropy_from_metadata(sampled_output_metadata)
        resolved_entropy = 0.0 if entropy is None else float(entropy)
        history.metadata["estimated_step_index"] = step_index
        history.metadata["estimated_entropy"] = resolved_entropy
        history.metadata["estimated_sampled_actions"] = sampled_actions
        history.metadata["estimated_sampled_raw_outputs"] = sampled_outputs
        history.metadata["estimated_sampled_output_metadata"] = sampled_output_metadata
        return resolved_entropy

    async def sample_tdp(
        self,
        sample: Any,
        trajectory_index: int,
        per_step_samples: int,
        *,
        include_counterfactuals: bool = False,
    ) -> TrajectoryDependentDecisionProcess:
        if per_step_samples <= 0:
            raise ValueError("per_step_samples must be positive")

        tdp_cache_key = self._tdp_cache_key(
            sample,
            trajectory_index=trajectory_index,
            include_counterfactuals=include_counterfactuals,
        )
        cached_tdp: TrajectoryDependentDecisionProcess | None = None
        forced_sample_indices: dict[int, int] | None = None
        sampling_cursors: dict[int, int] | None = None
        if self.shared_sampling_storage is not None:
            cached_tdp = self._load_cached_tdp(sample, tdp_cache_key, trajectory_index=trajectory_index)
            if cached_tdp is not None:
                cached_per_step_samples = cached_tdp.metadata.get("per_step_samples")
                if isinstance(cached_per_step_samples, int) and cached_per_step_samples >= per_step_samples:
                    return cached_tdp
                if (
                    isinstance(cached_per_step_samples, int)
                    and cached_per_step_samples < per_step_samples
                    and cached_tdp.metadata.get(
                        "fixed_trajectory_candidate_extension_requested"
                    )
                    is True
                ):
                    extended_tdp = await self._extend_cached_tdp_candidates_only(
                        sample=sample,
                        trajectory_index=trajectory_index,
                        cached_tdp=cached_tdp,
                        per_step_samples=per_step_samples,
                        include_counterfactuals=include_counterfactuals,
                    )
                    self._store_cached_tdp(
                        sample,
                        tdp_cache_key,
                        extended_tdp,
                        trajectory_index=trajectory_index,
                    )
                    return extended_tdp
                if (
                    isinstance(cached_per_step_samples, int)
                    and cached_per_step_samples < per_step_samples
                    and os.getenv(_REQUIRE_FROZEN_TDP_CACHE_ENV_VAR, "").strip().lower()
                    in {"1", "true", "yes", "on"}
                ):
                    raise ValueError(
                        "Strict frozen-pool execution refuses an unmarked cached TDP for "
                        f"sample={sample.sample_id} trajectory={trajectory_index}."
                    )
                forced_sample_indices = self._tdp_sample_indices(cached_tdp)
                if isinstance(cached_per_step_samples, int) and cached_per_step_samples > 0:
                    sampling_cursors = {1: trajectory_index * cached_per_step_samples}

        if os.getenv(_REQUIRE_FROZEN_TDP_CACHE_ENV_VAR, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise ValueError(
                "Strict frozen-pool execution refuses trajectory regeneration after a cache miss for "
                f"sample={sample.sample_id} trajectory={trajectory_index}."
            )

        backbone_cache_key = self._backbone_cache_key(sample)
        cached_backbone: BackboneTrajectory | None = None
        if trajectory_index == 0 and self.shared_sampling_storage is not None:
            cached_backbone = self._load_cached_backbone(sample, backbone_cache_key)

        rollout = await self._sample_rollout(
            sample=sample,
            rollout_id=f"tdp-{trajectory_index}",
            per_step_samples=per_step_samples,
            sampling_cursors=(
                sampling_cursors
                if sampling_cursors is not None
                else ({1: trajectory_index * per_step_samples} if trajectory_index > 0 else None)
            ),
            forced_sample_indices=forced_sample_indices,
        )
        counterfactual_records = (
            await self._build_tdp_counterfactual_records(
                sample=sample,
                rollout=rollout,
                per_step_samples=per_step_samples,
            )
            if include_counterfactuals
            else [[] for _ in rollout.steps]
        )

        if (
            trajectory_index == 0
            and self.shared_sampling_storage is not None
            and cached_backbone is None
        ):
            self._store_cached_backbone(sample, backbone_cache_key, self._build_backbone_from_rollout(sample, rollout))

        tdp = TrajectoryDependentDecisionProcess(
            sample_id=f"{sample.sample_id}-tdp-{trajectory_index}",
            prompt=rollout.prompt,
            steps=[
                self._build_tdp_step_record(
                    step,
                    counterfactual_records=counterfactual_records[step.index - 1],
                )
                for step in rollout.steps
            ],
            final_answer=rollout.final_answer,
            metadata={
                **rollout.metadata,
                "per_step_samples": per_step_samples,
                "include_counterfactuals": include_counterfactuals,
                "counterfactual_per_step_samples": per_step_samples if include_counterfactuals else 0,
                "trajectory_index": trajectory_index,
            },
        )
        if self.shared_sampling_storage is not None:
            self._store_cached_tdp(sample, tdp_cache_key, tdp, trajectory_index=trajectory_index)
        return tdp

    async def _extend_cached_tdp_candidates_only(
        self,
        *,
        sample: Any,
        trajectory_index: int,
        cached_tdp: TrajectoryDependentDecisionProcess,
        per_step_samples: int,
        include_counterfactuals: bool = False,
    ) -> TrajectoryDependentDecisionProcess:
        """Append candidates at frozen chat states without changing the trajectory."""
        if include_counterfactuals:
            raise ValueError(
                "Frozen candidate-only TDP extension does not support counterfactual regeneration."
            )

        prefix_steps: list[StepRecord] = []
        extended_steps: list[TDPStepRecord] = []
        for cached_step in cached_tdp.steps:
            existing_actions = list(cached_step.sampled_decisions)
            existing_outputs = cached_step.metadata.get("sampled_raw_outputs")
            existing_metadata = cached_step.metadata.get("sampled_output_metadata")
            chosen_output_index = cached_step.metadata.get("chosen_output_index")
            existing_count = len(existing_actions)
            if (
                existing_count <= 0
                or not isinstance(existing_outputs, list)
                or not isinstance(existing_metadata, list)
                or len(existing_outputs) != existing_count
                or len(existing_metadata) != existing_count
                or not isinstance(chosen_output_index, int)
                or not 0 <= chosen_output_index < existing_count
            ):
                raise ValueError(
                    "Frozen TDP candidate payloads are inconsistent for "
                    f"sample={sample.sample_id} trajectory={trajectory_index} "
                    f"step={cached_step.index}."
                )
            if existing_count > per_step_samples:
                raise ValueError(
                    f"Frozen TDP already has N={existing_count}, exceeding "
                    f"requested N={per_step_samples} for sample={sample.sample_id} "
                    f"trajectory={trajectory_index} step={cached_step.index}."
                )

            additional_count = per_step_samples - existing_count
            additional_samples: list[StepSample] = []
            if additional_count:
                conversation = self._build_conversation_from_steps(
                    sample,
                    prefix_steps,
                )
                messages = conversation + [
                    self._step_instruction_message(cached_step.index)
                ]
                generated = await self._generate_step_samples(
                    messages=messages,
                    sample_count=additional_count,
                )
                additional_samples = self._prepare_step_samples(
                    sample_id=str(sample.sample_id),
                    context_id=f"tdp-{trajectory_index}-fixed-extension",
                    step_index=cached_step.index,
                    sampled_step_samples=[
                        self.shared_step_sampler._coerce_sample(value)
                        for value in generated
                    ],
                )
                if len(additional_samples) != additional_count:
                    raise ValueError(
                        f"Expected {additional_count} additional candidates for "
                        f"sample={sample.sample_id} trajectory={trajectory_index} "
                        f"step={cached_step.index}, received "
                        f"{len(additional_samples)}."
                    )

            additional_outputs = [sample.output for sample in additional_samples]
            parsed_additional = [
                self._parse_step(output) for output in additional_outputs
            ]
            additional_actions = [
                parsed.action for parsed in parsed_additional
            ]
            additional_metadata = [
                self._merge_parser_metadata(step_sample.metadata, parsed)
                for step_sample, parsed in zip(
                    additional_samples,
                    parsed_additional,
                    strict=True,
                )
            ]
            combined_metadata = [
                *[dict(value) for value in existing_metadata],
                *additional_metadata,
            ]
            entropy, _ = compute_predictive_entropy_from_metadata(
                combined_metadata
            )
            measurements = dict(cached_step.uncertainty_measurements)
            measurements["pe"] = 0.0 if entropy is None else float(entropy)
            metadata = dict(cached_step.metadata)
            metadata.update(
                {
                    "sampled_raw_outputs": [
                        *[str(value) for value in existing_outputs],
                        *additional_outputs,
                    ],
                    "sampled_output_metadata": combined_metadata,
                    "fixed_trajectory_candidate_extension": True,
                    "fixed_trajectory_source_candidate_count": existing_count,
                    "fixed_trajectory_added_candidate_count": additional_count,
                }
            )
            extended_steps.append(
                TDPStepRecord(
                    index=cached_step.index,
                    realized_decision=cached_step.realized_decision,
                    sampled_decisions=[
                        *existing_actions,
                        *additional_actions,
                    ],
                    uncertainty_measurements=measurements,
                    counterfactual_records=list(
                        cached_step.counterfactual_records
                    ),
                    metadata=metadata,
                )
            )
            prefix_steps.append(
                StepRecord(
                    index=cached_step.index,
                    thought=self._coerce_optional_string(
                        cached_step.metadata.get("thought")
                    ),
                    action=cached_step.realized_decision,
                    observation=self._coerce_optional_string(
                        cached_step.metadata.get("observation")
                    ),
                    messages=[],
                    metadata=dict(cached_step.metadata),
                )
            )

        metadata = dict(cached_tdp.metadata)
        metadata.update(
            {
                "trajectory_index": trajectory_index,
                "per_step_samples": per_step_samples,
                "include_counterfactuals": False,
                "counterfactual_per_step_samples": 0,
                "fixed_trajectory_candidate_extension": True,
                "fixed_trajectory_source_per_step_samples": (
                    cached_tdp.metadata.get("per_step_samples")
                ),
            }
        )
        return TrajectoryDependentDecisionProcess(
            sample_id=cached_tdp.sample_id,
            prompt=cached_tdp.prompt,
            steps=extended_steps,
            final_answer=cached_tdp.final_answer,
            metadata=metadata,
        )

    async def hard_finalize_tdp(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> str | None:
        """Ask the backbone for a validated JSON final answer from an unfinished chat trajectory."""
        if isinstance(tdp.final_answer, str) and tdp.final_answer.strip():
            return tdp.final_answer.strip()

        rejected_attempts: list[dict[str, str]] = []
        tool_answer = await self._hard_finalize_with_answer_tool(sample, tdp, rejected_attempts)
        if tool_answer:
            if rejected_attempts:
                tdp.metadata["hard_finalization_rejected_attempts"] = rejected_attempts
            return tool_answer

        retry_reason: str | None = None
        previous_output: str | None = None
        for _attempt in range(2):
            messages = self._build_conversation_from_tdp(sample, tdp)
            user_prompt = (
                "The trajectory stopped before a Finish action. Based only on the question, "
                "the reasoning steps, and the observations above, infer the answer that should "
                'be submitted now. Return exactly one JSON object in this format: {"answer":"<short final answer>"}. '
                "Do not include markdown, reasoning, tool calls, or any text outside the JSON object. "
                "The answer value must be the submitted answer itself, not a description of the task."
            )
            if retry_reason is not None:
                user_prompt += (
                    "\n\nYour previous output was rejected because: "
                    f"{retry_reason}. Previous output: {previous_output!r}. "
                    'Return only valid JSON now, for example {"answer":"Paris"} or {"answer":"yes"}.'
                )
            messages.append({"role": "user", "content": user_prompt})
            output = await self._invoke_generation_callable(
                self._resolve_generation_callable(),
                messages=messages,
                temperature=0.0,
            )
            raw_output = self._coerce_output_text(output)
            answer, invalid_reason = self._parse_hard_finalization_answer(raw_output)
            if answer:
                if rejected_attempts:
                    tdp.metadata["hard_finalization_rejected_attempts"] = rejected_attempts
                tdp.metadata["hard_finalization_raw_output"] = raw_output
                return answer
            retry_reason = invalid_reason
            previous_output = raw_output
            rejected_attempts.append({"output": raw_output, "reason": invalid_reason})

        tdp.metadata["hard_finalization_rejected_attempts"] = rejected_attempts
        return None

    async def _hard_finalize_with_answer_tool(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
        rejected_attempts: list[dict[str, str]],
    ) -> str | None:
        completion_callable = getattr(self.model_client, "acompletion_with_tools", None)
        if completion_callable is None:
            return None

        messages = self._build_conversation_from_tdp(sample, tdp)
        messages.append(
            {
                "role": "user",
                "content": (
                    "The trajectory stopped before a Finish action. Infer the exact final answer now. "
                    "Use the submit_final_answer tool exactly once. Do not write prose before or after the tool call."
                ),
            }
        )
        output = completion_callable(messages, [self._hard_finalization_answer_tool()], temperature=0.0)
        if inspect.isawaitable(output):
            output = await output

        answer, invalid_reason = self._extract_hard_finalization_tool_answer(output)
        if answer:
            tdp.metadata["hard_finalization_tool_output"] = output
            return answer

        rejected_attempts.append({"output": json.dumps(output, sort_keys=True, ensure_ascii=True), "reason": invalid_reason})
        return None

    @staticmethod
    def _hard_finalization_answer_tool() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "submit_final_answer",
                "description": "Submit the short final answer inferred from the partial trajectory.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The exact final answer to submit, with no explanation.",
                        }
                    },
                    "required": ["answer"],
                },
            },
        }

    @staticmethod
    def _extract_hard_finalization_tool_answer(payload: Any) -> tuple[str | None, str]:
        if not isinstance(payload, Mapping):
            return None, "tool finalization returned a non-object payload"
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return None, "tool finalization did not call submit_final_answer"
        function = tool_calls[0].get("function") if isinstance(tool_calls[0], Mapping) else None
        if not isinstance(function, Mapping):
            return None, "tool call did not contain function arguments"
        if function.get("name") != "submit_final_answer":
            return None, "tool finalization called the wrong function"
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return None, "tool finalization arguments were not a JSON string"
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return None, f"tool finalization arguments were not valid JSON: {exc.msg}"
        if not isinstance(parsed, Mapping):
            return None, "tool finalization arguments were not a JSON object"
        raw_answer = parsed.get("answer")
        if not isinstance(raw_answer, str):
            return None, "tool finalization did not contain a string answer field"
        answer = raw_answer.strip().strip('"').strip("'")
        invalid_reason = ChatLocalBranchingExecutor._validate_hard_finalization_answer(answer)
        if invalid_reason is not None:
            return None, invalid_reason
        return answer, ""

    def _default_model_signature(self) -> dict[str, Any]:
        return {
            "client_type": f"{type(self.model_client).__module__}.{type(self.model_client).__qualname__}",
            "model": getattr(self.model_client, "model", None),
        }

    def _prompt_template_signature(self) -> dict[str, Any]:
        return {
            "type": f"{type(self.prompt_template).__module__}.{type(self.prompt_template).__qualname__}",
            "system_prompt": self.prompt_template.system_prompt,
        }

    def _generation_signature(self) -> dict[str, Any]:
        return {
            "executor": self.name,
            "local_branching_version": _CHAT_LOCAL_BRANCHING_CACHE_VERSION,
            "sample_temperature": self.sample_temperature,
            "prompt_template": self._prompt_template_signature(),
            "step_output_format": self.step_output_format,
            "step_retry_attempts": self.step_retry_attempts,
        }

    def _executor_signature(self) -> dict[str, Any]:
        return {
            **self._generation_signature(),
            "max_steps": self.max_steps,
            "backbone_per_step_samples": self.backbone_per_step_samples,
            "next_step_entropy_samples": self.next_step_entropy_samples,
        }

    def _step_sampling_executor_signature(self) -> dict[str, Any]:
        return self._generation_signature()

    def _sample_signature(self, sample: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _sampling_cache_key(self, sample: Any, *, kind: str, **payload: Any) -> dict[str, Any]:
        return {
            "kind": kind,
            "sample": self._sample_signature(sample),
            "model": self.model_signature,
            "executor": self._executor_signature(),
            **payload,
        }

    def _backbone_cache_key(self, sample: Any) -> dict[str, Any]:
        return {
            "kind": "backbone",
            "sample": self._backbone_sample_signature(sample),
            "model": self.model_signature,
            "executor": self._generation_signature(),
        }

    def _backbone_sample_signature(self, sample: Any) -> dict[str, Any]:
        signature = dict(self._sample_signature(sample))
        signature.pop("sample_id", None)
        return signature

    def _backbone_cache_storage_id(self, sample: Any) -> str:
        return f"shared-backbone-{sampling_fingerprint(self._backbone_sample_signature(sample))}"

    def _tdp_cache_key(
        self,
        sample: Any,
        *,
        trajectory_index: int,
        include_counterfactuals: bool = False,
    ) -> dict[str, Any]:
        key: dict[str, Any] = {
            "kind": "tdp",
            "sample": self._backbone_sample_signature(sample),
            "model": self.model_signature,
            "executor": self._generation_signature(),
            "trajectory_index": trajectory_index,
            "include_counterfactuals": include_counterfactuals,
        }
        return key

    def _tdp_cache_storage_id(self, sample: Any, trajectory_index: int) -> str:
        return f"shared-tdp-{trajectory_index}-{sampling_fingerprint(self._backbone_sample_signature(sample))}"

    def _tdp_sample_indices(self, tdp: TrajectoryDependentDecisionProcess) -> dict[int, int]:
        cached_per_step_samples = tdp.metadata.get("per_step_samples")
        trajectory_index = tdp.metadata.get("trajectory_index")
        if not isinstance(cached_per_step_samples, int) or cached_per_step_samples <= 0:
            cached_per_step_samples = 0
        if not isinstance(trajectory_index, int) or trajectory_index < 0:
            trajectory_index = 0
        sample_indices: dict[int, int] = {}
        for step in tdp.steps:
            chosen_output_index = step.metadata.get("chosen_output_index")
            if not isinstance(chosen_output_index, int) or chosen_output_index < 0:
                continue
            if step.index == 1 and trajectory_index > 0 and cached_per_step_samples > 0:
                sample_indices[step.index] = trajectory_index * cached_per_step_samples + chosen_output_index
            else:
                sample_indices[step.index] = chosen_output_index
        return sample_indices

    def _step_sampling_cache_key(self, sample: Any, *, step_index: int, messages: list[dict[str, str]]) -> dict[str, Any]:
        del sample, step_index
        return {
            "kind": "step_samples",
            "step_sampling_version": _CHAT_LOCAL_BRANCHING_STEP_SAMPLING_VERSION,
            "model": self.model_signature,
            "executor": self._step_sampling_executor_signature(),
            "messages": messages,
        }

    def _step_sampling_pool_id(self) -> str:
        return "shared-context-pool"

    def _load_cached_backbone(self, sample: Any, cache_key: dict[str, Any]) -> BackboneTrajectory | None:
        payload = self._load_cached_payload(
            f"{self.name}/backbone",
            sample,
            cache_key,
            cache_sample_id=self._backbone_cache_storage_id(sample),
        )
        if payload is None:
            return None
        LOGGER.info("Loaded %s backbone from shared sampling storage for sample_id=%s", self.name, sample.sample_id)
        return self._deserialize_backbone(payload)

    def _store_cached_backbone(self, sample: Any, cache_key: dict[str, Any], backbone: BackboneTrajectory) -> None:
        self._store_cached_payload(
            f"{self.name}/backbone",
            sample,
            cache_key,
            {"backbone": self._serialize(backbone)},
            cache_sample_id=self._backbone_cache_storage_id(sample),
        )

    def _load_cached_local_history(self, sample: Any, cache_key: dict[str, Any]) -> LocalBranchHistory | None:
        payload = self._load_cached_payload(f"{self.name}/local_history", sample, cache_key)
        if payload is None:
            return None
        LOGGER.info("Loaded %s local history from shared sampling storage for sample_id=%s", self.name, sample.sample_id)
        return self._deserialize_local_history(payload)

    def _store_cached_local_history(self, sample: Any, cache_key: dict[str, Any], history: LocalBranchHistory) -> None:
        self._store_cached_payload(f"{self.name}/local_history", sample, cache_key, {"history": self._serialize(history)})

    def _load_cached_entropy_estimate(
        self,
        sample: Any,
        cache_key: dict[str, Any],
        history: LocalBranchHistory,
    ) -> float | None:
        payload = self._load_cached_payload(f"{self.name}/next_step_entropy", sample, cache_key)
        if payload is None:
            return None
        entropy = payload.get("entropy")
        if not isinstance(entropy, (int, float)):
            return None
        metadata = payload.get("history_metadata")
        if isinstance(metadata, dict):
            history.metadata.update(metadata)
        LOGGER.info(
            "Loaded %s next-step entropy from shared sampling storage for sample_id=%s step=%s",
            self.name,
            sample.sample_id,
            history.metadata.get("estimated_step_index", cache_key.get("step_index")),
        )
        return float(entropy)

    def _store_cached_entropy_estimate(
        self,
        sample: Any,
        cache_key: dict[str, Any],
        *,
        entropy: float,
        history: LocalBranchHistory,
    ) -> None:
        metadata = {
            key: history.metadata.get(key)
            for key in (
                "estimated_step_index",
                "estimated_entropy",
                "estimated_sampled_actions",
                "estimated_sampled_raw_outputs",
                "estimated_sampled_output_metadata",
            )
            if key in history.metadata
        }
        self._store_cached_payload(
            f"{self.name}/next_step_entropy",
            sample,
            cache_key,
            {"entropy": entropy, "history_metadata": metadata},
        )

    def _load_cached_tdp(
        self,
        sample: Any,
        cache_key: dict[str, Any],
        *,
        trajectory_index: int,
    ) -> TrajectoryDependentDecisionProcess | None:
        payload = self._load_cached_payload(
            f"{self.name}/tdp",
            sample,
            cache_key,
            cache_sample_id=self._tdp_cache_storage_id(sample, trajectory_index),
        )
        if payload is None:
            return None
        LOGGER.info("Loaded %s TDP from shared sampling storage for sample_id=%s trajectory=%s", self.name, sample.sample_id, trajectory_index)
        return self._deserialize_tdp(payload)

    def _store_cached_tdp(
        self,
        sample: Any,
        cache_key: dict[str, Any],
        tdp: TrajectoryDependentDecisionProcess,
        *,
        trajectory_index: int,
    ) -> None:
        self._store_cached_payload(
            f"{self.name}/tdp",
            sample,
            cache_key,
            {"tdp": self._serialize(tdp)},
            cache_sample_id=self._tdp_cache_storage_id(sample, trajectory_index),
        )

    def _serialize(self, value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        return value

    def _load_cached_payload(
        self,
        category: str,
        sample: Any,
        cache_key: dict[str, Any],
        *,
        cache_sample_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self.shared_sampling_storage is None:
            return None
        return self.shared_sampling_storage.load(
            category,
            sample_id=cache_sample_id or sample.sample_id,
            key=cache_key,
        )

    def _store_cached_payload(
        self,
        category: str,
        sample: Any,
        cache_key: dict[str, Any],
        value: dict[str, Any],
        *,
        cache_sample_id: str | None = None,
    ) -> None:
        if self.shared_sampling_storage is None:
            return
        self.shared_sampling_storage.store(
            category,
            sample_id=cache_sample_id or sample.sample_id,
            key=cache_key,
            value=value,
        )

    def _deserialize_backbone(self, payload: dict[str, Any]) -> BackboneTrajectory:
        backbone_payload = payload.get("backbone")
        if not isinstance(backbone_payload, dict):
            raise ValueError("Invalid cached backbone payload.")
        return BackboneTrajectory(
            sample_id=str(backbone_payload.get("sample_id", "")),
            prompt=str(backbone_payload.get("prompt", "")),
            steps=[self._deserialize_step_record(step) for step in backbone_payload.get("steps", [])],
            final_answer=self._coerce_optional_string(backbone_payload.get("final_answer")),
            metadata=self._deserialize_metadata(backbone_payload.get("metadata")),
        )

    def _build_backbone_from_rollout(self, sample: Any, rollout: SampledChatRollout) -> BackboneTrajectory:
        return BackboneTrajectory(
            sample_id=f"{sample.sample_id}-backbone",
            prompt=sample.question,
            steps=[self._build_backbone_step_record(step) for step in rollout.steps],
            final_answer=rollout.final_answer,
            metadata={
                **rollout.metadata,
                "step_entropies": [step.entropy for step in rollout.steps],
                "per_step_samples": self.backbone_per_step_samples,
            },
        )

    def _deserialize_local_history(self, payload: dict[str, Any]) -> LocalBranchHistory:
        history_payload = payload.get("history")
        if not isinstance(history_payload, dict):
            raise ValueError("Invalid cached local history payload.")
        return LocalBranchHistory(
            step_index=int(history_payload.get("step_index", 0)),
            window_start=int(history_payload.get("window_start", 0)),
            prefix_steps=[self._deserialize_step_record(step) for step in history_payload.get("prefix_steps", [])],
            branched_steps=[self._deserialize_step_record(step) for step in history_payload.get("branched_steps", [])],
            metadata=self._deserialize_metadata(history_payload.get("metadata")),
        )

    def _deserialize_tdp(self, payload: dict[str, Any]) -> TrajectoryDependentDecisionProcess:
        tdp_payload = payload.get("tdp")
        if not isinstance(tdp_payload, dict):
            raise ValueError("Invalid cached TDP payload.")
        return TrajectoryDependentDecisionProcess(
            sample_id=str(tdp_payload.get("sample_id", "")),
            prompt=str(tdp_payload.get("prompt", "")),
            steps=[self._deserialize_tdp_step_record(step) for step in tdp_payload.get("steps", [])],
            final_answer=self._coerce_optional_string(tdp_payload.get("final_answer")),
            metadata=self._deserialize_metadata(tdp_payload.get("metadata")),
        )

    def _deserialize_step_record(self, payload: Any) -> StepRecord:
        if not isinstance(payload, dict):
            raise ValueError("Invalid cached step record.")
        return StepRecord(
            index=int(payload.get("index", 0)),
            thought=self._coerce_optional_string(payload.get("thought")),
            action=self._coerce_optional_string(payload.get("action")),
            observation=self._coerce_optional_string(payload.get("observation")),
            messages=self._deserialize_messages(payload.get("messages")),
            metadata=self._deserialize_metadata(payload.get("metadata")),
        )

    def _deserialize_tdp_step_record(self, payload: Any) -> TDPStepRecord:
        if not isinstance(payload, dict):
            raise ValueError("Invalid cached TDP step record.")
        raw_measurements = payload.get("uncertainty_measurements")
        uncertainty_measurements = (
            {str(key): float(value) for key, value in raw_measurements.items()}
            if isinstance(raw_measurements, dict)
            else {}
        )
        return TDPStepRecord(
            index=int(payload.get("index", 0)),
            realized_decision=str(payload.get("realized_decision", "")),
            sampled_decisions=[str(item) for item in payload.get("sampled_decisions", [])],
            uncertainty_measurements=uncertainty_measurements,
            counterfactual_records=self._deserialize_counterfactual_records(payload.get("counterfactual_records")),
            metadata=self._deserialize_metadata(payload.get("metadata")),
        )

    def _deserialize_counterfactual_records(self, payload: Any) -> list[TDPCounterfactualRecord]:
        if not isinstance(payload, list):
            return []
        records: list[TDPCounterfactualRecord] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            branches_payload = item.get("branches")
            branches: list[TDPCounterfactualBranch] = []
            if isinstance(branches_payload, list):
                for branch_payload in branches_payload:
                    if not isinstance(branch_payload, dict):
                        continue
                    branches.append(
                        TDPCounterfactualBranch(
                            source_decision=str(branch_payload.get("source_decision", "")),
                            target_sampled_decisions=[
                                str(value) for value in branch_payload.get("target_sampled_decisions", [])
                            ],
                            target_sampled_output_metadata=[
                                dict(value)
                                for value in branch_payload.get("target_sampled_output_metadata", [])
                                if isinstance(value, dict)
                            ],
                            metadata=self._deserialize_metadata(branch_payload.get("metadata")),
                        )
                    )
            records.append(
                TDPCounterfactualRecord(
                    source_step_index=int(item.get("source_step_index", 0)),
                    realized_source_decision=str(item.get("realized_source_decision", "")),
                    branches=branches,
                    metadata=self._deserialize_metadata(item.get("metadata")),
                )
            )
        return records

    def _deserialize_messages(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        messages: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            messages.append({str(key): value for key, value in item.items()})
        return messages

    def _deserialize_metadata(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        return dict(payload)

    def _coerce_optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    async def _sample_rollout(
        self,
        sample: Any,
        rollout_id: str,
        per_step_samples: int,
        initial_conversation: list[dict[str, str]] | None = None,
        start_step_index: int = 1,
        stop_step_index: int | None = None,
        sampling_cursors: dict[int, int] | None = None,
        forced_sample_indices: dict[int, int] | None = None,
    ) -> SampledChatRollout:
        rng = random.Random(f"{sample.sample_id}:{rollout_id}")
        prompt_messages = self._base_messages(sample)
        conversation = list(initial_conversation) if initial_conversation is not None else list(prompt_messages)
        sampled_steps: list[SampledChatStep] = []
        final_answer: str | None = None
        last_step_index = self.max_steps if stop_step_index is None else stop_step_index
        resolved_sampling_cursors = dict(sampling_cursors or {})
        resolved_forced_sample_indices = dict(forced_sample_indices or {})
        forced_sample_fallbacks: list[dict[str, Any]] = []
        stopped_by_callback = False

        if start_step_index <= 0:
            raise ValueError("start_step_index must be positive")
        if last_step_index < start_step_index - 1:
            raise ValueError("stop_step_index must be greater than or equal to start_step_index - 1")

        for step_index in range(start_step_index, last_step_index + 1):
            forced_sample_index = resolved_forced_sample_indices.get(step_index)
            sampling_cursor = resolved_sampling_cursors.get(step_index, 0)
            if forced_sample_index is not None:
                if forced_sample_index < 0:
                    raise ValueError("forced sample indices must be non-negative")
                if step_index not in resolved_sampling_cursors:
                    sampling_cursor = max(
                        0, forced_sample_index - (per_step_samples - 1)
                    )

            messages = conversation + [self._step_instruction_message(step_index)]
            sampled_step_samples = await self._sample_step_samples(
                sample=sample,
                messages=messages,
                per_step_samples=per_step_samples,
                step_index=step_index,
                sampling_cursor=sampling_cursor,
            )
            prepared_step_samples = self._prepare_step_samples(
                sample_id=sample.sample_id,
                context_id=rollout_id,
                step_index=step_index,
                sampled_step_samples=sampled_step_samples,
            )
            sampled_outputs = [step_sample.output for step_sample in prepared_step_samples]
            parsed_steps = [
                self._parse_step(output)
                for output in sampled_outputs
            ]
            if not parsed_steps:
                raise ValueError(f"No sampled outputs were available for step {step_index}.")
            step_metadata: dict[str, Any] = {}
            if forced_sample_index is not None:
                chosen_output_index = forced_sample_index - sampling_cursor
                if not 0 <= chosen_output_index < len(parsed_steps):
                    fallback_index = min(max(chosen_output_index, 0), len(parsed_steps) - 1)
                    fallback = {
                        "step_index": step_index,
                        "forced_sample_index": forced_sample_index,
                        "sampling_cursor": sampling_cursor,
                        "requested_output_index": chosen_output_index,
                        "fallback_output_index": fallback_index,
                        "available_sample_count": len(parsed_steps),
                    }
                    LOGGER.warning(
                        "Forced sample index %s for step %s was not available in the sampled pool; "
                        "using fallback output index %s from %s available samples.",
                        forced_sample_index,
                        step_index,
                        fallback_index,
                        len(parsed_steps),
                    )
                    forced_sample_fallbacks.append(fallback)
                    step_metadata["forced_sample_fallback"] = fallback
                    chosen_output_index = fallback_index
            else:
                chosen_output_index = rng.randrange(len(parsed_steps))
            chosen_step = parsed_steps[chosen_output_index]
            sampled_actions = [parsed_step.action for parsed_step in parsed_steps]
            sampled_output_metadata = [
                self._merge_parser_metadata(step_sample.metadata, parsed_step)
                for step_sample, parsed_step in zip(prepared_step_samples, parsed_steps)
            ]
            entropy, _ = compute_predictive_entropy_from_metadata(sampled_output_metadata)
            observation = None

            if self._is_finish_action(chosen_step.action):
                final_answer = self._extract_finish_answer(chosen_step.action)
            else:
                observation = self._synthesize_observation(sample, chosen_step.action)

            sampled_steps.append(
                SampledChatStep(
                    index=step_index,
                    chosen_step=chosen_step,
                    sampled_outputs=sampled_outputs,
                    sampled_actions=sampled_actions,
                    sampled_output_metadata=sampled_output_metadata,
                    chosen_output_index=chosen_output_index,
                    entropy=0.0 if entropy is None else float(entropy),
                    observation=observation,
                    metadata=step_metadata,
                )
            )

            conversation.append({"role": "assistant", "content": self._format_step_message(chosen_step)})
            if observation is not None:
                conversation.append({"role": "user", "content": f"Observation: {observation}"})
            if final_answer is not None:
                break
            if self.step_stop_callback is not None and step_index < last_step_index:
                should_stop = self.step_stop_callback(
                    sample=sample,
                    rollout_id=rollout_id,
                    step=sampled_steps[-1],
                    steps=tuple(sampled_steps),
                    conversation=tuple(dict(message) for message in conversation),
                )
                if inspect.isawaitable(should_stop):
                    should_stop = await should_stop
                if bool(should_stop):
                    stopped_by_callback = True
                    break

        if final_answer is None and sampled_steps:
            final_answer = self._extract_finish_answer(sampled_steps[-1].chosen_step.action)

        rollout_metadata = {
            "executor": self.name,
            "question": sample.question,
            "rollout_id": rollout_id,
            "start_step_index": start_step_index,
            "stop_step_index": last_step_index,
            "base_prompt_messages": prompt_messages,
            "sampling_cursors": resolved_sampling_cursors,
            "forced_sample_fallback_count": len(forced_sample_fallbacks),
            "stopped_by_callback": stopped_by_callback,
        }
        if forced_sample_fallbacks:
            rollout_metadata["forced_sample_fallbacks"] = forced_sample_fallbacks

        return SampledChatRollout(
            prompt=sample.question,
            steps=sampled_steps,
            final_answer=final_answer,
            metadata=rollout_metadata,
        )

    def _prepare_step_samples(
        self,
        *,
        sample_id: str,
        context_id: str,
        step_index: int,
        sampled_step_samples: list[StepSample],
    ) -> list[StepSample]:
        usable_step_samples = [step_sample for step_sample in sampled_step_samples if step_sample.output.strip()]
        dropped_outputs = len(sampled_step_samples) - len(usable_step_samples)
        if dropped_outputs:
            LOGGER.warning(
                "Empty outputs detected for sample_id=%s context=%s step=%s dropped=%s total=%s",
                sample_id,
                context_id,
                step_index,
                dropped_outputs,
                len(sampled_step_samples),
            )
        if usable_step_samples:
            return usable_step_samples

        LOGGER.warning(
            "All outputs were empty for sample_id=%s context=%s step=%s; using unknown fallback",
            sample_id,
            context_id,
            step_index,
        )
        return [StepSample(output="")]

    def _build_conversation_from_steps(self, sample: Any, steps: list[StepRecord]) -> list[dict[str, str]]:
        conversation = self._base_messages(sample)
        for step in steps:
            if step.messages:
                conversation.extend(
                    {"role": str(message.get("role", "assistant")), "content": str(message.get("content", ""))}
                    for message in step.messages
                )
                continue

            parsed_step = ParsedChatStep(
                thought=step.thought,
                action=step.action or "",
                raw_content=str(step.metadata.get("raw_content", "")),
            )
            conversation.append({"role": "assistant", "content": self._format_step_message(parsed_step)})
            if step.observation is not None:
                conversation.append({"role": "user", "content": f"Observation: {step.observation}"})
        return conversation

    def _build_conversation_from_tdp(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> list[dict[str, str]]:
        conversation = self._base_messages(sample)
        if len(tdp.steps) > 5:
            conversation.append(
                {
                    "role": "user",
                    "content": f"{len(tdp.steps) - 5} earlier steps are omitted for brevity.",
                }
            )
        for step in tdp.steps[-5:]:
            raw_content = step.metadata.get("raw_content")
            thought = step.metadata.get("thought")
            parsed_step = ParsedChatStep(
                thought=str(thought) if thought is not None else None,
                action=self._compact_hard_finalization_text(step.realized_decision, limit=700),
                raw_content=str(raw_content) if raw_content is not None else "",
            )
            conversation.append({"role": "assistant", "content": self._format_step_message(parsed_step)})
            observation = step.metadata.get("observation")
            if isinstance(observation, str) and observation:
                conversation.append(
                    {
                        "role": "user",
                        "content": f"Observation: {self._compact_hard_finalization_text(observation, limit=700)}",
                    }
                )
        return conversation

    @staticmethod
    def _parse_hard_finalization_answer(output: str) -> tuple[str | None, str]:
        candidate = ChatLocalBranchingExecutor._strip_json_fence(output).strip()
        parsed: Any
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None, "output was not exactly one valid JSON object"

        if not isinstance(parsed, dict):
            return None, "JSON value was not an object"
        raw_answer = parsed.get("answer")
        if not isinstance(raw_answer, str):
            return None, "JSON object did not contain a string answer field"
        answer = raw_answer.strip().strip('"').strip("'")
        invalid_reason = ChatLocalBranchingExecutor._validate_hard_finalization_answer(answer)
        if invalid_reason is not None:
            return None, invalid_reason
        return answer, ""

    @staticmethod
    def _strip_json_fence(output: str) -> str:
        stripped = output.strip()
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            return fence_match.group(1)
        return stripped

    @staticmethod
    def _validate_hard_finalization_answer(answer: str) -> str | None:
        if not answer:
            return "answer was empty"
        if "\n" in answer or "\r" in answer:
            return "answer contained multiple lines"
        if len(answer) > 200:
            return "answer was too long to be a final answer"
        lowered = answer.lower()
        if lowered.startswith(
            (
                "the user",
                "based on",
                "let me",
                "i ",
                "we ",
                "to determine",
                "looking at",
                "the trajectory",
                "the task",
                "step ",
            )
        ):
            return "answer looked like reasoning or task description"
        if any(marker in answer for marker in ("```", "**", "Thought:", "Action:", "Observation:")):
            return "answer contained formatting or trajectory text"
        word_count = len(answer.split())
        path_like = answer.startswith(("/", "./", "../", "~")) or "\\" in answer
        numeric_like = bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer))
        if word_count > 12 and not path_like and not numeric_like:
            return "answer was too verbose to be a final answer"
        return None

    @staticmethod
    def _compact_hard_finalization_text(text: str, *, limit: int) -> str:
        compacted = " ".join(str(text).split())
        if len(compacted) <= limit:
            return compacted
        half = max(1, (limit - 20) // 2)
        return f"{compacted[:half]} ... {compacted[-half:]}"

    def _build_backbone_step_record(self, step: SampledChatStep) -> StepRecord:
        chosen_output_metadata = self._resolve_chosen_output_metadata(step)
        step_messages = [{"role": "assistant", "content": self._format_step_message(step.chosen_step)}]
        if step.observation is not None:
            step_messages.append({"role": "user", "content": f"Observation: {step.observation}"})

        return StepRecord(
            index=step.index,
            thought=step.chosen_step.thought,
            action=step.chosen_step.action,
            observation=step.observation,
            messages=step_messages,
            metadata={
                **step.metadata,
                "pe": step.entropy,
                "sampled_actions": list(step.sampled_actions),
                "sampled_raw_outputs": list(step.sampled_outputs),
                "sampled_output_metadata": list(step.sampled_output_metadata),
                "chosen_output_index": step.chosen_output_index,
                "chosen_output_metadata": chosen_output_metadata,
                "raw_content": step.chosen_step.raw_content,
                "parser_metadata": dict(step.chosen_step.metadata),
            },
        )

    def _build_tdp_step_record(
        self,
        step: SampledChatStep,
        *,
        counterfactual_records: list[TDPCounterfactualRecord] | None = None,
    ) -> TDPStepRecord:
        chosen_output_metadata = self._resolve_chosen_output_metadata(step)
        metadata: dict[str, Any] = {
            **step.metadata,
            "thought": step.chosen_step.thought,
            "raw_content": step.chosen_step.raw_content,
            "parser_metadata": dict(step.chosen_step.metadata),
            "sampled_raw_outputs": list(step.sampled_outputs),
            "sampled_output_metadata": list(step.sampled_output_metadata),
            "chosen_output_index": step.chosen_output_index,
            "chosen_output_metadata": chosen_output_metadata,
        }
        if step.observation is not None:
            metadata["observation"] = step.observation

        uncertainty_measurements = {"pe": step.entropy}
        ppl = self._perplexity_from_output_metadata(chosen_output_metadata)
        if ppl is not None:
            uncertainty_measurements["ppl"] = ppl

        return TDPStepRecord(
            index=step.index,
            realized_decision=step.chosen_step.action,
            sampled_decisions=list(step.sampled_actions),
            uncertainty_measurements=uncertainty_measurements,
            counterfactual_records=list(counterfactual_records or []),
            metadata=metadata,
        )

    async def _build_tdp_counterfactual_records(
        self,
        *,
        sample: Any,
        rollout: SampledChatRollout,
        per_step_samples: int,
    ) -> list[list[TDPCounterfactualRecord]]:
        stepwise_records: list[list[TDPCounterfactualRecord]] = [[] for _ in rollout.steps]
        if len(rollout.steps) <= 1:
            return stepwise_records

        realized_step_records = [self._build_backbone_step_record(step) for step in rollout.steps]
        for target_step_index in range(2, len(rollout.steps) + 1):
            for source_step_index in range(1, target_step_index):
                stepwise_records[target_step_index - 1].append(
                    await self._sample_counterfactual_record(
                        sample=sample,
                        rollout=rollout,
                        realized_step_records=realized_step_records,
                        source_step_index=source_step_index,
                        target_step_index=target_step_index,
                        per_step_samples=per_step_samples,
                    )
                )
        return stepwise_records

    async def _sample_counterfactual_record(
        self,
        *,
        sample: Any,
        rollout: SampledChatRollout,
        realized_step_records: list[StepRecord],
        source_step_index: int,
        target_step_index: int,
        per_step_samples: int,
    ) -> TDPCounterfactualRecord:
        source_step = rollout.steps[source_step_index - 1]
        prefix_steps = realized_step_records[: source_step_index - 1]
        branches: list[TDPCounterfactualBranch] = []
        sample_id = str(getattr(sample, "sample_id", "sample"))

        for branch_index, sampled_output in enumerate(source_step.sampled_outputs):
            parsed_source_step = self._parse_step(sampled_output)
            branch_metadata: dict[str, Any] = {
                "branch_index": branch_index,
                "target_step_index": target_step_index,
            }
            target_sampled_decisions: list[str] = []
            target_sampled_output_metadata: list[dict[str, Any]] = []
            terminated_before_target = self._is_finish_action(parsed_source_step.action)

            if not terminated_before_target:
                conversation = self._build_conversation_from_steps(sample, prefix_steps)
                source_observation = self._synthesize_observation(sample, parsed_source_step.action)
                self._append_step_to_conversation(conversation, parsed_source_step, source_observation)

                for intermediate_step in rollout.steps[source_step_index: target_step_index - 1]:
                    self._append_step_to_conversation(
                        conversation,
                        intermediate_step.chosen_step,
                        intermediate_step.observation,
                    )
                    if self._is_finish_action(intermediate_step.chosen_step.action):
                        terminated_before_target = True
                        break

                if not terminated_before_target:
                    messages = conversation + [self._step_instruction_message(target_step_index)]
                    sampled_step_samples = await self._sample_step_samples(
                        sample=sample,
                        messages=messages,
                        per_step_samples=per_step_samples,
                        step_index=target_step_index,
                    )
                    prepared_step_samples = self._prepare_step_samples(
                        sample_id=sample_id,
                        context_id=f"counterfactual-{source_step_index}-{target_step_index}-{branch_index}",
                        step_index=target_step_index,
                        sampled_step_samples=sampled_step_samples,
                    )
                    parsed_target_steps = [self._parse_step(step_sample.output) for step_sample in prepared_step_samples]
                    target_sampled_output_metadata = [
                        self._merge_parser_metadata(step_sample.metadata, parsed_step)
                        for step_sample, parsed_step in zip(prepared_step_samples, parsed_target_steps)
                    ]
                    target_sampled_decisions = [parsed_step.action for parsed_step in parsed_target_steps]

            if terminated_before_target:
                branch_metadata["terminated_before_target"] = True

            branches.append(
                TDPCounterfactualBranch(
                    source_decision=source_step.sampled_actions[branch_index],
                    target_sampled_decisions=target_sampled_decisions,
                    target_sampled_output_metadata=target_sampled_output_metadata,
                    metadata=branch_metadata,
                )
            )

        return TDPCounterfactualRecord(
            source_step_index=source_step_index,
            realized_source_decision=source_step.chosen_step.action,
            branches=branches,
            metadata={"target_step_index": target_step_index},
        )

    def _append_step_to_conversation(
        self,
        conversation: list[dict[str, str]],
        step: ParsedChatStep,
        observation: str | None,
    ) -> None:
        conversation.append({"role": "assistant", "content": self._format_step_message(step)})
        if observation is not None:
            conversation.append({"role": "user", "content": f"Observation: {observation}"})

    def _prompt_template_messages(self, sample: Any) -> list[dict[str, str]]:
        return list(self.prompt_template.build_messages(sample))

    def _base_messages(self, sample: Any) -> list[dict[str, str]]:
        raise NotImplementedError

    def _step_instruction_message(self, step_index: int) -> dict[str, str]:
        raise NotImplementedError

    async def _sample_step_samples(
        self,
        *,
        sample: Any,
        messages: list[dict[str, str]],
        per_step_samples: int,
        step_index: int,
        sampling_cursor: int = 0,
    ) -> list[StepSample]:
        async def sample_fn(sample_count: int) -> list[StepSample | str | dict[str, Any]]:
            return await self._generate_step_samples(messages=messages, sample_count=sample_count)

        samples = await self.shared_step_sampler.sample(
            category=f"{self.name}/step_samples",
            sample_id=self._step_sampling_pool_id(),
            key=self._step_sampling_cache_key(sample, step_index=step_index, messages=messages),
            sample_count=per_step_samples,
            cursor=sampling_cursor,
            sample_fn=sample_fn,
        )
        return samples

    async def _generate_step_samples(
        self,
        *,
        messages: list[dict[str, str]],
        sample_count: int,
    ) -> list[StepSample | str | dict[str, Any]]:
        if self.step_output_format == "structured_json" and hasattr(self.model_client, "sample_many_detailed"):
            return await self._generate_structured_json_step_samples(messages=messages, sample_count=sample_count)

        if hasattr(self.model_client, "sample_many_detailed"):
            outputs = await self._invoke_generation_callable(
                getattr(self.model_client, "sample_many_detailed"),
                messages=messages,
                temperature=self.sample_temperature,
                n=sample_count,
            )
            if isinstance(outputs, list):
                return [self._coerce_step_sample(output) for output in outputs]

        if hasattr(self.model_client, "sample_many"):
            outputs = await self._invoke_generation_callable(
                getattr(self.model_client, "sample_many"),
                messages=messages,
                temperature=self.sample_temperature,
                n=sample_count,
            )
            if isinstance(outputs, list):
                return [self._coerce_step_sample(output) for output in outputs]

        outputs: list[StepSample | str | dict[str, Any]] = []
        generation_callable = self._resolve_generation_callable()
        for _ in range(sample_count):
            output = await self._invoke_generation_callable(generation_callable, messages=messages, temperature=self.sample_temperature)
            outputs.append(self._coerce_step_sample(output))
        return outputs

    async def _generate_structured_json_step_samples(
        self,
        *,
        messages: list[dict[str, str]],
        sample_count: int,
    ) -> list[StepSample]:
        detailed_callable = getattr(self.model_client, "sample_many_detailed")
        accepted: list[StepSample] = []
        rejected: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = sample_count * (1 + self.step_retry_attempts)

        while len(accepted) < sample_count and attempts < max_attempts:
            request_count = min(sample_count - len(accepted), max_attempts - attempts)
            attempts += request_count
            outputs = await detailed_callable(
                messages,
                temperature=self.sample_temperature,
                n=request_count,
                extra_body=self._structured_step_extra_body(),
            )
            if not isinstance(outputs, list):
                outputs = [outputs]
            for output in outputs:
                step_sample = self._coerce_step_sample(output)
                parsed_step = self._parse_step(step_sample.output)
                invalid_reason = self._invalid_step_reason(parsed_step, step_sample.metadata)
                if invalid_reason is None:
                    metadata = dict(step_sample.metadata)
                    metadata["step_output_format"] = "structured_json"
                    metadata["step_retry_attempts_used"] = len(rejected)
                    if rejected:
                        metadata["step_rejected_attempts"] = list(rejected)
                    accepted.append(StepSample(output=step_sample.output, metadata=metadata))
                    if len(accepted) >= sample_count:
                        break
                else:
                    rejected.append(
                        {
                            "reason": invalid_reason,
                            "truncated": bool(step_sample.metadata.get("truncated")),
                            "preview": step_sample.output[:240],
                        }
                    )

        while len(accepted) < sample_count:
            accepted.append(
                StepSample(
                    output='{"thought":"No valid structured step after retries.","action":"Finish[Unknown]"}',
                    metadata={
                        "step_output_format": "structured_json",
                        "step_generation_failed": True,
                        "step_retry_attempts_used": len(rejected),
                        "step_rejected_attempts": list(rejected),
                    },
                )
            )
        return accepted

    def _structured_step_extra_body(self) -> dict[str, Any]:
        return {
            "structured_outputs": {
                "json": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "action": {"type": "string"},
                    },
                    "required": ["thought", "action"],
                    "additionalProperties": False,
                }
            }
        }

    def _invalid_step_reason(self, step: ParsedChatStep, metadata: Mapping[str, Any]) -> str | None:
        if metadata.get("truncated"):
            return "truncated"
        if not step.action:
            return "missing action"
        if not self._is_valid_step_action(step.action):
            return f"invalid action: {step.action[:120]}"
        return None

    def _resolve_generation_callable(self) -> Any:
        for attribute in ("ainference", "inference", "agenerate", "generate"):
            candidate = getattr(self.model_client, attribute, None)
            if candidate is not None:
                return candidate
        if callable(self.model_client):
            return self.model_client
        raise TypeError(
            f"{type(self).__name__} requires a model_client with ainference, inference, agenerate, generate, or __call__."
        )

    async def _invoke_generation_callable(
        self,
        callable_obj: Any,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        n: int | None = None,
    ) -> Any:
        attempts = []
        if n is not None:
            attempts.extend(
                [
                    lambda: callable_obj(messages, temperature=temperature, n=n),
                    lambda: callable_obj(history=messages, temperature=temperature, n=n),
                    lambda: callable_obj(messages, n=n),
                    lambda: callable_obj(history=messages, n=n),
                ]
            )
        attempts.extend(
            [
                lambda: callable_obj(messages, temperature=temperature),
                lambda: callable_obj(history=messages, temperature=temperature),
                lambda: callable_obj(messages),
                lambda: callable_obj(history=messages),
            ]
        )

        last_error: Exception | None = None
        for attempt in attempts:
            try:
                result = attempt()
            except TypeError as exc:
                last_error = exc
                continue
            if inspect.isawaitable(result):
                return await result
            return result

        raise TypeError("Unsupported model_client call signature") from last_error

    def _coerce_output_text(self, output: Any) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            for key in ("content", "text", "response", "output"):
                value = output.get(key)
                if isinstance(value, str):
                    return value.strip()
        return str(output).strip()

    def _coerce_step_sample(self, output: Any) -> StepSample:
        if isinstance(output, StepSample):
            return output
        if isinstance(output, dict):
            metadata = output.get("metadata")
            return StepSample(
                output=self._coerce_output_text(output),
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )
        return StepSample(output=self._coerce_output_text(output))

    def _merge_parser_metadata(self, metadata: Mapping[str, Any], parsed_step: ParsedChatStep) -> dict[str, Any]:
        merged = dict(metadata)
        merged["parser_metadata"] = dict(parsed_step.metadata)
        return merged

    def _resolve_chosen_output_metadata(self, step: SampledChatStep) -> dict[str, Any]:
        if 0 <= step.chosen_output_index < len(step.sampled_output_metadata):
            metadata = step.sampled_output_metadata[step.chosen_output_index]
            if isinstance(metadata, dict):
                return dict(metadata)
        return {}

    def _backbone_sample_indices(self, backbone: BackboneTrajectory) -> dict[int, int]:
        sample_indices: dict[int, int] = {}
        for step in backbone.steps:
            chosen_output_index = step.metadata.get("chosen_output_index")
            if isinstance(chosen_output_index, int) and chosen_output_index >= 0:
                sample_indices[step.index] = chosen_output_index
        return sample_indices

    def _perplexity_from_output_metadata(self, metadata: dict[str, Any]) -> float | None:
        logprob_sum = metadata.get("token_logprob_sum")
        token_count = metadata.get("token_count")
        if not isinstance(logprob_sum, (int, float)) or not isinstance(token_count, int):
            return None
        if token_count <= 0:
            return None
        return float(math.exp(-float(logprob_sum) / float(token_count)))

    def _parse_step(self, raw_output: str) -> ParsedChatStep:
        content = raw_output.strip()
        if not content:
            return ParsedChatStep(
                thought=None,
                action="Finish[Unknown]",
                raw_content="",
                metadata={"action_source": "empty", "parser_recovered_action": False, "num_action_candidates": 0},
            )

        if content.startswith("{"):
            try:
                parsed_json = json.loads(self._strip_json_fence(content).strip())
            except json.JSONDecodeError:
                parsed_json = None
            if isinstance(parsed_json, Mapping):
                raw_action = parsed_json.get("action")
                raw_thought = parsed_json.get("thought")
                action = str(raw_action).strip() if raw_action is not None else ""
                thought = str(raw_thought).strip() if raw_thought is not None else None
                if self._is_valid_step_action(action):
                    return ParsedChatStep(
                        thought=thought or None,
                        action=action,
                        raw_content=content,
                        metadata={
                            "action_source": "json",
                            "parser_recovered_action": False,
                            "num_action_candidates": 1,
                            "role_leak_detected": self._detect_role_leak(content),
                        },
                    )

        candidates = self._extract_step_action_candidates(content)

        if candidates:
            action_line_candidates = [candidate for candidate in candidates if candidate.get("source") == "action_line"]
            chosen = action_line_candidates[-1] if action_line_candidates else candidates[-1]
            action = str(chosen["action"])
            source = str(chosen["source"])
            parser_recovered_action = not self._first_line_is_action(content, action)
            action_start = int(chosen["start"])
        else:
            action = content.splitlines()[0].strip().strip("`")
            source = "fallback_first_line"
            parser_recovered_action = False
            action_start = 0

        thought = self._extract_step_thought(content, action_start=action_start)
        return ParsedChatStep(
            thought=thought,
            action=action,
            raw_content=content,
            metadata={
                "action_source": source,
                "parser_recovered_action": parser_recovered_action,
                "num_action_candidates": len(candidates),
                "role_leak_detected": self._detect_role_leak(content),
            },
        )

    def _extract_step_action_candidates(self, content: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()

        for match in self._ACTION_LINE_PATTERN.finditer(content):
            raw_action = match.group(1).strip().strip("`")
            bracket_match = self._BRACKET_ACTION_PATTERN.search(raw_action)
            if bracket_match is None:
                continue
            action = bracket_match.group(1).strip()
            if not self._is_valid_step_action(action):
                continue
            key = (match.start(), action.lower())
            if key not in seen:
                candidates.append({"action": action, "source": "action_line", "start": match.start()})
                seen.add(key)

        for match in self._BRACKET_ACTION_PATTERN.finditer(content):
            action = match.group(1).strip()
            if not self._is_valid_step_action(action):
                continue
            key = (match.start(), action.lower())
            if key not in seen:
                candidates.append({"action": action, "source": "bracket", "start": match.start()})
                seen.add(key)

        return candidates

    def _extract_step_thought(self, content: str, *, action_start: int) -> str | None:
        matches = [match for match in self._THOUGHT_LINE_PATTERN.finditer(content) if match.start() <= action_start]
        if not matches:
            return None
        thought = matches[-1].group(1).strip().strip("`").strip()
        thought = re.sub(r"\s+", " ", thought)
        if not thought:
            return None
        lowered = thought.lower()
        if "action:" in lowered or "thought:" in lowered:
            return None
        if any(self._is_placeholder_action(match.group(1)) for match in self._BRACKET_ACTION_PATTERN.finditer(thought)):
            return None
        if len(thought) > 180:
            return None
        return thought

    def _is_valid_step_action(self, action: str) -> bool:
        stripped = action.strip()
        if self._BRACKET_ACTION_PATTERN.fullmatch(stripped) is None:
            return False
        argument = self._extract_action_argument(stripped).strip()
        if self._is_placeholder_action(stripped):
            return False
        if self._is_finish_action(stripped):
            return self._is_valid_finish_action(argument)
        return True

    def _is_placeholder_action(self, action: str) -> bool:
        argument = self._extract_action_argument(action).strip()
        normalized_argument = re.sub(r"\s+", " ", argument.lower())
        return normalized_argument in self._PLACEHOLDER_ACTION_ARGUMENTS

    def _is_valid_finish_action(self, answer: str) -> bool:
        return bool(answer.strip()) and answer.strip().lower() not in self._PLACEHOLDER_ACTION_ARGUMENTS

    @staticmethod
    def _detect_role_leak(content: str) -> bool:
        lowered = content.lower()
        return any(marker in lowered for marker in ("assistantanalysis", "assistantfinal", "<|assistant", "<|final"))

    @staticmethod
    def _first_line_is_action(content: str, action: str) -> bool:
        first_line = content.strip().splitlines()[0].strip().strip("`") if content.strip() else ""
        return first_line == action or first_line.lower() == f"action: {action}".lower()

    def _format_step_message(self, step: ParsedChatStep) -> str:
        if step.thought:
            return f"Thought: {step.thought}\nAction: {step.action}"
        return f"Action: {step.action}"

    def _is_finish_action(self, action: str) -> bool:
        return action.strip().lower().startswith("finish[")

    def _extract_finish_answer(self, action: str) -> str | None:
        match = re.match(r"\s*Finish\[(.*)\]\s*", action, re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).strip()

    def _extract_action_argument(self, action: str) -> str:
        match = re.match(r"\s*[A-Za-z]+\[(.*)\]\s*", action)
        if match is None:
            return action.strip()
        return match.group(1).strip()

    def _synthesize_observation(self, sample: Any, action: str) -> str:
        raise NotImplementedError
