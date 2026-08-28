from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..datasets.strategyqa import StrategyQASample
from ..models import ModelGenerationError
from ..sampling import SharedSamplingStorage
from ..trajectory import TrajectoryDependentDecisionProcess
from .chat_local_branching import ChatLocalBranchingExecutor


@dataclass
class StrategyQAPromptTemplate:
    system_prompt: str = (
        "You are a reasoning agent answering StrategyQA yes/no questions. "
        "Expose reasoning in short discrete steps and end with a final yes/no answer."
    )

    def build_messages(self, sample: StrategyQASample) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {sample.question}"},
        ]


class StrategyQALocalBranchingExecutor(ChatLocalBranchingExecutor):
    name = "strategyqa"

    def __init__(
        self,
        model_client: Any,
        prompt_template: StrategyQAPromptTemplate | None = None,
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
        step_stop_callback: Any | None = None,
    ) -> None:
        super().__init__(
            model_client,
            prompt_template or StrategyQAPromptTemplate(),
            max_steps=max_steps,
            sample_temperature=sample_temperature,
            backbone_per_step_samples=backbone_per_step_samples,
            next_step_entropy_samples=next_step_entropy_samples,
            collect_sample_logprobs=collect_sample_logprobs,
            shared_sampling_storage=shared_sampling_storage,
            model_signature=model_signature,
            step_output_format=step_output_format,
            step_retry_attempts=step_retry_attempts,
            step_stop_callback=step_stop_callback,
        )

    def _sample_signature(self, sample: StrategyQASample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "facts": list(sample.facts),
        }

    def _base_messages(self, sample: StrategyQASample) -> list[dict[str, str]]:
        messages = self._prompt_template_messages(sample)
        if sample.facts:
            evidence_lines = "\n".join(f"- {fact}" for fact in sample.facts)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Available evidence for the question:\n"
                        f"{evidence_lines}\n\n"
                        "Use the evidence when deciding each next step."
                    ),
                }
            )
        return messages

    def _step_instruction_message(self, step_index: int) -> dict[str, str]:
        if self.step_output_format == "structured_json":
            return {
                "role": "user",
                "content": (
                    f"Produce exactly one next reasoning step for step {step_index}. "
                    "Return only one JSON object and nothing else. "
                    'Schema: {"thought":"<=12 words", '
                    '"action":"Search[entity] or Lookup[keyword] or Finish[Yes/No]"}. '
                    "No markdown. No extra explanation. "
                    "Use Finish[...] when you can answer the StrategyQA question."
                ),
            }
        return {
            "role": "user",
            "content": (
                f"Produce exactly one next reasoning step for step {step_index}. "
                "Respond with two lines in this exact format:\n"
                "Thought: <brief reasoning>\n"
                "Action: Search[entity] or Lookup[keyword] or Finish[Yes/No]\n"
                "Use Finish[...] when you can answer the StrategyQA question."
            ),
        }

    def _is_valid_finish_action(self, answer: str) -> bool:
        return answer.strip().lower() in {"yes", "no"}

    def _synthesize_observation(self, sample: StrategyQASample, action: str) -> str:
        if not sample.facts:
            return "No supporting evidence available."
        query = self._extract_action_argument(action)
        query_tokens = set(re.findall(r"\w+", query.lower()))

        def score_fact(fact: str) -> tuple[int, int]:
            normalized_fact = fact.lower()
            fact_tokens = set(re.findall(r"\w+", normalized_fact))
            overlap = len(query_tokens & fact_tokens)
            exact_match = 1 if query and query.lower() in normalized_fact else 0
            return (exact_match, overlap)

        best_fact = max(sample.facts, key=score_fact)
        return best_fact
