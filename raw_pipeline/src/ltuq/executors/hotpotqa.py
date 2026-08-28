from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, cast

from ..datasets.hotpotqa import HotpotQASample, format_hotpotqa_context
from ..models import ModelGenerationError
from ..sampling import SharedSamplingStorage
from ..trajectory import TrajectoryDependentDecisionProcess
from .chat_local_branching import ChatLocalBranchingExecutor


@dataclass
class HotpotQAPromptTemplate:
    system_prompt: str = (
        "You are a multi-hop QA agent answering HotpotQA questions. "
        "Reason in short discrete steps grounded in the available evidence and end with a short final answer."
    )
    include_context: bool = True
    max_context_paragraphs: int = 10

    def build_messages(self, sample: HotpotQASample) -> list[dict[str, str]]:
        user_content = f"Question: {sample.question}"
        if self.include_context and sample.context:
            context = format_hotpotqa_context(sample, max_paragraphs=self.max_context_paragraphs)
            user_content = f"Question: {sample.question}\n\nContext:\n{context}\n\nAnswer:"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


class HotpotQALocalBranchingExecutor(ChatLocalBranchingExecutor):
    name = "hotpotqa"

    def __init__(
        self,
        model_client: Any,
        prompt_template: HotpotQAPromptTemplate | None = None,
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
    ) -> None:
        super().__init__(
            model_client,
            cast(Any, prompt_template or HotpotQAPromptTemplate()),
            max_steps=max_steps,
            sample_temperature=sample_temperature,
            backbone_per_step_samples=backbone_per_step_samples,
            next_step_entropy_samples=next_step_entropy_samples,
            collect_sample_logprobs=collect_sample_logprobs,
            shared_sampling_storage=shared_sampling_storage,
            model_signature=model_signature,
            step_output_format=step_output_format,
            step_retry_attempts=step_retry_attempts,
        )

    def _prompt_template_signature(self) -> dict[str, Any]:
        prompt_template = cast(HotpotQAPromptTemplate, self.prompt_template)
        return {
            "type": f"{type(prompt_template).__module__}.{type(prompt_template).__qualname__}",
            "system_prompt": prompt_template.system_prompt,
            "include_context": prompt_template.include_context,
            "max_context_paragraphs": prompt_template.max_context_paragraphs,
        }

    def _sample_signature(self, sample: HotpotQASample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "answer": sample.answer,
            "qa_type": sample.qa_type,
            "level": sample.level,
            "context": [
                {
                    "title": paragraph.title,
                    "sentences": list(paragraph.sentences),
                }
                for paragraph in sample.context
            ],
        }

    def _base_messages(self, sample: HotpotQASample) -> list[dict[str, str]]:
        messages = self._prompt_template_messages(sample)
        if sample.supporting_facts:
            fact_lines = "\n".join(
                f"- {fact.title} [sentence {fact.sentence_index}]" for fact in sample.supporting_facts
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Known supporting facts in the provided context:\n"
                        f"{fact_lines}\n\n"
                        "Use the context to ground each action."
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
                    '"action":"Search[entity] or Lookup[keyword] or Finish[short answer]"}. '
                    "No markdown. No extra explanation. "
                    "Use Search[...] or Lookup[...] to inspect the provided context and Finish[...] when you can answer the HotpotQA question."
                ),
            }
        return {
            "role": "user",
            "content": (
                f"Produce exactly one next reasoning step for step {step_index}. "
                "Respond with two lines in this exact format:\n"
                "Thought: <brief reasoning>\n"
                "Action: Search[entity] or Lookup[keyword] or Finish[short answer]\n"
                "Use Search[...] or Lookup[...] to inspect the provided context and Finish[...] when you can answer the HotpotQA question."
            ),
        }

    def _synthesize_observation(self, sample: HotpotQASample, action: str) -> str:
        if not sample.context:
            return "No supporting context available."

        query = self._extract_action_argument(action)
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            query_tokens = set(re.findall(r"\w+", sample.question.lower()))

        best_paragraph = None
        best_sentence = None
        best_score = (-1, -1, -1)

        for paragraph_index, paragraph in enumerate(sample.context, start=1):
            title_tokens = set(re.findall(r"\w+", paragraph.title.lower()))
            title_overlap = len(query_tokens & title_tokens)
            for sentence_index, sentence in enumerate(paragraph.sentences):
                sentence_tokens = set(re.findall(r"\w+", sentence.lower()))
                overlap = len(query_tokens & sentence_tokens)
                exact_match = 1 if query and query.lower() in sentence.lower() else 0
                score = (exact_match, overlap, title_overlap)
                if score > best_score:
                    best_score = score
                    best_paragraph = (paragraph_index, paragraph)
                    best_sentence = (sentence_index, sentence)

        if best_paragraph is None:
            paragraph = sample.context[0]
            preview = " ".join(paragraph.sentences[:2]).strip()
            return f"[{1}] {paragraph.title}: {preview}".strip()

        paragraph_index, paragraph = best_paragraph
        if best_sentence is None:
            preview = " ".join(paragraph.sentences[:2]).strip()
            return f"[{paragraph_index}] {paragraph.title}: {preview}".strip()

        sentence_index, sentence = best_sentence
        return f"[{paragraph_index}] {paragraph.title} [sentence {sentence_index}]: {sentence}"
