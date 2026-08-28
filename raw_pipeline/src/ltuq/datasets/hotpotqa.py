from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class HotpotQAParagraph:
    title: str
    sentences: tuple[str, ...] = ()


@dataclass(frozen=True)
class HotpotQASupportingFact:
    title: str
    sentence_index: int


@dataclass(frozen=True)
class HotpotQASample:
    sample_id: str
    question: str
    answer: str
    context: tuple[HotpotQAParagraph, ...] = ()
    supporting_facts: tuple[HotpotQASupportingFact, ...] = ()
    qa_type: str = ""
    level: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_sentences(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def _coerce_context(row: dict[str, Any]) -> tuple[HotpotQAParagraph, ...]:
    context = row.get("context") or []
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        return tuple(
            HotpotQAParagraph(title=str(title), sentences=_coerce_sentences(sentence_group))
            for title, sentence_group in zip(titles, sentences)
        )

    paragraphs: list[HotpotQAParagraph] = []
    for item in context:
        if isinstance(item, dict):
            paragraphs.append(
                HotpotQAParagraph(
                    title=str(item.get("title", "")),
                    sentences=_coerce_sentences(item.get("sentences", [])),
                )
            )
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            paragraphs.append(
                HotpotQAParagraph(title=str(item[0]), sentences=_coerce_sentences(item[1]))
            )
    return tuple(paragraphs)


def _coerce_supporting_facts(row: dict[str, Any]) -> tuple[HotpotQASupportingFact, ...]:
    supporting_facts = row.get("supporting_facts") or {}
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or []
        sentence_ids = supporting_facts.get("sent_id") or []
        return tuple(
            HotpotQASupportingFact(title=str(title), sentence_index=int(sentence_id))
            for title, sentence_id in zip(titles, sentence_ids)
        )

    facts: list[HotpotQASupportingFact] = []
    for item in supporting_facts:
        if isinstance(item, dict):
            facts.append(
                HotpotQASupportingFact(
                    title=str(item.get("title", "")),
                    sentence_index=int(item.get("sent_id", 0)),
                )
            )
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            facts.append(HotpotQASupportingFact(title=str(item[0]), sentence_index=int(item[1])))
    return tuple(facts)


def format_hotpotqa_context(
    sample: HotpotQASample,
    max_paragraphs: int | None = None,
) -> str:
    paragraphs = sample.context if max_paragraphs is None else sample.context[:max_paragraphs]
    blocks: list[str] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        blocks.append(f"[{index}] {paragraph.title}\n" + " ".join(paragraph.sentences))
    return "\n\n".join(blocks)


def row_to_hotpotqa_sample(row: dict[str, Any], offset: int = 0) -> HotpotQASample:
    row_dict = dict(row)
    sample_id = str(row_dict.get("id", offset))
    return HotpotQASample(
        sample_id=sample_id,
        question=str(row_dict.get("question", "")),
        answer=str(row_dict.get("answer", "")),
        context=_coerce_context(row_dict),
        supporting_facts=_coerce_supporting_facts(row_dict),
        qa_type=str(row_dict.get("type", "")),
        level=str(row_dict.get("level", "")),
        metadata=row_dict,
    )


def load_hotpotqa_split(
    split: str = "validation",
    config_name: str = "distractor",
    dataset_name: str = "hotpotqa/hotpot_qa",
    revision: str | None = None,
) -> list[HotpotQASample]:
    dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        revision=revision,
    )
    return [row_to_hotpotqa_sample(dict(row), offset) for offset, row in enumerate(dataset)]
