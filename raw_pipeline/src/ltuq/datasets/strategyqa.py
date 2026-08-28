from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class StrategyQASample:
    sample_id: str
    question: str
    answer: Any
    facts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_facts(row: dict[str, Any]) -> tuple[str, ...]:
    facts = row.get("facts") or row.get("evidence") or row.get("decomposition") or []
    if isinstance(facts, str):
        return (facts,)
    if isinstance(facts, list):
        return tuple(str(item) for item in facts)
    return ()


def load_strategyqa_split(
    split: str = "train",
    dataset_name: str = "ChilleD/StrategyQA",
    revision: str | None = None,
) -> list[StrategyQASample]:
    dataset = load_dataset(dataset_name, split=split, revision=revision)
    samples: list[StrategyQASample] = []

    for offset, row in enumerate(dataset):
        row_dict = dict(row)
        sample_id = str(row_dict.get("id", offset))
        samples.append(
            StrategyQASample(
                sample_id=sample_id,
                question=str(row_dict.get("question", "")),
                answer=row_dict.get("answer"),
                facts=_coerce_facts(row_dict),
                metadata=row_dict,
            )
        )

    return samples
