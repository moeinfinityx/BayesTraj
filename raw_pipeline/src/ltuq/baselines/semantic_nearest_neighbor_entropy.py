"""Semantic Nearest-Neighbor Entropy adapted from the authors' public code.

The official implementation is MIT licensed and available at
https://github.com/BigML-CS-UCLA/SNNE.  LTUQ keeps its ``only_denom`` score,
includes diagonal self-similarity, uses ROUGE-L F1, and fixes temperature at
one so no correctness labels are used for hyperparameter selection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
import math
from typing import Any, Final

from ..trajectory import TrajectoryDependentDecisionProcess


SNNE_BASELINE: Final[str] = "snne"
DEFAULT_SNNE_TEMPERATURE: Final[float] = 1.0
SNNE_VARIANT: Final[str] = "only_denom"
SNNE_INCLUDE_SELF_SIMILARITY: Final[bool] = True
SNNE_SIMILARITY: Final[str] = "rougeL_fmeasure"
SNNE_FORMULA_VERSION: Final[str] = "nguyen-acl-findings-2025-official-only-denom-v1"
SNNE_OFFICIAL_REPOSITORY: Final[str] = "https://github.com/BigML-CS-UCLA/SNNE"
SNNE_OFFICIAL_COMMIT: Final[str] = "d3e95fd5ffc6edbc669c9704a97e7b335aaa416d"


PairwiseSimilarity = Callable[[str, str], float]


@dataclass(frozen=True)
class SNNEResponseDiagnostics:
    index: int
    sample_id: str
    response: str
    response_source: str


@dataclass(frozen=True)
class SNNEPairDiagnostics:
    source_index: int
    target_index: int
    similarity: float


@dataclass(frozen=True)
class SNNEDiagnostics:
    score: float | None
    temperature: float
    num_responses: int
    formula_version: str = SNNE_FORMULA_VERSION
    variant: str = SNNE_VARIANT
    include_self_similarity: bool = SNNE_INCLUDE_SELF_SIMILARITY
    similarity_name: str = SNNE_SIMILARITY
    official_repository: str = SNNE_OFFICIAL_REPOSITORY
    official_commit: str = SNNE_OFFICIAL_COMMIT
    responses: list[SNNEResponseDiagnostics] = field(default_factory=list)
    pairs: list[SNNEPairDiagnostics] = field(default_factory=list)
    row_logsumexp: list[float] = field(default_factory=list)
    unavailable_reason: str | None = None
    similarity_provenance: dict[str, Any] = field(default_factory=dict)


class RougeLSimilarity:
    """ROUGE-L F1 backend matching the official SNNE lexical path."""

    def __init__(self) -> None:
        try:
            from rouge_score.rouge_scorer import RougeScorer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("snne requires the rouge-score package.") from exc
        self._scorer = RougeScorer(["rougeL"], use_stemmer=False)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "rouge_score.rouge_scorer.RougeScorer",
            "metric": "rougeL",
            "statistic": "fmeasure",
            "use_stemmer": False,
            "package_version": version("rouge-score"),
        }

    def __call__(self, left: str, right: str) -> float:
        return float(self._scorer.score(left, right)["rougeL"].fmeasure)


def validate_snne_temperature(temperature: float) -> float:
    resolved = float(temperature)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("SNNE temperature must be finite and positive.")
    return resolved


def _trajectory_response(tdp: TrajectoryDependentDecisionProcess) -> tuple[str, str] | None:
    if isinstance(tdp.final_answer, str) and tdp.final_answer.strip():
        return tdp.final_answer.strip(), "final_answer"
    if tdp.steps and tdp.steps[-1].realized_decision.strip():
        return tdp.steps[-1].realized_decision.strip(), "terminal_realized_decision"
    return None


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def compute_snne_diagnostics(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    *,
    similarity_fn: PairwiseSimilarity | None = None,
    temperature: float = DEFAULT_SNNE_TEMPERATURE,
) -> SNNEDiagnostics:
    """Compute official-code SNNE over stored trajectory final responses."""

    resolved_temperature = validate_snne_temperature(temperature)
    resolved_similarity = similarity_fn or RougeLSimilarity()
    provenance = getattr(
        resolved_similarity,
        "provenance",
        {"backend": getattr(resolved_similarity, "__qualname__", type(resolved_similarity).__name__)},
    )
    if len(tdps) < 2:
        return SNNEDiagnostics(
            score=None,
            temperature=resolved_temperature,
            num_responses=len(tdps),
            unavailable_reason="requires_at_least_two_complete_trajectories",
            similarity_provenance=dict(provenance),
        )
    if len({tdp.prompt.strip() for tdp in tdps}) != 1:
        return SNNEDiagnostics(
            score=None,
            temperature=resolved_temperature,
            num_responses=len(tdps),
            unavailable_reason="trajectories_do_not_share_one_prompt",
            similarity_provenance=dict(provenance),
        )

    extracted = [_trajectory_response(tdp) for tdp in tdps]
    if any(item is None for item in extracted):
        missing = [index for index, item in enumerate(extracted) if item is None]
        return SNNEDiagnostics(
            score=None,
            temperature=resolved_temperature,
            num_responses=len(tdps),
            unavailable_reason=f"missing_final_response_at_indices:{missing}",
            similarity_provenance=dict(provenance),
        )
    resolved = [item for item in extracted if item is not None]
    responses = [
        SNNEResponseDiagnostics(
            index=index,
            sample_id=tdps[index].sample_id,
            response=response,
            response_source=source,
        )
        for index, (response, source) in enumerate(resolved)
    ]

    size = len(responses)
    similarities = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    pairs: list[SNNEPairDiagnostics] = []
    for i in range(size):
        for j in range(i + 1, size):
            value = float(resolved_similarity(responses[i].response, responses[j].response))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("SNNE similarities must be finite and in [0, 1].")
            similarities[i][j] = value
            similarities[j][i] = value
            pairs.append(SNNEPairDiagnostics(source_index=i, target_index=j, similarity=value))

    row_logsumexp = [
        _logsumexp([similarities[i][j] / resolved_temperature for j in range(size)])
        for i in range(size)
    ]
    score = -sum(row_logsumexp) / size
    return SNNEDiagnostics(
        score=score,
        temperature=resolved_temperature,
        num_responses=size,
        responses=responses,
        pairs=pairs,
        row_logsumexp=row_logsumexp,
        similarity_provenance=dict(provenance),
    )


def compute_snne(
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    *,
    similarity_fn: PairwiseSimilarity | None = None,
    temperature: float = DEFAULT_SNNE_TEMPERATURE,
) -> float | None:
    return compute_snne_diagnostics(
        tdps,
        similarity_fn=similarity_fn,
        temperature=temperature,
    ).score
