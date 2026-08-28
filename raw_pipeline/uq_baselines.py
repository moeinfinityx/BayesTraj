"""Post-hoc uncertainty baselines for saved LLM-EUP response artifacts.

The implementations follow the published definitions of:

* Kernel Language Entropy (KLE)
* Sum of eigenvalues of the graph Laplacian (EigV)
* Confidence and Consistency-based UQ (CoCoA)
* Semantic Density

The estimators intentionally operate on arrays rather than repository-specific
objects.  This keeps the numerical definitions testable and lets the runner
reuse response generations without loading the generation backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import expm


EPS = 1e-12


def _as_square(matrix: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square, got shape={value.shape}.")
    if value.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values.")
    return value


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    value = _as_square(matrix, "matrix")
    return 0.5 * (value + value.T)


def semantic_similarity(
    entailment: np.ndarray,
    neutral: np.ndarray,
    contradiction: np.ndarray,
) -> np.ndarray:
    """Bidirectional graded semantic similarity used by Semantic Density.

    The paper defines distance as ``p(contra) + .5 p(neutral)``.  Similarity is
    therefore ``p(entail) + .5 p(neutral)``.  We average both NLI directions.
    """

    entailment = _as_square(entailment, "entailment")
    neutral = _as_square(neutral, "neutral")
    contradiction = _as_square(contradiction, "contradiction")
    if not (entailment.shape == neutral.shape == contradiction.shape):
        raise ValueError("NLI probability matrices must have identical shapes.")
    similarity = entailment + 0.5 * neutral
    similarity = np.clip(symmetrize(similarity), 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def normalized_laplacian(affinity: np.ndarray) -> np.ndarray:
    affinity = np.clip(symmetrize(affinity), 0.0, None)
    degree = affinity.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > EPS
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = np.eye(affinity.shape[0], dtype=np.float64)
    laplacian -= inverse_sqrt[:, None] * affinity * inverse_sqrt[None, :]
    return symmetrize(laplacian)


def eigv_score(affinity: np.ndarray) -> float:
    """Sum of clipped ``1-lambda`` values of the normalized Laplacian."""

    eigenvalues = np.linalg.eigvalsh(normalized_laplacian(affinity))
    return float(np.clip(1.0 - eigenvalues, 0.0, None).sum())


def _manual_kle_affinity(
    entailment: np.ndarray,
    neutral: np.ndarray,
    contradiction: np.ndarray,
) -> np.ndarray:
    """Construct the graph used by the official KLE implementation.

    Each directed NLI decision contributes 1 for entailment, .5 for neutral,
    and 0 for contradiction.  The official unweighted graph keeps an edge when
    the two directed decisions sum to at least 1.5.
    """

    probs = np.stack([contradiction, neutral, entailment], axis=-1)
    decisions = np.argmax(probs, axis=-1)
    directed = np.choose(decisions, [0.0, 0.5, 1.0])
    pair_weight = directed + directed.T
    affinity = (pair_weight >= 1.5).astype(np.float64)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def von_neumann_entropy(
    kernel: np.ndarray,
    *,
    scale: bool = True,
) -> float:
    kernel = _as_square(kernel, "kernel")
    diagonal = np.sqrt(np.clip(np.diag(kernel), EPS, None))
    density = kernel / np.outer(diagonal, diagonal)
    density /= float(kernel.shape[0])
    density = symmetrize(density)
    eigenvalues = np.clip(np.linalg.eigvalsh(density), 0.0, None)
    eigenvalues = eigenvalues[eigenvalues > 1e-8]
    entropy = float(-np.sum(eigenvalues * np.log(eigenvalues)))
    if scale and kernel.shape[0] > 1:
        entropy /= float(np.log(kernel.shape[0]))
    return entropy


def kle_heat_score(
    entailment: np.ndarray,
    neutral: np.ndarray,
    contradiction: np.ndarray,
    normalized_graph_laplacian: bool = False,
    scale: bool = True,
) -> float:
    """KLE with the heat kernel from the official NeurIPS implementation."""

    heat_time = 0.4
    affinity = _manual_kle_affinity(entailment, neutral, contradiction)
    degree = affinity.sum(axis=1)
    if normalized_graph_laplacian:
        laplacian = normalized_laplacian(affinity)
    else:
        laplacian = np.diag(degree) - affinity
    kernel = expm(-float(heat_time) * laplacian)
    return von_neumann_entropy(kernel, scale=scale)


def cocoa_score(
    target_sequence_logprob: float,
    target_to_samples_similarity: Sequence[float],
    *,
    confidence: str = "maxprob",
) -> float:
    """CoCoA risk for one selected output and its alternative samples."""

    similarities = np.asarray(target_to_samples_similarity, dtype=np.float64).reshape(-1)
    if similarities.size == 0:
        raise ValueError("CoCoA needs at least one alternative response.")
    if not np.all(np.isfinite(similarities)):
        raise ValueError("CoCoA similarities contain non-finite values.")
    consistency_risk = float(np.mean(1.0 - np.clip(similarities, 0.0, 1.0)))
    logprob = float(target_sequence_logprob)
    if confidence == "maxprob":
        confidence_risk = -np.expm1(min(logprob, 0.0))
    elif confidence == "ppl":
        # For PPL CoCoA, the caller supplies the mean token log probability.
        confidence_risk = -np.expm1(min(logprob, 0.0))
    else:
        raise ValueError(f"Unsupported CoCoA confidence={confidence!r}.")
    return float(confidence_risk * consistency_risk)


def semantic_density_score(
    selected_index: int,
    length_normalized_logprobs: Sequence[float],
    entailment: np.ndarray,
    neutral: np.ndarray,
    contradiction: np.ndarray,
    *,
    unique_texts: Optional[Sequence[str]] = None,
) -> float:
    """Negative semantic density, so larger values consistently mean uncertainty."""

    similarity = semantic_similarity(entailment, neutral, contradiction)
    n = similarity.shape[0]
    selected_index = int(selected_index)
    if selected_index < 0 or selected_index >= n:
        raise IndexError(f"selected_index={selected_index} outside [0, {n}).")
    logprobs = np.asarray(length_normalized_logprobs, dtype=np.float64).reshape(-1)
    if logprobs.shape[0] != n:
        raise ValueError(
            "length_normalized_logprobs/matrix size mismatch: "
            f"{logprobs.shape[0]} vs {n}."
        )
    indices = np.arange(n)
    if unique_texts is not None:
        if len(unique_texts) != n:
            raise ValueError("unique_texts length must match the NLI matrices.")
        seen = set()
        keep = []
        for idx, text in enumerate(unique_texts):
            normalized = str(text).strip()
            if normalized not in seen:
                seen.add(normalized)
                keep.append(idx)
        indices = np.asarray(keep, dtype=np.int64)
    weights = np.exp(np.clip(logprobs[indices], -745.0, 0.0))
    denominator = float(weights.sum())
    if denominator <= EPS:
        weights = np.ones_like(weights)
        denominator = float(weights.sum())
    density = float(
        np.sum(weights * similarity[selected_index, indices]) / denominator
    )
    return -density


@dataclass(frozen=True)
class BaselineScores:
    kle: np.ndarray
    eigv: np.ndarray
    cocoa_maxprob: np.ndarray
    cocoa_ppl: np.ndarray
    semantic_density: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "kle": self.kle,
            "eigv": self.eigv,
            "cocoa_maxprob": self.cocoa_maxprob,
            "cocoa_ppl": self.cocoa_ppl,
            "semantic_density": self.semantic_density,
        }


def evaluate_saved_responses(
    *,
    responses: Sequence[Sequence[str]],
    sample_sequence_logprobs: np.ndarray,
    sample_normalized_logprobs: np.ndarray,
    pred_sequence_logprobs: Sequence[float],
    pred_normalized_logprobs: Sequence[float],
    nli_probabilities: np.ndarray,
) -> BaselineScores:
    """Evaluate all four baselines.

    ``nli_probabilities`` has shape ``[N, 1+R, 1+R, 3]``.  Candidate zero is
    the greedy prediction and candidates 1..R are sampled responses.  The last
    dimension is ordered contradiction, neutral, entailment.
    """

    sample_sequence_logprobs = np.asarray(sample_sequence_logprobs, dtype=np.float64)
    sample_normalized_logprobs = np.asarray(sample_normalized_logprobs, dtype=np.float64)
    pred_sequence_logprobs = np.asarray(pred_sequence_logprobs, dtype=np.float64).reshape(-1)
    pred_normalized_logprobs = np.asarray(pred_normalized_logprobs, dtype=np.float64).reshape(-1)
    nli_probabilities = np.asarray(nli_probabilities, dtype=np.float64)
    n_questions = len(responses)
    expected = (n_questions, sample_sequence_logprobs.shape[1] + 1)
    if sample_sequence_logprobs.shape != sample_normalized_logprobs.shape:
        raise ValueError("Sample sequence-score arrays must have identical shapes.")
    if sample_sequence_logprobs.shape[0] != n_questions:
        raise ValueError("responses/sample sequence-score question count mismatch.")
    if pred_sequence_logprobs.shape[0] != n_questions:
        raise ValueError("pred sequence-score question count mismatch.")
    if pred_normalized_logprobs.shape[0] != n_questions:
        raise ValueError("pred normalized-score question count mismatch.")
    if nli_probabilities.shape != (expected[0], expected[1], expected[1], 3):
        raise ValueError(
            "Unexpected NLI tensor shape: "
            f"{nli_probabilities.shape}; expected "
            f"{(expected[0], expected[1], expected[1], 3)}."
        )

    output: Mapping[str, list] = {
        "kle": [],
        "eigv": [],
        "cocoa_maxprob": [],
        "cocoa_ppl": [],
        "semantic_density": [],
    }
    for qid, row_responses in enumerate(responses):
        n_resp = len(row_responses)
        if n_resp != sample_sequence_logprobs.shape[1]:
            raise ValueError(
                f"Question {qid} has {n_resp} responses; "
                f"score file has {sample_sequence_logprobs.shape[1]}."
            )
        # KLE and EigV are sample-set estimators and exclude the greedy output.
        sample_nli = nli_probabilities[qid, 1:, 1:, :]
        contra = sample_nli[:, :, 0]
        neutral = sample_nli[:, :, 1]
        entail = sample_nli[:, :, 2]
        output["kle"].append(kle_heat_score(entail, neutral, contra))
        output["eigv"].append(eigv_score(symmetrize(entail)))

        # CoCoA compares the greedy/selected response with sampled alternatives.
        full_nli = nli_probabilities[qid]
        full_similarity = semantic_similarity(
            full_nli[:, :, 2],
            full_nli[:, :, 1],
            full_nli[:, :, 0],
        )
        target_similarity = full_similarity[0, 1:]
        output["cocoa_maxprob"].append(
            cocoa_score(
                pred_sequence_logprobs[qid],
                target_similarity,
                confidence="maxprob",
            )
        )
        output["cocoa_ppl"].append(
            cocoa_score(
                pred_normalized_logprobs[qid],
                target_similarity,
                confidence="ppl",
            )
        )

        # Published Semantic Density selects a response and evaluates its
        # density among probability-weighted alternatives.  Candidate zero is
        # the greedy response and is included exactly once.
        candidate_norm_logprobs = np.concatenate(
            ([pred_normalized_logprobs[qid]], sample_normalized_logprobs[qid])
        )
        candidate_texts = ["__greedy_candidate__"] + list(row_responses)
        output["semantic_density"].append(
            semantic_density_score(
                0,
                candidate_norm_logprobs,
                full_nli[:, :, 2],
                full_nli[:, :, 1],
                full_nli[:, :, 0],
                unique_texts=candidate_texts,
            )
        )

    return BaselineScores(
        **{
            key: np.asarray(values, dtype=np.float64)
            for key, values in output.items()
        }
    )
