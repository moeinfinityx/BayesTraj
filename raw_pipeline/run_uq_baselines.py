#!/usr/bin/env python
"""Run KLE, EigV, CoCoA, and Semantic Density on saved LLM-EUP artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from uq_baselines import evaluate_saved_responses


DEFAULT_NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"


def _bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}.")


def _read_jsonl(path: Path, field: str):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if field not in row:
                raise KeyError(f"{path}:{line_no} missing field {field!r}.")
            rows.append((int(row.get("i", len(rows))), row[field]))
    if field != "responses":
        rows.sort(key=lambda item: item[0])
    return [value for _, value in rows]


def _find_preds(split_dir: Path) -> Path:
    preferred = split_dir / "preds.jsonl"
    if preferred.exists():
        return preferred
    candidates = sorted(split_dir.glob("preds_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No preds JSONL found in {split_dir}.")
    return candidates[0]


def _resolve_split_dir(artifact_dir: Path, split: str) -> Path:
    direct = artifact_dir.expanduser().resolve()
    if (direct / "questions.jsonl").exists():
        return direct
    candidate = direct / split
    if (candidate / "questions.jsonl").exists():
        return candidate
    raise FileNotFoundError(
        f"{artifact_dir} is neither a split directory nor an artifact root "
        f"containing split={split!r}."
    )


def _load_score_file(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence-score artifact: {path}.")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "sequence_logprob",
            "length_normalized_sequence_logprob",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} missing arrays: {missing}.")
        return {key: np.asarray(data[key]) for key in data.files}


def _load_labels(split_dir: Path) -> Tuple[np.ndarray, str]:
    for name, kind in (
        ("ambig_label.npy", "ambig"),
        ("ood_label.npy", "ood"),
    ):
        path = split_dir / name
        if path.exists():
            return np.asarray(np.load(path), dtype=np.int64).reshape(-1), kind
    y_path = split_dir / "y.npy"
    if not y_path.exists():
        raise FileNotFoundError(
            f"No ambig_label.npy, ood_label.npy, or y.npy in {split_dir}."
        )
    return 1 - np.asarray(np.load(y_path), dtype=np.int64).reshape(-1), "1-y"


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")


def _label_indices(model) -> Tuple[int, int, int]:
    label2id = {
        str(key).lower(): int(value)
        for key, value in (getattr(model.config, "label2id", None) or {}).items()
    }
    id2label = {
        int(key): str(value).lower()
        for key, value in (getattr(model.config, "id2label", None) or {}).items()
    }
    for index, label in id2label.items():
        label2id.setdefault(label, index)

    def find(name: str, fallback: int) -> int:
        for label, index in label2id.items():
            if name in label:
                return int(index)
        return fallback

    contradiction = find("contradiction", 0)
    neutral = find("neutral", 1)
    entailment = find("entailment", 2)
    if len({contradiction, neutral, entailment}) != 3:
        raise ValueError(
            f"Could not resolve distinct NLI labels from {model.config.id2label!r}."
        )
    return contradiction, neutral, entailment


def _cached_weight_snapshot(model_name: str) -> Path | None:
    repository = "models--" + model_name.replace("/", "--")
    snapshot_root = Path(HF_HUB_CACHE) / repository / "snapshots"
    if not snapshot_root.exists():
        return None
    candidates = sorted(
        {
            path.parent
            for pattern in ("model.safetensors", "pytorch_model.bin")
            for path in snapshot_root.glob(f"*/{pattern}")
        }
    )
    return candidates[-1] if candidates else None


def _load_nli_components(model_name: str):
    """Load a three-way NLI checkpoint, including split HF cache snapshots."""

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                fix_mistral_regex=True,
            )
        except AttributeError:
            # Transformers 5.5 attempts to apply its Mistral regex patch to
            # DeBERTa's SentencePiece tokenizer and dereferences a nonexistent
            # ``backend_tokenizer`` attribute.  The patch is Mistral-specific;
            # loading the same DeBERTa tokenizer without it is the correct
            # compatibility fallback.
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
        )
        return tokenizer, model
    except (OSError, AttributeError) as direct_error:
        # Hugging Face may store a converted safetensors revision separately
        # from the main revision containing config/tokenizer files.
        weight_snapshot = _cached_weight_snapshot(model_name)
        if weight_snapshot is not None:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                    fix_mistral_regex=True,
                )
            except AttributeError:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
            model = AutoModelForSequenceClassification.from_pretrained(
                weight_snapshot,
                config=config,
                local_files_only=True,
            )
            return tokenizer, model
        if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
            "TRANSFORMERS_OFFLINE"
        ) == "1":
            raise OSError(
                f"NLI model {model_name!r} is incomplete in the local cache."
            ) from direct_error
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                fix_mistral_regex=True,
            )
        except AttributeError:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model


def _question_conditioned(question: str, response: str) -> str:
    return f"{str(question).strip()} {str(response).strip()}".strip()


def collect_nli_probabilities(
    *,
    questions: Sequence[str],
    preds: Sequence[str],
    responses: Sequence[Sequence[str]],
    model_name: str,
    device: str,
    batch_size: int,
    cache_path: Path,
    force: bool,
) -> np.ndarray:
    n_questions = len(questions)
    n_resp = len(responses[0])
    expected_shape = (n_questions, n_resp + 1, n_resp + 1, 3)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            probs = np.asarray(cached["probabilities"], dtype=np.float32)
            cached_model = str(cached["model_name"].item())
        if probs.shape != expected_shape:
            raise ValueError(
                f"Cached NLI tensor shape={probs.shape}, expected={expected_shape}."
            )
        if cached_model != model_name:
            raise ValueError(
                f"Cached NLI model={cached_model!r}, requested={model_name!r}."
            )
        print(f"[NLI] Reusing {cache_path}", flush=True)
        return probs

    resolved_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer, model = _load_nli_components(model_name)
    model = model.to(resolved_device)
    model.eval()
    contra_idx, neutral_idx, entail_idx = _label_indices(model)

    partial_path = cache_path.with_suffix(".partial.npy")
    completed_path = cache_path.with_suffix(".completed.npy")
    if (
        partial_path.exists()
        and completed_path.exists()
        and not force
    ):
        probabilities = np.lib.format.open_memmap(
            partial_path,
            mode="r+",
            dtype=np.float32,
            shape=expected_shape,
        )
        completed = np.load(completed_path)
        if completed.shape != (n_questions,):
            raise ValueError(f"Invalid NLI completion checkpoint: {completed_path}.")
    else:
        probabilities = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
        completed = np.zeros(n_questions, dtype=bool)
        np.save(completed_path, completed)

    order = [contra_idx, neutral_idx, entail_idx]
    with torch.inference_mode():
        for qid in range(n_questions):
            if bool(completed[qid]):
                continue
            candidates = [preds[qid]] + list(responses[qid])
            conditioned = [
                _question_conditioned(questions[qid], response)
                for response in candidates
            ]
            premises = []
            hypotheses = []
            for first in conditioned:
                for second in conditioned:
                    premises.append(first)
                    hypotheses.append(second)
            row_probs = []
            for start in range(0, len(premises), batch_size):
                encoded = tokenizer(
                    premises[start : start + batch_size],
                    hypotheses[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[:, order]
                row_probs.append(probs.detach().float().cpu().numpy())
            size = n_resp + 1
            probabilities[qid] = np.concatenate(row_probs, axis=0).reshape(size, size, 3)
            completed[qid] = True
            probabilities.flush()
            np.save(completed_path, completed)
            if (qid + 1) % 10 == 0 or qid + 1 == n_questions:
                print(f"[NLI] {qid + 1}/{n_questions} questions", flush=True)

    final = np.asarray(probabilities, dtype=np.float32).copy()
    np.savez_compressed(
        cache_path,
        probabilities=final,
        model_name=np.asarray(model_name),
        candidate_order=np.asarray("greedy_then_samples"),
        label_order=np.asarray(["contradiction", "neutral", "entailment"]),
    )
    return final


def _metric(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(scores)
    labels = np.asarray(labels, dtype=np.int64)[finite]
    scores = np.asarray(scores, dtype=np.float64)[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "aupr": float("nan"), "n": int(labels.size)}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "n": int(labels.size),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KLE, EigV, CoCoA, and Semantic Density."
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_resp", type=int, default=10)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--nli_model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--nli_device", default="auto")
    parser.add_argument("--nli_batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", type=_bool, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dir = _resolve_split_dir(Path(args.artifact_dir), args.split)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else split_dir / "uq_baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metrics.json"
    scores_path = output_dir / "scores.npz"
    if (results_path.exists() or scores_path.exists()) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {results_path} or {scores_path}; "
            "choose a new --output_dir or pass --force=True."
        )

    questions = _read_jsonl(split_dir / "questions.jsonl", "question")
    preds_path = _find_preds(split_dir)
    preds = _read_jsonl(preds_path, "pred")
    responses_path = split_dir / "se_resp_res.jsonl"
    if not responses_path.exists():
        responses_path = split_dir / "se_resp.jsonl"
    responses = _read_jsonl(responses_path, "responses")
    sample_scores_path = split_dir / f"se_resp_res_sequence_scores_{args.n_resp}.npz"
    sample_scores = _load_score_file(sample_scores_path)
    pred_scores_path = split_dir / "preds_sequence_scores.npz"
    pred_scores = _load_score_file(pred_scores_path)
    labels, label_kind = _load_labels(split_dir)
    lengths = [
        len(questions),
        len(preds),
        len(responses),
        len(labels),
        sample_scores["sequence_logprob"].shape[0],
        pred_scores["sequence_logprob"].shape[0],
    ]
    n = min(lengths)
    if len(set(lengths)) != 1:
        raise ValueError(f"Artifact question-count mismatch: {lengths}.")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        n = min(n, args.limit)
    questions = questions[:n]
    preds = preds[:n]
    responses = responses[:n]
    labels = labels[:n]
    for qid, row in enumerate(responses):
        if len(row) != args.n_resp:
            raise ValueError(
                f"Question {qid} contains {len(row)} responses; expected {args.n_resp}."
            )

    cache_name = (
        f"pairwise_nli_{args.n_resp}_{_model_slug(args.nli_model)}"
        f"_n{n}.npz"
    )
    nli_path = output_dir / cache_name
    nli = collect_nli_probabilities(
        questions=questions,
        preds=preds,
        responses=responses,
        model_name=args.nli_model,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        cache_path=nli_path,
        force=args.force,
    )

    scores = evaluate_saved_responses(
        responses=responses,
        sample_sequence_logprobs=sample_scores["sequence_logprob"][:n],
        sample_normalized_logprobs=sample_scores[
            "length_normalized_sequence_logprob"
        ][:n],
        pred_sequence_logprobs=pred_scores["sequence_logprob"][:n],
        pred_normalized_logprobs=pred_scores[
            "length_normalized_sequence_logprob"
        ][:n],
        nli_probabilities=nli,
    ).as_dict()

    metrics = {name: _metric(labels, values) for name, values in scores.items()}
    np.savez_compressed(
        scores_path,
        labels=labels,
        **scores,
    )
    payload = {
        "artifact_split_dir": str(split_dir),
        "split": args.split,
        "label_kind": label_kind,
        "n_questions": int(n),
        "n_responses": int(args.n_resp),
        "similarity_source": "nli",
        "nli_model": args.nli_model,
        "nli_cache": str(nli_path),
        "heat_time": 0.4,
        "methods": {
            "kle": "heat-kernel KLE",
            "eigv": "sum of clipped 1-lambda normalized-Laplacian eigenvalues",
            "cocoa_maxprob": "CoCoA with sequence probability confidence",
            "cocoa_ppl": "CoCoA with mean-token probability confidence",
            "semantic_density": "negative probability-weighted semantic density",
        },
        "metrics": metrics,
        "inputs": {
            "questions": str(split_dir / "questions.jsonl"),
            "preds": str(preds_path),
            "responses": str(responses_path),
            "sample_scores": str(sample_scores_path),
            "pred_scores": str(pred_scores_path),
        },
    }
    temporary = results_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, results_path)
    payload["output_sha256"] = {
        "metrics": _sha256(results_path),
        "scores": _sha256(scores_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
