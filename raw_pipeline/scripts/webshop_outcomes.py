"""Label-free WebShop outcome functionals used by BayesTraj."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Sequence

import numpy as np


def entropy(values: Sequence[str]) -> float:
    counts = Counter(values); total = float(len(values))
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def is_purchase(bucket: str) -> bool:
    return bucket.startswith("purchase:")


def hierarchical_entropy(buckets: Sequence[str]) -> float:
    purchase = [bucket for bucket in buckets if is_purchase(bucket)]
    failure = [bucket for bucket in buckets if not is_purchase(bucket)]
    total = float(len(buckets)); completion = ["purchase" if is_purchase(bucket) else "no-purchase" for bucket in buckets]
    return entropy(completion) + 0.25 * len(purchase) / total * (entropy(purchase) if purchase else 0.0) + len(failure) / total * (entropy(failure) if failure else 0.0)


def _seed(sample_id: str, prefix: int) -> int:
    payload = f"webshop-hierarchical|{sample_id}|{prefix}|gamma=0.25|0.5"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**32)


def posterior_moments(buckets: Sequence[str], sample_id: str, prefix: int, draws: int) -> tuple[float, float]:
    support = [*sorted(set(buckets)), "other:purchase", "other:no-purchase"]
    parameters = np.asarray([buckets.count(key) + 0.5 for key in support[:-2]] + [0.25, 0.25])
    purchase_mask = np.asarray([is_purchase(key) or key == "other:purchase" for key in support])
    def functional(rows):
        p = np.sum(rows[:, purchase_mask], axis=1); q = 1 - p
        root = -p * np.log(np.maximum(p, 1e-300)) - q * np.log(np.maximum(q, 1e-300))
        detail = -np.sum(rows[:, purchase_mask] * np.log(np.maximum(rows[:, purchase_mask] / np.maximum(p[:, None], 1e-300), 1e-300)), axis=1)
        failures = -np.sum(rows[:, ~purchase_mask] * np.log(np.maximum(rows[:, ~purchase_mask] / np.maximum(q[:, None], 1e-300), 1e-300)), axis=1)
        return root + 0.25 * detail + failures
    probabilities = np.random.default_rng(_seed(sample_id, prefix)).dirichlet(parameters, size=draws)
    values = functional(probabilities)
    return float(np.mean(values)), float(np.var(values))


def target(buckets: Sequence[str]) -> float:
    return hierarchical_entropy(buckets)
