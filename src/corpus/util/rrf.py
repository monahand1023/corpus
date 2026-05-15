"""Reciprocal Rank Fusion — combines results from multiple rankers (dense
vector search + BM25) into a single ranked list without requiring score
normalization. Formula: score(item) = sum over rankers of weight / (k + rank).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    key: Callable[[T], Any] | None = None,
) -> list[T]:
    if key is None:
        key = lambda item: item  # noqa: E731
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[object, float] = {}
    first_seen: dict[object, T] = {}

    for lst, weight in zip(ranked_lists, weights, strict=True):
        for rank, item in enumerate(lst):
            kid = key(item)
            scores[kid] = scores.get(kid, 0.0) + weight / (k + rank + 1)
            first_seen.setdefault(kid, item)

    ordered_ids = sorted(scores.keys(), key=lambda i: -scores[i])
    return [first_seen[i] for i in ordered_ids]
