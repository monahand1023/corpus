"""Pure retrieval-quality metrics.

NO retriever/config/corpus imports — every function takes only string keys plus a
scalar ``k`` and returns floats, so this module can be reused BYTE-IDENTICAL across
separate RAG deployments. Keep it that way: nothing corpus-specific belongs here.

Conventions:
- ``expected_keys`` is an OR-set of relevant ``source_key`` strings — a query
  "hits" if ANY expected key appears in the results (binary relevance).
- ``ranked_found_keys`` is the retrieved ``source_key`` list in rank order (rank 1
  first), already deduped by source (as the Retriever returns it).
- Empty ``expected_keys`` marks a negative/informational query; callers exclude
  those from scoring, and the functions here still return 0.0 defensively.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def recall_at_k(expected_keys: Sequence[str], ranked_found_keys: Sequence[str], k: int) -> float:
    """1.0 if any expected key appears in the top-k, else 0.0."""
    if not expected_keys:
        return 0.0
    expected = set(expected_keys)
    return 1.0 if any(key in expected for key in ranked_found_keys[:k]) else 0.0


def mrr(expected_keys: Sequence[str], ranked_found_keys: Sequence[str]) -> float:
    """Reciprocal rank (1-indexed) of the FIRST expected key; 0.0 if none.

    No cutoff — the whole ranked list is considered.
    """
    if not expected_keys:
        return 0.0
    expected = set(expected_keys)
    for rank, key in enumerate(ranked_found_keys, start=1):
        if key in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(expected_keys: Sequence[str], ranked_found_keys: Sequence[str], k: int) -> float:
    """Binary-gain nDCG@k.

    Each expected key has gain 1. DCG sums 1/log2(rank+1) over hits in the top-k;
    the ideal DCG places min(#expected, k) hits at the top. 0.0 if no expected keys.
    """
    if not expected_keys:
        return 0.0
    expected = set(expected_keys)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, key in enumerate(ranked_found_keys[:k], start=1)
        if key in expected
    )
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class QueryScore:
    recall: float
    rr: float
    ndcg: float


def score_query(
    expected_keys: Sequence[str], ranked_found_keys: Sequence[str], k: int
) -> QueryScore:
    """Compute all three metrics for one query in a single call."""
    return QueryScore(
        recall=recall_at_k(expected_keys, ranked_found_keys, k),
        rr=mrr(expected_keys, ranked_found_keys),
        ndcg=ndcg_at_k(expected_keys, ranked_found_keys, k),
    )


@dataclass(frozen=True)
class MetricSummary:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    n: int


def aggregate(scores: Sequence[QueryScore]) -> MetricSummary:
    """Macro-average each metric across scored queries. n=0 -> all zeros."""
    n = len(scores)
    if n == 0:
        return MetricSummary(recall_at_k=0.0, mrr=0.0, ndcg_at_k=0.0, n=0)
    return MetricSummary(
        recall_at_k=sum(s.recall for s in scores) / n,
        mrr=sum(s.rr for s in scores) / n,
        ndcg_at_k=sum(s.ndcg for s in scores) / n,
        n=n,
    )
