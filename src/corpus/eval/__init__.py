"""Retrieval- and (later) generation-quality evaluation.

`metrics.py` is intentionally dependency-free so it can be reused byte-identical
across separate RAG deployments — do not add corpus-specific imports to it.
"""

from __future__ import annotations

from corpus.eval.metrics import (
    MetricSummary,
    QueryScore,
    aggregate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_query,
)

__all__ = [
    "MetricSummary",
    "QueryScore",
    "aggregate",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "score_query",
]
