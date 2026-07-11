"""Retrieval- and (later) generation-quality evaluation.

`metrics.py` is intentionally dependency-free so it can be reused byte-identical
across separate RAG deployments — do not add corpus-specific imports to it.
"""

from __future__ import annotations

from .generation import GENERATOR_DEFAULT_MODEL, Answer, answer_from_context
from .judge import (
    JUDGE_DEFAULT_MODEL,
    AxisVerdict,
    JudgeVerdict,
    aggregate_verdicts,
    judge_answer,
)
from .metrics import (
    MetricSummary,
    QueryScore,
    aggregate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_query,
)
from .validation import (
    JudgeCase,
    ValidationReport,
    cohens_kappa,
    run_validation_study,
)

__all__ = [
    "GENERATOR_DEFAULT_MODEL",
    "JUDGE_DEFAULT_MODEL",
    "Answer",
    "AxisVerdict",
    "JudgeCase",
    "JudgeVerdict",
    "MetricSummary",
    "QueryScore",
    "ValidationReport",
    "aggregate",
    "aggregate_verdicts",
    "answer_from_context",
    "cohens_kappa",
    "judge_answer",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "run_validation_study",
    "score_query",
]
