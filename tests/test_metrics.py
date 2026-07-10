from __future__ import annotations

import math

from corpus.eval.metrics import (
    MetricSummary,
    QueryScore,
    aggregate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_query,
)


def test_recall_hit_within_k() -> None:
    assert recall_at_k(["a"], ["x", "a", "y"], k=5) == 1.0


def test_recall_miss() -> None:
    assert recall_at_k(["a"], ["x", "y", "z"], k=5) == 0.0


def test_recall_respects_cutoff() -> None:
    # hit sits at rank 3; k=2 excludes it
    assert recall_at_k(["a"], ["x", "y", "a"], k=2) == 0.0
    assert recall_at_k(["a"], ["x", "y", "a"], k=3) == 1.0


def test_recall_or_set() -> None:
    # any expected key counts
    assert recall_at_k(["a", "b"], ["x", "b"], k=5) == 1.0


def test_recall_empty_expected_is_zero() -> None:
    assert recall_at_k([], ["x", "y"], k=5) == 0.0


def test_mrr_rank_one() -> None:
    assert mrr(["a"], ["a", "b"]) == 1.0


def test_mrr_rank_three() -> None:
    assert mrr(["a"], ["x", "y", "a"]) == 1.0 / 3.0


def test_mrr_first_of_expected_wins() -> None:
    # b appears at rank 2, a at rank 4; reciprocal of the FIRST match
    assert mrr(["a", "b"], ["x", "b", "y", "a"]) == 0.5


def test_mrr_no_hit() -> None:
    assert mrr(["a"], ["x", "y"]) == 0.0


def test_ndcg_single_hit_rank_one() -> None:
    assert ndcg_at_k(["a"], ["a", "b", "c"], k=5) == 1.0


def test_ndcg_single_hit_rank_three() -> None:
    # DCG = 1/log2(4) = 0.5 ; IDCG = 1/log2(2) = 1.0 ; nDCG = 0.5
    assert ndcg_at_k(["a"], ["x", "y", "a"], k=5) == 0.5


def test_ndcg_hit_outside_k_is_zero() -> None:
    assert ndcg_at_k(["a"], ["x", "y", "a"], k=2) == 0.0


def test_ndcg_two_hits_ideal_order() -> None:
    # both expected at ranks 1,2 -> perfect -> 1.0
    assert ndcg_at_k(["a", "b"], ["a", "b", "c"], k=5) == 1.0


def test_ndcg_two_hits_suboptimal_order() -> None:
    # hits at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG (2 hits ideal) = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    expected = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(ndcg_at_k(["a", "b"], ["a", "x", "b"], k=5), expected)


def test_ndcg_empty_expected_is_zero() -> None:
    assert ndcg_at_k([], ["x"], k=5) == 0.0


def test_score_query_bundles_all_three() -> None:
    s = score_query(["a"], ["x", "a"], k=5)
    assert s == QueryScore(recall=1.0, rr=0.5, ndcg=ndcg_at_k(["a"], ["x", "a"], 5))


def test_aggregate_macro_averages() -> None:
    scores = [
        QueryScore(recall=1.0, rr=1.0, ndcg=1.0),
        QueryScore(recall=0.0, rr=0.0, ndcg=0.0),
    ]
    summary = aggregate(scores)
    assert summary == MetricSummary(recall_at_k=0.5, mrr=0.5, ndcg_at_k=0.5, n=2)


def test_aggregate_empty_is_zeros() -> None:
    assert aggregate([]) == MetricSummary(recall_at_k=0.0, mrr=0.0, ndcg_at_k=0.0, n=0)
