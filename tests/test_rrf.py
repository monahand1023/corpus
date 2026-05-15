from __future__ import annotations

import pytest

from corpus.util.rrf import reciprocal_rank_fusion


def test_empty() -> None:
    assert reciprocal_rank_fusion([[]]) == []


def test_single_list_preserves_order() -> None:
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]


def test_shared_item_outranks_unique_low_ranks() -> None:
    lists = [["only_a", "shared", "x_a"], ["only_b", "shared", "x_b"]]
    out = reciprocal_rank_fusion(lists)
    assert out[0] == "shared"


def test_weights_shift_ordering() -> None:
    lists = [["a", "b"], ["b", "a"]]
    out = reciprocal_rank_fusion(lists, weights=[3.0, 0.5])
    assert out[0] == "a"


def test_key_extraction_dedupes_by_id() -> None:
    lists = [
        [{"id": 1, "v": "A"}, {"id": 2, "v": "B"}],
        [{"id": 2, "v": "B-alt"}, {"id": 3, "v": "C"}],
    ]
    out = reciprocal_rank_fusion(lists, key=lambda x: x["id"])
    # id=2 appears in both, ranks first
    assert out[0]["id"] == 2
    # First-seen instance kept
    assert next(i for i in out if i["id"] == 2)["v"] == "B"


def test_mismatched_weights_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
