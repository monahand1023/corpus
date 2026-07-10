from __future__ import annotations

import math

import pytest

from corpus.embedder.factory import make_embedder
from corpus.embedder.hash import HashEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_dim_is_respected() -> None:
    e = HashEmbedder(dim=128)
    v = e.embed_query("hello world")
    assert len(v) == 128


def test_deterministic_across_instances() -> None:
    # Two fresh instances must agree — guards against builtin hash() (salted).
    a = HashEmbedder(dim=64).embed_query("reciprocal rank fusion")
    b = HashEmbedder(dim=64).embed_query("reciprocal rank fusion")
    assert a == b


def test_query_vector_is_l2_normalized() -> None:
    v = HashEmbedder(dim=256).embed_query("some non trivial text here")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


def test_lexical_overlap_beats_disjoint() -> None:
    e = HashEmbedder(dim=512)
    q = e.embed_query("the quick brown fox")
    near = e.embed_query("the quick brown dog")
    far = e.embed_query("zzz yyy www vvv")
    assert _cosine(q, near) > _cosine(q, far)


def test_embed_documents_maps_empty_to_none() -> None:
    e = HashEmbedder(dim=32)
    out = e.embed_documents(["real text", "", "   "])
    assert out[0] is not None and len(out[0]) == 32
    assert out[1] is None
    assert out[2] is None


def test_embed_query_rejects_empty() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=32).embed_query("   ")


def test_bad_dim_rejected() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)


def test_factory_builds_hash_embedder() -> None:
    e = make_embedder(provider="hash", model="hash-v1", dim=128)
    assert isinstance(e, HashEmbedder)
    assert len(e.embed_query("x")) == 128
