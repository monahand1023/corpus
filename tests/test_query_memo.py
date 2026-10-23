"""`MemoizedQueryEmbedder`: one vector per distinct query, for comparisons.

A hosted embedder does not return a bit-identical vector for a fixed query
(measured on voyage-4-large: three distinct vectors from four identical
calls). Two retrieval configs compared on separately-fetched vectors differ
by that noise as well as by the config; sharing one vector per query leaves
only the config.
"""

from __future__ import annotations

from collections.abc import Sequence

from corpus.embedder.base import Embedder
from corpus.embedder.memo import MemoizedQueryEmbedder


class _Noisy:
    """Returns a different vector on every call, like a hosted embedder."""

    total_tokens_used = 0
    counts_tokens = False

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        self.total_tokens_used += 3
        return [float(len(self.query_calls))]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        return [[0.0] for _ in texts]


def test_a_repeated_query_is_embedded_once_and_gets_the_same_vector() -> None:
    inner = _Noisy()
    memo = MemoizedQueryEmbedder(inner)

    first = memo.embed_query("q")
    second = memo.embed_query("q")

    assert first == second
    assert inner.query_calls == ["q"]


def test_distinct_queries_are_each_embedded() -> None:
    inner = _Noisy()
    memo = MemoizedQueryEmbedder(inner)

    memo.embed_query("a")
    memo.embed_query("b")

    assert inner.query_calls == ["a", "b"]


def test_a_caller_mutating_its_vector_does_not_poison_the_memo() -> None:
    memo = MemoizedQueryEmbedder(_Noisy())

    memo.embed_query("q").append(99.0)

    assert memo.embed_query("q") == [1.0]


def test_documents_and_token_accounting_pass_through() -> None:
    inner = _Noisy()
    memo = MemoizedQueryEmbedder(inner)

    assert memo.embed_documents(["x", "y"]) == [[0.0], [0.0]]
    memo.embed_query("q")
    memo.embed_query("q")
    assert memo.total_tokens_used == 3  # billed once
    assert memo.counts_tokens is False


def test_it_satisfies_the_embedder_protocol() -> None:
    assert isinstance(MemoizedQueryEmbedder(_Noisy()), Embedder)
