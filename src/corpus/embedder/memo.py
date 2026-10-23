"""One query vector per distinct query, for comparing retrieval configs.

A hosted embedder is not bit-reproducible: measured on voyage-4-large, four
identical calls returned three distinct vectors (cosine ~0.998). Small, but
enough to reorder near-ties -- so two configs compared on separately-fetched
vectors differ by that noise as well as by the config, and the comparison
cannot say which. Wrapping the embedder gives every config the same vector.

Deliberately in-memory and per-instance: it lives as long as one comparison.
Do NOT share one across `corpus-eval --repeat` runs, whose job is to measure
exactly the noise this removes.
"""

from __future__ import annotations

from collections.abc import Sequence

from corpus.embedder.base import Embedder


class MemoizedQueryEmbedder:
    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._vectors: dict[str, list[float]] = {}

    # Properties with setters, not copies: the protocol declares these as
    # settable attributes, and the inner embedder is the one that counts.
    @property
    def total_tokens_used(self) -> int:
        return self._inner.total_tokens_used

    @total_tokens_used.setter
    def total_tokens_used(self, value: int) -> None:
        self._inner.total_tokens_used = value

    @property
    def counts_tokens(self) -> bool:
        return getattr(self._inner, "counts_tokens", True)

    @counts_tokens.setter
    def counts_tokens(self, value: bool) -> None:
        self._inner.counts_tokens = value

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._vectors:
            self._vectors[text] = self._inner.embed_query(text)
        # A copy: a caller that mutates its vector must not change the next
        # config's.
        return list(self._vectors[text])
