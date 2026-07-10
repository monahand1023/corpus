"""Deterministic, zero-dependency hashing embedder.

NOT a semantic model: it hashes token features into a fixed-dim vector (signed
feature hashing) and L2-normalizes, so cosine similarity approximates weighted
lexical overlap. Its purpose is a STABLE, reproducible, API-key-free embedding
backend for tests, CI, and a "try corpus with no key" on-ramp — NOT a quality
claim. For real retrieval quality use the `voyage` or `gemini` providers.

Determinism matters: we hash with `hashlib.blake2b`, not Python's builtin
`hash()`, because the latter is salted per process (PYTHONHASHSEED) and would make
vectors differ run-to-run — poison for a regression gate.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """Signed feature-hashing embedder. `dim` must match the store's embedding_dim."""

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError(f"HashEmbedder dim must be positive, got {dim}")
        self._dim = dim
        self.total_tokens_used = 0

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = _tokenize(text)
        for tok in tokens:
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
            bucket = h % self._dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[bucket] += sign
        self.total_tokens_used += len(tokens)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        return [self._embed_one(t) if t and t.strip() else None for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query")
        return self._embed_one(text)
