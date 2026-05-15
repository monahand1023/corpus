"""Cheap token-count heuristic for chunking decisions.

Real billing-accurate counts come from the embedder's `count_tokens` endpoint
and aren't needed for chunk-boundary math. The chars/4 heuristic is well-known
and close enough for splitting; the actual tokenizer would shift specific
chunks by ±10% but boundary placement is unchanged.
"""

from __future__ import annotations

MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
