"""Local cross-encoder re-ranker.

After the dense + BM25 fusion gives us ~40 candidates, a cross-encoder
re-scores each (query, chunk) pair directly — more discriminating than
cosine similarity alone, at the cost of one model forward-pass per candidate.

Default model: `BAAI/bge-reranker-v2-m3`. Multilingual, ~568 MB on disk,
~5ms per (query, chunk) pair on M-series Macs. Top of MTEB reranking
leaderboards as of 2026.

Lazy-loaded — importing this module is cheap; the model only materializes
on first `.rerank()` call. So unused imports pay nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corpus.db.sqlite import StoredChunk

logger = logging.getLogger(__name__)


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name
        self._model = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info("loading reranker model %s (first call; ~5s)", self._model_name)
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        candidates: list[StoredChunk],
        top_n: int | None = None,
    ) -> list[StoredChunk]:
        if not candidates:
            return []
        self._ensure_loaded()
        assert self._model is not None
        pairs = [[query, c.content] for c in candidates]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -float(x[1]))
        out = [c for c, _ in reranked]
        return out[:top_n] if top_n else out
