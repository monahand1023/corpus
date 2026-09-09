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
    from sentence_transformers import CrossEncoder

    from corpus.db.sqlite import StoredChunk

logger = logging.getLogger(__name__)

# Cross-encoders truncate long inputs anyway; keep the pair text bounded so a
# huge chunk doesn't dominate the batch.
_MAX_RERANK_CHARS = 4000


def _rerank_text(c: StoredChunk) -> str:
    """Text the cross-encoder scores against: the doc summary + content when a
    per-doc summary has been attached (by the Retriever at query time), else bare
    content. Prepending the summary gives the cross-encoder doc-level context so a
    chunk that's relevant only in light of its document isn't judged on its bare
    text and demoted out."""
    summary = getattr(c, "summary", None)
    text = f"{summary}\n\n{c.content}" if summary else c.content
    return text[:_MAX_RERANK_CHARS]


class BGEReranker:
    # Device is explicit and defaults to CPU — never left for
    # sentence-transformers to auto-select. On Apple Silicon, an unspecified
    # device silently resolves to MPS (the GPU). corpus is consumed by
    # long-running local processes (e.g. an overnight MLX vision-captioning
    # job) that hold the GPU for hours; co-resident MLX/Metal and PyTorch-MPS
    # processes are a documented source of Metal command-buffer faults. A
    # search query re-ranking ~40 candidates must never be able to take down
    # someone else's multi-hour GPU job just because sentence-transformers
    # found a GPU sitting there.
    #
    # Benchmarked 2026-09-08 on an M4 Pro (bge-reranker-v2-m3, real ~18-chunk
    # candidate pools drawn from a 46k-chunk corpus via corpus-voyage.db):
    # CPU ~390ms/pair, MPS ~142ms/pair (MPS ~2.7x faster, not the ~1x/"both
    # sub-second" this module originally assumed — see
    # .superpowers/sdd/2026-09-09-reranker.md for the full numbers). At the
    # default rerank_pool_size=30 that's ~10-12s/query on CPU vs. ~4-5s on
    # MPS — NEITHER device is sub-second here; this model is far heavier than
    # the ~5ms/pair back-of-envelope estimate that originally motivated "CPU
    # is fine, default to it." CPU remains the default anyway, on safety
    # grounds alone: it is the one option that can never contend with a
    # co-resident GPU job, and that property doesn't get weaker just because
    # CPU is also slower than hoped. But the latency is real, not
    # negligible — factor it in before turning `--rerank` on by default
    # anywhere. Set `[reranker] device = "mps"` in corpus.toml if you know
    # the GPU is free and want the ~2.7x speedup.
    #
    # For scale: with the pragma tuning in `corpus.config.PerformanceConfig`
    # applied, a full vector KNN search costs ~50ms — so one CPU rerank pair
    # (~390ms) alone costs ~8x an entire tuned search, before multiplying by
    # however many candidates are in the pool. Reranking is not a cheap
    # finishing touch on top of search; it's the dominant cost by almost an
    # order of magnitude per pair. That gap — not just the GPU-contention
    # risk above — is why `rerank=False` stays the default rather than
    # something to flip on casually.
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model: CrossEncoder | None = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "loading reranker model %s on device=%s (first call; ~5s)",
            self._model_name,
            self._device,
        )
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name, device=self._device)

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
        pairs = [(query, _rerank_text(c)) for c in candidates]
        # CrossEncoder.predict accepts a batch of (query, text) pairs — documented
        # usage — but its type stubs only admit a single pair or flat list.
        scores = self._model.predict(pairs)  # type: ignore[arg-type]
        reranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -float(x[1]))
        out = [c for c, _ in reranked]
        return out[:top_n] if top_n else out
