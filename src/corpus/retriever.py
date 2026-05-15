"""Query-time pipeline: embed → hybrid search → dedupe → diversity → (rerank).

Reference patterns (used by `expand_context` and the BM25-weight heuristic)
are loaded from `corpus.toml` — NOT hardcoded. Default: empty list, in which
case `expand_context` returns siblings + parent only, and BM25 weight stays
at a low default for prose queries.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from corpus.db.sqlite import ChunkStore, StoredChunk
from corpus.embedder.base import Embedder
from corpus.util.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from corpus.reranker.local import BGEReranker


@dataclass
class RetrievalResult:
    query: str
    chunks: list[StoredChunk]


# Always-on heuristic: queries containing identifier-looking tokens (quoted
# phrases, backticked code, anything with hyphens or hashes) deserve more BM25.
# This is generic — unlike the references list below, which is corpus-specific.
_GENERIC_ID_HINTS = [
    re.compile(r"`[^`]+`"),
    re.compile(r"\".+?\""),
]


def _has_generic_id_hint(query: str) -> bool:
    return any(p.search(query) for p in _GENERIC_ID_HINTS)


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        reranker: BGEReranker | None = None,
        reference_patterns: Sequence[tuple[re.Pattern[str], str]] | None = None,
    ):
        """`reference_patterns` is a list of (compiled_regex, source_type)
        tuples. Loaded from corpus.toml's [[references]] blocks. Empty list
        means no cross-document reference chasing in `expand_context` and
        only generic ID-hint heuristics for BM25 weighting."""
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._refs: list[tuple[re.Pattern[str], str]] = list(reference_patterns or [])

    def _auto_fts_weight(self, query: str) -> float:
        """High BM25 weight when query looks identifier-shaped; low for prose."""
        if _has_generic_id_hint(query):
            return 1.0
        for pat, _ in self._refs:
            if pat.search(query):
                return 1.0
        return 0.25

    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_sources: Sequence[str] | None = None,
        dedupe_by_source: bool = True,
        max_per_source_type: int | None = 3,
        hybrid: bool = True,
        vector_weight: float = 1.0,
        fts_weight: float | None = None,
        rerank: bool = False,
        rerank_pool_size: int = 30,
    ) -> RetrievalResult:
        embedding = self._embedder.embed_query(question)
        over_fetch = max(top_k * 8, 40)
        vector_hits = self._store.vector_search(
            embedding, top_k=over_fetch, filter_sources=filter_sources
        )

        if hybrid:
            effective_fts_weight = (
                fts_weight if fts_weight is not None else self._auto_fts_weight(question)
            )
            fts_hits = self._store.fts_search(
                question, top_k=over_fetch, filter_sources=filter_sources
            )
            fused = reciprocal_rank_fusion(
                [vector_hits, fts_hits],
                weights=[vector_weight, effective_fts_weight],
                key=lambda c: c.id,
            )
        else:
            fused = list(vector_hits)

        if rerank and self._reranker is not None:
            fused = self._reranker.rerank(question, fused[:rerank_pool_size])

        if not dedupe_by_source and max_per_source_type is None:
            return RetrievalResult(query=question, chunks=fused[:top_k])

        seen_sources: set[tuple[str, str]] = set()
        per_type_count: dict[str, int] = {}
        result: list[StoredChunk] = []
        for c in fused:
            if dedupe_by_source:
                key = (c.source_type, c.source_key)
                if key in seen_sources:
                    continue
                seen_sources.add(key)
            if max_per_source_type is not None:
                stype = c.source_type
                if per_type_count.get(stype, 0) >= max_per_source_type:
                    continue
                per_type_count[stype] = per_type_count.get(stype, 0) + 1
            result.append(c)
            if len(result) >= top_k:
                break
        return RetrievalResult(query=question, chunks=result)

    def timeline(
        self,
        topic: str,
        top_k: int = 20,
        since: str | None = None,
        until: str | None = None,
        filter_sources: Sequence[str] | None = None,
    ) -> list[StoredChunk]:
        result = self.query(
            topic, top_k=top_k * 3, filter_sources=filter_sources
        )
        candidates = result.chunks
        if since:
            candidates = [c for c in candidates if (_chunk_updated_at(c) or "") >= since]
        if until:
            candidates = [c for c in candidates if (_chunk_updated_at(c) or "") <= until]
        candidates.sort(key=lambda c: _chunk_updated_at(c) or "")
        return candidates[:top_k]

    def recent_activity(
        self,
        days: int = 7,
        filter_sources: Sequence[str] | None = None,
        top_k: int = 20,
    ) -> list[StoredChunk]:
        since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
        rows = self._store.find_recent(since=since, filter_sources=filter_sources, limit=top_k * 5)
        seen: set[tuple[str, str]] = set()
        out: list[StoredChunk] = []
        for c in rows:
            key = (c.source_type, c.source_key)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= top_k:
                break
        return out

    def expand_context(
        self,
        chunk_id: str,
        include: Sequence[str] = ("siblings", "references", "parent"),
        max_results: int = 10,
    ) -> list[StoredChunk]:
        """Chase references from a chunk.

        - `siblings`: other chunks of the same (source_type, source_key)
        - `references`: chunks of any source whose IDs match the configured
          reference patterns in corpus.toml. If no patterns configured, this
          step returns nothing.
        - `parent`: if the chunk's metadata.extra.parent is set, the parent's
          chunks. Requires connectors that populate this field.
        """
        seed = self._store.get_by_id(chunk_id)
        if seed is None:
            return []

        include_set = set(include)
        seen: set[str] = {seed.id}
        results: list[StoredChunk] = []

        if "siblings" in include_set:
            for c in self._store.get_by_source_key(seed.source_type, seed.source_key):
                if c.id not in seen and len(results) < max_results:
                    seen.add(c.id)
                    results.append(c)

        if "references" in include_set and self._refs:
            for pattern, ref_source_type in self._refs:
                matches = {m.group(0) for m in pattern.finditer(seed.content)}
                # Don't self-cite.
                matches = {m for m in matches if m != seed.source_key}
                for key in sorted(matches):
                    if len(results) >= max_results:
                        break
                    for c in self._store.get_by_source_key(ref_source_type, key):
                        if c.id not in seen:
                            seen.add(c.id)
                            results.append(c)
                            break

        if "parent" in include_set:
            parent_key = (seed.metadata.get("extra") or {}).get("parent")
            if parent_key:
                for c in self._store.get_by_source_key(seed.source_type, parent_key):
                    if c.id not in seen and len(results) < max_results:
                        seen.add(c.id)
                        results.append(c)

        return results[:max_results]

    def close(self) -> None:
        self._store.close()


def _chunk_updated_at(c: StoredChunk) -> str | None:
    md = c.metadata or {}
    return md.get("updated_at") or md.get("created_at")
