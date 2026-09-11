"""Query-time pipeline: embed → hybrid search → dedupe → diversity → (rerank).

Reference patterns (used by `expand_context` and the BM25-weight heuristic)
are loaded from `corpus.toml` — NOT hardcoded. Default: empty list, in which
case `expand_context` returns siblings + parent only, and BM25 weight stays
at a low default for prose queries.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from corpus.db.sqlite import ChunkStore, StoredChunk
from corpus.embedder.base import Embedder
from corpus.util.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from corpus.reranker.local import BGEReranker


# User-supplied [[references]] regexes (from corpus.toml) run against query text
# and chunk content. They're trusted config, but a catastrophic-backtracking
# pattern on long input could stall the worker thread — so we cap the number of
# characters any reference pattern is scanned against as a defensive bound.
MAX_REGEX_SCAN_CHARS = 20_000


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


# Ceiling on how far `timeline` will widen its candidate pool chasing a date
# range. Bounds the worst case: a range that matches nothing would otherwise
# widen until it had scanned every chunk in the store.
_TIMELINE_MAX_POOL = 2_000


def build_vector_pool(
    store: ChunkStore,
    embedding: Sequence[float],
    *,
    source_types: Sequence[str],
    global_k: int,
    per_source_k: int,
    floor: int,
    filter_sources: Sequence[str] | None = None,
) -> list[StoredChunk]:
    """One global k-NN, then a targeted top-up for under-served source types.

    Every chunk has a distance to the query, so a source holding most of the
    corpus fills the pool by sheer volume regardless of relevance -- and a type
    absent from the POOL cannot be recovered downstream, because neither the
    diversity cap nor its backfill invents candidates. Measured on a large
    archive that is dominated by a single source type, a global-only pool starved a type completely
    on 1 query in 12.

    The obvious fix -- one k-NN per source type, always -- pays for that on
    every query. Measured medians, same queries, same archives:

        3 source types    global 1,581 ms   fan-out 4,009 ms   adaptive 1,832 ms
       44 source types    fan-out 2,012 ms                     adaptive 1,724 ms

    Fan-out cost +154% on the narrow archive to fix 8% of queries, while
    adaptive costs +16% there and is FASTER on the wide one, because a single
    large k-NN beats 44 small filtered ones. Adaptive reached the same final
    answer as full fan-out on 10 of 12 queries and full pool coverage on both.

    `floor` should be the per-type diversity cap: exactly how many chunks a
    type could contribute to the answer, so a type below it is under-served by
    construction while a type at or above it already has all the cap will use.

    `global_k` is clamped by `vector_search` to VEC0_MAX_K, so a very wide
    archive simply gets a smaller global pass and leans harder on the top-ups.
    Correctness does not depend on the budget fitting.
    """
    hits = store.vector_search(embedding, top_k=global_k, filter_sources=filter_sources)
    have: Counter[str] = Counter(c.source_type for c in hits)
    for stype in source_types:
        if have[stype] < floor:
            hits.extend(
                store.vector_search(embedding, top_k=per_source_k, filter_sources=[stype])
            )
    # A chunk can arrive from BOTH the global pass and its type's top-up. The
    # per-source fan-out this replaced queried each type exactly once and so
    # could not produce duplicates; this can, and RRF ranks by position -- the
    # same id appearing twice would be scored as two candidates and
    # double-weighted against everything else.
    seen: set[str] = set()
    deduped: list[StoredChunk] = []
    for c in hits:
        if c.id not in seen:
            seen.add(c.id)
            deduped.append(c)
    # Concatenated lists are not globally ordered; RRF consumes rank position,
    # so restore a global ordering before fusing. Ties break on `id`, not
    # insertion order, which would otherwise systematically favour whichever
    # source type sorts first.
    deduped.sort(key=lambda c: (c.distance if c.distance is not None else float("inf"), c.id))
    return deduped


def assemble_results(
    fused: Sequence[StoredChunk],
    top_k: int,
    *,
    dedupe_by_source: bool = True,
    max_per_source_type: int | None = 3,
    reject: Callable[[StoredChunk], bool] | None = None,
    source_key: Callable[[StoredChunk], Hashable] | None = None,
    content_key: Callable[[StoredChunk], Hashable] | None = None,
) -> list[StoredChunk]:
    """Turn a fused candidate list into the final top_k.

    This existed in three near-identical copies -- here, and in two private
    forks -- which is how the same truncation bug came to be fixed three
    times. The forks differ only in WHICH chunks they reject and how they
    identify a duplicate, so those are the hooks; the selection logic is
    shared.

    `reject` drops a chunk outright (domain noise, date/person filters).
    `source_key` overrides the dedupe identity -- one fork keys email by
    thread so a 40-message thread cannot fill the answer. `content_key`
    adds a second dedupe pass over content itself, collapsing the same text
    republished across document versions.

    The per-type cap is a PREFERENCE for spread, not a budget on the answer.
    A fixed cap meeting a variable top_k silently truncates, and the
    arithmetic is brutal on a narrow archive: measured on a live large
    store with 3 source types and cap 3, an MCP search could not return more
    than 9 however large top_k was, while advertising "Default 15, max 30" --
    and a query whose matches were mostly one type returned 5.

    So the cap is honoured first, then any remaining slots are filled from
    what it displaced, best-scoring first. Spread still wins every slot it
    can actually use; top_k goes back to meaning what it says; and a short
    result now means the candidate pool really was exhausted, which is the
    only honest reason to return fewer than were asked for.
    """
    seen_sources: set[Hashable] = set()
    seen_content: set[Hashable] = set()
    per_type_count: dict[str, int] = {}
    result: list[StoredChunk] = []
    overflow: list[StoredChunk] = []

    for c in fused:
        if reject is not None and reject(c):
            continue
        if dedupe_by_source:
            key = source_key(c) if source_key is not None else (c.source_type, c.source_key)
            if key in seen_sources:
                continue
            seen_sources.add(key)
        if content_key is not None:
            ckey = content_key(c)
            if ckey in seen_content:
                continue
            seen_content.add(ckey)
        if max_per_source_type is not None:
            stype = c.source_type
            if per_type_count.get(stype, 0) >= max_per_source_type:
                # Passed every other gate -- held back only by the cap, so it
                # is a legitimate candidate if slots go unfilled.
                overflow.append(c)
                continue
            per_type_count[stype] = per_type_count.get(stype, 0) + 1
        result.append(c)
        if len(result) >= top_k:
            return result

    # Appended, not merged by score: the diversity-selected chunks keep the
    # front of the context window, where they do the most good.
    if len(result) < top_k:
        result.extend(overflow[: top_k - len(result)])
    return result


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
        scan = query[:MAX_REGEX_SCAN_CHARS]
        for pat, _ in self._refs:
            if pat.search(scan):
                return 1.0
        return 0.25

    def _attach_summaries(self, chunks: list[StoredChunk]) -> None:
        """Populate each chunk's `.summary` from the per-doc `summaries` table so
        the reranker can score against summary+content (query-time only — NO
        re-embedding of the corpus). Lookups are cached by (source_type,
        source_key) so a pool of N chunks from M docs costs M queries, not N."""
        cache: dict[tuple[str, str], str | None] = {}
        for c in chunks:
            key = (c.source_type, c.source_key)
            if key not in cache:
                row = self._store.get_summary(c.source_type, c.source_key)
                cache[key] = row["summary"] if row else None
            c.summary = cache[key]

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

        # Candidate pool for VECTOR search: one global k-NN, then a targeted
        # top-up for any source type the global pass under-served. Ranking,
        # fusion and the cap are unchanged -- they still run over the union.
        #
        # This replaced an unconditional per-source fan-out (one k-NN per type,
        # always). Fan-out exists for a real defect: every chunk has a distance
        # to the query, so a source holding most of the corpus fills the pool by
        # sheer volume regardless of relevance, and a type absent from the pool
        # cannot be recovered downstream -- neither the diversity cap nor its
        # backfill invents candidates. Measured on a large archive that is
        # dominated by a single source type, a global-only pool starved a type completely on 1 query
        # in 12.
        #
        # But paying for the fix on every query is the wrong trade. Measured
        # medians, same queries, same archives:
        #
        #   3 source types    global 1,581 ms   fan-out 4,009 ms   adaptive 1,832 ms
        #  44 source types    (fan-out was the status quo, 2,012 ms)  adaptive 1,724 ms
        #
        # Fan-out cost +154% on the narrow archive to fix 8% of queries. Adaptive
        # costs +16% there and is FASTER on the wide one, because one large k-NN
        # beats 44 small filtered ones. It reached the same final answer as full
        # fan-out on 10 of 12 queries, and full pool coverage on both archives.
        #
        # The top-up floor is `max_per_source_type`: exactly how many chunks a
        # type could contribute to the answer, so a type below it is under-served
        # by construction. A type at or above it already has everything the cap
        # will let it use.
        source_types = (
            list(dict.fromkeys(filter_sources)) if filter_sources else self._store.source_types()
        )
        if not source_types:
            return RetrievalResult(query=question, chunks=[])
        per_source = max(top_k * 8 // len(source_types), top_k * 2, 10)

        vector_hits = build_vector_pool(
            self._store,
            embedding,
            source_types=source_types,
            global_k=per_source * len(source_types),
            per_source_k=per_source,
            floor=max_per_source_type or 1,
            # `source_types`, not `filter_sources`: the former is deduplicated,
            # so a caller passing ["notes", "notes"] does not widen the SQL
            # filter with a redundant term.
            filter_sources=source_types if filter_sources else None,
        )

        if hybrid:
            effective_fts_weight = (
                fts_weight if fts_weight is not None else self._auto_fts_weight(question)
            )
            # ONE global call, not per source type (see comment above). Sized
            # to contribute roughly as many candidates in total as the
            # per-source vector loop does, so the fused pool doesn't shrink on
            # the BM25 side relative to the vector side.
            fts_hits = self._store.fts_search(
                question, top_k=per_source * len(source_types), filter_sources=filter_sources
            )
            fused = reciprocal_rank_fusion(
                [vector_hits, fts_hits],
                weights=[vector_weight, effective_fts_weight],
                key=lambda c: c.id,
            )
        else:
            fused = list(vector_hits)

        if rerank and self._reranker is not None:
            pool = fused[:rerank_pool_size]
            self._attach_summaries(pool)
            fused = self._reranker.rerank(question, pool)

        # The cap spreads results ACROSS source types. A query already
        # filtered to one type has nothing to spread across, so the cap can
        # only subtract: `top_k=10, filter_sources=['notes']` returned 3
        # results with 40 matching chunks in the store. Disabled for that
        # case only.
        if filter_sources is not None and len(set(filter_sources)) == 1:
            max_per_source_type = None

        return RetrievalResult(
            query=question,
            chunks=assemble_results(
                fused,
                top_k,
                dedupe_by_source=dedupe_by_source,
                max_per_source_type=max_per_source_type,
            ),
        )

    def timeline(
        self,
        topic: str,
        top_k: int = 20,
        since: str | None = None,
        until: str | None = None,
        filter_sources: Sequence[str] | None = None,
    ) -> list[StoredChunk]:
        # The date filter runs in Python AFTER retrieval, so a topic whose
        # semantically-nearest chunks all fall outside the range leaves
        # nothing: measured with 200 near-but-old chunks and 10 further-but-
        # recent ones, `since` returned ZERO of the 10. That is the ordinary
        # shape of a timeline query — "what happened lately about X" asks for
        # recent material, which is rarely the most semantically central.
        #
        # So the candidate pool widens while the filter is starving it. The
        # unfiltered case and the uncrowded case still cost one query; only a
        # date range that actually excluded the pool pays for more, and
        # widening stops as soon as the pool comes back short, which means
        # there is nothing further to find.
        pool = top_k * 3
        candidates: list[StoredChunk] = []
        while True:
            result = self.query(
                topic,
                top_k=pool,
                filter_sources=filter_sources,
                # No diversity cap. A timeline is one topic ordered by DATE,
                # so spreading across source types is not what is being asked
                # for — and the default cap of 3 per type silently limited
                # every timeline to 3 x (number of types) candidates no matter
                # what top_k said, which also defeats the widening below.
                max_per_source_type=None,
            )
            candidates = result.chunks
            if since:
                candidates = [c for c in candidates if (_chunk_updated_at(c) or "") >= since]
            if until:
                candidates = [c for c in candidates if (_chunk_updated_at(c) or "") <= until]
            enough = len(candidates) >= top_k
            exhausted = len(result.chunks) < pool
            if enough or exhausted or pool >= _TIMELINE_MAX_POOL:
                break
            pool = min(pool * 8, _TIMELINE_MAX_POOL)

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
                matches = {
                    m.group(0)
                    for m in pattern.finditer(seed.content[:MAX_REGEX_SCAN_CHARS])
                }
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
