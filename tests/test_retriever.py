from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import corpus.retriever
from corpus.db.sqlite import ChunkStore, StoredChunk
from corpus.retriever import Retriever, assemble_results
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 1024


def fake_embedding(seed: int) -> list[float]:
    return [math.sin((seed + 1) * (i + 1) * 0.001) for i in range(DIM)]


def make_dupe(source_type: str, source_key: str, idx: int, content: str) -> Chunk:
    return Chunk(
        id=chunk_id(source_type, source_key, ChunkKind.HEADER, idx),
        content=content,
        content_hash=sha256(content),
        metadata=ChunkMetadata(
            source_type=source_type,
            source_key=source_key,
            chunk_kind=ChunkKind.HEADER,
            chunk_index=idx,
            title=f"{source_type}:{source_key}",
        ),
    )


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    store = ChunkStore(tmp_path / "ret.db", embedding_dim=DIM)
    items = []
    for i in range(6):
        items.append((make_dupe("notes", f"doc-{i}", 0, f"note {i}"), fake_embedding(i)))
    for i in range(3):
        items.append((make_dupe("notes", "doc-0", i + 1, f"doc-0 body {i}"), fake_embedding(i + 10)))
    for i in range(3):
        items.append((make_dupe("papers", f"p-{i}", 0, f"paper {i}"), fake_embedding(i + 20)))
    store.upsert_batch(items)

    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)
    yield r
    store.close()


def test_dedupe_by_source_key(retriever: Retriever) -> None:
    """The doc-0 group has 4 chunks; dedupe should collapse to 1."""
    result = retriever.query("anything", top_k=10, max_per_source_type=None, hybrid=False)
    keys = [c.source_key for c in result.chunks]
    assert keys.count("doc-0") == 1


def test_max_per_source_type_cap(retriever: Retriever) -> None:
    """Where the cap alone can satisfy top_k, it binds exactly.

    4 slots across 2 source types at cap 2 needs no backfill, so nothing
    here relaxes the cap.
    """
    result = retriever.query("anything", top_k=4, max_per_source_type=2, hybrid=False)
    counts: dict[str, int] = {}
    for c in result.chunks:
        counts[c.source_type] = counts.get(c.source_type, 0) + 1
    assert len(result.chunks) == 4
    for n in counts.values():
        assert n <= 2


def test_diversity_makes_room_for_papers(retriever: Retriever) -> None:
    """With cap=2, top-6 should include at least one paper not just notes."""
    result = retriever.query("anything", top_k=6, max_per_source_type=2, hybrid=False)
    sources = {c.source_type for c in result.chunks}
    assert "papers" in sources


def test_reference_pattern_drives_expand_context(tmp_path: Path) -> None:
    """expand_context references mode uses regex from config, not hardcoded."""
    store = ChunkStore(tmp_path / "ref.db", embedding_dim=DIM)
    items = [
        # Seed mentions TKT-9
        (make_dupe("tickets", "TKT-1", 0, "depends on TKT-9 to ship"), fake_embedding(1)),
        (make_dupe("tickets", "TKT-9", 0, "the dependency ticket"), fake_embedding(2)),
        (make_dupe("tickets", "TKT-5", 0, "unrelated"), fake_embedding(3)),
    ]
    store.upsert_batch(items)
    embedder = MagicMock()
    pattern = re.compile(r"\bTKT-\d+\b")
    r = Retriever(
        store=store,
        embedder=embedder,
        reference_patterns=[(pattern, "tickets")],
    )

    seed_id = chunk_id("tickets", "TKT-1", ChunkKind.HEADER, 0)
    related = r.expand_context(seed_id, include=["references"])
    keys = {c.source_key for c in related}
    assert "TKT-9" in keys
    assert "TKT-1" not in keys  # self-cite excluded
    assert "TKT-5" not in keys  # not mentioned
    store.close()


def test_no_reference_patterns_means_no_cross_doc_references(tmp_path: Path) -> None:
    """With references=[], expand_context returns siblings + parent only."""
    store = ChunkStore(tmp_path / "noref.db", embedding_dim=DIM)
    items = [
        (make_dupe("notes", "A", 0, "mentions DOC-9"), fake_embedding(1)),
        (make_dupe("notes", "DOC-9", 0, "would-be reference"), fake_embedding(2)),
    ]
    store.upsert_batch(items)
    embedder = MagicMock()
    r = Retriever(store=store, embedder=embedder, reference_patterns=[])

    seed_id = chunk_id("notes", "A", ChunkKind.HEADER, 0)
    related = r.expand_context(seed_id, include=["references"])
    assert related == []  # no patterns configured
    store.close()


def test_auto_fts_weight_boosts_on_reference_match(tmp_path: Path) -> None:
    """When a query contains a configured reference pattern, fts_weight jumps
    to 1.0 (BM25-heavy). Prose queries stay at 0.25."""
    store = ChunkStore(tmp_path / "fw.db", embedding_dim=DIM)
    embedder = MagicMock()
    pattern = re.compile(r"\bTKT-\d+\b")
    r = Retriever(
        store=store,
        embedder=embedder,
        reference_patterns=[(pattern, "tickets")],
    )
    # Query mentioning a reference pattern → BM25-heavy
    assert r._auto_fts_weight("what happened with TKT-42") == 1.0
    # Plain prose query → semantic-dominant
    assert r._auto_fts_weight("how does the payment flow work") == 0.25
    # Generic ID hint (backticked code) also boosts, even without a configured pattern
    assert r._auto_fts_weight("what does `parse_markdown` do") == 1.0
    store.close()


def test_auto_fts_weight_no_patterns_uses_generic_hints_only(tmp_path: Path) -> None:
    store = ChunkStore(tmp_path / "fw2.db", embedding_dim=DIM)
    embedder = MagicMock()
    r = Retriever(store=store, embedder=embedder, reference_patterns=[])
    # No configured patterns — TKT-42 is just prose now
    assert r._auto_fts_weight("what happened with TKT-42") == 0.25
    # But quoted phrases still trigger the generic hint
    assert r._auto_fts_weight('search for "exact phrase"') == 1.0
    store.close()


# ---------------------------------------------------------------------------
# Helpers for timeline / recent_activity tests
# ---------------------------------------------------------------------------


def make_chunk_with_date(
    source_type: str,
    source_key: str,
    idx: int,
    content: str,
    days_ago: int,
) -> Chunk:
    now = datetime.now(UTC)
    updated_at = (now - timedelta(days=days_ago)).isoformat()
    return Chunk(
        id=chunk_id(source_type, source_key, ChunkKind.HEADER, idx),
        content=content,
        content_hash=sha256(content),
        metadata=ChunkMetadata(
            source_type=source_type,
            source_key=source_key,
            chunk_kind=ChunkKind.HEADER,
            chunk_index=idx,
            title=f"{source_type}:{source_key}",
            updated_at=updated_at,
        ),
    )


@pytest.fixture
def retriever_with_dates(tmp_path: Path) -> Retriever:
    """Fixture with chunks that have known updated_at dates for timeline/recent_activity tests."""
    store = ChunkStore(tmp_path / "dated.db", embedding_dim=DIM)
    items = [
        # notes: varying ages
        (make_chunk_with_date("notes", "old-doc", 0, "old note content", 30), fake_embedding(0)),
        (make_chunk_with_date("notes", "mid-doc", 0, "mid note content", 10), fake_embedding(1)),
        (make_chunk_with_date("notes", "new-doc", 0, "new note content", 2), fake_embedding(2)),
        # papers: also recent
        (make_chunk_with_date("papers", "paper-a", 0, "paper content a", 3), fake_embedding(3)),
        (make_chunk_with_date("papers", "paper-b", 0, "paper content b", 25), fake_embedding(4)),
        # notes: multiple chunks for same source_key (for dedup test)
        (make_chunk_with_date("notes", "multi-doc", 0, "multi doc chunk 0", 1), fake_embedding(5)),
        (make_chunk_with_date("notes", "multi-doc", 1, "multi doc chunk 1", 1), fake_embedding(6)),
        (make_chunk_with_date("notes", "multi-doc", 2, "multi doc chunk 2", 1), fake_embedding(7)),
    ]
    store.upsert_batch(items)

    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)
    yield r
    store.close()


# ---------------------------------------------------------------------------
# timeline() tests
# ---------------------------------------------------------------------------


def test_timeline_sorts_ascending_by_date(retriever_with_dates: Retriever) -> None:
    """timeline() returns chunks sorted oldest-first by updated_at."""
    results = retriever_with_dates.timeline("note content", top_k=20)
    dates = [c.metadata.get("updated_at") or "" for c in results]
    assert dates == sorted(dates), "timeline results should be sorted ascending by updated_at"


def test_timeline_since_filters_old(retriever_with_dates: Retriever) -> None:
    """Passing since= excludes chunks older than the cutoff date."""
    since = (datetime.now(UTC) - timedelta(days=15)).date().isoformat()
    results = retriever_with_dates.timeline("note content", top_k=20, since=since)
    source_keys = {c.source_key for c in results}
    # old-doc is 30 days ago — should be excluded
    assert "old-doc" not in source_keys
    # paper-b is 25 days ago — should be excluded
    assert "paper-b" not in source_keys


def test_timeline_until_filters_future(retriever_with_dates: Retriever) -> None:
    """Passing until= excludes chunks more recent than the cutoff date."""
    until = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()
    results = retriever_with_dates.timeline("note content", top_k=20, until=until)
    source_keys = {c.source_key for c in results}
    # new-doc is 2 days ago — should be excluded
    assert "new-doc" not in source_keys
    # multi-doc is 1 day ago — should be excluded
    assert "multi-doc" not in source_keys


def test_timeline_empty_when_no_candidates(tmp_path: Path) -> None:
    """timeline() on an empty store returns an empty list."""
    store = ChunkStore(tmp_path / "empty.db", embedding_dim=DIM)
    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)
    results = r.timeline("anything at all", top_k=10)
    assert results == []
    store.close()


# ---------------------------------------------------------------------------
# recent_activity() tests
# ---------------------------------------------------------------------------


def test_recent_activity_returns_within_window(retriever_with_dates: Retriever) -> None:
    """Chunks updated within the window appear; older chunks don't."""
    results = retriever_with_dates.recent_activity(days=7)
    source_keys = {c.source_key for c in results}
    # Within 7 days: new-doc (2d), paper-a (3d), multi-doc (1d)
    assert "new-doc" in source_keys
    assert "paper-a" in source_keys
    assert "multi-doc" in source_keys
    # Outside 7 days: old-doc (30d), mid-doc (10d), paper-b (25d)
    assert "old-doc" not in source_keys
    assert "mid-doc" not in source_keys
    assert "paper-b" not in source_keys


def test_recent_activity_dedupes_by_source_key(retriever_with_dates: Retriever) -> None:
    """multi-doc has 3 chunks but recent_activity should include it only once."""
    results = retriever_with_dates.recent_activity(days=7)
    multi_doc_entries = [c for c in results if c.source_key == "multi-doc"]
    assert len(multi_doc_entries) == 1, "same source_key should appear at most once"


def test_recent_activity_filter_sources(retriever_with_dates: Retriever) -> None:
    """filter_sources=["notes"] should exclude papers chunks."""
    results = retriever_with_dates.recent_activity(days=30, filter_sources=["notes"])
    source_types = {c.source_type for c in results}
    assert "papers" not in source_types
    assert "notes" in source_types


def test_recent_activity_top_k_caps_output(retriever_with_dates: Retriever) -> None:
    """top_k limits the number of returned chunks."""
    results = retriever_with_dates.recent_activity(days=365, top_k=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# _attach_summaries() — context-aware reranking support (#4)
# ---------------------------------------------------------------------------


def test_attach_summaries_populates_and_caches_by_source_key(tmp_path: Path) -> None:
    """Per-doc summary is attached to each chunk; lookups are cached by
    (source_type, source_key) so M docs cost M queries regardless of N chunks."""
    store = ChunkStore(tmp_path / "sum.db", embedding_dim=DIM)
    items = [
        (make_dupe("notes", "doc-A", 0, "A chunk 0"), fake_embedding(0)),
        (make_dupe("notes", "doc-A", 1, "A chunk 1"), fake_embedding(1)),
        (make_dupe("notes", "doc-B", 0, "B chunk 0"), fake_embedding(2)),
    ]
    store.upsert_batch(items)
    store.upsert_summary("notes", "doc-A", "summary of A", "h", "model", 1)
    # doc-B intentionally has no summary.

    embedder = MagicMock()
    r = Retriever(store=store, embedder=embedder)

    chunks = store.get_by_source_key("notes", "doc-A") + store.get_by_source_key("notes", "doc-B")

    calls: list[tuple[str, str]] = []
    real_get = store.get_summary

    def counting_get(stype: str, skey: str):
        calls.append((stype, skey))
        return real_get(stype, skey)

    store.get_summary = counting_get  # type: ignore[method-assign]
    r._attach_summaries(chunks)

    by_key = {(c.source_key, c.metadata["chunk_index"]): c for c in chunks}
    # Both doc-A chunks get the summary; doc-B gets None.
    assert by_key[("doc-A", 0)].summary == "summary of A"
    assert by_key[("doc-A", 1)].summary == "summary of A"
    assert by_key[("doc-B", 0)].summary is None
    # 3 chunks, 2 distinct docs → exactly 2 DB lookups (cached).
    assert len(calls) == 2
    store.close()


def test_rerank_pool_gets_summaries_attached(tmp_path: Path) -> None:
    """query(rerank=True) attaches summaries to the rerank pool before scoring."""
    store = ChunkStore(tmp_path / "rr.db", embedding_dim=DIM)
    items = [(make_dupe("notes", "doc-A", 0, "A body"), fake_embedding(0))]
    store.upsert_batch(items)
    store.upsert_summary("notes", "doc-A", "the summary", "h", "model", 1)

    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))

    captured: list = []

    class SpyReranker:
        def rerank(self, query, candidates, top_n=None):
            captured.extend(candidates)
            return candidates

    r = Retriever(store=store, embedder=embedder, reranker=SpyReranker())
    r.query("anything", top_k=5, hybrid=False, rerank=True)

    assert captured, "reranker received no candidates"
    assert captured[0].summary == "the summary"
    store.close()


def test_auto_fts_weight_bounds_regex_scan_length() -> None:
    """Reference patterns must run against a bounded slice of input, so a
    catastrophic-backtracking pattern can't be fed an unbounded string
    (ReDoS blast-radius limit)."""
    from corpus.retriever import MAX_REGEX_SCAN_CHARS

    seen: dict[str, int] = {}

    class Spy:
        def search(self, s: str) -> None:
            seen["len"] = len(s)
            return None

    r = Retriever.__new__(Retriever)
    r._refs = [(Spy(), "tickets")]  # type: ignore[list-item]
    r._auto_fts_weight("x" * (MAX_REGEX_SCAN_CHARS + 5000))
    assert seen["len"] <= MAX_REGEX_SCAN_CHARS


@pytest.fixture
def lopsided_retriever(tmp_path: Path) -> Retriever:
    """One huge source and one tiny source, to expose pool starvation."""
    store = ChunkStore(tmp_path / "lop.db", embedding_dim=DIM)
    items = []
    for i in range(200):
        items.append((make_dupe("photos", f"img-{i}", 0, f"photo {i}"), fake_embedding(i)))
    for i in range(3):
        items.append((make_dupe("notes", f"n-{i}", 0, f"note {i}"), fake_embedding(i + 5000)))
    store.upsert_batch(items)
    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)
    yield r
    store.close()


def test_small_source_not_starved_by_large_one(lopsided_retriever: Retriever) -> None:
    """A 3-chunk source must still be reachable beside a 200-chunk source."""
    result = lopsided_retriever.query(
        "anything", top_k=5, max_per_source_type=3, hybrid=False
    )
    types = {c.source_type for c in result.chunks}
    assert "notes" in types, f"small source starved; got only {types}"


def test_top_k_is_filled_despite_cap(lopsided_retriever: Retriever) -> None:
    """The cap must not leave result slots empty when other sources have content."""
    result = lopsided_retriever.query(
        "anything", top_k=5, max_per_source_type=3, hybrid=False
    )
    assert len(result.chunks) == 5, f"expected 5 results, got {len(result.chunks)}"


def test_source_types_lists_distinct_types(tmp_path: Path) -> None:
    store = ChunkStore(tmp_path / "st.db", embedding_dim=DIM)
    store.upsert_batch([
        (make_dupe("photos", "a", 0, "x"), fake_embedding(1)),
        (make_dupe("notes", "b", 0, "y"), fake_embedding(2)),
        (make_dupe("notes", "c", 0, "z"), fake_embedding(3)),
    ])
    assert sorted(store.source_types()) == ["notes", "photos"]
    store.close()


# ---------------------------------------------------------------------------
# Fix round 1 regressions: tie-break determinism (Finding 3) and duplicate
# filter_sources (Finding 4) in the per-source-type pool construction.
# ---------------------------------------------------------------------------


def test_tie_break_is_deterministic_regardless_of_filter_sources_order(tmp_path: Path) -> None:
    """Equal-distance hits must break ties on chunk id, not on insertion
    order from the per-source-type fetch loop. Before the fix, list.sort's
    stability preserved insertion order -- which followed whatever order
    filter_sources was passed in, so the same corpus and query could return
    a DIFFERENT winner depending purely on argument order. After the fix,
    the tie-break is fully determined by id and independent of fetch order."""
    store = ChunkStore(tmp_path / "tie.db", embedding_dim=DIM)
    tied_embedding = fake_embedding(0)
    items = [
        (make_dupe("alpha", "doc", 0, "alpha content"), tied_embedding),
        (make_dupe("zeta", "doc", 0, "zeta content"), tied_embedding),
    ]
    store.upsert_batch(items)
    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=tied_embedding)
    r = Retriever(store=store, embedder=embedder)

    result_az = r.query(
        "anything", top_k=2, filter_sources=["alpha", "zeta"],
        max_per_source_type=None, dedupe_by_source=False, hybrid=False,
    )
    result_za = r.query(
        "anything", top_k=2, filter_sources=["zeta", "alpha"],
        max_per_source_type=None, dedupe_by_source=False, hybrid=False,
    )
    ids_az = [c.id for c in result_az.chunks]
    ids_za = [c.id for c in result_za.chunks]
    assert ids_az == ids_za, (
        f"tie-break depends on filter_sources argument order: {ids_az} vs {ids_za}"
    )
    assert ids_az == sorted(ids_az), f"tie not broken deterministically by id: {ids_az}"
    store.close()


def test_duplicate_filter_sources_do_not_double_fetch_or_change_results(tmp_path: Path) -> None:
    """Repeated entries in filter_sources must not re-fetch (and re-append)
    a source's candidates once per repetition -- that would inflate its RRF
    rank contribution and shrink per_source sizing via len() counting
    duplicates as extra source types."""
    store = ChunkStore(tmp_path / "dup.db", embedding_dim=DIM)
    items = []
    for i in range(5):
        items.append((make_dupe("notes", f"n-{i}", 0, f"note {i}"), fake_embedding(i)))
    for i in range(5):
        items.append((make_dupe("photos", f"p-{i}", 0, f"photo {i}"), fake_embedding(i + 100)))
    store.upsert_batch(items)
    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)

    calls: list[list[str] | None] = []
    real_vector_search = store.vector_search

    def counting_vector_search(embedding, top_k, filter_sources=None):  # type: ignore[no-untyped-def]
        calls.append(list(filter_sources) if filter_sources else None)
        return real_vector_search(embedding, top_k, filter_sources=filter_sources)

    store.vector_search = counting_vector_search  # type: ignore[method-assign]

    result_dup = r.query(
        "anything", top_k=5, filter_sources=["notes", "notes", "photos"],
        max_per_source_type=None, hybrid=False,
    )
    # Assert the INTENT, not the pool-building shape: a duplicated entry must
    # not cause the same source type to be fetched twice. (This previously
    # pinned "exactly one call with filter_sources == ['notes']", which was a
    # detail of the per-source fan-out the adaptive pool replaced.)
    for c in calls:
        assert c is None or len(c) == len(set(c)), f"duplicate types in one fetch: {c}"
    notes_fetches = [c for c in calls if c is not None and "notes" in c]
    assert len(notes_fetches) == 1, f"notes fetched {len(notes_fetches)} times: {calls}"

    result_deduped = r.query(
        "anything", top_k=5, filter_sources=["notes", "photos"],
        max_per_source_type=None, hybrid=False,
    )
    assert [c.id for c in result_dup.chunks] == [c.id for c in result_deduped.chunks]
    store.close()


def test_fts_search_called_once_not_per_source_type(tmp_path: Path) -> None:
    """FTS must be issued as ONE global query, not once per source type.

    ChunkStore.fts_search's docstring/comment records why: a per-source
    rowid pre-filter was tried and measured ~1561x slower at 40k chunks / 4
    source types (it defeats FTS5's rank-ordered LIMIT short-circuit). That
    guard lives in fts_search's implementation, but nothing pins the
    CALL COUNT from the retriever side -- a future refactor to a per-source
    loop that keeps the post-filter would still return correct results (so
    every correctness test would keep passing) while costing ~96ms vs ~24ms
    at 40k chunks. Spy on fts_search to catch that regression directly,
    following the pattern in
    test_duplicate_filter_sources_do_not_double_fetch_or_change_results."""
    store = ChunkStore(tmp_path / "fts_calls.db", embedding_dim=DIM)
    items = []
    for i in range(3):
        items.append((make_dupe("notes", f"n-{i}", 0, f"note {i} about testing"), fake_embedding(i)))
    for i in range(3):
        items.append((make_dupe("papers", f"p-{i}", 0, f"paper {i} about testing"), fake_embedding(i + 10)))
    for i in range(3):
        items.append((make_dupe("photos", f"ph-{i}", 0, f"photo {i} about testing"), fake_embedding(i + 20)))
    store.upsert_batch(items)

    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)

    calls: list[list[str] | None] = []
    real_fts_search = store.fts_search

    def counting_fts_search(query, top_k, filter_sources=None):  # type: ignore[no-untyped-def]
        calls.append(list(filter_sources) if filter_sources else None)
        return real_fts_search(query, top_k, filter_sources=filter_sources)

    store.fts_search = counting_fts_search  # type: ignore[method-assign]

    # No filter_sources -> source_types() resolves to all 3 registered types.
    # A per-source loop (the regression this test guards against) would call
    # fts_search 3 times here instead of once.
    r.query("testing", top_k=5, hybrid=True)
    assert len(calls) == 1, f"fts_search called {len(calls)} times, expected exactly 1: {calls}"

    calls.clear()
    r.query("testing", top_k=5, filter_sources=["notes", "papers"], hybrid=True)
    assert len(calls) == 1, f"fts_search called {len(calls)} times, expected exactly 1: {calls}"
    store.close()


def test_single_source_filter_is_not_capped(retriever: Retriever) -> None:
    """The cap spreads results ACROSS source types. A query already filtered
    to one type has nothing to spread across, so the cap could only subtract:
    top_k=10 filtered to one source returned 3 results with far more matching
    chunks available."""
    result = retriever.query(
        "anything", top_k=6, filter_sources=["notes"], max_per_source_type=2, hybrid=False
    )

    assert len(result.chunks) > 2
    assert {c.source_type for c in result.chunks} == {"notes"}


def test_the_cap_backfills_instead_of_truncating(retriever: Retriever) -> None:
    """A fixed cap must not silently shrink a variable top_k.

    This previously asserted the opposite -- that the cap binds even when it
    leaves the caller short. Live evidence overturned it: on a large
    archive with 3 source types and cap 3, an MCP search advertising "max 30"
    could not structurally return more than 9, and typically returned 5.

    The store holds 9 distinct source keys (6 notes + 3 papers), so top_k=8
    is satisfiable only by going past 2-per-type.
    """
    result = retriever.query("anything", top_k=8, max_per_source_type=2, hybrid=False)

    assert len(result.chunks) == 8
    # Spread still won every slot it could use: both types appear, and the
    # cap shaped the front of the list before backfill took over.
    assert {c.source_type for c in result.chunks} == {"notes", "papers"}
    head = result.chunks[:4]
    counts: dict[str, int] = {}
    for c in head:
        counts[c.source_type] = counts.get(c.source_type, 0) + 1
    assert all(n <= 2 for n in counts.values()), counts


def test_a_short_result_now_means_the_pool_was_exhausted(retriever: Retriever) -> None:
    """Backfill must not invent results. 9 distinct source keys exist; asking
    for 20 returns 9, not 20."""
    result = retriever.query("anything", top_k=20, max_per_source_type=2, hybrid=False)

    assert len(result.chunks) == 9


# --- timeline ---------------------------------------------------------------
#
# "What happened lately about X" asks for RECENT material, which is rarely the
# most semantically central. The date filter ran in Python after retrieval, so
# a topic whose nearest chunks all fell outside the range returned nothing.


def _dated_store(tmp_path):
    from corpus.db.sqlite import ChunkStore
    from corpus.types import Chunk, ChunkKind, ChunkMetadata
    from corpus.util.hash import chunk_id

    store = ChunkStore(tmp_path / "dated.db", embedding_dim=DIM)

    def mk(key, text, updated, emb):
        return (
            Chunk(
                id=chunk_id("notes", key, ChunkKind.BODY, 0),
                content=text,
                content_hash=f"h{key}",
                metadata=ChunkMetadata(
                    source_type="notes", source_key=key, chunk_kind=ChunkKind.BODY,
                    chunk_index=0, title=key, updated_at=updated,
                ),
            ),
            emb,
        )

    items = [mk(f"old{i}", f"migration status {i}", "2020-01-01", [0.10] * DIM) for i in range(200)]
    items += [mk(f"new{i}", f"migration status {i}", "2026-09-01", [0.40] * DIM) for i in range(10)]
    store.upsert_batch(items)
    return store


class _NearOld:
    total_tokens_used = 0

    def embed_query(self, q):  # nearest the OLD chunks
        return [0.10] * DIM

    def embed_documents(self, ds):
        return [[0.10] * DIM for _ in ds]


def test_timeline_finds_recent_items_buried_below_nearer_old_ones(tmp_path: Path) -> None:
    store = _dated_store(tmp_path)
    try:
        r = Retriever(store=store, embedder=_NearOld())

        results = r.timeline("migration status", top_k=5, since="2026-01-01")

        assert len(results) == 5
        assert all(c.source_key.startswith("new") for c in results)
    finally:
        store.close()


def test_timeline_is_not_limited_by_the_diversity_cap(tmp_path: Path) -> None:
    """A timeline is one topic ordered by DATE. The default cap of 3 per
    source type silently limited every timeline to 3 x (number of types)
    candidates no matter what top_k asked for."""
    store = _dated_store(tmp_path)
    try:
        r = Retriever(store=store, embedder=_NearOld())

        results = r.timeline("migration status", top_k=10)

        assert len(results) == 10
    finally:
        store.close()


def test_timeline_with_a_range_that_matches_nothing_returns_empty(tmp_path: Path) -> None:
    # The widening must terminate rather than chase an empty range forever.
    store = _dated_store(tmp_path)
    try:
        r = Retriever(store=store, embedder=_NearOld())

        assert r.timeline("migration status", top_k=5, since="2030-01-01") == []
    finally:
        store.close()


# --- assemble_results hooks -------------------------------------------------
#
# The selection logic is shared with two private forks that hook into it. The
# Retriever above never passes a hook, so without these the fork-only paths
# would be untested here and would drift again -- which is exactly how the
# same truncation bug ended up in three copies.


def _stored(source_type: str, source_key: str, content: str | None = None) -> StoredChunk:
    # Distinct by default. This used to default to "x" for every chunk, which
    # was harmless while dedupe keyed on id alone -- once content became a
    # dedupe key it collapsed whole fixtures to one result.
    if content is None:
        content = f"content of {source_type}:{source_key}"
    return StoredChunk(
        id=chunk_id(source_type, source_key, ChunkKind.HEADER, 0),
        source_type=source_type,
        source_key=source_key,
        content=content,
        metadata={},
        title=f"{source_type}:{source_key}",
        url=None,
    )


def test_reject_hook_drops_chunks_before_they_consume_a_slot() -> None:
    fused = [_stored("a", f"k{i}") for i in range(6)]
    out = assemble_results(
        fused, top_k=3, max_per_source_type=None,
        reject=lambda c: c.source_key in {"k0", "k1"},
    )
    assert [c.source_key for c in out] == ["k2", "k3", "k4"]


def test_source_key_hook_overrides_the_dedupe_identity() -> None:
    """One fork keys email by thread, so a 40-message thread cannot fill the
    answer on its own."""
    fused = [_stored("email", f"msg-{i}") for i in range(5)] + [_stored("doc", "d1")]
    out = assemble_results(
        fused, top_k=10, max_per_source_type=None,
        source_key=lambda c: (c.source_type, "thread-1" if c.source_type == "email" else c.source_key),
    )
    assert [(c.source_type, c.source_key) for c in out] == [("email", "msg-0"), ("doc", "d1")]


def test_content_key_hook_collapses_republished_text() -> None:
    fused = [
        _stored("doc", "v1", "the same paragraph"),
        _stored("doc", "v2", "the same paragraph"),
        _stored("doc", "v3", "something else"),
    ]
    out = assemble_results(
        fused, top_k=10, max_per_source_type=None, content_key=lambda c: c.content,
    )
    assert [c.source_key for c in out] == ["v1", "v3"]


def test_backfill_only_promotes_chunks_that_passed_every_other_gate() -> None:
    """A chunk rejected or deduped must never reappear via the overflow list."""
    fused = [
        _stored("a", "a1", "one"),
        _stored("a", "a2", "two"),               # over a cap of 1 -> overflow
        _stored("a", "noise", "three"),          # rejected outright
        _stored("a", "a3", "dupe"),              # over the cap -> overflow
        _stored("a", "a4", "dupe"),              # same content as a3 -> collapsed
    ]
    out = assemble_results(
        fused, top_k=5, max_per_source_type=1,
        reject=lambda c: c.source_key == "noise",
        content_key=lambda c: c.content,
    )
    keys = [c.source_key for c in out]
    assert "noise" not in keys and "a4" not in keys
    # a1 takes the single capped slot; a2 and a3 backfill in score order.
    assert keys == ["a1", "a2", "a3"]


# --- adaptive candidate pool -------------------------------------------------
#
# Every chunk has a distance to the query, so a source type holding most of the
# corpus fills the pool by volume regardless of relevance. A type absent from
# the POOL cannot be recovered downstream: neither the diversity cap nor its
# backfill invents candidates. Measured on a large archive that is predominantly one
# type, a global-only pool starved a type completely on 1 query in 12.


@pytest.fixture
def skewed(tmp_path: Path) -> Retriever:
    """One type swamps the store, mimicking the live 82/9/9 split."""
    store = ChunkStore(tmp_path / "skew.db", embedding_dim=DIM)
    items = []
    # 200 "bulk" chunks packed tightly around the query vector.
    for i in range(200):
        items.append((make_dupe("bulk", f"b-{i}", 0, f"bulk {i}"), fake_embedding(i)))
    # 3 "rare" chunks further away -- they lose every global top-k race.
    for i in range(3):
        items.append((make_dupe("rare", f"r-{i}", 0, f"rare {i}"), fake_embedding(5000 + i)))
    store.upsert_batch(items)
    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)
    yield r
    store.close()


def test_a_starved_source_type_is_topped_up_into_the_pool(skewed: Retriever) -> None:
    result = skewed.query("anything", top_k=5, max_per_source_type=2, hybrid=False)

    assert "rare" in {c.source_type for c in result.chunks}, (
        "the swamped type never reached the candidate pool, so the cap and its "
        "backfill had nothing to select"
    )


def test_the_top_up_does_not_fire_when_every_type_is_represented(tmp_path: Path) -> None:
    """The cost must be paid only by queries that were actually starved."""
    store = ChunkStore(tmp_path / "even.db", embedding_dim=DIM)
    items = []
    for i in range(10):
        items.append((make_dupe("a", f"a-{i}", 0, f"a {i}"), fake_embedding(i)))
        items.append((make_dupe("b", f"b-{i}", 0, f"b {i}"), fake_embedding(i + 50)))
    store.upsert_batch(items)
    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=fake_embedding(0))
    r = Retriever(store=store, embedder=embedder)

    calls: list[object] = []
    real = store.vector_search

    def counting(embedding, top_k, filter_sources=None):  # type: ignore[no-untyped-def]
        calls.append(filter_sources)
        return real(embedding, top_k, filter_sources=filter_sources)

    store.vector_search = counting  # type: ignore[method-assign]
    r.query("anything", top_k=5, max_per_source_type=3, hybrid=False)

    assert len(calls) == 1, f"expected one global fetch, got {len(calls)}: {calls}"
    store.close()


def test_the_pool_never_contains_the_same_chunk_twice(skewed: Retriever) -> None:
    """A chunk can arrive from BOTH the global pass and its type's top-up.

    RRF ranks by position, so the same id appearing twice would be scored as
    two candidates and double-weighted. The old fan-out queried each type
    exactly once and could not produce duplicates; this can.
    """
    captured: list[list[StoredChunk]] = []
    real_rrf = corpus.retriever.reciprocal_rank_fusion

    def spy(lists, weights, key):  # type: ignore[no-untyped-def]
        captured.extend(lists)
        return real_rrf(lists, weights=weights, key=key)

    corpus.retriever.reciprocal_rank_fusion = spy
    try:
        skewed.query("anything", top_k=5, max_per_source_type=2, hybrid=True)
    finally:
        corpus.retriever.reciprocal_rank_fusion = real_rrf

    assert captured, "fusion was never reached"
    vector_list = captured[0]
    ids = [c.id for c in vector_list]
    assert len(ids) == len(set(ids)), "duplicate chunk in the vector candidate list"


# --- the same passage must not occupy two result slots -----------------------


def _dupe(cid: str, key: str, text: str, distance: float) -> StoredChunk:
    """A stored chunk with an explicit id, so two documents can share content."""
    return StoredChunk(
        id=cid,
        source_type="notes",
        source_key=key,
        content=text,
        metadata={},
        title=key,
        url=None,
        distance=distance,
    )


def test_identical_content_from_two_documents_takes_one_slot() -> None:
    """Measured on live archives: 6.5% of one index, 2.8% of another, are
    passages that also appear under a different document -- a file copied into
    two folders, a note and its backup, mail received at two addresses.

    Deduping by chunk id alone cannot see it: the ids differ because the
    documents differ. The passage then spends two slots of a top-k saying the
    same thing twice.
    """
    from corpus.retriever import assemble_results

    fused = [
        _dupe("1", "live/note.md", "the same paragraph", 0.10),
        _dupe("2", "backup/note.md", "the same paragraph", 0.11),
        _dupe("3", "other.md", "a different paragraph", 0.20),
    ]
    out = assemble_results(fused, top_k=3)
    assert [c.id for c in out] == ["1", "3"], [c.id for c in out]


def test_the_first_copy_offered_is_the_one_kept() -> None:
    """`assemble_results` consumes the order it is given; it does not sort.

    Callers hand it a ranked list (build_vector_pool sorts by distance, RRF
    emits by rank), so in practice the survivor is the best-scoring copy. The
    first draft of this test asserted score-based selection and failed --
    worth pinning the real contract rather than the assumed one.
    """
    from corpus.retriever import assemble_results

    fused = [
        _dupe("better", "b.md", "same text", 0.10),
        _dupe("worse", "a.md", "same text", 0.50),
    ]
    assert [c.id for c in assemble_results(fused, top_k=5)] == ["better"]


def test_distinct_passages_are_untouched() -> None:
    from corpus.retriever import assemble_results

    fused = [
        _dupe("1", "a.md", "first passage", 0.10),
        _dupe("2", "b.md", "second passage", 0.20),
    ]
    assert len(assemble_results(fused, top_k=5)) == 2


def test_a_caller_can_still_override_the_content_identity() -> None:
    # The hook predates this default and one fork relies on it.
    from corpus.retriever import assemble_results

    fused = [
        _dupe("1", "a.md", "Text With Case", 0.10),
        _dupe("2", "b.md", "text with case", 0.20),
    ]
    out = assemble_results(fused, top_k=5, content_key=lambda c: c.content.lower())
    assert len(out) == 1


def test_a_timeline_keeps_repeated_text_at_different_dates() -> None:
    """A timeline's unit is a DATED OCCURRENCE, not a unique passage.

    A recurring report, a daily-log template or a repeated status update is
    genuinely the same text at two dates, and those are two events. Content
    dedupe would collapse them -- and worse, because the date filter runs
    AFTER retrieval, it can discard the RECENT copy in favour of an older one
    the filter then removes, returning nothing for a topic that has plenty.

    Found because a fixture happened to give old and new chunks identical
    text; the interaction it exposed is real.
    """
    from corpus.retriever import assemble_results

    fused = [
        _dupe("old", "report-2020.md", "weekly status: all green", 0.10),
        _dupe("new", "report-2026.md", "weekly status: all green", 0.20),
    ]
    # The default collapses them...
    assert len(assemble_results(fused, top_k=5)) == 1
    # ...and the timeline's override keeps both, so the date filter can choose.
    both = assemble_results(fused, top_k=5, content_key=lambda c: c.id)
    assert [c.id for c in both] == ["old", "new"]
