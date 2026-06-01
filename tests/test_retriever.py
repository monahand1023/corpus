from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corpus.db.sqlite import ChunkStore
from corpus.retriever import Retriever
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 1024


def fake_embedding(seed: int) -> list[float]:
    return [math.sin((seed + 1) * (i + 1) * 0.001) for i in range(DIM)]


def make_chunk(source_type: str, source_key: str, idx: int, content: str) -> Chunk:
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
        items.append((make_chunk("notes", f"doc-{i}", 0, f"note {i}"), fake_embedding(i)))
    for i in range(3):
        items.append((make_chunk("notes", "doc-0", i + 1, f"doc-0 body {i}"), fake_embedding(i + 10)))
    for i in range(3):
        items.append((make_chunk("papers", f"p-{i}", 0, f"paper {i}"), fake_embedding(i + 20)))
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
    result = retriever.query("anything", top_k=10, max_per_source_type=2, hybrid=False)
    counts: dict[str, int] = {}
    for c in result.chunks:
        counts[c.source_type] = counts.get(c.source_type, 0) + 1
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
        (make_chunk("tickets", "TKT-1", 0, "depends on TKT-9 to ship"), fake_embedding(1)),
        (make_chunk("tickets", "TKT-9", 0, "the dependency ticket"), fake_embedding(2)),
        (make_chunk("tickets", "TKT-5", 0, "unrelated"), fake_embedding(3)),
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
        (make_chunk("notes", "A", 0, "mentions DOC-9"), fake_embedding(1)),
        (make_chunk("notes", "DOC-9", 0, "would-be reference"), fake_embedding(2)),
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
        (make_chunk("notes", "doc-A", 0, "A chunk 0"), fake_embedding(0)),
        (make_chunk("notes", "doc-A", 1, "A chunk 1"), fake_embedding(1)),
        (make_chunk("notes", "doc-B", 0, "B chunk 0"), fake_embedding(2)),
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
    items = [(make_chunk("notes", "doc-A", 0, "A body"), fake_embedding(0))]
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
