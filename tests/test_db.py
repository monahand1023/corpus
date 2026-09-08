from __future__ import annotations

import math
from pathlib import Path

import pytest

from corpus.db.sqlite import ChunkStore, EmbeddingDimMismatch
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 1024


def make_chunk(key: str, idx: int, kind: ChunkKind, content: str) -> Chunk:
    return Chunk(
        id=chunk_id("notes", key, kind, idx),
        content=content,
        content_hash=sha256(content),
        metadata=ChunkMetadata(
            source_type="notes",
            source_key=key,
            chunk_kind=kind,
            chunk_index=idx,
            title=f"{key}",
        ),
    )


def fake_embedding(seed: int) -> list[float]:
    return [math.sin((seed + 1) * (i + 1) * 0.001) for i in range(DIM)]


def make_chunk_typed(source_type: str, key: str, idx: int, kind: ChunkKind, content: str) -> Chunk:
    """Like make_chunk, but with a caller-supplied source_type (make_chunk
    hardcodes "notes")."""
    return Chunk(
        id=chunk_id(source_type, key, kind, idx),
        content=content,
        content_hash=sha256(content),
        metadata=ChunkMetadata(
            source_type=source_type,
            source_key=key,
            chunk_kind=kind,
            chunk_index=idx,
            title=f"{key}",
        ),
    )


@pytest.fixture
def store(tmp_path: Path) -> ChunkStore:
    s = ChunkStore(tmp_path / "test.db", embedding_dim=DIM)
    yield s
    s.close()


def test_upsert_skips_unchanged(store: ChunkStore) -> None:
    items = [(make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"content {i}"), fake_embedding(i)) for i in range(3)]
    r1 = store.upsert_batch(items)
    assert r1.upserted == 3 and r1.skipped == 0
    r2 = store.upsert_batch(items)
    assert r2.upserted == 0 and r2.skipped == 3


def test_vector_search_returns_closest(store: ChunkStore) -> None:
    items = [(make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i)) for i in range(5)]
    store.upsert_batch(items)
    results = store.vector_search(fake_embedding(2), top_k=3)
    assert len(results) == 3
    assert results[0].source_key == "DOC-2"


def test_vector_search_filter_sources_is_a_prefilter_not_a_postfilter(store: ChunkStore) -> None:
    """Regression for source starvation: filter_sources must constrain the
    ANN search itself, not filter a fixed-size global over-fetch window
    after the fact. With 200 "photos" chunks crowding the neighborhood and
    only 3 "notes" chunks, the notes rank ~96th-102nd nearest globally to
    the query -- a post-filter over a top_k*3=60 window finds none of them.
    A true pre-filter must find all three regardless of global rank."""
    items = [
        (make_chunk_typed("photos", f"img-{i}", 0, ChunkKind.SECTION, f"photo {i}"), fake_embedding(i))
        for i in range(200)
    ]
    items += [
        (make_chunk_typed("notes", f"n-{i}", 0, ChunkKind.SECTION, f"note {i}"), fake_embedding(i + 5000))
        for i in range(3)
    ]
    store.upsert_batch(items)
    results = store.vector_search(fake_embedding(0), top_k=20, filter_sources=["notes"])
    keys = {c.source_key for c in results}
    assert keys == {"n-0", "n-1", "n-2"}, f"notes starved by post-filter; got {keys}"


def test_vector_search_top_k_exceeding_filtered_row_count_does_not_raise(store: ChunkStore) -> None:
    """top_k may exceed the number of rows matching filter_sources; the
    pre-filtered ANN query (k applied to an already-constrained rowid set)
    must not raise, just return what's available."""
    items = [(make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i)) for i in range(3)]
    store.upsert_batch(items)
    results = store.vector_search(fake_embedding(0), top_k=50, filter_sources=["notes"])
    assert len(results) == 3


def test_fts_search_filter_sources_is_a_prefilter_not_a_postfilter(store: ChunkStore) -> None:
    """Same starvation bug applies to BM25 search: filter_sources must
    constrain the FTS MATCH itself, not a fixed-size global over-fetch
    window filtered afterward."""
    items = [
        (make_chunk_typed("photos", f"img-{i}", 0, ChunkKind.SECTION, f"needle {i}"), fake_embedding(i))
        for i in range(200)
    ]
    items += [
        (make_chunk_typed("notes", f"n-{i}", 0, ChunkKind.SECTION, f"needle {i}"), fake_embedding(i + 5000))
        for i in range(3)
    ]
    store.upsert_batch(items)
    results = store.fts_search("needle", top_k=20, filter_sources=["notes"])
    keys = {c.source_key for c in results}
    assert keys == {"n-0", "n-1", "n-2"}, f"notes starved by post-filter; got {keys}"


def test_delete_orphans(store: ChunkStore) -> None:
    items = [(make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i)) for i in range(5)]
    store.upsert_batch(items)
    seen_ids = {items[0][0].id, items[1][0].id}
    deleted = store.delete_orphans("notes", seen_ids)
    assert deleted == 3
    assert store.stats()["total"] == 2


def test_summary_roundtrip(store: ChunkStore) -> None:
    store.upsert_summary("notes", "DOC-1", "a summary", "h1", "model-x", 100)
    out = store.get_summary("notes", "DOC-1")
    assert out is not None
    assert out["summary"] == "a summary"
    assert out["doc_hash"] == "h1"


def test_embedding_dim_guard(tmp_path: Path) -> None:
    """Switching dims on an existing DB must fail loudly, not silently corrupt."""
    s1 = ChunkStore(tmp_path / "guard.db", embedding_dim=1024)
    s1.close()
    with pytest.raises(EmbeddingDimMismatch):
        ChunkStore(tmp_path / "guard.db", embedding_dim=512)


def test_delete_orphans_clears_vec_and_fts(store: ChunkStore) -> None:
    """Orphan deletion must remove rows from chunks_vec and chunks_fts too,
    not just the chunks table — otherwise the virtual tables accumulate
    dangling rows that corrupt future joins."""
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"orphan test content {i}"), fake_embedding(i))
        for i in range(5)
    ]
    store.upsert_batch(items)

    # Sanity: all three tables have 5 rows
    conn = store._conn
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 5
    assert conn.execute("SELECT COUNT(*) c FROM chunks_vec").fetchone()["c"] == 5
    assert conn.execute("SELECT COUNT(*) c FROM chunks_fts").fetchone()["c"] == 5

    # Keep only the first 2 — drop 3 as orphans
    seen_ids = {items[0][0].id, items[1][0].id}
    deleted = store.delete_orphans("notes", seen_ids)
    assert deleted == 3

    # All three tables should now have exactly 2 rows
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 2
    assert conn.execute("SELECT COUNT(*) c FROM chunks_vec").fetchone()["c"] == 2
    assert conn.execute("SELECT COUNT(*) c FROM chunks_fts").fetchone()["c"] == 2


def test_delete_orphans_handles_many_rows(store: ChunkStore) -> None:
    """>1000 orphans must delete without hitting SQLite's bind-variable cap.

    The old implementation deleted `chunks` via a single
    `WHERE id IN (?,?,...)` with one bind variable per orphan — that crashes
    with 'too many SQL variables' past ~32k. Per-rowid deletes in the loop have
    no such limit. 1001 rows proves the loop path is taken and stays correct."""
    n = 1001
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"orphan {i}"), fake_embedding(i))
        for i in range(n)
    ]
    store.upsert_batch(items)
    assert store.stats()["total"] == n

    # Keep only one — everything else is an orphan.
    seen_ids = {items[0][0].id}
    deleted = store.delete_orphans("notes", seen_ids)
    assert deleted == n - 1

    conn = store._conn
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM chunks_vec").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM chunks_fts").fetchone()["c"] == 1


def test_delete_by_source_clears_vec_and_fts(store: ChunkStore) -> None:
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i))
        for i in range(4)
    ]
    store.upsert_batch(items)
    store.delete_by_source("notes")
    conn = store._conn
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM chunks_vec").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM chunks_fts").fetchone()["c"] == 0


def test_connections_are_per_thread(store: ChunkStore) -> None:
    """Each thread accessing the store should get its own sqlite connection."""
    import threading

    main_conn = store._conn
    other_conn_box: list = []

    def grab() -> None:
        other_conn_box.append(store._conn)

    t = threading.Thread(target=grab)
    t.start()
    t.join()

    assert other_conn_box, "thread did not run"
    assert other_conn_box[0] is not main_conn, "threads shared a connection"
