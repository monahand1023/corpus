from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import corpus.db.sqlite
from corpus.db.sqlite import (
    DEFAULT_CACHE_SIZE_MB,
    DEFAULT_MMAP_SIZE_MB,
    FTS_VERSION,
    ChunkStore,
    EmbeddingDimMismatch,
    OrphanPruneRefused,
    ReadOnlyStoreError,
)
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


def test_fts_search_filter_sources_is_a_postfilter_by_design(store: ChunkStore) -> None:
    """DELIBERATE, MEASURED DESIGN CHOICE (do not "fix" this back to a
    rowid-IN pre-filter like vector_search's): a rowid-IN pre-filter was
    tried here and measured at ~1561x slower than this global-query form
    (mean 7.3s/query, p99 17.5s at 40k chunks / 4 source types) because it
    defeats FTS5's rank-ordered LIMIT short-circuit, forcing SQLite to fall
    back to `USE TEMP B-TREE FOR ORDER BY` -- a full MATCH + sort of every
    matching row. See the comment in ChunkStore.fts_search for the full
    mechanism.

    This is safe unlike vector_search's starvation risk: BM25 only ranks
    chunks that actually contain the query's terms, so if the dominant
    source's content is NOT what the query matches on, a small genuinely
    matching source is never crowded out -- the over-fetch window only ever
    has as many candidates as chunks that matched at all."""
    items = [
        (make_chunk_typed("photos", f"img-{i}", 0, ChunkKind.SECTION, f"unrelated content {i}"), fake_embedding(i))
        for i in range(200)
    ]
    items += [
        (make_chunk_typed("notes", f"n-{i}", 0, ChunkKind.SECTION, f"needle {i}"), fake_embedding(i + 5000))
        for i in range(3)
    ]
    store.upsert_batch(items)
    # None of the 200 "photos" chunks contain "needle" -- only the 3 "notes"
    # chunks match the query at all, so they survive the post-filter.
    results = store.fts_search("needle", top_k=20, filter_sources=["notes"])
    keys = {c.source_key for c in results}
    assert keys == {"n-0", "n-1", "n-2"}, f"a genuinely-matching small source was starved; got {keys}"


def test_fts_search_widens_rather_than_starving_a_filtered_source(store: ChunkStore) -> None:
    """A large source that ALSO matches the query used to fill the fixed
    over-fetch window ahead of a smaller matching source, and the filtered
    query returned NOTHING — silently, with matching content in the store.

    That was pinned here as an accepted cost of avoiding a measured ~1561x
    pre-filter regression, on the reasoning that a genuinely-matching source
    would usually survive the window. Measured, it does not: it takes only a
    few times more rows in the other source to crowd it out completely.

    The window now widens when a filter is starving it. The uncrowded path
    still costs one query; only a filter that actually got crowded out pays
    for another, so the regression this trade-off existed to avoid is still
    avoided."""
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
    assert keys == {"n-0", "n-1", "n-2"}, f"filtered source starved; got {keys}"


def test_fts_filter_finds_a_source_that_sorts_last(store: ChunkStore) -> None:
    """The case that killed the widening approach before this one: when every
    chunk matches equally, BM25 ties and rows order by rowid, so a source
    inserted LAST sits below any bounded window. Widening bought latency and
    still returned nothing; the pre-filtered fallback returns the rows."""
    items = [
        (make_chunk_typed("photos", f"img-{i}", 0, ChunkKind.SECTION, "needle"), fake_embedding(i))
        for i in range(300)
    ]
    items += [
        (make_chunk_typed("notes", f"n-{i}", 0, ChunkKind.SECTION, "needle"), fake_embedding(i + 9000))
        for i in range(5)
    ]
    store.upsert_batch(items)

    results = store.fts_search("needle", top_k=5, filter_sources=["notes"])

    assert {c.source_key for c in results} == {f"n-{i}" for i in range(5)}


def test_fts_filter_on_a_source_with_no_matches_returns_empty(store: ChunkStore) -> None:
    items = [
        (make_chunk_typed("photos", f"img-{i}", 0, ChunkKind.SECTION, f"needle {i}"), fake_embedding(i))
        for i in range(300)
    ]
    store.upsert_batch(items)

    assert store.fts_search("needle", top_k=20, filter_sources=["notes"]) == []


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
    no such limit. 1001 rows proves the loop path is taken and stays correct.

    force=True: this test is about the delete mechanics, not the
    blast-radius guard (see test_delete_orphans_refuses_when_orphan_ratio_
    exceeds_threshold) — dropping 1000/1001 rows would otherwise trip it."""
    n = 1001
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"orphan {i}"), fake_embedding(i))
        for i in range(n)
    ]
    store.upsert_batch(items)
    assert store.stats()["total"] == n

    # Keep only one — everything else is an orphan.
    seen_ids = {items[0][0].id}
    deleted = store.delete_orphans("notes", seen_ids, force=True)
    assert deleted == n - 1

    conn = store._conn
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM chunks_vec").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM chunks_fts").fetchone()["c"] == 1


def test_delete_orphans_refuses_when_orphan_ratio_exceeds_threshold(store: ChunkStore) -> None:
    """The blast-radius guard: pruning almost all of a source's existing
    chunks in one call is refused (nothing deleted) rather than silently
    carried out, above the small-source floor."""
    n = 100
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i))
        for i in range(n)
    ]
    store.upsert_batch(items)

    with pytest.raises(OrphanPruneRefused) as exc_info:
        store.delete_orphans("notes", set())  # every chunk would be an orphan
    err = exc_info.value
    assert err.source_type == "notes"
    assert err.existing == n
    assert err.orphans == n
    assert err.ratio == 1.0
    assert "notes" in str(err)
    assert store.stats()["total"] == n, "refusal must delete nothing"


def test_delete_orphans_force_bypasses_the_guard(store: ChunkStore) -> None:
    """force=True (wired to --prune-anyway) deletes anyway, for a genuine
    bulk deletion."""
    n = 100
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i))
        for i in range(n)
    ]
    store.upsert_batch(items)

    deleted = store.delete_orphans("notes", set(), force=True)
    assert deleted == n
    assert store.stats()["total"] == 0


def test_delete_orphans_floor_allows_small_source_to_legitimately_drop_to_zero(
    store: ChunkStore,
) -> None:
    """A source with 3 chunks that legitimately drops to 0 is not blocked by
    the floor -- default min_chunks_for_guard=50 means the ratio check never
    even runs for a source this small."""
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i))
        for i in range(3)
    ]
    store.upsert_batch(items)

    deleted = store.delete_orphans("notes", set())
    assert deleted == 3
    assert store.stats()["total"] == 0


def test_delete_orphans_under_threshold_prunes_normally(store: ChunkStore) -> None:
    """A routine re-ingest that drops well under the ratio threshold prunes
    without any guard involvement."""
    n = 100
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i))
        for i in range(n)
    ]
    store.upsert_batch(items)
    seen_ids = {items[i][0].id for i in range(90)}  # drop 10/100 = 10% < 20%

    deleted = store.delete_orphans("notes", seen_ids)
    assert deleted == 10
    assert store.stats()["total"] == 90


def test_delete_orphans_guard_thresholds_are_configurable(store: ChunkStore) -> None:
    """max_orphan_ratio / min_chunks_for_guard are per-call overrides (the
    caller resolves them from corpus.toml's [pruning] section)."""
    n = 20
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"x {i}"), fake_embedding(i))
        for i in range(n)
    ]
    store.upsert_batch(items)

    # Default floor (50) would never guard a 20-chunk source; a caller-supplied
    # floor of 5 makes the guard apply here.
    with pytest.raises(OrphanPruneRefused):
        store.delete_orphans("notes", set(), min_chunks_for_guard=5)

    # A caller-supplied ratio of 1.0 (never refuse) lets the same deletion
    # through even though it drops 100% of the source.
    deleted = store.delete_orphans("notes", set(), min_chunks_for_guard=5, max_orphan_ratio=1.0)
    assert deleted == n


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


def test_fts_finds_two_character_japanese_word(store: ChunkStore) -> None:
    store.upsert_batch([
        (make_chunk("jp", 0, ChunkKind.BODY, "東京で会議をしました"), fake_embedding(1)),
    ])
    assert len(store.fts_search("東京", top_k=5)) == 1
    assert len(store.fts_search("会議", top_k=5)) == 1


def test_fts_finds_accented_word(store: ChunkStore) -> None:
    store.upsert_batch([
        (make_chunk("es", 0, ChunkKind.BODY, "la reunión en el café"), fake_embedding(2)),
    ])
    assert len(store.fts_search("café", top_k=5)) == 1
    assert len(store.fts_search("cafe", top_k=5)) == 1


def test_fts_does_not_raise_on_operator_like_query(store: ChunkStore) -> None:
    store.upsert_batch([
        (make_chunk("en", 0, ChunkKind.BODY, "cats and dogs"), fake_embedding(3)),
    ])
    # Previously raised fts5 syntax errors that were swallowed as "no hits".
    # Operator-like tokens must be treated as literal TERMS, not FTS5
    # operators: "OR"/"AND" become quoted terms that don't match "cats and
    # dogs" (no literal "or"/"and" word in it) but don't break the query
    # either, so the doc is still found via "cats"/"dogs". A bare "NOT" is
    # quoted to a literal term with nothing to match, giving zero hits
    # rather than raising or silently matching everything.
    assert len(store.fts_search("cats OR AND OR dogs", top_k=5)) == 1
    assert store.fts_search("NOT", top_k=5) == []


def test_fts_ranks_true_cjk_match_above_incidental_bigram_matches(store: ChunkStore) -> None:
    """A CJK query is OR-joined bigrams (東京の会議 -> 東京/京の/の会/会議), so any
    document sharing just ONE bigram with the query also matches -- match
    alone doesn't prove the ranking is useful. This asserts ranked ORDER: the
    document that actually contains the query phrase (matching all four
    bigrams) must outrank documents that only incidentally share a single
    bigram with the query."""
    store.upsert_batch([
        (make_chunk("true", 0, ChunkKind.BODY, "東京の会議は来週開催されます"), fake_embedding(1)),
        (make_chunk("incidental1", 0, ChunkKind.BODY, "彼は東京に出張に行きました"), fake_embedding(2)),
        (make_chunk("incidental2", 0, ChunkKind.BODY, "午後の会議室は予約済みです"), fake_embedding(3)),
        (make_chunk("incidental3", 0, ChunkKind.BODY, "会議の議題については後で連絡します"), fake_embedding(4)),
    ])
    results = store.fts_search("東京の会議", top_k=10)
    assert len(results) == 4, "all four docs share at least one bigram and should match"
    assert results[0].source_key == "true", (
        f"expected the true phrase match to rank first, got order: "
        f"{[r.source_key for r in results]}"
    )


def test_fts_english_stemming_still_works(store: ChunkStore) -> None:
    store.upsert_batch([
        (make_chunk("en", 1, ChunkKind.BODY, "the quarterly planning meeting"), fake_embedding(4)),
    ])
    assert len(store.fts_search("meetings", top_k=5)) == 1


def test_fts_migration_rebuilds_unnormalized_index(tmp_path: Path) -> None:
    """A DB written before normalization must be repaired on next open."""
    path = tmp_path / "old.db"
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (make_chunk("jp", 0, ChunkKind.BODY, "東京で会議をしました"), fake_embedding(1)),
    ])
    # Simulate a pre-migration DB: raw content in FTS, no version stamp.
    conn = store._conn
    conn.execute("DELETE FROM chunks_fts")
    rows = conn.execute("SELECT rowid, content FROM chunks").fetchall()
    for row in rows:
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
            (row["rowid"], row["content"]),
        )
    conn.execute("DELETE FROM schema_meta WHERE key = 'fts_version'")
    conn.commit()
    assert store.fts_search("東京", top_k=5) == []  # broken, as an old DB would be
    store.close()

    reopened = ChunkStore(path, embedding_dim=DIM)
    assert len(reopened.fts_search("東京", top_k=5)) == 1  # migration repaired it
    reopened.close()


def _make_unmigrated_store(path: Path) -> None:
    """Build a store, then roll it back to a pre-migration state: raw
    (unnormalized) FTS content and no fts_version stamp. Shared by the
    read-only and migration-logging tests below."""
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (make_chunk("jp", 0, ChunkKind.BODY, "東京で会議をしました"), fake_embedding(1)),
    ])
    conn = store._conn
    conn.execute("DELETE FROM chunks_fts")
    rows = conn.execute("SELECT rowid, content FROM chunks").fetchall()
    for row in rows:
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
            (row["rowid"], row["content"]),
        )
    conn.execute("DELETE FROM schema_meta WHERE key = 'fts_version'")
    conn.commit()
    store.close()


def test_read_only_requires_an_existing_file(tmp_path: Path) -> None:
    """Read-only mode never creates a store -- opening one that doesn't
    exist yet is a clear, immediate error, not a silently created empty DB."""
    with pytest.raises(FileNotFoundError):
        ChunkStore(tmp_path / "does_not_exist.db", embedding_dim=DIM, read_only=True)


def test_read_only_never_migrates(tmp_path: Path) -> None:
    """Opening a store read-only must NEVER run _migrate_fts -- the whole
    point is that inspecting a store (e.g. a backup) cannot silently rewrite
    it. Regression for: opening a ~70k-chunk backup read-write to compare
    its pre-migration behavior performed the migration, destroying the very
    state being compared."""
    path = tmp_path / "old.db"
    _make_unmigrated_store(path)

    ro = ChunkStore(path, embedding_dim=DIM, read_only=True)
    try:
        # Still broken, exactly as the unmigrated on-disk state was -- proves
        # no migration ran on open.
        assert ro.fts_search("東京", top_k=5) == []
        row = ro._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'fts_version'"
        ).fetchone()
        assert row is None, "read-only open must not have written an fts_version stamp"
    finally:
        ro.close()

    # A normal (writable) open of the SAME file afterward still migrates
    # correctly -- read-only mode didn't corrupt anything, just deferred it.
    rw = ChunkStore(path, embedding_dim=DIM)
    try:
        assert len(rw.fts_search("東京", top_k=5)) == 1
    finally:
        rw.close()


def test_read_only_allows_reads(tmp_path: Path) -> None:
    path = tmp_path / "readable.db"
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (make_chunk("DOC-0", 0, ChunkKind.SECTION, "readable content"), fake_embedding(0)),
    ])
    store.close()

    ro = ChunkStore(path, embedding_dim=DIM, read_only=True)
    try:
        assert ro.stats()["total"] == 1
        assert len(ro.fts_search("readable", top_k=5)) == 1
        assert len(ro.vector_search(fake_embedding(0), top_k=5)) == 1
        assert ro.get_by_source_key("notes", "DOC-0") != []
    finally:
        ro.close()


def test_read_only_raises_on_every_mutating_method(tmp_path: Path) -> None:
    """Anything that would mutate must raise a clear, read-only-specific
    error -- not SQLite's generic 'attempt to write a readonly database'
    from wherever the write happens to land."""
    path = tmp_path / "guarded.db"
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (make_chunk("DOC-0", 0, ChunkKind.SECTION, "content"), fake_embedding(0)),
    ])
    store.close()

    ro = ChunkStore(path, embedding_dim=DIM, read_only=True)
    try:
        chunk = make_chunk("DOC-1", 0, ChunkKind.SECTION, "new content")
        with pytest.raises(ReadOnlyStoreError, match="read_only=True"):
            ro.upsert(chunk, fake_embedding(1))
        with pytest.raises(ReadOnlyStoreError, match="read_only=True"):
            ro.upsert_batch([(chunk, fake_embedding(1))])
        with pytest.raises(ReadOnlyStoreError, match="read_only=True"):
            ro.delete_by_source("notes")
        with pytest.raises(ReadOnlyStoreError, match="read_only=True"):
            ro.delete_orphans("notes", set())
        with pytest.raises(ReadOnlyStoreError, match="read_only=True"):
            ro.upsert_summary("notes", "DOC-0", "summary", "hash", "model-x")
        # None of the attempts above actually changed anything.
        assert ro.stats()["total"] == 1
    finally:
        ro.close()


def test_migration_logs_warning_with_path_and_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A migration that runs against a store with existing chunks must WARN
    (not just INFO) with the path, both before and after, including the
    rebuilt count -- opening a file should never silently rewrite the user's
    data. Regression for: the old code logged this at INFO and only after
    the fact, with no distinct 'about to happen' signal."""
    path = tmp_path / "warns.db"
    _make_unmigrated_store(path)

    with caplog.at_level(logging.WARNING, logger="corpus.db.sqlite"):
        store = ChunkStore(path, embedding_dim=DIM)
    store.close()

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) >= 2, "expected a before-migration AND after-migration WARNING"
    joined = "\n".join(r.getMessage() for r in warnings)
    assert str(path) in joined
    assert "1" in joined  # one chunk rebuilt


def test_fresh_store_creation_does_not_log_a_migration_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A brand-new, empty store has no prior data at risk -- creating one for
    the first time must not print a scary 'migrating' WARNING on every
    ordinary first ingest."""
    with caplog.at_level(logging.WARNING, logger="corpus.db.sqlite"):
        store = ChunkStore(tmp_path / "brand_new.db", embedding_dim=DIM)
    store.close()

    assert not any(r.levelname == "WARNING" for r in caplog.records)


# ---------------------------------------------------------------------------
# In-tree-index guard: corpus refuses (with a loud warning, not a hard error)
# to treat "inside corpus's own source tree" as a correct place for a
# consumer's index. See `_corpus_engine_roots` / `_warn_if_inside_corpus_repo`
# in corpus.db.sqlite for the full rationale.
# ---------------------------------------------------------------------------


def test_opening_a_store_inside_a_simulated_engine_root_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`tmp_path` stands in for "corpus's own repo root" here so the test
    never has to construct a store inside the real checkout -- the detection
    logic itself (resolve + is_relative_to) doesn't care what the root is,
    only `_corpus_engine_roots` decides what counts as one.

    `_TEMP_ROOT` also has to be monkeypatched to something that does NOT
    cover `tmp_path`: real `tmp_path` fixtures live under the system temp
    directory by construction, and that exemption is checked (correctly)
    before engine roots are even consulted -- otherwise this test would
    "pass" for the wrong reason, without ever exercising the engine-root
    branch it means to test."""
    import corpus.db.sqlite as sqlite_module

    monkeypatch.setattr(sqlite_module, "_corpus_engine_roots", lambda: [tmp_path])
    monkeypatch.setattr(sqlite_module, "_TEMP_ROOT", tmp_path / "_unrelated_decoy_temp_root")
    db_path = tmp_path / "corpus.db"

    with caplog.at_level(logging.WARNING, logger="corpus.db.sqlite"):
        store = ChunkStore(db_path, embedding_dim=DIM)
    store.close()

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(db_path) in message
    assert "mistake" in message


def test_warning_does_not_prevent_opening_an_existing_in_tree_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check is advisory, not enforced -- an upgrade must never lock a
    user out of data that already lives at a warned-about path."""
    import corpus.db.sqlite as sqlite_module

    monkeypatch.setattr(sqlite_module, "_corpus_engine_roots", lambda: [tmp_path])
    monkeypatch.setattr(sqlite_module, "_TEMP_ROOT", tmp_path / "_unrelated_decoy_temp_root")
    db_path = tmp_path / "corpus.db"

    store = ChunkStore(db_path, embedding_dim=DIM)
    try:
        store.upsert_batch([(make_chunk("DOC-0", 0, ChunkKind.SECTION, "x"), fake_embedding(0))])
        assert store.stats()["total"] == 1
    finally:
        store.close()

    # Reopening the same in-tree path still works too.
    reopened = ChunkStore(db_path, embedding_dim=DIM)
    try:
        assert reopened.stats()["total"] == 1
    finally:
        reopened.close()


def test_ordinary_tmp_path_store_does_not_trigger_the_in_tree_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No monkeypatching here -- this exercises the real
    `_corpus_engine_roots()` against a genuine `tmp_path`, which must never
    match (it isn't inside corpus's package dir or checkout root, and is
    additionally under the system temp root, which is skipped outright)."""
    with caplog.at_level(logging.WARNING, logger="corpus.db.sqlite"):
        store = ChunkStore(tmp_path / "ordinary.db", embedding_dim=DIM)
    store.close()

    assert not any("mistake" in r.getMessage() for r in caplog.records)


def test_memory_path_is_exempt_from_the_in_tree_warning(caplog: pytest.LogCaptureFixture) -> None:
    from corpus.db.sqlite import _warn_if_inside_corpus_repo

    with caplog.at_level(logging.WARNING, logger="corpus.db.sqlite"):
        _warn_if_inside_corpus_repo(Path(":memory:"))

    assert not caplog.records


def test_corpus_engine_roots_does_not_escalate_a_non_editable_install(tmp_path: Path) -> None:
    """A normal (non-editable) install's package directory sits directly
    under `site-packages` with no `src` parent -- there is no repo root to
    protect, so only the package directory itself should come back."""
    from unittest.mock import patch

    from corpus.db.sqlite import _corpus_engine_roots

    db_subdir = tmp_path / "site-packages" / "corpus" / "db"
    db_subdir.mkdir(parents=True)
    fake_module_file = db_subdir / "sqlite.py"
    fake_module_file.write_text("# stub")

    with patch("corpus.db.sqlite.__file__", str(fake_module_file)):
        roots = _corpus_engine_roots()

    assert roots == [db_subdir.parent]  # ".../site-packages/corpus" -- the package dir


def test_corpus_engine_roots_ignores_a_src_layout_from_an_unrelated_project(
    tmp_path: Path,
) -> None:
    """A `src/<pkg>` shape alone isn't enough to escalate to a repo root --
    the pyproject.toml one level up must actually declare *this* project.
    Otherwise a vendored copy or an unrelated project using the same layout
    convention would wrongly gain repo-root-wide warnings."""
    from unittest.mock import patch

    from corpus.db.sqlite import _corpus_engine_roots

    pkg_dir = tmp_path / "src" / "corpus" / "db"
    pkg_dir.mkdir(parents=True)
    fake_module_file = pkg_dir / "sqlite.py"
    fake_module_file.write_text("# stub")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "unrelated-project"\n')

    with patch("corpus.db.sqlite.__file__", str(fake_module_file)):
        roots = _corpus_engine_roots()

    assert roots == [pkg_dir.parent]


# ---------------------------------------------------------------------------
# Memory-tuning pragmas (cache_size / mmap_size / temp_store). See
# corpus.config.PerformanceConfig for the benchmark behind the defaults --
# mmap_size is the pragma that actually matters for vector search; these
# tests only check that ChunkStore applies whatever it's configured with,
# not the resulting speedup (that needs a store far too large for the unit
# test suite; see the scratch benchmark referenced in the config docstring).
# ---------------------------------------------------------------------------

def test_performance_pragmas_apply_defaults(store: ChunkStore) -> None:
    conn = store._conn
    assert conn.execute("PRAGMA cache_size").fetchone()[0] == -(DEFAULT_CACHE_SIZE_MB * 1024)
    assert conn.execute("PRAGMA mmap_size").fetchone()[0] == DEFAULT_MMAP_SIZE_MB * 1024 * 1024
    assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2  # 2 == MEMORY


def test_performance_pragmas_are_configurable(tmp_path: Path) -> None:
    store = ChunkStore(
        tmp_path / "perf.db",
        embedding_dim=DIM,
        cache_size_mb=8,
        mmap_size_mb=32,
        temp_store_memory=False,
    )
    try:
        conn = store._conn
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -8192
        assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 32 * 1024 * 1024
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 0  # 0 == DEFAULT, not MEMORY
    finally:
        store.close()


def test_performance_pragmas_apply_on_read_only_connections_too(tmp_path: Path) -> None:
    """mmap_size composes with read_only=True -- benchmarked exclusively
    through read-only connections, matching how corpus-mcp / corpus-query
    actually use it, per the brief item that asked this to be checked
    rather than assumed."""
    path = tmp_path / "perf_ro.db"
    w = ChunkStore(path, embedding_dim=DIM, mmap_size_mb=16)
    w.upsert_batch([(make_chunk("DOC-0", 0, ChunkKind.SECTION, "x"), fake_embedding(0))])
    w.close()

    ro = ChunkStore(path, embedding_dim=DIM, read_only=True, mmap_size_mb=16)
    try:
        assert ro._conn.execute("PRAGMA mmap_size").fetchone()[0] == 16 * 1024 * 1024
        # Still fully functional, not just configured.
        assert len(ro.fts_search("x", top_k=5)) == 1
    finally:
        ro.close()


# --- expensive FTS migrations are opt-in ------------------------------------
#
# A small store rebuilds on open; that is cheap and a migration you must
# remember is one that gets skipped. A large store does not, because the
# rebuild is ONE transaction holding SQLite's only writer lock from first
# delete to final commit — minutes on a million-chunk store, during which any
# concurrent writer fails on its busy timeout. A library constructor is not a
# maintenance command.


def _stale(path: Path) -> None:
    """Stamp a store's FTS index as written by an older normalization."""
    con = sqlite3.connect(path)
    con.execute("UPDATE schema_meta SET value = '1' WHERE key = 'fts_version'")
    con.commit()
    con.close()


def test_a_small_store_still_migrates_on_open(tmp_path: Path) -> None:
    db = tmp_path / "small.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.upsert_batch([(make_chunk("jp", 0, ChunkKind.BODY, "東京で会議をしました"), fake_embedding(1))])
    store.close()
    _stale(db)

    reopened = ChunkStore(db, embedding_dim=DIM)

    assert reopened.fts_version() == FTS_VERSION
    assert len(reopened.fts_search("東京", top_k=5)) == 1
    reopened.close()


def test_a_large_store_is_left_alone_and_reported(tmp_path: Path, caplog) -> None:
    db = tmp_path / "big.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.upsert_batch([(make_chunk("jp", 0, ChunkKind.BODY, "東京で会議"), fake_embedding(1))])
    store.close()
    _stale(db)

    with patch("corpus.db.sqlite.AUTO_FTS_MIGRATION_MAX_CHUNKS", 0), caplog.at_level("WARNING"):
        reopened = ChunkStore(db, embedding_dim=DIM)

    # Left stale rather than silently rebuilt, and said so loudly.
    assert reopened.fts_version() == "1"
    assert "STALE" in caplog.text
    assert "corpus-migrate-fts" in caplog.text
    # The mixed-version writer hazard is the one SQLite cannot catch, so the
    # warning has to name it.
    assert "older writer" in caplog.text
    reopened.close()


def test_the_explicit_flag_performs_the_large_migration(tmp_path: Path) -> None:
    db = tmp_path / "big2.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.upsert_batch([(make_chunk("jp", 0, ChunkKind.BODY, "東京で会議"), fake_embedding(1))])
    store.close()
    _stale(db)

    with patch("corpus.db.sqlite.AUTO_FTS_MIGRATION_MAX_CHUNKS", 0):
        migrated = ChunkStore(db, embedding_dim=DIM, allow_expensive_migration=True)

    assert migrated.fts_version() == FTS_VERSION
    assert len(migrated.fts_search("東京", top_k=5)) == 1
    migrated.close()


def test_reingesting_changed_content_requeues_the_chunk_for_context(
    tmp_path: Path,
) -> None:
    """Re-ingest drops a stale context so the chunk is contextualized again.

    `upsert` overwrites the vector and the FTS row with content-only
    versions, which is correct -- the old ones described text that no longer
    exists. But it used to leave `context` set, and that combination stranded
    the chunk permanently: the contextual-retrieval benefit was gone from both
    retrieval arms, the stored blurb described the previous content, and a
    non-NULL context kept it off `chunks_missing_context` so no later run
    could repair it. Paid for once, silently reverted by the next ingest.
    """
    store = ChunkStore(tmp_path / "reingest.db", embedding_dim=DIM)
    chunk = make_chunk("doc", 0, ChunkKind.BODY, "approved, option B")
    store.upsert_batch([(chunk, fake_embedding(1))])
    store.set_context(chunk.id, "From the Zanzibar rollout thread.", fake_embedding(2))
    assert store.chunks_missing_context("notes") == []

    changed = make_chunk("doc", 0, ChunkKind.BODY, "approved, option C instead")
    store.upsert_batch([(changed, fake_embedding(3))])

    # Back on the queue, with no stale blurb left behind.
    assert len(store.chunks_missing_context("notes")) == 1
    row = store._conn.execute(
        "SELECT context FROM chunks WHERE id = ?", (chunk.id,)
    ).fetchone()
    assert row["context"] is None
    store.close()


def test_reingesting_unchanged_content_keeps_its_context(tmp_path: Path) -> None:
    """The other half: an unchanged chunk must NOT lose its context, or every
    routine re-ingest would bill a full re-contextualization of the archive."""
    store = ChunkStore(tmp_path / "unchanged.db", embedding_dim=DIM)
    chunk = make_chunk("doc", 0, ChunkKind.BODY, "approved, option B")
    store.upsert_batch([(chunk, fake_embedding(1))])
    store.set_context(chunk.id, "From the Zanzibar rollout thread.", fake_embedding(2))

    store.upsert_batch([(make_chunk("doc", 0, ChunkKind.BODY, "approved, option B"), fake_embedding(1))])

    assert store.chunks_missing_context("notes") == []
    assert len(store.fts_search("Zanzibar", top_k=5)) == 1
    store.close()


def test_the_rebuild_preserves_generated_context(tmp_path: Path) -> None:
    """A contextualized chunk's FTS row covers context + content, because
    `set_context` writes it that way. A rebuild that re-reads only `content`
    strips every generated context back out of BM25 -- silently, and for
    precisely the chunks a contextualization run was paid for.

    Measured on a real archive before the fix: every contextualized chunk lost their
    context from the index, and nothing reported a problem -- the rebuild
    counted every row and stamped itself successful.
    """
    db = tmp_path / "ctx.db"
    store = ChunkStore(db, embedding_dim=DIM)
    chunk = make_chunk("doc", 0, ChunkKind.BODY, "approved, option B")
    store.upsert_batch([(chunk, fake_embedding(1))])
    store.set_context(chunk.id, "From the Zanzibar rollout thread.", fake_embedding(2))
    # The context is reachable before the rebuild ...
    assert len(store.fts_search("Zanzibar", top_k=5)) == 1
    store.close()

    _stale(db)
    migrated = ChunkStore(db, embedding_dim=DIM)

    # ... and must still be after it.
    assert migrated.fts_version() == FTS_VERSION
    assert len(migrated.fts_search("Zanzibar", top_k=5)) == 1
    assert len(migrated.fts_search("option", top_k=5)) == 1
    migrated.close()


def test_an_interrupted_migration_rolls_back_cleanly(tmp_path: Path) -> None:
    """The rebuild is one transaction, so a crash mid-way leaves the OLD index
    and the OLD stamp — never a durable half-migrated state — and the next
    attempt starts clean. Simulated by raising partway through the rebuild."""
    db = tmp_path / "crash.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.upsert_batch(
        [
            (make_chunk("jp", i, ChunkKind.BODY, f"東京で会議{i}"), fake_embedding(i))
            for i in range(6)
        ]
    )
    store.close()
    _stale(db)

    calls = {"n": 0}
    real = corpus.db.sqlite.normalize_for_fts

    def explode(text: str) -> str:
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated crash mid-rebuild")
        return real(text)

    with patch("corpus.db.sqlite.normalize_for_fts", explode), pytest.raises(RuntimeError):
        ChunkStore(db, embedding_dim=DIM)

    # Old stamp intact, index intact, integrity intact, and a clean retry works.
    con = sqlite3.connect(db)
    assert con.execute(
        "SELECT value FROM schema_meta WHERE key='fts_version'"
    ).fetchone()[0] == "1"
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()

    retried = ChunkStore(db, embedding_dim=DIM)
    assert retried.fts_version() == FTS_VERSION
    assert len(retried.fts_search("東京", top_k=10)) == 6
    retried.close()


def test_using_a_closed_store_says_so(tmp_path: Path) -> None:
    """close() cannot reach into another thread's threading.local() to clear
    its connection reference, so a thread that already had one would other-
    wise fail with sqlite3's "Cannot operate on a closed database" from a
    call site unrelated to closing. Use-after-close is a caller bug and the
    error should name it."""
    store = ChunkStore(tmp_path / "closed.db", embedding_dim=DIM)
    store.upsert_batch([(make_chunk("a", 0, ChunkKind.BODY, "text"), fake_embedding(1))])
    store.close()

    with pytest.raises(ReadOnlyStoreError, match="closed"):
        store.fts_search("text", top_k=1)


def test_close_is_idempotent(tmp_path: Path) -> None:
    store = ChunkStore(tmp_path / "twice.db", embedding_dim=DIM)
    store.close()
    store.close()  # must not raise


def test_a_closed_store_is_not_silently_reopened(tmp_path: Path) -> None:
    # Quietly resurrecting one would hide the caller's mistake.
    db = tmp_path / "noreopen.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.close()

    with pytest.raises(ReadOnlyStoreError):
        store.stats()

    # A fresh store on the same file is the correct way back.
    reopened = ChunkStore(db, embedding_dim=DIM)
    assert reopened.stats()["total"] == 0
    reopened.close()


def test_a_large_top_k_does_not_crash_the_vector_search(store: ChunkStore) -> None:
    """sqlite-vec's vec0 refuses a KNN query with k above 4096, raising
    `OperationalError: k value in knn query too large`. Nothing bounded it, so
    a caller asking for a large enough top_k crashed the whole retrieval
    instead of getting the most it could give — reachable from any code path
    that widens a candidate pool."""
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"text {i}"), fake_embedding(i))
        for i in range(20)
    ]
    store.upsert_batch(items)

    results = store.vector_search(fake_embedding(1), top_k=99_999)

    assert len(results) == 20


def test_a_large_top_k_with_a_source_filter_also_survives(store: ChunkStore) -> None:
    items = [
        (make_chunk(f"DOC-{i}", 0, ChunkKind.SECTION, f"text {i}"), fake_embedding(i))
        for i in range(20)
    ]
    store.upsert_batch(items)

    results = store.vector_search(fake_embedding(1), top_k=99_999, filter_sources=["notes"])

    assert len(results) == 20


def test_every_fts_write_goes_through_the_single_text_builder() -> None:
    """No writer may compute its own FTS text.

    Three defects in one day came from four write paths independently
    deciding what text belongs in `chunks_fts` and disagreeing. The
    invariant is now `_fts_text`, and this asserts it structurally rather
    than by hoping the next person reads the docstring -- the failures in
    this class are all silent, so a review is unlikely to catch a fourth.
    """
    source = Path(corpus.db.sqlite.__file__).read_text().splitlines()

    inserts = [i for i, line in enumerate(source) if "INSERT INTO chunks_fts(" in line]
    assert inserts, "expected at least one chunks_fts writer"
    for i in inserts:
        window = "\n".join(source[i : i + 4])
        assert "_fts_text(" in window, (
            f"chunks_fts write at line {i + 1} does not use _fts_text:\n{window}"
        )

    # And the normalizer itself is reachable from exactly one place, so the
    # "raw write is unreachable by CJK queries" failure cannot reappear either.
    calls = [
        i for i, line in enumerate(source)
        if "normalize_for_fts(" in line and not line.lstrip().startswith("#")
        and "import" not in line
    ]
    assert len(calls) == 1, (
        "normalize_for_fts should be called only inside _fts_text; found at lines "
        f"{[i + 1 for i in calls]}"
    )
