"""`chunks_vec` partitioned by `source_type`.

The unpartitioned layout made a source-filtered k-NN cost a full scan: the
filter was `rowid IN (SELECT ... WHERE source_type = ?)`, which constrains
the result but not the work. `build_vector_pool` runs one such scan per
under-served type, and on a wide archive nearly every type is under-served,
so a query paid for dozens of full scans. Measured on a large archive:
~1.06s to build the pool unpartitioned, ~0.11s partitioned, same pools.

Partitioning changes what every writer must supply, so these tests pin the
layout, every write path, the migration of existing stores (automatic only
when small, never on a read-only open), and the repair of vectors an older
writer filed without a type.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlite_vec

from corpus.db.sqlite import ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _emb(seed: int) -> list[float]:
    return [math.sin((seed + 1) * (j + 1) * 0.37) for j in range(DIM)]


def _chunk(stype: str, key: str, idx: int = 0) -> Chunk:
    text = f"{stype} {key} {idx}"
    return Chunk(
        id=chunk_id(stype, key, ChunkKind.BODY, idx),
        content=text,
        content_hash=sha256(text),
        metadata=ChunkMetadata(
            source_type=stype, source_key=key, chunk_kind=ChunkKind.BODY,
            chunk_index=idx, title=key,
        ),
    )


def _fill(path: Path) -> ChunkStore:
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch(
        [(_chunk("notes", f"n{i}"), _emb(i)) for i in range(4)]
        + [(_chunk("mail", f"m{i}"), _emb(10 + i)) for i in range(3)]
    )
    return store


def _raw(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    return con


def _make_legacy(path: Path) -> None:
    """Rewrite a store's vectors into the pre-partition layout."""
    con = _raw(path)
    rows = con.execute("SELECT rowid, embedding FROM chunks_vec").fetchall()
    con.execute("DROP TABLE chunks_vec")
    con.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{DIM}])")
    con.executemany("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", rows)
    con.commit()
    con.close()


def _vector_types(path: Path) -> dict[int, str | None]:
    con = _raw(path)
    try:
        return dict(con.execute("SELECT rowid, source_type FROM chunks_vec").fetchall())
    finally:
        con.close()


def _chunk_types(path: Path) -> dict[int, str]:
    con = sqlite3.connect(path)
    try:
        return dict(con.execute("SELECT rowid, source_type FROM chunks").fetchall())
    finally:
        con.close()


def _keys(store: ChunkStore, seed: int, sources: list[str] | None = None) -> list[str]:
    return [c.source_key for c in store.vector_search(_emb(seed), top_k=10, filter_sources=sources)]


# --- the layout and its writers ---------------------------------------------


def test_a_new_store_is_partitioned_and_every_vector_carries_its_type(tmp_path: Path) -> None:
    db = tmp_path / "new.db"
    store = _fill(db)
    assert store.vec_partitioned()
    store.close()

    assert _vector_types(db) == _chunk_types(db)


def test_a_filtered_search_returns_only_the_named_types(tmp_path: Path) -> None:
    store = _fill(tmp_path / "f.db")

    assert sorted(_keys(store, 0, ["mail"])) == ["m0", "m1", "m2"]
    assert sorted(_keys(store, 0, ["notes", "mail"])) == sorted(_keys(store, 0))
    store.close()


def test_set_context_keeps_the_vector_in_its_partition(tmp_path: Path) -> None:
    db = tmp_path / "ctx.db"
    store = _fill(db)
    target = _chunk("mail", "m1")

    assert store.set_context(target.id, "From the quarterly thread.", _emb(99))

    assert "m1" in _keys(store, 99, ["mail"])
    store.close()
    assert _vector_types(db) == _chunk_types(db)


def test_reingesting_a_chunk_keeps_its_vector_typed(tmp_path: Path) -> None:
    db = tmp_path / "re.db"
    store = _fill(db)
    store.upsert_batch([(_chunk("notes", "n1"), _emb(42))])
    store.close()

    assert _vector_types(db) == _chunk_types(db)


# --- migrating an existing store ---------------------------------------------


def test_a_small_legacy_store_is_migrated_on_open_with_the_same_results(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    store = _fill(db)
    before = {s: _keys(store, s, f) for s in (0, 11) for f in (None, ["mail"])}  # type: ignore[misc]
    store.close()
    _make_legacy(db)

    migrated = ChunkStore(db, embedding_dim=DIM)

    assert migrated.vec_partitioned()
    after = {s: _keys(migrated, s, f) for s in (0, 11) for f in (None, ["mail"])}  # type: ignore[misc]
    assert after == before
    migrated.close()
    assert _vector_types(db) == _chunk_types(db)


def test_a_read_only_open_never_migrates_and_still_filters(tmp_path: Path) -> None:
    db = tmp_path / "ro.db"
    _fill(db).close()
    _make_legacy(db)

    ro = ChunkStore(db, embedding_dim=DIM, read_only=True)

    assert not ro.vec_partitioned()
    assert sorted(_keys(ro, 0, ["mail"])) == ["m0", "m1", "m2"]
    ro.close()


def test_a_legacy_store_still_takes_writes_before_it_is_migrated(tmp_path: Path) -> None:
    """A large store stays on the old layout until someone runs the command,
    and must keep ingesting correctly meanwhile."""
    db = tmp_path / "big-writes.db"
    _fill(db).close()
    _make_legacy(db)

    with patch("corpus.db.sqlite.AUTO_VEC_MIGRATION_MAX_CHUNKS", 0):
        store = ChunkStore(db, embedding_dim=DIM)
        store.upsert_batch([(_chunk("mail", "m9"), _emb(50))])
        assert not store.vec_partitioned()
        assert "m9" in _keys(store, 50, ["mail"])
        store.close()


def test_a_large_legacy_store_is_left_alone_and_reported(tmp_path: Path, caplog) -> None:
    db = tmp_path / "big.db"
    _fill(db).close()
    _make_legacy(db)

    with patch("corpus.db.sqlite.AUTO_VEC_MIGRATION_MAX_CHUNKS", 0), caplog.at_level(logging.WARNING):
        store = ChunkStore(db, embedding_dim=DIM)

    assert not store.vec_partitioned()
    assert "corpus-migrate-vec" in caplog.text
    store.close()


def test_the_explicit_flag_migrates_a_large_store(tmp_path: Path) -> None:
    db = tmp_path / "big2.db"
    _fill(db).close()
    _make_legacy(db)

    with patch("corpus.db.sqlite.AUTO_VEC_MIGRATION_MAX_CHUNKS", 0):
        store = ChunkStore(db, embedding_dim=DIM, allow_expensive_migration=True)

    assert store.vec_partitioned()
    store.close()
    assert _vector_types(db) == _chunk_types(db)


def test_migration_drops_vectors_whose_chunk_is_gone(tmp_path: Path) -> None:
    """A vector with no chunk row is unreachable garbage: it has no type to
    file it under, and every search joins it away anyway."""
    db = tmp_path / "orphan.db"
    _fill(db).close()
    _make_legacy(db)
    con = _raw(db)
    con.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (9999, ?)",
                (sqlite_vec.serialize_float32(_emb(7)),))
    con.commit()
    con.close()

    ChunkStore(db, embedding_dim=DIM).close()

    assert _vector_types(db) == _chunk_types(db)


def test_an_interrupted_migration_rolls_back_to_the_old_layout(tmp_path: Path) -> None:
    db = tmp_path / "crash.db"
    _fill(db).close()
    _make_legacy(db)

    with patch("corpus.db.sqlite.create_vec_table", side_effect=RuntimeError("crash")), \
            pytest.raises(RuntimeError):
        ChunkStore(db, embedding_dim=DIM)

    con = _raw(db)
    assert "partition key" not in con.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunks_vec'"
    ).fetchone()[0]
    assert con.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 7
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()

    retried = ChunkStore(db, embedding_dim=DIM)
    assert retried.vec_partitioned()
    retried.close()


# --- vectors an older writer filed without a type ------------------------------


def test_an_untyped_vector_from_an_older_writer_is_refiled_on_open(tmp_path: Path) -> None:
    """Code from before partitioning inserts `(rowid, embedding)` only, and
    sqlite-vec accepts that with a NULL partition: the vector still appears
    unfiltered but vanishes from every filtered search, with no error. A
    read-write open by current code puts it back."""
    db = tmp_path / "untyped.db"
    _fill(db).close()
    con = _raw(db)
    rowid = con.execute("SELECT rowid FROM chunks WHERE source_key = 'm1'").fetchone()[0]
    blob = con.execute("SELECT embedding FROM chunks_vec WHERE rowid = ?", (rowid,)).fetchone()[0]
    con.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
    con.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (rowid, blob))
    con.commit()
    con.close()
    assert _vector_types(db)[rowid] is None

    store = ChunkStore(db, embedding_dim=DIM)

    assert "m1" in _keys(store, 11, ["mail"])
    store.close()
    assert _vector_types(db) == _chunk_types(db)
