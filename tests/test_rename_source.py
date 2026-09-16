"""Rename a source without re-ingesting or re-embedding it.

`source_type` is not just a label -- it is the first field of the chunk id:

    id = f"{source_type}:{source_key}:{chunk_kind}:{chunk_index}"

So renaming the column alone leaves ids that no longer match what the next
ingest computes, and that ingest inserts a second copy of every chunk instead
of recognising the existing rows. The rename has to recompute ids, and update
the copy of `source_type` inside the metadata JSON, or a reader that trusts
the metadata disagrees with the column beside it.

Vectors and BM25 rows are keyed by `rowid`, which never changes, so they are
left alone -- and that is the whole point: renaming a source must not cost an
embedding bill.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from corpus.db.sqlite import ChunkStore
from corpus.rename import SourceNotFound, TargetExists, rename_source
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _store(tmp_path: Path, sources: dict[str, int]) -> Path:
    db = tmp_path / "c.db"
    store = ChunkStore(db, embedding_dim=DIM)
    rows = []
    for name, n in sources.items():
        for i in range(n):
            rows.append((
                Chunk(
                    id=chunk_id(name, f"doc{i}", ChunkKind.BODY, i),
                    content=f"{name} body {i} with distinct wording.",
                    content_hash=sha256(f"{name}{i}"),
                    metadata=ChunkMetadata(
                        source_type=name, source_key=f"doc{i}",
                        chunk_kind=ChunkKind.BODY, chunk_index=i, title=f"doc{i}",
                    ),
                ),
                [math.sin((i + 1) * (j + 1) * 0.1) for j in range(DIM)],
            ))
    store.upsert_batch(rows)
    store.upsert_summary(
        source_type="old", source_key="doc0", summary="paid for",
        doc_hash="h", model="m", token_count=5,
    )
    store.close()
    return db


def _rows(db: Path, source: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT id, source_type, metadata FROM chunks WHERE source_type = ?"
            " ORDER BY chunk_index",
            (source,),
        ).fetchall()
    finally:
        conn.close()


# --- the rename ---------------------------------------------------------------


def test_the_column_moves(tmp_path):
    db = _store(tmp_path, {"old": 3})
    assert rename_source(db, "old", "new") == 3
    assert len(_rows(db, "new")) == 3
    assert _rows(db, "old") == []


def test_the_ids_are_recomputed_so_the_next_ingest_recognises_them(tmp_path):
    """The defect a column-only rename leaves behind: every chunk is
    re-inserted as a new row on the next ingest, silently doubling the
    source."""
    db = _store(tmp_path, {"old": 3})
    rename_source(db, "old", "new")
    for row in _rows(db, "new"):
        expected = chunk_id("new", f"doc{row['id'].split(':')[3]}", ChunkKind.BODY,
                            int(row["id"].split(":")[3]))
        assert row["id"] == expected, f"stale id {row['id']}"
        assert row["id"].startswith("new:")


def test_the_metadata_copy_moves_too(tmp_path):
    """A reader that trusts the metadata must not disagree with the column
    beside it."""
    db = _store(tmp_path, {"old": 2})
    rename_source(db, "old", "new")
    for row in _rows(db, "new"):
        assert '"source_type":"new"' in row["metadata"].replace(" ", "")


def test_vectors_and_bm25_rows_are_untouched(tmp_path):
    """The point of the whole operation: renaming must not cost an embedding
    bill."""
    import sqlite_vec

    def counts(path):
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.enable_load_extension(False)
        try:
            return (
                c.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0],
                c.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0],
            )
        finally:
            c.close()

    db = _store(tmp_path, {"old": 4})
    before = counts(db)
    rename_source(db, "old", "new")
    assert before == counts(db) == (4, 4)


def test_summaries_follow_the_rename(tmp_path):
    """A summary is keyed by source_type too. Leaving it behind orphans paid
    work that `get_summary` can then never find."""
    db = _store(tmp_path, {"old": 2})
    rename_source(db, "old", "new")
    store = ChunkStore(db, embedding_dim=DIM, read_only=True)
    try:
        assert store.get_summary("new", "doc0") is not None
        assert store.get_summary("old", "doc0") is None
    finally:
        store.close()


def test_other_sources_are_not_touched(tmp_path):
    db = _store(tmp_path, {"old": 2, "keep": 3})
    rename_source(db, "old", "new")
    assert len(_rows(db, "keep")) == 3


# --- refusals ------------------------------------------------------------------


def test_renaming_a_source_that_is_not_there_is_an_error(tmp_path):
    db = _store(tmp_path, {"old": 1})
    with pytest.raises(SourceNotFound):
        rename_source(db, "absent", "new")


def test_renaming_onto_an_existing_source_is_refused(tmp_path):
    """Merging two sources is a different operation with different
    consequences -- chunk ids would collide and one source would silently
    overwrite the other."""
    db = _store(tmp_path, {"old": 2, "taken": 2})
    with pytest.raises(TargetExists):
        rename_source(db, "old", "taken")
    assert len(_rows(db, "old")) == 2, "a refused rename changed something"


def test_an_invalid_name_is_refused(tmp_path):
    """`source_type` has a pattern in SourceConfig. A rename that bypasses it
    produces a source no corpus.toml can ever name again."""
    db = _store(tmp_path, {"old": 1})
    with pytest.raises(ValueError):
        rename_source(db, "old", "Not A Valid Name")
