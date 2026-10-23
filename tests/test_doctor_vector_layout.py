"""corpus-doctor reports a vector index that filters by full scan.

Both conditions degrade search without an error anywhere: an unpartitioned
index makes every source-filtered search read every vector, and a vector an
older writer filed with no type is missing from every filtered search. Each
is a FAIL with the fix named, because a warning nobody acts on changes
nothing.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import sqlite_vec

from corpus.cli.doctor import _check_vector_layout
from corpus.db.sqlite import ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _store(path: Path) -> Path:
    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id("notes", f"k{i}", ChunkKind.BODY, i),
                content=f"note {i}",
                content_hash=sha256(f"n{i}"),
                metadata=ChunkMetadata(
                    source_type="notes", source_key=f"k{i}", chunk_kind=ChunkKind.BODY,
                    chunk_index=i, title=f"k{i}",
                ),
            ),
            [math.sin((i + 1) * (j + 1) * 0.3) for j in range(DIM)],
        )
        for i in range(3)
    ])
    store.close()
    return path


def _raw(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    return con


def test_a_partitioned_index_passes(tmp_path, capsys) -> None:
    assert _check_vector_layout(str(_store(tmp_path / "ok.db"))) is True
    assert "[  ok  ]" in capsys.readouterr().out


def test_an_unpartitioned_index_fails_and_names_the_command(tmp_path, capsys) -> None:
    db = _store(tmp_path / "legacy.db")
    con = _raw(db)
    rows = con.execute("SELECT rowid, embedding FROM chunks_vec").fetchall()
    con.execute("DROP TABLE chunks_vec")
    con.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{DIM}])")
    con.executemany("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", rows)
    con.commit()
    con.close()

    assert _check_vector_layout(str(db)) is False
    assert "corpus-migrate-vec" in capsys.readouterr().out


def test_untyped_vectors_fail_and_say_where_they_came_from(tmp_path, capsys) -> None:
    db = _store(tmp_path / "untyped.db")
    con = _raw(db)
    rid, blob = con.execute("SELECT rowid, embedding FROM chunks_vec LIMIT 1").fetchone()
    con.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rid,))
    con.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (rid, blob))
    con.commit()
    con.close()

    assert _check_vector_layout(str(db)) is False
    out = capsys.readouterr().out
    assert "1 " in out and "older" in out


def test_no_database_is_skipped(capsys) -> None:
    assert _check_vector_layout(None) is True
    assert "SKIPPED" in capsys.readouterr().out
