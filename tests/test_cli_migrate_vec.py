"""corpus-migrate-vec partitions a large store's vector index deliberately.

Same contract as corpus-migrate-fts: a dry run must not cause the migration
it reports on, and a migration that did not land must be a failure, because
the operator's next act is to let writers back on.
"""

from __future__ import annotations

import math
from pathlib import Path

import sqlite_vec

from corpus.config import CorpusConfig, EmbedderConfig
from corpus.db.sqlite import ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _config(db: Path) -> CorpusConfig:
    return CorpusConfig(db_path=db, embedder=EmbedderConfig(provider="hash", dim=DIM))


def _legacy_store(path: Path) -> None:
    import sqlite3

    store = ChunkStore(path, embedding_dim=DIM)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id(st, f"k{i}", ChunkKind.BODY, i),
                content=f"{st} {i}",
                content_hash=sha256(f"{st}{i}"),
                metadata=ChunkMetadata(
                    source_type=st, source_key=f"k{i}", chunk_kind=ChunkKind.BODY,
                    chunk_index=i, title=f"k{i}",
                ),
            ),
            [math.sin((i + 1) * (j + 1) * 0.3) for j in range(DIM)],
        )
        for i, st in enumerate(["notes", "notes", "mail"])
    ])
    store.close()
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    rows = con.execute("SELECT rowid, embedding FROM chunks_vec").fetchall()
    con.execute("DROP TABLE chunks_vec")
    con.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{DIM}])")
    con.executemany("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", rows)
    con.commit()
    con.close()


def _partitioned(path: Path) -> bool:
    store = ChunkStore(path, embedding_dim=DIM, read_only=True)
    try:
        return store.vec_partitioned()
    finally:
        store.close()


def _run(monkeypatch, argv, *, cfg):
    import corpus.cli.migrate_vec as mod

    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg)
    monkeypatch.setattr("sys.argv", argv)
    return mod.main()


def test_a_missing_database_is_an_error_not_a_traceback(tmp_path, monkeypatch, capsys):
    code = _run(monkeypatch, ["corpus-migrate-vec"], cfg=_config(tmp_path / "absent.db"))

    assert code == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_dry_run_does_not_perform_the_migration_it_reports_on(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    _legacy_store(db)

    assert _run(monkeypatch, ["corpus-migrate-vec", "--dry-run"], cfg=_config(db)) == 0
    assert not _partitioned(db), "a dry run migrated the store"


def test_a_partitioned_store_is_left_alone(tmp_path, monkeypatch, capsys):
    db = tmp_path / "current.db"
    _legacy_store(db)
    ChunkStore(db, embedding_dim=DIM).close()  # small store migrates on open
    assert _partitioned(db)

    assert _run(monkeypatch, ["corpus-migrate-vec"], cfg=_config(db)) == 0
    assert "Nothing to do" in capsys.readouterr().out


def test_it_migrates_a_store_too_large_to_migrate_on_open(tmp_path, monkeypatch):
    db = tmp_path / "big.db"
    _legacy_store(db)
    monkeypatch.setattr("corpus.db.sqlite.AUTO_VEC_MIGRATION_MAX_CHUNKS", 0)

    assert _run(monkeypatch, ["corpus-migrate-vec"], cfg=_config(db)) == 0
    assert _partitioned(db)


def test_a_migration_that_did_not_land_is_reported_as_a_failure(tmp_path, monkeypatch, capsys):
    db = tmp_path / "stuck.db"
    _legacy_store(db)
    monkeypatch.setattr(ChunkStore, "_migrate_vec", lambda self, conn: None)

    code = _run(monkeypatch, ["corpus-migrate-vec"], cfg=_config(db))

    assert code == 1
    assert "ERROR" in capsys.readouterr().out
