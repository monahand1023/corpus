"""`corpus-rename` changes ids across a live index, and had no tests.

`corpus.rename` is covered; the command wrapped around it was not, and the
command is where the guards live: the confirmation before an irreversible
change, and the reminder that corpus.toml has to be edited to match. Forget
that second step and the next ingest treats every renamed chunk as an orphan
and deletes it.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from corpus.config import CorpusConfig, EmbedderConfig
from corpus.db.sqlite import ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

DIM = 8


def _config(db: Path) -> CorpusConfig:
    return CorpusConfig(db_path=db, embedder=EmbedderConfig(provider="hash", dim=DIM))


def _store(tmp_path: Path, n: int = 3) -> Path:
    db = tmp_path / "c.db"
    store = ChunkStore(db, embedding_dim=DIM)
    store.upsert_batch([
        (
            Chunk(
                id=chunk_id("old", f"d{i}", ChunkKind.BODY, i),
                content=f"body {i} with its own wording",
                content_hash=sha256(f"c{i}"),
                metadata=ChunkMetadata(
                    source_type="old", source_key=f"d{i}",
                    chunk_kind=ChunkKind.BODY, chunk_index=i, title=f"d{i}",
                ),
            ),
            [math.sin((i + 1) * (j + 1) * 0.1) for j in range(DIM)],
        )
        for i in range(n)
    ])
    store.close()
    return db


def _run(monkeypatch, argv, *, cfg, answer=None):
    import corpus.cli.rename as mod

    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg)
    if answer is not None:
        monkeypatch.setattr("builtins.input", lambda _p="": answer)
    monkeypatch.setattr("sys.argv", argv)
    return mod.main()


def _names(db: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT DISTINCT source_type FROM chunks")}
    finally:
        conn.close()


def test_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    db = _store(tmp_path)
    before = db.read_bytes()
    assert _run(monkeypatch, ["corpus-rename", "old", "new", "--dry-run"], cfg=_config(db)) == 0
    assert "3" in capsys.readouterr().out
    assert db.read_bytes() == before, "a dry run wrote to the database"


def test_declining_the_prompt_renames_nothing(tmp_path, monkeypatch):
    db = _store(tmp_path)
    assert _run(
        monkeypatch, ["corpus-rename", "old", "new"], cfg=_config(db), answer="n"
    ) == 1
    assert _names(db) == {"old"}


def test_confirming_renames(tmp_path, monkeypatch, capsys):
    db = _store(tmp_path)
    assert _run(
        monkeypatch, ["corpus-rename", "old", "new"], cfg=_config(db), answer="y"
    ) == 0
    assert _names(db) == {"new"}


def test_it_says_to_update_corpus_toml(tmp_path, monkeypatch, capsys):
    """Forget that and the next ingest sees every renamed chunk as an orphan
    and deletes it. The reminder is the only thing standing between a rename
    and a silently emptied source."""
    db = _store(tmp_path)
    _run(monkeypatch, ["corpus-rename", "old", "new", "--yes"], cfg=_config(db))
    out = capsys.readouterr().out
    assert "corpus.toml" in out
    assert "orphan" in out.lower()


def test_a_missing_database_is_an_error(tmp_path, monkeypatch):
    assert _run(
        monkeypatch, ["corpus-rename", "old", "new", "--dry-run"],
        cfg=_config(tmp_path / "absent.db"),
    ) == 1


def test_renaming_a_source_that_is_not_there_exits_nonzero(tmp_path, monkeypatch, capsys):
    db = _store(tmp_path)
    code = _run(
        monkeypatch, ["corpus-rename", "absent", "new", "--yes"], cfg=_config(db)
    )
    assert code == 1
    assert _names(db) == {"old"}


def test_an_invalid_target_name_is_refused_before_anything_changes(tmp_path, monkeypatch):
    db = _store(tmp_path)
    code = _run(
        monkeypatch, ["corpus-rename", "old", "Not Valid", "--yes"], cfg=_config(db)
    )
    assert code == 2
    assert _names(db) == {"old"}
