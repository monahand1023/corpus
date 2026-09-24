"""corpus-reset deletes data, and had no tests.

Two destructive paths: `--source` drops one source's chunks, `--all` unlinks
the database file and its WAL/SHM sidecars. The confirmation prompt is the
only thing between a mistyped command and an archive that costs hours of
GPU and real API spend to rebuild, so the prompt's behaviour is the point.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corpus.config import CorpusConfig, EmbedderConfig


def _config(tmp_path: Path) -> CorpusConfig:
    return CorpusConfig(
        db_path=tmp_path / "c.db",
        embedder=EmbedderConfig(provider="hash", dim=64),
    )


def _run(monkeypatch, argv, *, answer=None, store=None, cfg=None):
    import corpus.cli.reset as mod

    monkeypatch.setattr(mod, "load_config_or_exit", lambda _p: cfg)
    if store is not None:
        monkeypatch.setattr(mod, "ChunkStore", SimpleNamespace(from_config=lambda *a, **k: store))
    if answer is not None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    monkeypatch.setattr("sys.argv", argv)
    return mod.main()


# --- the nuclear option ------------------------------------------------------


def test_all_deletes_the_database_and_its_sidecars(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    db = Path(cfg.db_path)
    db.write_bytes(b"db")
    for suffix in ("-wal", "-shm", "-journal"):
        Path(str(db) + suffix).write_bytes(b"x")

    assert _run(monkeypatch, ["corpus-reset", "--all", "--yes"], cfg=cfg) == 0
    assert not db.exists()
    for suffix in ("-wal", "-shm", "-journal"):
        assert not Path(str(db) + suffix).exists(), f"{suffix} left behind"


def test_declining_the_prompt_deletes_nothing(tmp_path, monkeypatch) -> None:
    """The only guard on an irreversible action."""
    cfg = _config(tmp_path)
    Path(cfg.db_path).write_bytes(b"db")

    code = _run(monkeypatch, ["corpus-reset", "--all"], answer="n", cfg=cfg)

    assert code == 1, "declining must be a non-zero exit"
    assert Path(cfg.db_path).exists(), "the database was deleted after 'n'"


@pytest.mark.parametrize("answer", ["", "no", "N", "yes", "Y", " y "])
def test_only_an_exact_y_confirms(tmp_path, monkeypatch, answer) -> None:
    """`yes` does NOT confirm, which is worth pinning rather than assuming:
    a user who types the whole word keeps their database."""
    cfg = _config(tmp_path)
    Path(cfg.db_path).write_bytes(b"db")

    _run(monkeypatch, ["corpus-reset", "--all"], answer=answer, cfg=cfg)
    survived = Path(cfg.db_path).exists()
    assert survived is (answer.strip().lower() != "y")


def test_all_on_a_missing_database_is_not_an_error(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    assert _run(monkeypatch, ["corpus-reset", "--all", "--yes"], cfg=cfg) == 0


# --- the surgical option -----------------------------------------------------


def test_source_drops_only_that_source(tmp_path, monkeypatch, capsys) -> None:
    cfg = _config(tmp_path)
    Path(cfg.db_path).write_bytes(b"db")
    store = MagicMock()
    store.stats.return_value = {"by_source": {"notes": 42, "papers": 7}}
    store.delete_by_source.return_value = 42

    code = _run(monkeypatch, ["corpus-reset", "--source", "notes", "--yes"],
                store=store, cfg=cfg)

    assert code == 0
    store.delete_by_source.assert_called_once_with("notes")
    assert "42" in capsys.readouterr().out


def test_declining_drops_no_chunks(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    Path(cfg.db_path).write_bytes(b"db")
    store = MagicMock()
    store.stats.return_value = {"by_source": {"notes": 42}}

    code = _run(monkeypatch, ["corpus-reset", "--source", "notes"],
                answer="n", store=store, cfg=cfg)

    assert code == 1
    store.delete_by_source.assert_not_called()


def test_an_unknown_source_deletes_nothing_and_says_so(
    tmp_path, monkeypatch, capsys
) -> None:
    """A typo'd source name must not prompt, and must not delete."""
    cfg = _config(tmp_path)
    Path(cfg.db_path).write_bytes(b"db")
    store = MagicMock()
    store.stats.return_value = {"by_source": {"notes": 42}}

    code = _run(monkeypatch, ["corpus-reset", "--source", "notez", "--yes"],
                store=store, cfg=cfg)

    assert code == 0
    store.delete_by_source.assert_not_called()
    assert "no chunks" in capsys.readouterr().out.lower()


def test_the_store_is_closed_even_when_the_user_declines(
    tmp_path, monkeypatch
) -> None:
    cfg = _config(tmp_path)
    Path(cfg.db_path).write_bytes(b"db")
    store = MagicMock()
    store.stats.return_value = {"by_source": {"notes": 1}}

    _run(monkeypatch, ["corpus-reset", "--source", "notes"],
         answer="n", store=store, cfg=cfg)

    store.close.assert_called_once()
