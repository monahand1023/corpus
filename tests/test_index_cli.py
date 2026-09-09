"""Tests for the `corpus-index` CLI: survey -> plan -> confirm -> ingest,
end to end, via `corpus.cli.index.main_argv`.

Uses the keyless `hash` embedder (same pattern as test_ingest_cli.py /
test_survey_cli.py) so the whole flow runs with no API key and no network.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from corpus.cli.index import main_argv
from corpus.db.sqlite import ChunkStore
from corpus.types import Chunk, ChunkKind, ChunkMetadata
from corpus.util.hash import chunk_id, sha256

_DIM = 8


def _write_config(cfg_path: Path, db_path: Path) -> None:
    cfg_path.write_text(
        f'[corpus]\ndb_path = "{db_path.as_posix()}"\n'
        '[embedder]\nprovider = "hash"\nmodel = "hash-v1"\ndim = 8\n'
        '[retriever]\ntop_k = 5\n'
    )


def _touch(root: Path, rel: str, content: str = "hello world this is content\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _run(argv: list[str]) -> int:
    return main_argv(argv)


# ---------------------------------------------------------------------------
# dry-run: shows the plan, never writes, never ingests
# ---------------------------------------------------------------------------


def test_dry_run_shows_plan_writes_nothing_ingests_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    db = tmp_path / "corpus.db"
    _write_config(cfg, db)
    src = tmp_path / "Inbox"
    _touch(src, "note.md", "some real markdown content here")
    _touch(src, "deck.foo", "an unsupported format")

    original_cfg_text = cfg.read_text()
    rc = _run([str(src), "--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "(dry run" in out
    assert "Gap" in out
    assert ".foo" in out
    assert "Plan —" in out
    assert "inbox_markdown" in out
    assert "=== Ingesting" not in out
    assert cfg.read_text() == original_cfg_text, "dry-run must never write corpus.toml"
    assert not db.exists(), "dry-run must never ingest"


def test_missing_directory_errors_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")

    rc = _run([str(tmp_path / "nope"), "--config", str(cfg), "--dry-run"])

    assert rc == 1
    assert "not a directory" in capsys.readouterr().err


def test_nothing_ingestible_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")
    src = tmp_path / "Empty"
    src.mkdir()

    rc = _run([str(src), "--config", str(cfg), "--yes"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Nothing ingestible found" in out


# ---------------------------------------------------------------------------
# confirmation gate
# ---------------------------------------------------------------------------


def test_declining_confirmation_aborts_without_writing_or_ingesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    db = tmp_path / "corpus.db"
    _write_config(cfg, db)
    src = tmp_path / "Inbox"
    _touch(src, "note.md")

    original_cfg_text = cfg.read_text()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    rc = _run([str(src), "--config", str(cfg)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Aborted" in out
    assert "=== Ingesting" not in out
    assert cfg.read_text() == original_cfg_text
    assert not db.exists()


def test_eof_on_confirmation_is_treated_as_decline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")
    src = tmp_path / "Inbox"
    _touch(src, "note.md")

    def _raise_eof(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)

    rc = _run([str(src), "--config", str(cfg)])

    assert rc == 1
    assert "Aborted" in capsys.readouterr().out


def test_yes_flag_skips_prompt_and_ingests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    db = tmp_path / "corpus.db"
    _write_config(cfg, db)
    src = tmp_path / "Inbox"
    _touch(src, "note.md", "some real markdown content about corpus")

    def _fail_if_called(_prompt: str = "") -> str:
        raise AssertionError("--yes must skip the confirmation prompt entirely")

    monkeypatch.setattr("builtins.input", _fail_if_called)

    rc = _run([str(src), "--config", str(cfg), "--yes"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "=== Ingesting inbox_markdown ===" in out
    assert "documents:        1" in out
    assert 'name = "inbox_markdown"' in cfg.read_text()
    assert db.exists()


def test_confirming_with_y_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")
    src = tmp_path / "Inbox"
    _touch(src, "note.md", "content")

    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    rc = _run([str(src), "--config", str(cfg)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "=== Ingesting inbox_markdown ===" in out


# ---------------------------------------------------------------------------
# reproducibility: re-running is idempotent on config, still re-ingests
# ---------------------------------------------------------------------------


def test_rerun_on_same_directory_does_not_duplicate_config_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")
    src = tmp_path / "Inbox"
    _touch(src, "note.md", "content")

    rc1 = _run([str(src), "--config", str(cfg), "--yes"])
    assert rc1 == 0
    capsys.readouterr()

    _touch(src, "note2.md", "more content")
    rc2 = _run([str(src), "--config", str(cfg), "--yes"])
    out2 = capsys.readouterr().out

    assert rc2 == 0
    assert "Already configured (unchanged): inbox_markdown" in out2
    assert cfg.read_text().count('name = "inbox_markdown"') == 1
    assert "documents:        2" in out2  # second run picks up the new file


def test_name_collision_from_different_folder_is_refused_but_others_proceed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    cfg.write_text(
        f'[corpus]\ndb_path = "{(tmp_path / "corpus.db").as_posix()}"\n'
        '[embedder]\nprovider = "hash"\nmodel = "hash-v1"\ndim = 8\n'
        '[retriever]\ntop_k = 5\n'
        "\n[[sources]]\n"
        'name = "inbox_markdown"\n'
        'type = "markdown"\n'
        f'path = "{other_dir.as_posix()}"\n'
    )
    src = tmp_path / "Inbox"  # normalizes to the same "inbox_markdown" name
    _touch(src, "note.md", "content")
    _touch(src, "note.txt", "other content")

    rc = _run([str(src), "--config", str(cfg), "--yes"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "REFUSED to touch" in out
    assert "inbox_markdown" in out
    # The non-colliding text source still gets added and ingested.
    assert "=== Ingesting inbox_text ===" in out
    assert 'name = "inbox_text"' in cfg.read_text()


def test_name_prefix_avoids_collision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")
    src = tmp_path / "Inbox"
    _touch(src, "note.md", "content")

    rc = _run([str(src), "--config", str(cfg), "--yes", "--name-prefix", "personal"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "=== Ingesting personal_inbox_markdown ===" in out


# ---------------------------------------------------------------------------
# --check-overlap
# ---------------------------------------------------------------------------


def _fake_embedding(seed: int) -> list[float]:
    return [math.sin((seed + 1) * (i + 1) * 0.001) for i in range(_DIM)]


def test_check_overlap_missing_db_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")
    src = tmp_path / "Inbox"
    _touch(src, "note.md")

    rc = _run(
        [str(src), "--config", str(cfg), "--dry-run", "--check-overlap", str(tmp_path / "nope.db")]
    )

    assert rc == 1
    assert "database not found" in capsys.readouterr().err


def test_check_overlap_reports_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_db = tmp_path / "archive.db"
    store = ChunkStore(archive_db, embedding_dim=_DIM)
    content = "this exact distinctive phrase should be found during the overlap check"
    try:
        chunk = Chunk(
            id=chunk_id("archive", "doc1", ChunkKind.SECTION, 0),
            content=content,
            content_hash=sha256(content),
            metadata=ChunkMetadata(
                source_type="archive",
                source_key="doc1",
                chunk_kind=ChunkKind.SECTION,
                chunk_index=0,
                title="doc1",
            ),
        )
        store.upsert_batch([(chunk, _fake_embedding(0))])
    finally:
        store.close()

    cfg = tmp_path / "corpus.toml"
    _write_config(cfg, tmp_path / "corpus.db")
    src = tmp_path / "Inbox"
    _touch(src, "note.txt", content)

    rc = _run([str(src), "--config", str(cfg), "--dry-run", "--check-overlap", str(archive_db)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Overlap vs." in out
    assert "estimated overlap" in out
