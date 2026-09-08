"""FIX 1 (regression): a source whose connector needs a missing optional
extra must fail ONLY that source with a friendly, actionable message --
`--all` must continue on to ingest every subsequent source rather than
aborting with a raw ModuleNotFoundError traceback.

Before the fix: registry.py's `_build_*` factories never actually probed
the third-party import (each connector imports it lazily inside `load()`),
so the "install the extra" ImportError was dead code and a real
ModuleNotFoundError surfaced instead -- which `cli/ingest.py`'s
`except (ValueError, FileNotFoundError)` did not catch, aborting `--all`
and silently skipping every source after the broken one.

We simulate the missing module (never uninstall a real dependency) via
builtins.__import__, the same pattern used in test_factory.py.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from corpus.cli import ingest as ingest_cli


def _write_config(cfg_path: Path, db_path: Path, notes_dir: Path, docx_dir: Path) -> None:
    cfg_path.write_text(
        f'[corpus]\ndb_path = "{db_path.as_posix()}"\n'
        '[embedder]\nprovider = "hash"\nmodel = "hash-v1"\ndim = 256\n'
        '[retriever]\ntop_k = 5\n'
        '\n[[sources]]\n'
        'name = "docs"\n'
        'type = "docx"\n'
        f'path = "{docx_dir.as_posix()}"\n'
        '\n[[sources]]\n'
        'name = "notes"\n'
        'type = "markdown"\n'
        f'path = "{notes_dir.as_posix()}"\n'
    )


def _run(cfg: Path, extra: list[str]) -> tuple[int, str]:
    argv = sys.argv
    sys.argv = ["corpus-ingest", "--config", str(cfg), *extra]
    try:
        code = ingest_cli.main()
    finally:
        sys.argv = argv
    return code, ""


def test_missing_extra_fails_only_that_source_and_all_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "hello.md").write_text("# Hello\n\nSome content about corpus ingestion.\n")
    docx_dir = tmp_path / "docx_source"
    docx_dir.mkdir()

    cfg_path = tmp_path / "corpus.toml"
    _write_config(cfg_path, tmp_path / "test.db", notes_dir, docx_dir)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "docx" or name.startswith("docx."):
            raise ImportError("No module named 'docx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    exit_code, _ = _run(cfg_path, ["--all"])
    out = capsys.readouterr().out

    # The whole run reports failure (exit_code != 0)...
    assert exit_code == 1
    # ...but the docx source got the friendly, actionable message...
    assert "corpus-rag[docx]" in out
    assert "index left intact; skipping this source" in out
    # ...and crucially, --all did NOT abort: the "notes" source after the
    # broken one still ran and ingested its document.
    assert "=== Ingesting notes ===" in out
    assert "documents:        1" in out


def test_prune_anyway_is_refused_with_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--prune-anyway forces a destructive sweep past the pruning safety gate,
    so it is deliberately unavailable across every source at once: the operator
    has to name the source whose unreadable files they have actually reviewed."""
    notes_dir = tmp_path / "notes_source"
    notes_dir.mkdir()
    docx_dir = tmp_path / "docx_source"
    docx_dir.mkdir()
    cfg_path = tmp_path / "corpus.toml"
    _write_config(cfg_path, tmp_path / "test.db", notes_dir, docx_dir)

    exit_code, _ = _run(cfg_path, ["--all", "--prune-anyway"])
    out = capsys.readouterr().out

    assert exit_code == 2
    assert "Refusing --prune-anyway with --all" in out
    assert "--source NAME" in out
    # It must refuse BEFORE doing any ingestion work at all.
    assert "=== Ingesting" not in out
