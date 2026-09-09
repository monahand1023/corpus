"""Tests for the `corpus-survey` CLI entrypoint (argparse wiring, --json,
human-readable output, and CLI-level error handling)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from corpus.cli.survey import main_argv


def _touch(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_census_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _touch(tmp_path, "notes.md", "hello")
    _touch(tmp_path, "deck.foo", "world")

    rc = main_argv(["census", str(tmp_path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_scanned"] == 2
    assert payload["follows_symlinks"] is False
    assert {b["bucket"] for b in payload["indexable"]} == {".md"}
    assert {b["bucket"] for b in payload["gap"]} == {".foo"}


def test_census_human_output_mentions_symlink_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _touch(tmp_path, "notes.md")

    rc = main_argv(["census", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "not followed" in out
    assert "Gap" in out
    assert "Indexable" in out


def test_census_nonexistent_path_errors_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main_argv(["census", str(tmp_path / "nope")])

    assert rc == 1
    assert "not a directory" in capsys.readouterr().err


def test_census_exclude_flag_is_repeatable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _touch(tmp_path, "keep.md")
    _touch(tmp_path, "skip1/a.md")
    _touch(tmp_path, "skip2/b.md")

    rc = main_argv(
        ["census", str(tmp_path), "--exclude", "skip1", "--exclude", "skip2", "--json"]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_scanned"] == 1


def test_no_subcommand_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main_argv([])


def test_archives_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as zf:
        zf.writestr("notes.md", "content")
        zf.writestr("node_modules/pkg/readme.md", "vendored")

    rc = main_argv(["archives", str(tmp_path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["archive_count"] == 1
    assert payload["totals"]["dependency_noise"] == 1
    assert payload["archives"][0]["indexable_by_type"] == {"markdown": 1}


def test_archives_human_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as zf:
        zf.writestr("notes.md", "content")

    rc = main_argv(["archives", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "read-only" in out
    assert "bundle.zip" in out


def test_archives_nonexistent_path_errors_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main_argv(["archives", str(tmp_path / "nope")])

    assert rc == 1
    assert "not a directory" in capsys.readouterr().err


def test_media_json_output_without_ffprobe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import corpus.survey.media as media_mod

    monkeypatch.setattr(media_mod.shutil, "which", lambda _: None)
    _touch(tmp_path, "call.mp3")

    rc = main_argv(["media", str(tmp_path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ffprobe_available"] is False
    assert payload["total_files"] == 1
    assert payload["estimated_total_hours"] is None


def test_media_human_output_notes_ffprobe_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import corpus.survey.media as media_mod

    monkeypatch.setattr(media_mod.shutil, "which", lambda _: None)
    _touch(tmp_path, "call.mp3")

    rc = main_argv(["media", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "ffprobe not found" in out


def test_media_nonexistent_path_errors_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main_argv(["media", str(tmp_path / "nope")])

    assert rc == 1
    assert "not a directory" in capsys.readouterr().err
