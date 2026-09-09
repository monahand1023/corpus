"""Tests for `corpus-ingest --path DIR` — scan a folder, detect which built-in
connectors apply, and ingest each detected type as its own source.

The point of the feature is that dropping files in a folder and running one
command is enough; you never hand-write a [[sources]] block.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from corpus.connectors.registry import DEFAULT_GLOBS
from corpus.util.autodetect import detect_sources, normalize_source_name


def _touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    return p


def test_default_globs_covers_every_registered_connector() -> None:
    """If a connector is registered but has no default glob, --path silently
    cannot find its files."""
    from corpus.connectors.registry import CONNECTOR_REGISTRY

    assert set(DEFAULT_GLOBS) == set(CONNECTOR_REGISTRY)


def test_detects_one_source_per_present_file_type(tmp_path: Path) -> None:
    root = tmp_path / "Inbox"
    _touch(root, "a.md")
    _touch(root, "b.txt")
    _touch(root, "nested/c.pdf")
    _touch(root, "d.docx")

    found = detect_sources(root)
    by_type = {s.type: s for s in found}

    assert set(by_type) == {"markdown", "text", "pdf", "docx"}
    assert all(s.path == str(root) for s in found)


def test_types_with_no_files_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "Inbox"
    _touch(root, "only.md")
    found = detect_sources(root)
    assert [s.type for s in found] == ["markdown"]


def test_source_names_are_namespaced_by_folder(tmp_path: Path) -> None:
    """Two folders must NOT share a source_type. delete_orphans is scoped by
    source_type alone, so a bare `pdf` name would make ingesting the second
    folder delete the first folder's chunks."""
    a = tmp_path / "Field Notes - 2024"
    b = tmp_path / "Inbox"
    _touch(a, "one.pdf")
    _touch(b, "two.pdf")

    name_a = detect_sources(a)[0].name
    name_b = detect_sources(b)[0].name

    assert name_a != name_b, "distinct folders must get distinct source names"
    assert name_a == "field_notes_2024_pdf"
    assert name_b == "inbox_pdf"


def test_detected_names_are_valid_source_types(tmp_path: Path) -> None:
    """SourceConfig.name is constrained to ^[a-z][a-z0-9_]*$; a folder called
    '2024 Reports!' must still yield a usable identifier."""
    root = tmp_path / "2024 Reports!"
    _touch(root, "x.txt")
    (name,) = [s.name for s in detect_sources(root)]
    assert name == "reports_text"


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_sources(tmp_path / "nope")


def test_empty_directory_detects_nothing(tmp_path: Path) -> None:
    root = tmp_path / "Empty"
    root.mkdir()
    assert detect_sources(root) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Field Notes - 2024", "field_notes_2024"),
        ("Inbox", "inbox"),
        ("2024 Reports!", "reports"),
        ("...", "folder"),
    ],
)
def test_normalize_source_name(raw: str, expected: str) -> None:
    assert normalize_source_name(raw) == expected
