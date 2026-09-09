"""Tests for `corpus.survey.classify` (per-file classification) and
`corpus.survey.census` (the tree-wide census that groups classified files
into indexable / gap / noise buckets)."""

from __future__ import annotations

from pathlib import Path

from corpus.survey.census import run_census
from corpus.survey.classify import classify_file, indexable_extension_map


def _touch(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------


def test_indexable_extensions_map_to_their_connector() -> None:
    assert classify_file("notes.md").category == "indexable"
    assert classify_file("notes.md").detail == "markdown"
    assert classify_file("report.PDF").category == "indexable"  # case-insensitive
    assert classify_file("report.PDF").detail == "pdf"


def test_html_alias_extension_is_indexable() -> None:
    # .htm is an accepted spelling variant, not just .html — see
    # corpus.connectors.zip._EXTENSION_ALIASES, reused here for no-drift.
    c = classify_file("page.htm")
    assert c.category == "indexable"
    assert c.detail == "html"


def test_top_level_zip_is_indexable_unlike_nested_archive_members() -> None:
    c = classify_file("bundle.zip")
    assert c.category == "indexable"
    assert c.detail == "zip"


def test_unsupported_extension_is_a_gap() -> None:
    c = classify_file("deck.foo")
    assert c.category == "gap"
    assert c.bucket == ".foo"


def test_no_extension_file_is_a_gap_with_readable_bucket() -> None:
    c = classify_file("README")
    assert c.category == "gap"
    assert c.bucket == "(no extension)"


def test_ds_store_is_noise_not_a_gap() -> None:
    c = classify_file(".DS_Store")
    assert c.category == "noise"


def test_thumbs_db_is_noise() -> None:
    assert classify_file("Thumbs.db").category == "noise"


def test_appledouble_prefix_is_noise() -> None:
    assert classify_file("._report.pdf").category == "noise"


def test_minified_js_is_noise_not_gap() -> None:
    c = classify_file("app.min.js")
    assert c.category == "noise"


def test_compiled_artifact_extension_is_noise() -> None:
    assert classify_file("module.pyc").category == "noise"
    assert classify_file("lib.so").category == "noise"


def test_indexable_extension_map_covers_registered_connectors() -> None:
    from corpus.connectors.registry import CONNECTOR_REGISTRY

    mapping = indexable_extension_map()
    assert set(mapping.values()) >= set(CONNECTOR_REGISTRY)


# ---------------------------------------------------------------------------
# run_census
# ---------------------------------------------------------------------------


def test_census_splits_into_three_buckets(tmp_path: Path) -> None:
    _touch(tmp_path, "notes.md", "hello world")
    _touch(tmp_path, "deck.foo", "x" * 100)
    _touch(tmp_path, ".DS_Store", "junk")

    result = run_census(tmp_path)

    assert result.files_scanned == 3
    assert {b.bucket for b in result.indexable} == {".md"}
    assert {b.bucket for b in result.gap} == {".foo"}
    assert {b.bucket for b in result.noise} == {".ds_store"}


def test_census_counts_and_sizes_are_accurate(tmp_path: Path) -> None:
    _touch(tmp_path, "a.txt", "12345")
    _touch(tmp_path, "b.txt", "1234567890")

    result = run_census(tmp_path)

    [bucket] = result.indexable
    assert bucket.count == 2
    assert bucket.total_bytes == 15


def test_census_sorts_by_size_descending(tmp_path: Path) -> None:
    _touch(tmp_path, "big.foo", "x" * 1000)
    _touch(tmp_path, "small.bar", "x" * 10)

    result = run_census(tmp_path)

    assert [b.bucket for b in result.gap] == [".foo", ".bar"]


def test_census_respects_excludes(tmp_path: Path) -> None:
    _touch(tmp_path, "keep.md", "content")
    _touch(tmp_path, "node_modules/pkg/readme.md", "vendored")

    result = run_census(tmp_path)

    [bucket] = result.indexable
    assert bucket.count == 1


def test_census_reports_walk_errors(tmp_path: Path) -> None:
    link = tmp_path / "broken.txt"
    link.symlink_to(tmp_path / "missing.txt")

    result = run_census(tmp_path)

    assert result.files_scanned == 0
    assert result.walk_stats.file_symlinks_skipped == 1
