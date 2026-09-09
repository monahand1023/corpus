"""Tests for `corpus.planner`: turning a raw directory into a reviewable
ingest plan (`build_plan`) and safely merging detected sources into an
existing corpus.toml (`merge_sources_into_toml`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.config import SourceConfig
from corpus.planner import build_plan, merge_sources_into_toml
from corpus.util.text_yield import estimate_tokens_from_bytes


def _touch(root: Path, rel: str, content: str = "hello world this is a test\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------


def test_build_plan_detects_sources_with_counts_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "Inbox"
    _touch(root, "a.md", "one two three")
    _touch(root, "b.md", "four five six seven")
    _touch(root, "c.txt", "eight nine")

    plan = build_plan(root)

    by_type = {p.source.type: p for p in plan.sources}
    assert set(by_type) == {"markdown", "text"}
    assert by_type["markdown"].file_count == 2
    assert by_type["markdown"].total_bytes == len("one two three") + len("four five six seven")
    assert by_type["text"].file_count == 1


def test_build_plan_estimated_tokens_uses_calibrated_text_yield(tmp_path: Path) -> None:
    """Token estimate is calibrated per connector type (`corpus.util.text_yield`),
    not a flat bytes/4 -- see `corpus.planner.PlannedSource.estimated_tokens`."""
    root = tmp_path / "Inbox"
    _touch(root, "a.txt", "x" * 400)

    plan = build_plan(root)
    (p,) = plan.sources
    assert p.total_bytes == 400
    assert p.estimated_tokens == estimate_tokens_from_bytes("text", 400)


def test_build_plan_reports_gap(tmp_path: Path) -> None:
    root = tmp_path / "Inbox"
    _touch(root, "deck.foo", "unknown format")

    plan = build_plan(root)

    assert plan.sources == []
    assert {b.bucket for b in plan.gap} == {".foo"}


def test_build_plan_excludes_default_noise_dirs_from_counts(tmp_path: Path) -> None:
    root = tmp_path / "Project"
    _touch(root, "README.md", "real content, real content, real content")
    _touch(root, "node_modules/pkg/README.md", "vendored readme, not mine")

    plan = build_plan(root)

    (p,) = plan.sources
    assert p.source.type == "markdown"
    assert p.file_count == 1  # the node_modules copy is not counted
    assert plan.census.walk_stats.dirs_pruned >= 1


def test_build_plan_type_that_exists_only_in_noise_is_left_out_entirely(tmp_path: Path) -> None:
    """A connector type detected by `detect_sources` (which has no excludes
    concept) but with ZERO files outside noise directories must not appear
    in the plan at all — proposing a source with a misleading "0 files" row
    would be worse than not proposing it, since `corpus-ingest` would still
    glob the whole tree (including the noise dir) at actual ingest time."""
    root = tmp_path / "Project"
    _touch(root, "node_modules/pkg/notes.md", "vendored, not mine")
    _touch(root, "real.txt", "actual content here")

    plan = build_plan(root)

    types = {p.source.type for p in plan.sources}
    assert types == {"text"}, "markdown only exists inside node_modules and must be excluded"


def test_build_plan_no_default_excludes_includes_noise_dirs(tmp_path: Path) -> None:
    root = tmp_path / "Project"
    _touch(root, "README.md", "real content")
    _touch(root, "node_modules/pkg/README.md", "vendored")

    plan = build_plan(root, use_default_excludes=False)

    (p,) = plan.sources
    assert p.file_count == 2


def test_build_plan_name_prefix_disambiguates(tmp_path: Path) -> None:
    root = tmp_path / "Inbox"
    _touch(root, "a.md", "content")

    plan = build_plan(root, name_prefix="personal")

    (p,) = plan.sources
    assert p.source.name == "personal_inbox_markdown"


def test_build_plan_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_plan(tmp_path / "nope")


def test_build_plan_totals(tmp_path: Path) -> None:
    root = tmp_path / "Inbox"
    _touch(root, "a.md", "x" * 40)
    _touch(root, "b.txt", "x" * 60)

    plan = build_plan(root)

    assert plan.total_file_count == 2
    assert plan.total_bytes == 100
    assert plan.total_estimated_tokens == (
        estimate_tokens_from_bytes("markdown", 40) + estimate_tokens_from_bytes("text", 60)
    )


# ---------------------------------------------------------------------------
# merge_sources_into_toml
# ---------------------------------------------------------------------------


def _config_text(extra: str = "") -> str:
    return (
        '[corpus]\ndb_path = "./corpus.db"\n\n'
        '[embedder]\nprovider = "hash"\nmodel = "hash-v1"\ndim = 8\n'
        f"{extra}"
    )


def test_merge_adds_new_sources_and_writes_file(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text(_config_text())
    new = [SourceConfig(name="inbox_markdown", type="markdown", path=str(tmp_path / "Inbox"))]

    result = merge_sources_into_toml(cfg, existing_sources=[], new_sources=new)

    assert result.written is True
    assert [s.name for s in result.added] == ["inbox_markdown"]
    assert result.conflicts == []
    assert result.ingestible_names == ["inbox_markdown"]
    text = cfg.read_text()
    assert '[[sources]]' in text
    assert 'name = "inbox_markdown"' in text
    assert 'type = "markdown"' in text
    # Original content survives untouched.
    assert 'provider = "hash"' in text


def test_merge_preserves_existing_file_content_verbatim(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    original = _config_text('\n[[sources]]\nname = "notes"\ntype = "markdown"\npath = "/x"\n')
    cfg.write_text(original)
    new = [SourceConfig(name="inbox_markdown", type="markdown", path=str(tmp_path / "Inbox"))]

    merge_sources_into_toml(cfg, existing_sources=[], new_sources=new)

    text = cfg.read_text()
    assert text.startswith(original)


def test_merge_is_idempotent_for_identical_rerun(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text(_config_text())
    root = tmp_path / "Inbox"
    existing = [SourceConfig(name="inbox_markdown", type="markdown", path=str(root))]
    new = [SourceConfig(name="inbox_markdown", type="markdown", path=str(root))]

    result = merge_sources_into_toml(cfg, existing_sources=existing, new_sources=new)

    assert result.written is False
    assert result.added == []
    assert [s.name for s in result.unchanged] == ["inbox_markdown"]
    assert result.ingestible_names == ["inbox_markdown"]
    # No duplicate block: file is exactly what it was before the call.
    assert cfg.read_text() == _config_text()


def test_merge_refuses_name_collision_from_different_folder(tmp_path: Path) -> None:
    """Two DIFFERENT folders whose basenames both normalize to `inbox` must
    not silently clobber each other's [[sources]] entry — that would delete
    the first folder's chunks on the next ingest (source_type-scoped orphan
    pruning)."""
    cfg = tmp_path / "corpus.toml"
    original = _config_text()
    cfg.write_text(original)
    existing = [
        SourceConfig(name="inbox_pdf", type="pdf", path=str(tmp_path / "Work" / "Inbox"))
    ]
    new = [SourceConfig(name="inbox_pdf", type="pdf", path=str(tmp_path / "Personal" / "Inbox"))]

    result = merge_sources_into_toml(cfg, existing_sources=existing, new_sources=new)

    assert result.written is False
    assert result.added == []
    assert len(result.conflicts) == 1
    assert result.ingestible_names == []
    assert "inbox_pdf" in result.conflicts[0].detail  # type: ignore[operator]
    # File is byte-for-byte untouched.
    assert cfg.read_text() == original


def test_merge_mixed_added_unchanged_conflict(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text(_config_text())
    existing = [
        SourceConfig(name="inbox_markdown", type="markdown", path=str(tmp_path / "Inbox")),
        SourceConfig(name="inbox_pdf", type="pdf", path=str(tmp_path / "Elsewhere")),
    ]
    new = [
        SourceConfig(name="inbox_markdown", type="markdown", path=str(tmp_path / "Inbox")),  # unchanged
        SourceConfig(name="inbox_pdf", type="pdf", path=str(tmp_path / "Inbox")),  # conflict (diff path)
        SourceConfig(name="inbox_text", type="text", path=str(tmp_path / "Inbox")),  # added
    ]

    result = merge_sources_into_toml(cfg, existing_sources=existing, new_sources=new)

    assert [s.name for s in result.unchanged] == ["inbox_markdown"]
    assert [o.source.name for o in result.conflicts] == ["inbox_pdf"]
    assert [s.name for s in result.added] == ["inbox_text"]
    assert set(result.ingestible_names) == {"inbox_markdown", "inbox_text"}
    text = cfg.read_text()
    assert text.count('name = "inbox_text"') == 1
    assert 'name = "inbox_pdf"' not in text  # never re-rendered; only the original mention, which doesn't exist here


def test_merge_no_new_sources_does_not_write(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    original = _config_text()
    cfg.write_text(original)

    result = merge_sources_into_toml(cfg, existing_sources=[], new_sources=[])

    assert result.written is False
    assert cfg.read_text() == original
