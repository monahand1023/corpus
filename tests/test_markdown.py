from __future__ import annotations

from pathlib import Path

import pytest

from corpus.chunkers.markdown import (
    MAX_CHUNK_CHARS,
    _coalesce_small,
    chunk_markdown_body,
    parse_markdown,
)
from corpus.connectors.markdown import MarkdownChunker, MarkdownConnector


def test_parse_frontmatter() -> None:
    text = '---\ntitle: "Hello"\nid: abc123\n---\n# Heading\n\nbody'
    parsed = parse_markdown(text)
    assert parsed.frontmatter["title"] == "Hello"
    assert parsed.frontmatter["id"] == "abc123"
    assert parsed.body.startswith("# Heading")


def test_chunk_splits_at_h2() -> None:
    body = "# Title\nintro\n\n## A\n\nbody A\n\n## B\n\nbody B"
    chunks = chunk_markdown_body(body)
    assert any("## A" in c for c in chunks)
    assert any("## B" in c for c in chunks)


def test_chunk_preserves_code_fences() -> None:
    code = "```python\n" + ("x = 1\n" * 500) + "```"
    body = f"## Section\n\nintro\n\n{code}\n\n## Other\nbody"
    chunks = chunk_markdown_body(body)
    for c in chunks:
        assert c.count("```") % 2 == 0


def test_coalesce_packs_tiny_fragments() -> None:
    """Many tiny adjacent chunks should pack into far fewer ~max-size chunks,
    losing no content and preserving order."""
    tiny = [f"## row {i}" for i in range(500)]
    coalesced = _coalesce_small(tiny, MAX_CHUNK_CHARS)
    # Massive reduction in chunk count.
    assert len(coalesced) < len(tiny)
    # No chunk exceeds the cap.
    assert all(len(c) <= MAX_CHUNK_CHARS for c in coalesced)
    # Every original fragment survives, in order: joining all output back
    # contains each row id in ascending sequence.
    joined = "\n\n".join(coalesced)
    for i in range(500):
        assert f"## row {i}" in joined
    positions = [joined.index(f"## row {i}") for i in range(500)]
    assert positions == sorted(positions)


def test_coalesce_drops_empty_fragments() -> None:
    out = _coalesce_small(["a", "   ", "", "b"], MAX_CHUNK_CHARS)
    assert out == ["a\n\nb"]


def test_chunk_heading_heavy_body_coalesces() -> None:
    """A heading-heavy body must not explode into hundreds of ~10-char chunks."""
    body = "\n\n".join(f"## h{i}\nv{i}" for i in range(400))
    chunks = chunk_markdown_body(body)
    assert len(chunks) < 50  # would be ~400 without coalescing
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


def test_connector_loads_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        '---\ntitle: "Doc A"\nid: doc-a\n---\n# A\n\nContent of A.'
    )
    (tmp_path / "b.md").write_text("# B\n\nContent of B without frontmatter.")
    (tmp_path / "ignore.txt").write_text("text file, not markdown")

    conn = MarkdownConnector(source_type="notes", path=tmp_path)
    docs = list(conn.load())
    assert len(docs) == 2
    titles = {d.title for d in docs}
    assert "Doc A" in titles
    assert "b" in titles  # filename stem when frontmatter is missing


def test_connector_dedupes_identical_docs(tmp_path: Path) -> None:
    same = "# Same\n\nIdentical body."
    (tmp_path / "v1.md").write_text(same)
    (tmp_path / "v2.md").write_text(same)
    docs = list(MarkdownConnector(source_type="notes", path=tmp_path).load())
    assert len(docs) == 1


def test_chunker_emits_titled_chunks(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        '---\ntitle: "Doc A"\n---\n# A\n\nFirst paragraph.\n\nSecond paragraph.'
    )
    docs = list(MarkdownConnector(source_type="notes", path=tmp_path).load())
    chunker = MarkdownChunker(source_type="notes")
    chunks = chunker.chunk(docs[0])
    assert chunks
    # First chunk leads with title
    assert "Doc A" in chunks[0].content


def test_connector_missing_dir_raises() -> None:
    conn = MarkdownConnector(source_type="notes", path="/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        list(conn.load())


def test_connector_respects_glob(tmp_path: Path) -> None:
    (tmp_path / "include.md").write_text("# Include")
    (tmp_path / "exclude.txt").write_text("Exclude")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "deep.md").write_text("# Deep")
    docs = list(MarkdownConnector(source_type="notes", path=tmp_path, glob="*.md").load())
    titles = {d.title for d in docs}
    # Top-level only — sub/deep should NOT appear
    assert "include" in titles
    assert "deep" not in titles
