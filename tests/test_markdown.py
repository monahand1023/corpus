from __future__ import annotations

from itertools import pairwise
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


def test_cp932_encoded_file_is_decoded_correctly_not_replaced(tmp_path: Path) -> None:
    """See the identical regression test in test_text_connector.py — this
    connector shares the same `errors="replace"` -> fallback fix (see
    `corpus.util.encoding`)."""
    text = "# 議事録\n\n会議メモ: 予算は前年比で増加した。"
    (tmp_path / "memo.md").write_bytes(text.encode("cp932"))
    docs = list(MarkdownConnector(source_type="notes", path=tmp_path).load())
    assert len(docs) == 1
    assert "会議メモ" in docs[0].raw["body"]
    assert "�" not in docs[0].raw["body"]


# --- NUL stripping ----------------------------------------------------------
#
# Every connector routes through MarkdownChunker (registry.py's 13 _build_*
# functions all return one), so this is the single point every chunk of every
# source type passes through — these guard the whole ingest path, not just
# markdown.


def _chunk_body(tmp_path: Path, body: str) -> list[str]:
    (tmp_path / "a.md").write_text(body)
    docs = list(MarkdownConnector(source_type="notes", path=tmp_path).load())
    return [c.content for c in MarkdownChunker(source_type="notes").chunk(docs[0])]


def test_nul_characters_are_stripped_from_chunk_content(tmp_path: Path) -> None:
    # PDF extraction emits these when a page's font has no usable encoding —
    # 239 chunks in one real archive. FTS5 indexing and terminal display both
    # truncate at the first NUL, silently hiding the rest of a good chunk.
    joined = "".join(_chunk_body(tmp_path, "# A\n\nbefore\x00after"))

    assert "\x00" not in joined
    assert "before" in joined and "after" in joined


def test_nul_stripping_is_reflected_in_the_content_hash(tmp_path: Path) -> None:
    # The hash drives dedup and the re-embed decision, so it has to be taken
    # over the text actually stored, not the pre-normalized text.
    (tmp_path / "a.md").write_text("# A\n\nbody text\x00")
    dirty = list(MarkdownConnector(source_type="notes", path=tmp_path).load())
    (tmp_path / "a.md").write_text("# A\n\nbody text")
    clean = list(MarkdownConnector(source_type="notes", path=tmp_path).load())

    chunker = MarkdownChunker(source_type="notes")
    assert [c.content_hash for c in chunker.chunk(dirty[0])] == [
        c.content_hash for c in chunker.chunk(clean[0])
    ]


def test_form_feed_is_preserved(tmp_path: Path) -> None:
    # Deliberately not stripped alongside NUL: it is a real page separator in
    # extracted PDF text and carries structure worth keeping.
    assert "\x0c" in "".join(_chunk_body(tmp_path, "# A\n\npage one\x0cpage two"))


# --- chunk overlap ----------------------------------------------------------
#
# `OVERLAP_TOKENS` was declared in util/tokens.py from the start and never
# wired into the chunker, so chunks were butt-joined: text spanning a boundary
# was cut in half and matched neither piece. Nothing asserted the overlap
# existed, which is exactly how it stayed missing. These tests are that
# assertion.


def _long_body(paragraphs: int = 14, filler: str = "word") -> str:
    # Each paragraph is comfortably under the cap; together they exceed it, so
    # the size-splitter runs and has real paragraph boundaries to choose from.
    return "# T\n\n" + "\n\n".join(
        f"Paragraph {i} begins here {' '.join([filler] * 60)} and paragraph {i} ends."
        for i in range(paragraphs)
    )


def test_adjacent_chunks_overlap(tmp_path: Path) -> None:
    pieces = chunk_markdown_body(_long_body())

    assert len(pieces) > 1
    # The tail of each piece must reappear at the head of the next.
    overlaps = []
    for a, b in pairwise(pieces):
        shared = 0
        for n in range(min(len(a), len(b)), 0, -1):
            if b.startswith(a[-n:]):
                shared = n
                break
        overlaps.append(shared)
    assert all(o > 0 for o in overlaps), overlaps


def test_text_spanning_a_boundary_survives_in_one_piece() -> None:
    # The failure this prevents: a phrase split across two chunks matches
    # neither, so the document becomes unfindable by its own sentence.
    marker = "the migration slipped to Q3 because of the vendor"
    body = "# T\n\n" + "\n\n".join(f"Filler paragraph {i}. " + "pad " * 90 for i in range(8))
    body += "\n\n" + marker + "\n\n" + "\n\n".join(f"Tail {i}. " + "pad " * 90 for i in range(8))

    pieces = chunk_markdown_body(body)

    assert any(marker in p for p in pieces), "phrase was split across every chunk"


def test_overlap_does_not_prevent_forward_progress() -> None:
    # Paragraph breaks closer together than OVERLAP_CHARS would make the
    # step-back land at or before the cursor. Without the max() guard this
    # loops forever rather than failing, so a timeout here is the real signal.
    body = "# T\n\n" + "\n\n".join("x" * 20 for _ in range(600))

    pieces = chunk_markdown_body(body)

    assert pieces
    assert all(p.strip() for p in pieces)


def test_overlap_never_exceeds_the_chunk_cap() -> None:
    from corpus.chunkers.markdown import MAX_CHUNK_CHARS

    pieces = chunk_markdown_body(_long_body(paragraphs=30))

    assert all(len(p) <= MAX_CHUNK_CHARS for p in pieces), [len(p) for p in pieces]


def test_short_body_is_still_a_single_chunk() -> None:
    # Overlap must not manufacture extra chunks for content under the cap.
    assert len(chunk_markdown_body("# T\n\nOne short paragraph.")) == 1
