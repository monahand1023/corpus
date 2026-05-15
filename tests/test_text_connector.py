from __future__ import annotations

from pathlib import Path

import pytest

from corpus.connectors.markdown import MarkdownChunker
from corpus.connectors.text import TextConnector


def test_loads_plain_text(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Hello world.\n\nSecond paragraph.")
    (tmp_path / "b.txt").write_text("Another file.")
    (tmp_path / "ignore.md").write_text("Markdown file, not txt.")

    docs = list(TextConnector(source_type="notes", path=tmp_path).load())
    titles = {d.title for d in docs}
    assert titles == {"a", "b"}


def test_skips_empty_files(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "real.txt").write_text("Content here.")
    docs = list(TextConnector(source_type="notes", path=tmp_path).load())
    assert len(docs) == 1
    assert docs[0].title == "real"


def test_dedupes_identical_files(tmp_path: Path) -> None:
    body = "Identical content"
    (tmp_path / "v1.txt").write_text(body)
    (tmp_path / "v2.txt").write_text(body)
    docs = list(TextConnector(source_type="notes", path=tmp_path).load())
    assert len(docs) == 1


def test_chunker_emits_chunks_from_text_body(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Some content.\n\nMore content.")
    docs = list(TextConnector(source_type="notes", path=tmp_path).load())
    chunks = MarkdownChunker(source_type="notes").chunk(docs[0])
    assert chunks
    assert "Some content" in chunks[0].content


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(TextConnector(source_type="notes", path="/nonexistent").load())


def test_custom_glob(tmp_path: Path) -> None:
    (tmp_path / "in.txt").write_text("yes")
    (tmp_path / "out.log").write_text("no")
    docs = list(
        TextConnector(source_type="notes", path=tmp_path, glob="*.txt").load()
    )
    titles = {d.title for d in docs}
    assert titles == {"in"}
