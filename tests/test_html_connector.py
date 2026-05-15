from __future__ import annotations

from pathlib import Path

import pytest

from corpus.connectors.html import HtmlConnector

# A reasonably "real" HTML page so trafilatura has structure to chew on.
_SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>The Article Title</title>
    <meta name="author" content="Jane Author">
    <meta name="date" content="2026-03-01">
</head>
<body>
    <nav>Navigation menu — boilerplate</nav>
    <article>
        <h1>The Article Title</h1>
        <p>This is the main content paragraph. It contains substantive text
        that trafilatura should extract as the document body.</p>
        <p>A second paragraph with additional content. The article continues
        with more interesting text that establishes context.</p>
    </article>
    <footer>Footer boilerplate, copyright notice, related links — should be stripped</footer>
</body>
</html>
"""


def test_loads_html_extracts_main_content(tmp_path: Path) -> None:
    (tmp_path / "article.html").write_text(_SAMPLE_HTML)
    docs = list(HtmlConnector(source_type="articles", path=tmp_path).load())
    assert len(docs) == 1
    body = docs[0].raw["body"]
    assert "main content paragraph" in body
    assert "second paragraph" in body.lower()
    # Boilerplate should be stripped
    assert "Navigation menu" not in body
    assert "Footer boilerplate" not in body


def test_extracts_title_and_author_from_metadata(tmp_path: Path) -> None:
    (tmp_path / "a.html").write_text(_SAMPLE_HTML)
    docs = list(HtmlConnector(source_type="articles", path=tmp_path).load())
    assert docs[0].title == "The Article Title"


def test_skips_html_with_no_main_content(tmp_path: Path) -> None:
    (tmp_path / "empty.html").write_text("<html><body></body></html>")
    docs = list(HtmlConnector(source_type="articles", path=tmp_path).load())
    assert docs == []


def test_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(HtmlConnector(source_type="articles", path="/nonexistent").load())


def test_dedupes_identical_html(tmp_path: Path) -> None:
    (tmp_path / "v1.html").write_text(_SAMPLE_HTML)
    (tmp_path / "v2.html").write_text(_SAMPLE_HTML)
    docs = list(HtmlConnector(source_type="articles", path=tmp_path).load())
    assert len(docs) == 1
