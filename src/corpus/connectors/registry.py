"""Connector registry — maps the `type` field from corpus.toml's [[sources]]
to a (Connector, Chunker) pair.

To add a new source type:
  1. Write a Connector and a Chunker (see `markdown.py` for the reference)
  2. Add an entry below
  3. Reference it in your corpus.toml [[sources]] with `type = "your_name"`
"""

from __future__ import annotations

from typing import Any, Protocol

from corpus.config import SourceConfig
from corpus.connectors.markdown import MarkdownChunker, MarkdownConnector


class _ConnectorFactory(Protocol):
    def __call__(self, cfg: SourceConfig) -> Any: ...


def _build_markdown(cfg: SourceConfig) -> tuple[MarkdownConnector, MarkdownChunker]:
    connector = MarkdownConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or "**/*.md",
    )
    chunker = MarkdownChunker(source_type=cfg.name)
    return connector, chunker


def _build_text(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    from corpus.connectors.text import TextConnector

    connector = TextConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or "**/*.txt",
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_pdf(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        from corpus.connectors.pdf import PdfConnector
    except ImportError as e:
        raise ImportError(
            "PDF connector requires the [pdf] extra. "
            "Install with `pip install corpus-rag[pdf]` or `uv add pypdf`."
        ) from e
    connector = PdfConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or "**/*.pdf",
    )
    return connector, MarkdownChunker(source_type=cfg.name)


def _build_html(cfg: SourceConfig) -> tuple[Any, MarkdownChunker]:
    try:
        from corpus.connectors.html import HtmlConnector
    except ImportError as e:
        raise ImportError(
            "HTML connector requires the [html] extra. "
            "Install with `pip install corpus-rag[html]` or `uv add trafilatura`."
        ) from e
    connector = HtmlConnector(
        source_type=cfg.name,
        path=cfg.path,
        glob=cfg.glob or "**/*.html",
    )
    return connector, MarkdownChunker(source_type=cfg.name)


CONNECTOR_REGISTRY: dict[str, _ConnectorFactory] = {
    "markdown": _build_markdown,
    "text": _build_text,
    "pdf": _build_pdf,
    "html": _build_html,
}


def build_pipeline(cfg: SourceConfig) -> tuple[Any, Any]:
    """Return (connector, chunker) for a configured source."""
    if cfg.type not in CONNECTOR_REGISTRY:
        raise ValueError(
            f"Source '{cfg.name}' uses type='{cfg.type}', which is not registered. "
            f"Available: {sorted(CONNECTOR_REGISTRY)}. "
            f"Add your own by editing src/corpus/connectors/registry.py."
        )
    return CONNECTOR_REGISTRY[cfg.type](cfg)
