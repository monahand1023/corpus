from __future__ import annotations

from corpus.config import SourceConfig
from corpus.connectors.registry import CONNECTOR_REGISTRY, build_pipeline


def test_all_builtin_types_registered() -> None:
    assert "markdown" in CONNECTOR_REGISTRY
    assert "text" in CONNECTOR_REGISTRY
    assert "pdf" in CONNECTOR_REGISTRY
    assert "html" in CONNECTOR_REGISTRY


def test_build_markdown(tmp_path) -> None:
    cfg = SourceConfig(name="notes", type="markdown", path=str(tmp_path))
    connector, chunker = build_pipeline(cfg)
    assert connector.source_type == "notes"
    assert chunker.source_type == "notes"


def test_build_text(tmp_path) -> None:
    cfg = SourceConfig(name="notes", type="text", path=str(tmp_path))
    connector, _chunker = build_pipeline(cfg)
    assert connector.source_type == "notes"


def test_unknown_type_raises() -> None:
    cfg = SourceConfig(name="foo", type="nonexistent", path="/tmp")
    import pytest
    with pytest.raises(ValueError, match="not registered"):
        build_pipeline(cfg)
