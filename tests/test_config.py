from __future__ import annotations

from pathlib import Path

import pytest

from corpus.config import CorpusConfig


def test_load_minimal_config(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("""
[corpus]
db_path = "./test.db"

[[sources]]
name = "notes"
type = "markdown"
path = "~/notes"
""")
    config = CorpusConfig.load(cfg)
    assert config.db_path == Path("./test.db")
    assert len(config.sources) == 1
    assert config.sources[0].name == "notes"
    assert config.sources[0].type == "markdown"


def test_load_with_references(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("""
[corpus]
db_path = "./test.db"

[[references]]
pattern = '\\bDOC-\\d+\\b'
source_type = "tickets"
description = "Doc-style IDs"
""")
    config = CorpusConfig.load(cfg)
    assert len(config.references) == 1
    compiled = config.compiled_references()
    assert len(compiled) == 1
    pattern, source_type = compiled[0]
    assert source_type == "tickets"
    assert pattern.search("see DOC-42 for details")


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CorpusConfig.load(tmp_path / "nonexistent.toml")


def test_invalid_source_name_pattern(tmp_path: Path) -> None:
    """source_type names must be lowercase identifiers — UPPERCASE rejected."""
    from pydantic import ValidationError

    cfg = tmp_path / "corpus.toml"
    cfg.write_text("""
[corpus]
db_path = "./test.db"

[[sources]]
name = "BAD-NAME"
type = "markdown"
path = "~/notes"
""")
    with pytest.raises(ValidationError):
        CorpusConfig.load(cfg)


def test_defaults_apply(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("")
    config = CorpusConfig.load(cfg)
    assert config.embedder.model == "voyage-3-large"
    assert config.embedder.dim == 1024
    assert config.retriever.top_k == 5
    assert config.retriever.max_per_source_type == 3
