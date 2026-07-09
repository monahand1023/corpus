from __future__ import annotations

from pathlib import Path

import pytest

from corpus.config import ConfigError, CorpusConfig


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
    with pytest.raises(ConfigError):
        CorpusConfig.load(tmp_path / "nonexistent.toml")


def test_invalid_source_name_pattern(tmp_path: Path) -> None:
    """source_type names must be lowercase identifiers — UPPERCASE rejected."""
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("""
[corpus]
db_path = "./test.db"

[[sources]]
name = "BAD-NAME"
type = "markdown"
path = "~/notes"
""")
    with pytest.raises(ConfigError):
        CorpusConfig.load(cfg)


def test_embedder_dim_must_be_positive() -> None:
    from pydantic import ValidationError

    from corpus.config import EmbedderConfig

    with pytest.raises(ValidationError):
        EmbedderConfig(dim=0)
    with pytest.raises(ValidationError):
        EmbedderConfig(dim=-5)


def test_load_config_or_exit_clean_message_on_bad_toml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from corpus.cli._common import load_config_or_exit

    bad = tmp_path / "corpus.toml"
    bad.write_text("this is = = not valid toml")
    with pytest.raises(SystemExit) as e:
        load_config_or_exit(bad)
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "corpus.toml" in err


def test_load_config_or_exit_clean_message_on_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from corpus.cli._common import load_config_or_exit

    with pytest.raises(SystemExit) as e:
        load_config_or_exit(tmp_path / "nope.toml")
    assert e.value.code == 1
    assert "Traceback" not in capsys.readouterr().err


def test_defaults_apply(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("")
    config = CorpusConfig.load(cfg)
    assert config.embedder.model == "voyage-3-large"
    assert config.embedder.dim == 1024
    assert config.retriever.top_k == 5
    assert config.retriever.max_per_source_type == 3
