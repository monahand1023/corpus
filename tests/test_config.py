from __future__ import annotations

from pathlib import Path

import pytest

import corpus.config
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
    assert config.embedder.model == "voyage-4-large"
    assert config.embedder.dim == 1024
    assert config.retriever.top_k == 5
    assert config.retriever.max_per_source_type == 3
    assert config.reranker.device == "cpu"


def test_reranker_device_override(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("""
[corpus]
db_path = "./test.db"

[reranker]
device = "mps"
""")
    config = CorpusConfig.load(cfg)
    assert config.reranker.device == "mps"


def test_performance_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("")
    config = CorpusConfig.load(cfg)
    assert config.performance.cache_size_mb == 64
    assert config.performance.mmap_size_mb == 1024
    assert config.performance.temp_store_memory is True


def test_performance_override(tmp_path: Path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text("""
[corpus]
db_path = "./test.db"

[performance]
cache_size_mb = 16
mmap_size_mb = 4096
temp_store_memory = false
""")
    config = CorpusConfig.load(cfg)
    assert config.performance.cache_size_mb == 16
    assert config.performance.mmap_size_mb == 4096
    assert config.performance.temp_store_memory is False


def test_performance_sizes_must_be_positive() -> None:
    from pydantic import ValidationError

    from corpus.config import PerformanceConfig

    with pytest.raises(ValidationError):
        PerformanceConfig(cache_size_mb=0)
    with pytest.raises(ValidationError):
        PerformanceConfig(mmap_size_mb=-1)


def test_the_contextual_section_is_actually_read(tmp_path: Path) -> None:
    """`[contextual]` used to be declared on CorpusConfig and never read.

    A corpus.toml setting min_tokens / model / window_size parsed as valid
    TOML and was silently discarded -- the defaults always won, with no error
    and no warning. It was caught only because a dry-run's chunk count did not
    match an independent hand count, and it would otherwise have spent real
    money contextualizing chunks the operator had explicitly excluded.
    """
    cfg_path = tmp_path / "corpus.toml"
    cfg_path.write_text(
        '[corpus]\ndb_path = "./x.db"\n\n'
        "[contextual]\nmin_tokens = 260\nwindow_size = 12\n"
    )

    config = CorpusConfig.load(cfg_path)

    assert config.contextual.min_tokens == 260
    assert config.contextual.window_size == 12


def test_every_declared_section_is_read_from_the_toml() -> None:
    """Structural guard for the whole class of bug, not just `contextual`.

    A field declared on CorpusConfig but absent from `load`'s merge dict is
    unconfigurable and fails silently. Reading the source is the only way to
    catch that: every such field still type-checks, still validates, and still
    returns its default.
    """
    import ast
    import re

    source = Path(corpus.config.__file__).read_text()
    declared = [
        node.target.id
        for cls in ast.walk(ast.parse(source))
        if isinstance(cls, ast.ClassDef) and cls.name == "CorpusConfig"
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    read = set(re.findall(r'raw\.get\("([a-z_]+)"', source))
    # db_path comes out of the [corpus] section rather than one of its own.
    missing = [d for d in declared if d not in read and d != "db_path"]

    assert not missing, f"declared on CorpusConfig but never read from TOML: {missing}"


# ---------------------------------------------------------------------------
# [[references]] patterns are regexes, and were never validated as such
# ---------------------------------------------------------------------------


def test_an_invalid_reference_regex_is_a_config_error_not_a_later_traceback(
    tmp_path,
) -> None:
    """`ReferencePattern.pattern` is compiled lazily, by whichever command
    happens to call `compiled_references()` first. A typo in corpus.toml
    therefore surfaced as a raw `re.error` traceback at CLI startup -- and in
    the MCP server, as a process that dies before saying anything, which the
    client reports only as "server failed to start"."""
    cfg = tmp_path / "corpus.toml"
    cfg.write_text(
        '[corpus]\ndb_path = "x.db"\n\n'
        '[[references]]\npattern = "TICKET-[0-9"\nsource_type = "jira"\n'
    )
    with pytest.raises(ConfigError) as exc:
        CorpusConfig.load(cfg)
    assert "TICKET-[0-9" in str(exc.value)


def test_a_valid_reference_regex_still_loads(tmp_path) -> None:
    cfg = tmp_path / "corpus.toml"
    cfg.write_text(
        '[corpus]\ndb_path = "x.db"\n\n'
        '[[references]]\npattern = "TICKET-[0-9]+"\nsource_type = "jira"\n'
    )
    loaded = CorpusConfig.load(cfg)
    pattern, source_type = loaded.compiled_references()[0]
    assert pattern.search("see TICKET-42")
    assert source_type == "jira"
