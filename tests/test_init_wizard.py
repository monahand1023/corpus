from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from corpus.cli.init import (
    WizardAnswers,
    _normalize_source_name,
    _render_corpus_toml,
    _render_env,
)
from corpus.cli.init import (
    main as init_main,
)
from corpus.config import CorpusConfig


def test_normalize_source_name() -> None:
    assert _normalize_source_name("My Notes") == "my_notes"
    assert _normalize_source_name("UPPER") == "upper"
    assert _normalize_source_name("with-dashes") == "with_dashes"
    assert _normalize_source_name("123leading") == "leading"
    assert _normalize_source_name("") == "notes"  # falls back


def test_render_corpus_toml_is_valid(tmp_path: Path) -> None:
    answers = WizardAnswers(
        db_path=Path("./corpus.db"),
        source_name="notes",
        source_type="markdown",
        source_path=Path("~/Documents/notes"),
        provider="voyage",
        model="voyage-3-large",
        dim=1024,
    )
    rendered = _render_corpus_toml(answers)
    toml_file = tmp_path / "corpus.toml"
    toml_file.write_text(rendered)
    # Load via the real CorpusConfig — proves the rendered TOML is parseable
    config = CorpusConfig.load(toml_file)
    assert config.embedder.provider == "voyage"
    assert config.embedder.dim == 1024
    assert len(config.sources) == 1
    assert config.sources[0].name == "notes"
    assert config.sources[0].type == "markdown"


def test_render_env_includes_provider_specific_key() -> None:
    answers = WizardAnswers(
        db_path=Path("./corpus.db"),
        source_name="notes",
        source_type="markdown",
        source_path=Path("./notes"),
        provider="gemini",
        model="gemini-embedding-001",
        dim=1536,
    )
    out = _render_env(answers)
    assert "GEMINI_API_KEY=" in out
    assert "ANTHROPIC_API_KEY=" in out
    assert "aistudio" in out  # signup URL


def test_main_refuses_to_overwrite_existing(tmp_path: Path) -> None:
    (tmp_path / "corpus.toml").write_text("existing")
    with patch("sys.argv", ["corpus-init", "--out-dir", str(tmp_path)]):
        rc = init_main()
    assert rc == 1
    # Original content preserved
    assert (tmp_path / "corpus.toml").read_text() == "existing"


def test_main_writes_files_with_force(tmp_path: Path) -> None:
    """End-to-end through main() with mocked input() — exercises the
    interactive flow without prompting the test runner."""
    (tmp_path / "corpus.toml").write_text("existing")
    inputs = iter([
        "./corpus.db",          # db path
        "my_archive",           # source name
        "markdown",             # source type
        str(tmp_path),          # source path (use tmp_path so it exists)
        "voyage",               # provider
        "voyage-3-large",       # model
        "1024",                 # dim
    ])
    with patch("sys.argv", ["corpus-init", "--out-dir", str(tmp_path), "--force"]), \
         patch("builtins.input", lambda _prompt: next(inputs)):
        rc = init_main()
    assert rc == 0
    written = (tmp_path / "corpus.toml").read_text()
    assert 'name = "my_archive"' in written
    assert 'provider = "voyage"' in written
    # And it should parse cleanly
    config = CorpusConfig.load(tmp_path / "corpus.toml")
    assert config.sources[0].name == "my_archive"
