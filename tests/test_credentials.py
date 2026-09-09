"""corpus.credentials: predictable `.env` resolution.

Uses synthetic `.env` files under `tmp_path` throughout -- never a real
credential value. Every test that exercises the cwd-fallback branch
(`find_dotenv(usecwd=True)`) chdirs into a fresh `tmp_path`, which pytest
places well outside any directory that might hold a real `.env` (this repo's
own included), so the upward filesystem walk stays isolated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.credentials import (
    MissingCredentialError,
    describe_search,
    require_env,
    resolve_dotenv,
)

VAR = "CORPUS_TEST_CREDENTIAL"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with VAR unset, regardless of the real shell env."""
    monkeypatch.delenv(VAR, raising=False)


# ---------------------------------------------------------------------------
# resolve_dotenv precedence
# ---------------------------------------------------------------------------


def test_env_var_already_set_wins_over_conflicting_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precedence 1: an already-set variable is never overwritten by a .env
    file, even one sitting right beside --config."""
    cfg_dir = tmp_path / "consumer"
    cfg_dir.mkdir()
    config_path = cfg_dir / "corpus.toml"
    config_path.write_text("[corpus]\n")
    (cfg_dir / ".env").write_text(f"{VAR}=from-dotenv\n")

    monkeypatch.setenv(VAR, "from-real-shell")
    resolve_dotenv(config_path)

    assert __import__("os").environ[VAR] == "from-real-shell"


def test_loads_dotenv_beside_config_when_cwd_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precedence 2: a .env next to --config is found and loaded, even when
    the process is run from an unrelated cwd with no .env of its own."""
    cfg_dir = tmp_path / "archive-repo"
    cfg_dir.mkdir()
    config_path = cfg_dir / "docs.toml"
    config_path.write_text("[corpus]\n")
    (cfg_dir / ".env").write_text(f"{VAR}=from-config-sibling\n")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loaded = resolve_dotenv(config_path)

    import os

    assert os.environ[VAR] == "from-config-sibling"
    assert (cfg_dir / ".env").resolve() in loaded


def test_falls_back_to_cwd_dotenv_when_no_config_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precedence 3: with no --config (or no .env beside it), a .env found by
    walking up from the current working directory is used."""
    cwd = tmp_path / "somewhere"
    cwd.mkdir()
    (cwd / ".env").write_text(f"{VAR}=from-cwd\n")
    monkeypatch.chdir(cwd)

    resolve_dotenv(None)

    import os

    assert os.environ[VAR] == "from-cwd"


def test_config_sibling_takes_precedence_over_cwd_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both a config-sibling .env and a cwd .env define the same
    variable, the config-sibling value wins -- it is the more specific,
    intentional pairing (an archive repo's own docs.toml + .env)."""
    cfg_dir = tmp_path / "archive-repo"
    cfg_dir.mkdir()
    config_path = cfg_dir / "docs.toml"
    config_path.write_text("[corpus]\n")
    (cfg_dir / ".env").write_text(f"{VAR}=from-config-sibling\n")

    cwd = tmp_path / "run-from-here"
    cwd.mkdir()
    (cwd / ".env").write_text(f"{VAR}=from-cwd\n")
    monkeypatch.chdir(cwd)

    resolve_dotenv(config_path)

    import os

    assert os.environ[VAR] == "from-config-sibling"


def test_nothing_found_leaves_variable_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "empty"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    loaded = resolve_dotenv(None)

    import os

    assert VAR not in os.environ
    assert loaded == []


def test_config_path_pointing_at_missing_file_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --config path need not exist yet (e.g. a typo, or corpus.toml not
    written yet) -- resolution should not blow up, just find nothing beside
    it and fall back to the cwd search."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text(f"{VAR}=from-cwd\n")
    monkeypatch.chdir(cwd)

    resolve_dotenv(tmp_path / "nonexistent" / "corpus.toml")

    import os

    assert os.environ[VAR] == "from-cwd"


def test_resolve_dotenv_idempotent_across_repeated_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text(f"{VAR}=from-cwd\n")
    monkeypatch.chdir(cwd)

    resolve_dotenv(None)
    resolve_dotenv(None)  # must not raise, must not change the outcome

    import os

    assert os.environ[VAR] == "from-cwd"


# ---------------------------------------------------------------------------
# require_env
# ---------------------------------------------------------------------------


def test_require_env_returns_value_once_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "consumer"
    cfg_dir.mkdir()
    config_path = cfg_dir / "corpus.toml"
    config_path.write_text("[corpus]\n")
    (cfg_dir / ".env").write_text(f"{VAR}=resolved-value\n")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert require_env(VAR, config_path=config_path) == "resolved-value"


def test_require_env_checks_multiple_names_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the Gemini pattern: GEMINI_API_KEY or GOOGLE_API_KEY, whichever
    is actually set."""
    monkeypatch.setenv("CORPUS_TEST_FALLBACK", "value-from-second-name")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    value = require_env("CORPUS_TEST_PRIMARY", "CORPUS_TEST_FALLBACK", config_path=None)

    assert value == "value-from-second-name"


def test_require_env_raises_with_variable_name_and_search_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "consumer"
    cfg_dir.mkdir()
    config_path = cfg_dir / "corpus.toml"
    config_path.write_text("[corpus]\n")
    # deliberately no .env anywhere

    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    with pytest.raises(MissingCredentialError) as exc_info:
        require_env(VAR, config_path=config_path)

    message = str(exc_info.value)
    assert VAR in message
    assert str(cfg_dir) in message  # names the location it looked beside --config
    assert "current directory" in message


def test_require_env_error_names_every_variable_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "empty"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    with pytest.raises(MissingCredentialError) as exc_info:
        require_env("CORPUS_TEST_A", "CORPUS_TEST_B", config_path=None)

    message = str(exc_info.value)
    assert "CORPUS_TEST_A" in message
    assert "CORPUS_TEST_B" in message


# ---------------------------------------------------------------------------
# describe_search
# ---------------------------------------------------------------------------


def test_describe_search_reports_missing_sibling_and_found_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "consumer"
    cfg_dir.mkdir()
    config_path = cfg_dir / "corpus.toml"
    config_path.write_text("[corpus]\n")

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text(f"{VAR}=x\n")
    monkeypatch.chdir(cwd)

    description = describe_search(config_path)

    assert "no .env" in description  # beside --config: none found
    assert str(cwd) in description  # cwd: found, and named


# ---------------------------------------------------------------------------
# Wired into a real CLI entry point (not just the standalone helper)
# ---------------------------------------------------------------------------


def test_ingest_cli_loads_dotenv_beside_its_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """corpus-ingest's main() must resolve credentials through
    corpus.credentials, using the --config it was actually given -- not a
    bare load_dotenv() at import time. The `hash` embedder needs no real key,
    so this proves the wiring (config-sibling .env reaches the process
    environment through the full CLI, not just the standalone helper) without
    touching a real provider or the network.
    """
    from corpus.cli import ingest as ingest_cli

    marker = "CORPUS_TEST_INGEST_MARKER"
    monkeypatch.delenv(marker, raising=False)

    archive = tmp_path / "archive-repo"
    archive.mkdir()
    notes_dir = archive / "notes"
    notes_dir.mkdir()
    (notes_dir / "hello.md").write_text("# Hello\n\nSome content.\n")
    (archive / ".env").write_text(f"{marker}=from-config-sibling\n")

    config_path = archive / "corpus.toml"
    config_path.write_text(
        f'[corpus]\ndb_path = "{(archive / "test.db").as_posix()}"\n'
        '[embedder]\nprovider = "hash"\nmodel = "hash-v1"\ndim = 256\n'
        '\n[[sources]]\n'
        'name = "notes"\n'
        'type = "markdown"\n'
        f'path = "{notes_dir.as_posix()}"\n'
    )

    # Run from an unrelated cwd with no .env of its own -- only the
    # --config-adjacent .env should be able to supply the marker.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    argv = ["corpus-ingest", "--config", str(config_path), "--source", "notes"]
    monkeypatch.setattr("sys.argv", argv)
    try:
        exit_code = ingest_cli.main()
        assert exit_code == 0
        import os

        assert os.environ.get(marker) == "from-config-sibling"
    finally:
        __import__("os").environ.pop(marker, None)
