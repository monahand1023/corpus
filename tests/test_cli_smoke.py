"""Tests for the parts of corpus-smoke that run without spawning a server.

The spawning half is exercised by running the command against a real config;
what is unit-testable here is the target discovery, which is where the two
defects found while writing it lived: resolving the launcher's working
directory, and resolving the script to run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from corpus.cli.smoke import (
    _is_corpus_server,
    _launch_dir,
    _looks_empty,
    _servers_from_claude_config,
    _sibling_script,
    _text_of,
)

# --- _launch_dir -----------------------------------------------------------
# `uv --directory <repo> run corpus-mcp` is how these servers are configured.
# The directory is not cosmetic: the `.env` holding the embedder API key sits
# beside the config there, so a server spawned in the wrong cwd exits during
# initialize with a credentials error. A smoke test that got this wrong would
# report a failure against a server that works perfectly under the real client.


def test_launch_dir_reads_uv_directory_flag() -> None:
    args = ["--directory", "/repo/a mail consumer", "run", "corpus-mcp"]
    assert _launch_dir(args) == Path("/repo/a mail consumer")


def test_launch_dir_accepts_the_other_launcher_spellings() -> None:
    assert _launch_dir(["--project", "/repo/a", "run", "x"]) == Path("/repo/a")
    assert _launch_dir(["-C", "/repo/b", "run", "x"]) == Path("/repo/b")


def test_launch_dir_expands_a_home_relative_directory() -> None:
    got = _launch_dir(["--directory", "~/repo/c", "run", "x"])
    assert got is not None
    assert not str(got).startswith("~")


def test_launch_dir_is_none_when_no_directory_is_given() -> None:
    assert _launch_dir(["run", "corpus-mcp", "--config", "x.toml"]) is None


def test_launch_dir_ignores_a_trailing_flag_with_no_value() -> None:
    # `[..., "--directory"]` with nothing after it must not raise IndexError.
    assert _launch_dir(["run", "--directory"]) is None


# --- _sibling_script -------------------------------------------------------
# A `uv tool install` of corpus puts a corpus-mcp in ~/.local/bin that shadows
# an editable checkout for anything resolving through PATH. It is a DIFFERENT
# BUILD: the first run of this command reported a credentials failure whose
# error text did not even match the current source, because a stale copy was
# answering. Preferring the script beside the running interpreter pins the
# smoke test to the same code the tests and the editable install use.


def test_sibling_script_prefers_the_interpreters_own_directory() -> None:
    # Every venv running these tests has a `python` beside its interpreter.
    resolved = _sibling_script(Path(sys.executable).name)
    assert resolved == sys.executable


def test_sibling_script_falls_back_to_bare_name_for_path_lookup() -> None:
    # Nothing by this name sits beside the interpreter, so PATH must be left
    # to resolve it rather than a nonexistent absolute path being returned.
    assert _sibling_script("corpus-definitely-not-installed") == (
        "corpus-definitely-not-installed"
    )


# --- _servers_from_claude_config -------------------------------------------
# Claude configs nest mcpServers under per-project keys, and the same server
# name can appear under several projects.


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "claude.json"
    path.write_text(json.dumps(payload))
    return path


def test_finds_servers_at_the_top_level(tmp_path: Path) -> None:
    cfg = _write(tmp_path, {"mcpServers": {"a": {"command": "x", "args": ["1"]}}})
    found = _servers_from_claude_config(cfg)
    assert found["a"]["command"] == "x"


def test_finds_servers_nested_under_projects(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path,
        {"projects": {"/repo/one": {"mcpServers": {"b": {"command": "y"}}}}},
    )
    assert "b" in _servers_from_claude_config(cfg)


def test_a_server_defined_twice_is_reported_once(tmp_path: Path) -> None:
    # Smoke-testing the same server once per project it appears in would
    # multiply the runtime by the number of projects for no added signal.
    cfg = _write(
        tmp_path,
        {
            "projects": {
                "/repo/one": {"mcpServers": {"dup": {"command": "first"}}},
                "/repo/two": {"mcpServers": {"dup": {"command": "second"}}},
            }
        },
    )
    found = _servers_from_claude_config(cfg)
    assert list(found) == ["dup"]


def test_entries_without_a_command_are_skipped(tmp_path: Path) -> None:
    # Remote/SSE server entries carry a url instead of a command; there is no
    # process to spawn, and treating one as spawnable would crash the run.
    cfg = _write(
        tmp_path,
        {"mcpServers": {"remote": {"url": "https://example.test"},
                        "local": {"command": "x"}}},
    )
    assert list(_servers_from_claude_config(cfg)) == ["local"]


def test_lists_inside_the_config_are_traversed(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path, {"anything": [{"mcpServers": {"deep": {"command": "z"}}}]}
    )
    assert "deep" in _servers_from_claude_config(cfg)


# --- _text_of --------------------------------------------------------------


@dataclass
class _Item:
    text: str | None


@dataclass
class _Result:
    content: list[_Item] | None


def test_text_of_joins_every_text_block() -> None:
    assert _text_of(_Result([_Item("one"), _Item("two")])) == "one\ntwo"


def test_text_of_survives_a_result_with_no_content() -> None:
    # An empty result must read as "no text", not raise -- the report is the
    # error handling, and a crash here would lose the other servers' results.
    assert _text_of(_Result(None)) == ""
    assert _text_of(object()) == ""


def test_text_of_skips_non_text_blocks() -> None:
    assert _text_of(_Result([_Item(None), _Item("kept")])) == "kept"


# --- _is_corpus_server -----------------------------------------------------
# A Claude config is shared with every other MCP server the user has
# installed. Sweeping those spawns unrelated processes and reports on software
# this project does not own, so the default sweep is scoped to this engine's
# own archives.


def test_a_consumer_running_the_engine_directly_is_included() -> None:
    assert _is_corpus_server(
        "uv", ["--directory", "/repo/sample-rag", "run", "corpus-mcp",
               "--config", "/repo/sample-rag/sample.toml"]
    )


def test_an_archive_with_its_own_entry_point_is_included() -> None:
    assert _is_corpus_server(
        "uv", ["--directory", "/repo/sample-rag", "run", "python", "-m",
               "sample_rag.mcp_server"]
    )


def test_a_live_variant_console_script_is_included() -> None:
    assert _is_corpus_server(
        "uv", ["--directory", "/repo/sample-rag", "run", "sample-rag-live"]
    )


def test_unrelated_servers_are_excluded() -> None:
    # Design tools, storefronts and auditors share the config file and have
    # nothing to do with this engine.
    assert not _is_corpus_server("npx", ["-y", "@vendor/design-tool"])
    assert not _is_corpus_server("uvx", ["some-website-auditor"])
    assert not _is_corpus_server("node", ["/opt/storefront/server.js"])


def test_a_ragged_english_word_does_not_count_as_an_archive() -> None:
    # The pattern has to key on the `_rag`/`-rag` package convention, not on
    # the letters "rag" appearing anywhere in a path.
    assert not _is_corpus_server("node", ["/opt/ragtime/server.js"])


# --- _looks_empty ----------------------------------------------------------
# The silent-success case: a server pointed at a wrong or empty database
# answers every question with a polite "nothing found" and passes any check
# that only asks whether a response arrived.


def test_a_real_result_block_is_not_empty() -> None:
    body = (
        "1. Quarterly planning notes (notes/2024-planning.md)\n"
        "   The team agreed to move the review to the first week of the month,\n"
        "   and to keep the existing budget split unchanged.\n"
    )
    assert not _looks_empty(body)


def test_common_empty_phrasings_are_caught() -> None:
    for body in [
        "No results found for that query." + " " * 200,
        "No matches in this archive." + " " * 200,
        "No emails found." + " " * 200,
        "Nothing found." + " " * 200,
        "Returned 0 documents." + " " * 200,
    ]:
        assert _looks_empty(body), body[:40]


def test_a_terse_status_line_counts_as_empty() -> None:
    # The live-mail server returned 74 characters for a probe its mailbox had
    # no answer for, and the first version of this check called that a pass.
    assert _looks_empty("Searched INBOX (0 of 1 folders matched the filter).")


def test_whitespace_only_is_empty() -> None:
    assert _looks_empty("   \n\t  ")


# --- per-server probes -----------------------------------------------------
# A server backed by a LIVE source rather than an index needs its own probe. A
# generic one against a live mailbox legitimately finds nothing, and "found
# nothing" is exactly what a broken server looks like -- so the run reported
# 5/6 healthy for a server that was working.


def _split_probes(raw_probes, names, default):
    """Mirrors the CLI's parsing; kept here so the rule is pinned."""
    probes = {}
    chosen_default = default
    for raw in raw_probes:
        name, sep, query = raw.partition("=")
        if sep and name in names:
            probes[name] = query
        else:
            chosen_default = raw
    return probes, chosen_default


def test_a_bare_probe_replaces_the_default_for_every_server() -> None:
    probes, default = _split_probes(["invoice"], {"a", "b"}, "original")
    assert probes == {}
    assert default == "invoice"


def test_a_named_probe_overrides_only_that_server() -> None:
    probes, default = _split_probes(["live=invoice"], {"live", "other"}, "original")
    assert probes == {"live": "invoice"}
    assert default == "original"


def test_named_and_bare_probes_combine() -> None:
    probes, default = _split_probes(
        ["everything", "live=invoice"], {"live", "other"}, "original"
    )
    assert probes == {"live": "invoice"}
    assert default == "everything"


def test_a_query_containing_an_equals_sign_is_not_a_server_override() -> None:
    # "revenue=2024" names no server, so it is a query, not an override.
    probes, default = _split_probes(["revenue=2024"], {"live"}, "original")
    assert probes == {}
    assert default == "revenue=2024"
