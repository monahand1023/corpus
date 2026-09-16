"""Tests for corpus-doctor's query-log resolution.

The check these cover was skipped on every run of five live archives, because
`--query-log` was the only way to supply a path and nobody passed it. The
command printed "A skipped check is not a passing one" and then skipped it by
default -- a check that never runs reads exactly like a check that passes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from corpus.cli.doctor import _resolve_query_logs
from corpus.config import CorpusConfig, QueryLogConfig
from corpus.query_log import configured_log_path


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(query_log=kw.pop("query_log", []), config=kw.pop("config", None))


def test_an_explicit_flag_still_wins(tmp_path) -> None:
    log = tmp_path / "given.jsonl"
    log.write_text("")
    paths, reason = _resolve_query_logs(_args(query_log=[str(log)]))
    assert paths == [str(log)] and reason == ""


def test_without_a_config_the_reason_names_both_ways_in() -> None:
    paths, reason = _resolve_query_logs(_args())
    assert paths == []
    assert "--config" in reason and "--query-log" in reason


def test_logging_disabled_says_so_rather_than_blaming_a_flag(tmp_path, monkeypatch) -> None:
    # "You forgot a flag" and "logging is off" need different fixes, and only
    # one of them was ever reported.
    cfg = CorpusConfig(db_path=tmp_path / "c.db", query_log=QueryLogConfig(enabled=False))
    monkeypatch.setattr("corpus.cli._common.load_config_or_exit", lambda _p: cfg)
    paths, reason = _resolve_query_logs(_args(config="corpus.toml"))
    assert paths == []
    assert "OFF in config" in reason and "enabled = true" in reason


def test_logging_on_but_nothing_served_is_its_own_reason(tmp_path, monkeypatch) -> None:
    cfg = CorpusConfig(db_path=tmp_path / "c.db", query_log=QueryLogConfig(enabled=True))
    monkeypatch.setattr("corpus.cli._common.load_config_or_exit", lambda _p: cfg)
    paths, reason = _resolve_query_logs(_args(config="corpus.toml"))
    assert paths == []
    assert "nothing served" in reason


def test_an_existing_configured_log_is_audited_without_any_flag(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "queries.jsonl").write_text('{"query": "x"}\n')
    cfg = CorpusConfig(db_path=tmp_path / "c.db", query_log=QueryLogConfig(enabled=True))
    monkeypatch.setattr("corpus.cli._common.load_config_or_exit", lambda _p: cfg)
    paths, reason = _resolve_query_logs(_args(config="corpus.toml"))
    assert paths == [str(tmp_path / "queries.jsonl")] and reason == ""


def test_the_auditor_resolves_the_same_path_the_writer_appends_to(tmp_path) -> None:
    """The property that makes this safe to do at all.

    Resolving the path independently would be WORSE than not resolving it: the
    auditor would read a file nothing writes to and report it clean. Both
    sides call `configured_log_path`, so this pins that they agree.
    """
    cfg = CorpusConfig(db_path=tmp_path / "data" / "c.db", query_log=QueryLogConfig(enabled=True))
    assert configured_log_path(cfg) == tmp_path / "data" / "queries.jsonl"

    explicit = CorpusConfig(
        db_path=tmp_path / "data" / "c.db",
        query_log=QueryLogConfig(enabled=True, path=Path("/somewhere/else.jsonl")),
    )
    assert configured_log_path(explicit) == Path("/somewhere/else.jsonl")
    assert configured_log_path(CorpusConfig(db_path=tmp_path / "c.db")) is None
