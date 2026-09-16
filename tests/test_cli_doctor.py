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


# --- a check that examined nothing must say so, not imply a pass -------------


def _index(tmp_path, rows) -> str:
    import sqlite3

    p = tmp_path / "index.db"
    c = sqlite3.connect(p)
    c.executescript(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, source_type TEXT NOT NULL,"
        " source_key TEXT NOT NULL, content TEXT NOT NULL);"
    )
    c.executemany(
        "INSERT INTO chunks (id, source_type, source_key, content) VALUES (?,?,?,?)",
        [(str(i), st, k, t) for i, (st, k, t) in enumerate(rows)],
    )
    c.commit()
    c.close()
    return str(p)


def test_index_quality_on_an_empty_index_does_not_report_ok(tmp_path, capsys) -> None:
    """It printed "[ ok ] 0 chunks, no transcription artefacts".

    A clean bill of health having examined nothing -- which is what a scan
    pointed at the wrong database looks like.
    """
    from corpus.cli.doctor import _check_index_quality

    ok = _check_index_quality(_index(tmp_path, []))
    out = capsys.readouterr().out

    assert "[  ok  ]" not in out, out
    assert "examined nothing" in out, out
    assert ok is False, "a vacuous check is not a passing check"


def test_index_quality_on_a_real_index_still_reports_ok(tmp_path, capsys) -> None:
    from corpus.cli.doctor import _check_index_quality

    ok = _check_index_quality(
        _index(tmp_path, [("notes", "a.md", "we fed the ducks by the pond for an hour")])
    )
    out = capsys.readouterr().out

    assert "[  ok  ]" in out, out
    assert ok is True


def test_a_broken_detector_fails_the_doctor_rather_than_being_skipped(
    tmp_path, capsys, monkeypatch
) -> None:
    """The swallow-as-skip trap, one level up.

    `_check_index_quality` catches broad Exception and reports SKIPPED. A
    detector that cannot fire would be swallowed as "skipped (could not
    scan)" -- turning the loudest possible signal into the quietest, which is
    the pattern this whole mechanism exists to remove.
    """
    from corpus.cli.doctor import _check_index_quality

    monkeypatch.setattr(
        "corpus.survey.index_quality.subtitle_boilerplate", lambda _t: False
    )
    ok = _check_index_quality(
        _index(tmp_path, [("notes", "a.md", "ordinary text long enough to scan")])
    )
    out = capsys.readouterr().out

    assert ok is False, "a broken detector must fail, not skip"
    assert "SKIPPED" not in out, out
    assert "[ FAIL ]" in out, out
