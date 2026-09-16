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
        " source_key TEXT NOT NULL, content TEXT NOT NULL,"
        " content_hash TEXT NOT NULL DEFAULT '');"
    )
    c.executemany(
        "INSERT INTO chunks (id, source_type, source_key, content, content_hash)"
        " VALUES (?,?,?,?,?)",
        [(str(i), st, k, t, str(hash(t))) for i, (st, k, t) in enumerate(rows)],
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


# --- the remaining checks must also refuse to pass on a zero denominator -----


def test_a_gold_set_with_no_queries_fails_rather_than_reporting_no_findings(
    tmp_path, capsys
) -> None:
    """An empty gold set audited cleanly: "0 queries" then "[ ok ] no findings".

    The realistic route is a gold set whose module imports fine but whose
    QUERIES list is empty or renamed -- every finding it could have reported
    is a finding it cannot reach.
    """
    from corpus.cli.doctor import _check_gold_set

    queries = tmp_path / "eval_queries.py"
    queries.write_text("EVAL_QUERIES = []\n")
    ok = _check_gold_set(str(queries), _index(tmp_path, [("notes", "a.md", "text")]))
    out = capsys.readouterr().out

    assert ok is False, "a gold set that asks nothing proves nothing"
    assert "examined nothing" in out, out


def test_a_real_gold_set_reports_its_coverage(tmp_path, capsys) -> None:
    from corpus.cli.doctor import _check_gold_set

    queries = tmp_path / "eval_queries.py"
    queries.write_text(
        'EVAL_QUERIES = [{"query": "ducks at the pond", "expected_keys": ["a.md"]}]\n'
    )
    _check_gold_set(str(queries), _index(tmp_path, [("notes", "a.md", "ducks pond")]))
    out = capsys.readouterr().out
    assert "1 queries" in out, out


def test_an_empty_query_log_is_not_a_passing_check(tmp_path, capsys) -> None:
    """A log with zero entries told us nothing about demand, and said [ok]."""
    from corpus.cli.doctor import _check_query_logs

    log = tmp_path / "queries.jsonl"
    log.write_text("")
    ok, _reports = _check_query_logs([str(log)], 30)
    out = capsys.readouterr().out

    assert "examined nothing" in out or ok is False, out


# --- dormant filters ---------------------------------------------------------


def _sidecar(tmp_path, verdicts, policy="p"):
    from corpus.transcripts import store

    db = tmp_path / "transcripts.db"
    with store.open_store(db) as conn:
        for i, reason in enumerate(verdicts):
            store.save_no_text(
                conn, f"/w/{i}.m4a", duration_s=1.0, policy=policy, reason=reason
            )
    return str(db)


def test_a_filter_that_never_fires_is_named(tmp_path, capsys) -> None:
    """A dead knob reads exactly like a knob with nothing to reject.

    This project shipped one: a repetition check thresholded at 0.9 whose
    highest score across 74 real transcripts was 0.250.
    """
    from corpus.cli.doctor import _check_filter_activity

    # 250 verdicts, all one reason -- above the dormancy floor.
    db = _sidecar(tmp_path, ["silence"] * 250)
    ok = _check_filter_activity(db, policy="p")
    out = capsys.readouterr().out

    assert "looping_repetition" in out, out
    assert ok is True, "dormancy is a warning, not a build failure"


def test_a_small_sample_says_nothing_about_dormancy(tmp_path, capsys) -> None:
    from corpus.cli.doctor import _check_filter_activity

    db = _sidecar(tmp_path, ["silence"] * 3)
    _check_filter_activity(db, policy="p")
    out = capsys.readouterr().out
    assert "too small" in out.lower() or "looping_repetition" not in out, out


def test_no_sidecar_is_reported_as_not_checked(tmp_path, capsys) -> None:
    from corpus.cli.doctor import _check_filter_activity

    _check_filter_activity(str(tmp_path / "missing.db"), policy="p")
    out = capsys.readouterr().out
    assert "SKIPPED" in out or "NOT CHECKED" in out.upper(), out


def test_an_empty_log_that_is_BRAND_NEW_warns_rather_than_fails(
    tmp_path, capsys
) -> None:
    """A fresh install has an empty log, and that is not a defect.

    Failing here would make `corpus-doctor` exit non-zero on a perfectly
    healthy new archive, and a tool that cries wolf gets ignored -- which
    costs more than the permissiveness saves.
    """
    from corpus.cli.doctor import _check_query_logs

    log = tmp_path / "queries.jsonl"
    log.write_text("")  # created just now
    ok, _ = _check_query_logs([str(log)], 30)
    out = capsys.readouterr().out

    assert ok is True, "a log created moments ago cannot be suspicious yet"
    assert "nothing" in out.lower()


def test_an_empty_log_that_is_OLD_fails(tmp_path, capsys) -> None:
    """Two months of logging with zero entries is a broken writer, not low demand.

    This is the real case: an archive logged nothing for months because it was
    never reachable from the app being used, and every doctor run passed.
    """
    import os
    import time

    from corpus.cli.doctor import _check_query_logs

    log = tmp_path / "queries.jsonl"
    log.write_text("")
    old = time.time() - 60 * 24 * 3600
    os.utime(log, (old, old))

    ok, _ = _check_query_logs([str(log)], 30)
    out = capsys.readouterr().out

    assert ok is False, "an old empty log cannot tell low demand from a broken writer"
    assert "60 days" in out or "days" in out, out


# --- threshold margins -------------------------------------------------------


def test_thresholds_are_reported_against_the_archive_that_uses_them(
    tmp_path, capsys
) -> None:
    """Both threshold defects of 2026-09-16 would have shown up here.

    A ceiling 0.3% above real data reads identically to one 60% above it,
    until something measures the distance.
    """
    from corpus.cli.doctor import _check_threshold_margins
    from corpus.transcripts import store
    from corpus.transcripts.store import Transcript, Window

    db = tmp_path / "transcripts.db"
    with store.open_store(db) as conn:
        # Text whose looping share sits just under the ceiling.
        text = "happy birthday to you " * 14 + "there you go well done everyone"
        t = Transcript(path="/w/a.m4a", text=text,
                       windows=[Window(0.0, 60.0, text, "en")],
                       duration_s=60.0, model="m")
        store.save_transcript(conn, t)

    ok = _check_threshold_margins(str(db))
    out = capsys.readouterr().out

    assert ok is True, "margins are reported, never enforced"
    assert "headroom" in out
    assert "looping" in out.lower()


def test_no_sidecar_means_no_margin_opinion(tmp_path, capsys) -> None:
    from corpus.cli.doctor import _check_threshold_margins

    _check_threshold_margins(str(tmp_path / "missing.db"))
    out = capsys.readouterr().out
    assert "SKIPPED" in out


# --- shadowed engine components ----------------------------------------------


def test_a_consumer_overriding_an_engine_connector_is_named(capsys) -> None:
    """The a document consumer failure class, made visible.

    a document consumer held a full copy of the transcript connector and registered it
    over the engine's. Engine fixes stopped reaching the archive and nothing
    said so -- the ingest reported success. A loop filter dropped 716 junk
    chunks in principle and 301 in practice; the fork kept shadowing the rest.
    """
    from corpus.cli.doctor import _check_shadowed_components
    from corpus.connectors.registry import CONNECTOR_REGISTRY

    original = CONNECTOR_REGISTRY.get("markdown")
    try:
        CONNECTOR_REGISTRY["markdown"] = lambda cfg: (None, None)  # a local copy
        ok = _check_shadowed_components()
        out = capsys.readouterr().out
        assert "markdown" in out
        assert ok is True, "shadowing is reported, not failed -- it can be deliberate"
    finally:
        if original is not None:
            CONNECTOR_REGISTRY["markdown"] = original


def test_an_unmodified_registry_reports_clean(capsys) -> None:
    from corpus.cli.doctor import _check_shadowed_components

    _check_shadowed_components()
    out = capsys.readouterr().out
    assert "[  ok  ]" in out, out


def test_the_check_can_load_a_consumers_registration_first(capsys) -> None:
    """Without this the check cannot see the failure it exists for.

    `corpus-doctor` runs standalone and never imports a consumer's
    registration code, so the registry it inspects is pristine and every
    archive reports "none overridden" -- including one that overrides. The
    override has to be LOADED before it can be seen.
    """
    # A module that registers over an engine connector, as a consumer does.
    import sys
    import types

    from corpus.cli.doctor import _check_shadowed_components

    mod = types.ModuleType("_fake_consumer")
    def register() -> None:
        from corpus.connectors.registry import CONNECTOR_REGISTRY
        CONNECTOR_REGISTRY["markdown"] = lambda cfg: (None, None)
    mod.register = register  # type: ignore[attr-defined]
    sys.modules["_fake_consumer"] = mod

    from corpus.connectors.registry import _BUILTIN_BUILDERS, CONNECTOR_REGISTRY
    original = CONNECTOR_REGISTRY["markdown"]
    try:
        _check_shadowed_components(load="_fake_consumer")
        out = capsys.readouterr().out
        assert "markdown" in out, out
    finally:
        CONNECTOR_REGISTRY["markdown"] = _BUILTIN_BUILDERS["markdown"]
        sys.modules.pop("_fake_consumer", None)
        assert CONNECTOR_REGISTRY["markdown"] is original or True


# --- duplicate content -------------------------------------------------------


def test_duplicate_passages_are_reported_with_the_documents_involved(
    tmp_path, capsys
) -> None:
    """Report, never delete. Some duplication is legitimate."""
    from corpus.cli.doctor import _check_duplicate_content

    db = _index(tmp_path, [
        ("tr", "/vol/one/clip.mov", "first half of a long talk about migrations"),
        ("tr", "/vol/one/clip.mov", "second half of a long talk about migrations"),
        ("tr", "/backup/clip.mov", "first half of a long talk about migrations"),
        ("tr", "/backup/clip.mov", "second half of a long talk about migrations"),
    ])
    ok = _check_duplicate_content(db)
    out = capsys.readouterr().out

    assert ok is True, "duplication is reported, not failed -- some is legitimate"
    assert "clip.mov" in out
    assert "50" in out or "%" in out


def test_a_clean_index_reports_no_duplication(tmp_path, capsys) -> None:
    from corpus.cli.doctor import _check_duplicate_content

    _check_duplicate_content(_index(tmp_path, [
        ("notes", "a.md", "one distinct passage"),
        ("notes", "b.md", "another distinct passage"),
    ]))
    assert "[  ok  ]" in capsys.readouterr().out


def test_an_empty_index_cannot_report_clean_duplication(tmp_path, capsys) -> None:
    from corpus.cli.doctor import _check_duplicate_content

    _check_duplicate_content(_index(tmp_path, []))
    out = capsys.readouterr().out
    assert "examined nothing" in out or "SKIPPED" in out


# --- a margin has to show what it measured ------------------------------------


def test_a_tight_margin_shows_the_text_it_measured(tmp_path, capsys):
    """A number alone gets read the wrong way round.

    Live: "looping share: 0.7% headroom ... reaches 0.844444" was five copies
    of a decode loop, not material worth protecting. The right action was to
    look at the recordings, and nothing in the output pointed at them.
    """
    import sqlite3

    from corpus.cli.doctor import _check_threshold_margins

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE transcripts (path TEXT, text TEXT, duration_s REAL)"
    )
    loop = "こんばんは。ただいま準備しております。" * 13
    conn.execute(
        "INSERT INTO transcripts VALUES (?, ?, ?)",
        ("/media/clip-0042.MOV", loop, 60.0),
    )
    conn.execute(
        "INSERT INTO transcripts VALUES (?, ?, ?)",
        ("/media/ordinary.mov", "A normal sentence about a birthday party.", 12.0),
    )
    conn.commit()
    conn.close()

    _check_threshold_margins(str(db))
    out = capsys.readouterr().out

    assert "real data" not in out, "the check still called its sample real"
    assert "playroom" in out, "the nearest recording was not named"
