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
        # A dataclass, not a dict: corpus-eval reads these as ATTRIBUTES,
        # so a dict gold set raises AttributeError and never runs.
        'from dataclasses import dataclass\n'
        '@dataclass\n'
        'class Q:\n'
        '    query: str\n'
        '    expected_keys: list\n'
        'EVAL_QUERIES = [Q("ducks at the pond", ["a.md"])]\n'
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
    """The consumer-shadowing failure class, made visible.

    one consumer held a full copy of the transcript connector and registered it
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
    # Outside the finally, and without `or True`, which made this
    # unconditionally pass. Registry leakage between tests is a real failure
    # mode -- this test deliberately mutates a module-level dict -- and the
    # assertion guarding against it could not fail.
    assert CONNECTOR_REGISTRY["markdown"] is _BUILTIN_BUILDERS["markdown"]
    assert original is not None


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
        # An opaque synthetic path. This was a REAL filename from the owner's
        # archive -- a child's given name and the room he was filmed in -- and
        # the assertion below lifted a word straight out of it.
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
    assert "clip-0042" in out, "the nearest recording was not named"


# --- "never fired" and "cannot fire" are different facts ----------------------


def test_a_filter_gated_on_configuration_is_not_reported_as_merely_dormant(
    tmp_path, capsys
):
    """`only_unspoken_languages` is guarded by `if expected_languages and ...`,
    so with no languages supplied to corpus-transcribe it CANNOT fire. The
    dormancy check reported it beside filters that can fire and never have,
    under the heading "A filter that never fires is unnecessary or broken".

    Those need opposite responses: one is "delete this or find out why", the
    other is "you never turned it on". Reporting them identically is the same
    conflation -- could-not-run versus ran-and-found-nothing -- that this
    command exists to eliminate.
    """
    import sqlite3

    from corpus.cli.doctor import _check_filter_activity

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE no_text (path TEXT, reason TEXT, policy TEXT, duration_s REAL)"
    )
    conn.execute("CREATE TABLE dropped_windows (path TEXT, reason TEXT, policy TEXT)")
    # Enough verdicts to judge dormancy at all, none from the three quiet ones.
    for i in range(250):
        conn.execute(
            "INSERT INTO no_text VALUES (?, 'silence', 'p', 1.0)", (f"/f{i}.mov",)
        )
    conn.commit()
    conn.close()

    _check_filter_activity(str(db), policy="p")
    out = capsys.readouterr().out

    assert "only_unspoken_languages" in out
    assert "--language" in out, (
        "the report did not say the filter is unconfigured rather than dead"
    )


def test_the_configuration_note_names_a_real_filter():
    """A hand-maintained annotation beside a hand-maintained list is exactly
    what drifts. This catches a rename, and an explanation filed against the
    wrong population -- `impossible_speech_rate` is shadowed whole-file and
    perfectly live per-window, so the note is only true of one of them."""
    from corpus.cli.doctor import (
        FILTER_NOTES,
        PER_WINDOW_FILTERS,
        WHOLE_FILE_FILTERS,
    )

    known = {"whole-file": WHOLE_FILE_FILTERS, "per-window": PER_WINDOW_FILTERS}
    for population, name in FILTER_NOTES:
        assert population in known, f"{population} is not a judged population"
        assert name in known[population], f"{name} is not a {population} filter"


def test_the_per_window_population_is_still_judged_when_whole_file_is_clean(
    tmp_path, capsys
):
    """The two populations are judged in one loop, and the whole-file pass
    returned out of it.

    `if not dead: return True` ended the FUNCTION, not the iteration. So an
    archive whose whole-file filters have all fired -- the healthy case, and
    the one a maturing archive moves toward -- never had its per-window
    filters examined at all, and the report looked complete. A check that
    does not run is indistinguishable from a check that passed, which is the
    defect this whole command exists to remove.
    """
    from corpus.cli.doctor import PER_WINDOW_FILTERS, WHOLE_FILE_FILTERS
    from corpus.transcripts import store

    db = tmp_path / "transcripts.db"
    firing = [n for n in WHOLE_FILE_FILTERS if n != "only_unspoken_languages"]
    with store.open_store(db) as conn:
        # Whole-file: every filter that CAN fire has. Only the config-gated
        # one is idle, so `dead` is empty and the loop used to return here.
        for i in range(250):
            store.save_no_text(
                conn,
                f"/w/{i}.m4a",
                duration_s=1.0,
                policy="p",
                reason=firing[i % len(firing)],
            )
        # Per-window: one reason only, so the other four are genuinely dormant.
        for i in range(250):
            store.save_dropped_windows(
                conn,
                f"/w/{i}.m4a",
                [{"text": "x", "reason": "empty"}],
                policy="p",
            )

    ok = _check_filter_activity_out(str(db), capsys)

    assert "per-window" in ok, f"the per-window population was never judged:\n{ok}"
    for name in PER_WINDOW_FILTERS:
        if name != "empty":
            assert name in ok, f"{name} is dormant but was not named:\n{ok}"


def _check_filter_activity_out(db, capsys) -> str:
    from corpus.cli.doctor import _check_filter_activity

    _check_filter_activity(db, policy="p")
    return capsys.readouterr().out


def test_a_filter_the_window_pass_claims_first_is_not_called_broken(tmp_path, capsys):
    """"Never fired" and "cannot fire" again, from the other direction.

    A whole transcript that is only "Thank you for watching" never reaches the
    whole-file `subtitle_boilerplate` rule: the per-window pass drops that
    window as `caption_boilerplate` first, and the file exits as `empty`.
    Measured on the live archive -- 492 window drops for caption boilerplate
    and 436 files out as empty, against 0 whole-file subtitle_boilerplate
    under every policy that had per-window filtering, and 15 under the one
    policy that predated it.

    `impossible_speech_rate` is shadowed the same way and more strongly: the
    window check uses the same ceiling over a SHORTER span, so text that would
    fail over the file has already failed over its window.

    Calling either one "unnecessary or broken" sends someone to delete a rule
    that is doing its job one population over.
    """
    from corpus.cli.doctor import _check_filter_activity
    from corpus.transcripts import store

    db = tmp_path / "transcripts.db"
    with store.open_store(db) as conn:
        for i in range(250):
            store.save_no_text(
                conn, f"/w/{i}.m4a", duration_s=1.0, policy="p", reason="silence"
            )

    _check_filter_activity(str(db), policy="p")
    out = capsys.readouterr().out

    warned = [ln for ln in out.splitlines() if "never fired:" in ln]
    assert warned, out
    for name in ("subtitle_boilerplate", "impossible_speech_rate"):
        assert name not in " ".join(warned), (
            f"{name} is shadowed by the per-window pass, not broken:\n{out}"
        )
        assert name in out, f"{name} was dropped from the report entirely:\n{out}"


def test_the_doctor_knows_every_reason_the_pipeline_can_emit():
    """The two filter lists are hand-maintained beside a pipeline that is not.

    A reason missing from them can never be reported dormant -- so the filter
    that stopped working is exactly the one the dormancy check cannot see.
    This derives the vocabulary from the source and compares.

    Deliberately NOT a scan for every string literal: it reads the two
    functions that decide a verdict, which is where the vocabulary lives.
    """
    import ast
    from pathlib import Path

    from corpus.cli.doctor import PER_WINDOW_FILTERS, WHOLE_FILE_FILTERS

    src = Path("src/corpus/transcripts")

    def literals(path: Path, func: str) -> set[str]:
        tree = ast.parse((src / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func:
                out = set()
                for sub in ast.walk(node):
                    # `return "empty"` and `TranscriptVerdict(False, "empty")`
                    if (
                        isinstance(sub, ast.Return)
                        and isinstance(sub.value, ast.Constant)
                        and isinstance(sub.value.value, str)
                    ):
                        out.add(sub.value.value)
                    if isinstance(sub, ast.Call) and len(sub.args) > 1:
                        name = getattr(sub.func, "id", getattr(sub.func, "attr", ""))
                        if name == "TranscriptVerdict" and isinstance(
                            sub.args[1], ast.Constant
                        ):
                            out.add(sub.args[1].value)
                return out
        raise AssertionError(f"{func} not found in {path} -- it was renamed")

    whole = literals(Path("quality.py"), "judge_transcript")
    window = literals(Path("pipeline.py"), "_window_is_junk")

    assert whole <= set(WHOLE_FILE_FILTERS), (
        f"judge_transcript emits {whole - set(WHOLE_FILE_FILTERS)}, which the "
        "doctor cannot report dormant"
    )
    assert window <= set(PER_WINDOW_FILTERS), (
        f"_window_is_junk emits {window - set(PER_WINDOW_FILTERS)}, which the "
        "doctor cannot report dormant"
    )
