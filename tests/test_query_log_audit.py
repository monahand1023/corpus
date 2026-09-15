"""Tests for the query-log audit.

The log is an instrument, and it drifted twice without anyone noticing. Both
times the count went UP, which is what a log being used looks like, and both
times the archive with the most entries had almost no real use.
"""

from __future__ import annotations

import json
from pathlib import Path

from corpus.eval.query_log_audit import (
    audit_query_log,
    cross_archive_overlap,
    read_queries,
)


def _kinds(report) -> set[str]:
    return {f.kind for f in report.findings}


def _log(tmp_path: Path, queries: list[str], name: str = "queries.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps({"query": q, "tool": "search_knowledge"}) for q in queries)
    )
    return path


REAL = [
    "what did we decide about the billing provider",
    "when does the warranty on the boiler run out",
    "the quote for replacing the side gate",
]


# --- the two contaminations that actually happened -------------------------


def test_the_projects_own_health_probe_is_an_error(tmp_path: Path) -> None:
    # corpus-smoke drove every server with this, leaving three archives whose
    # logged queries were almost entirely the probe.
    report = audit_query_log(_log(tmp_path, [*REAL, "notes from last year"]))
    assert "synthetic-probe-in-log" in _kinds(report)
    assert not report.ok


def test_test_suite_fixtures_are_an_error(tmp_path: Path) -> None:
    # 276 entries that were 180 copies of "q", 32 of "find me something" and
    # 30 of "nothing here".
    queries = ["q"] * 180 + ["find me something"] * 32 + ["nothing here"] * 30
    report = audit_query_log(_log(tmp_path, queries))
    assert "fixture-queries-in-log" in _kinds(report)
    assert report.fixture == 242
    assert report.distinct_real == 0
    assert not report.ok


def test_a_clean_log_of_real_questions_passes(tmp_path: Path) -> None:
    report = audit_query_log(_log(tmp_path, REAL), min_real_for_conclusions=1)
    assert report.ok
    assert report.findings == []
    assert report.distinct_real == 3


# --- the generic shape, for fixtures this list has never seen --------------


def test_one_query_dominating_a_log_is_reported(tmp_path: Path) -> None:
    # A string no fixture list knows about is still suspicious when it is most
    # of the log. A person asking the same thing 40 times is possible;
    # something automated is likelier.
    queries = ["some unrecognised harness string"] * 40 + REAL
    report = audit_query_log(_log(tmp_path, queries), min_real_for_conclusions=1)
    assert "one-query-dominates" in _kinds(report)


def test_a_repeated_query_below_the_threshold_is_not_reported(tmp_path: Path) -> None:
    # A varied log where the most-repeated query is a small share of the whole:
    # people do re-ask things, and flagging that would make the check noise.
    varied = [f"a genuine question about topic number {i}" for i in range(40)]
    queries = ["what did we decide about the billing provider"] * 5 + varied
    report = audit_query_log(_log(tmp_path, queries), min_real_for_conclusions=1)
    assert "one-query-dominates" not in _kinds(report)


def test_a_very_short_query_counts_as_a_fixture(tmp_path: Path) -> None:
    report = audit_query_log(_log(tmp_path, ["q", "x", *REAL]))
    assert report.fixture == 2


# --- "no entries" must not read as "no usage" ------------------------------


def test_a_thin_log_says_it_cannot_support_conclusions(tmp_path: Path) -> None:
    # Logging that is configured but never exercised is indistinguishable from
    # logging that works and shows low use. The report has to say which
    # question it cannot answer.
    report = audit_query_log(_log(tmp_path, REAL))
    assert "too-thin-to-conclude" in _kinds(report)
    # A thin log is a warning, not an error: it is honest, merely unusable.
    assert report.ok


def test_a_missing_log_is_empty_rather_than_an_exception(tmp_path: Path) -> None:
    report = audit_query_log(tmp_path / "never-written.jsonl")
    assert report.total == 0
    assert report.ok


def test_unparseable_lines_are_counted_not_crashed_on(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query": "a real question about the floors"}\nnot json at all\n')
    report = audit_query_log(path, min_real_for_conclusions=1)
    assert report.unreadable == 1
    assert "unreadable-entries" in _kinds(report)


def test_read_queries_skips_entries_with_no_query_field(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text('{"tool": "corpus_stats"}\n{"query": "a real one about floors"}\n')
    queries, unreadable = read_queries(path)
    assert queries == ["a real one about floors"]
    assert unreadable == 1


# --- the measurement behind "one server or several?" -----------------------


def test_cross_archive_overlap_finds_queries_asked_of_two_archives(
    tmp_path: Path,
) -> None:
    # This is the criterion a deferred architecture decision rests on: a query
    # asked of two archives is one a single facade would have served in one
    # call.
    a = tmp_path / "alpha" / "data"
    b = tmp_path / "beta" / "data"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    report_a = audit_query_log(_log(a, ["shared question here", "alpha only thing"]))
    report_b = audit_query_log(_log(b, ["shared question here", "beta only thing"]))

    overlap = cross_archive_overlap([report_a, report_b])
    assert list(overlap) == ["shared question here"]
    assert sorted(overlap["shared question here"]) == ["alpha", "beta"]


def test_no_overlap_when_each_archive_is_asked_different_things(
    tmp_path: Path,
) -> None:
    a = tmp_path / "alpha" / "data"
    b = tmp_path / "beta" / "data"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    overlap = cross_archive_overlap([
        audit_query_log(_log(a, ["alpha only thing"])),
        audit_query_log(_log(b, ["beta only thing"])),
    ])
    assert overlap == {}


def test_overlap_ignores_fixtures_so_pollution_cannot_fake_demand(
    tmp_path: Path,
) -> None:
    # The pollution appeared in EVERY archive's log, so counting it would have
    # shown near-total cross-archive demand and argued for building a facade
    # nobody needed.
    a = tmp_path / "alpha" / "data"
    b = tmp_path / "beta" / "data"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    overlap = cross_archive_overlap([
        audit_query_log(_log(a, ["notes from last year", "q", "alpha only thing"])),
        audit_query_log(_log(b, ["notes from last year", "q", "beta only thing"])),
    ])
    assert overlap == {}


# --- the diagnostic must not stop at the first problem ---------------------


def test_an_unimportable_gold_set_fails_that_check_only(tmp_path, capsys) -> None:
    """A diagnostic that aborts hides every check after the one that broke.

    Gold sets commonly import their own package, so running the doctor with
    the wrong interpreter raises ModuleNotFoundError from inside the check --
    which took the whole run down and reported nothing about the query logs.
    """
    from corpus.cli.doctor import _check_gold_set

    broken = tmp_path / "eval_queries.py"
    broken.write_text("import a_package_that_does_not_exist\n")

    assert _check_gold_set(str(broken), str(tmp_path / "index.db")) is False
    out = capsys.readouterr().out
    assert "could not import" in out
