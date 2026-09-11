"""Query logging: opt-in, local, and never able to break a search.

Why it exists: every tuning question this engine raises needs realistic
queries to answer, and a SYNTHESISED gold set measures its own construction.
Two demonstrations of that on 2026-09-10 -- a title-mined set produced a
clean, monotone BM25-weight curve that reversed on a paraphrased set, and an
LLM-written set echoed its source documents' vocabulary. Served queries have
neither bias.
"""

from __future__ import annotations

import json
from pathlib import Path

from corpus.config import CorpusConfig
from corpus.query_log import QueryTimer, log_query


def _lines(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_logging_is_off_unless_asked_for(tmp_path: Path) -> None:
    # Logging someone's searches must be a decision they make, not a default
    # they discover later.
    cfg_path = tmp_path / "corpus.toml"
    cfg_path.write_text('[corpus]\ndb_path = "./x.db"\n')

    config = CorpusConfig.load(cfg_path)

    assert config.query_log.enabled is False


def test_a_served_query_is_recorded_with_what_it_returned(tmp_path: Path) -> None:
    log = tmp_path / "queries.jsonl"

    log_query(
        log,
        tool="search_knowledge",
        query="what did we decide about the migration",
        top_k=5,
        results=[("notes", "planning.md"), ("email", "thread-9")],
        elapsed_ms=42.4,
    )

    (rec,) = _lines(log)
    assert rec["query"] == "what did we decide about the migration"
    assert rec["tool"] == "search_knowledge"
    assert rec["top_k"] == 5
    assert rec["elapsed_ms"] == 42.4
    # (source_type, source_key) is exactly what a later relevance judgement
    # needs, without copying document text into a second place.
    assert rec["results"] == [
        {"source_type": "notes", "source_key": "planning.md"},
        {"source_type": "email", "source_key": "thread-9"},
    ]
    assert rec["ts"].endswith("+00:00")


def test_results_can_be_omitted_while_still_recording_the_query(tmp_path: Path) -> None:
    # The privacy dial: record THAT a search happened without recording which
    # documents it surfaced.
    log = tmp_path / "q.jsonl"

    log_query(log, tool="search_knowledge", query="salary review", results=None)

    (rec,) = _lines(log)
    assert rec["query"] == "salary review"
    assert "results" not in rec


def test_appends_rather_than_overwrites(tmp_path: Path) -> None:
    log = tmp_path / "q.jsonl"

    for q in ("first", "second", "third"):
        log_query(log, tool="search_knowledge", query=q)

    assert [r["query"] for r in _lines(log)] == ["first", "second", "third"]


def test_a_broken_log_path_never_breaks_the_search(tmp_path: Path) -> None:
    """The log is an aid, not part of the contract.

    A search failing because its log could not be written would be a strictly
    worse outcome than not logging -- so the failure is swallowed, and warned
    about once rather than once per query.
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("I am a file")

    # Writing "inside" a regular file: guaranteed OSError.
    log_query(blocker / "sub" / "q.jsonl", tool="search_knowledge", query="x")


def test_non_ascii_queries_are_stored_readably(tmp_path: Path) -> None:
    # These archives are Japanese and Spanish; \\u-escaped logs would be
    # unreadable exactly where they matter most.
    log = tmp_path / "q.jsonl"

    log_query(log, tool="search_knowledge", query="会議の予定は？")

    assert "会議の予定は？" in log.read_text()
    assert _lines(log)[0]["query"] == "会議の予定は？"


def test_the_timer_measures_elapsed_time() -> None:
    with QueryTimer() as t:
        sum(range(10000))

    assert t.elapsed_ms >= 0
