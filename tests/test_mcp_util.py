"""Shared MCP helpers: not leaking internals, and recording what was asked."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from corpus.mcp_util import QueryTimer, record_query, safe_tool


class _Chunk:
    def __init__(self, st: str, sk: str) -> None:
        self.source_type, self.source_key = st, sk


def test_a_failing_tool_returns_a_generic_message(caplog) -> None:
    """A raw exception would be serialised back to the MODEL -- file paths,
    SQL, stack frames. That is an information leak and useless to the caller,
    who cannot act on it."""

    @safe_tool
    async def boom(query: str) -> str:
        raise RuntimeError("/Users/someone/private.db is corrupt")

    out = asyncio.run(boom("x"))

    assert "private.db" not in out
    assert "boom" in out


def test_the_wrapper_preserves_the_signature() -> None:
    """FastMCP introspects the signature to build each tool's input schema.
    Without `functools.wraps` every wrapped tool would advertise
    `(*args, **kwargs)` and lose its parameters."""
    import inspect

    @safe_tool
    async def sample(query: str, top_k: int = 5) -> str:
        return "ok"

    assert list(inspect.signature(sample).parameters) == ["query", "top_k"]


def test_a_successful_tool_passes_its_result_through() -> None:
    @safe_tool
    async def fine(query: str) -> str:
        return f"found {query}"

    assert asyncio.run(fine("things")) == "found things"


def test_logging_off_writes_nothing(tmp_path: Path) -> None:
    # path=None is the default state everywhere: recording someone's searches
    # is a decision they make, not one they discover.
    record_query(None, tool="search", query="private question")

    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def serving(monkeypatch):
    """Make record_query behave as it does under a real server.

    record_query deliberately refuses to write while pytest is running, so a
    consumer's test suite cannot fill the query log with fixtures. Tests of
    the logging path itself have to opt out of that guard explicitly, which is
    the right shape: writing to the log is now something a test must ASK for.
    """
    monkeypatch.setenv("CORPUS_ALLOW_QUERY_LOG", "1")
    monkeypatch.delenv("CORPUS_SYNTHETIC_QUERY", raising=False)


def test_a_served_query_records_what_it_returned(tmp_path: Path, serving) -> None:
    log = tmp_path / "q.jsonl"

    record_query(
        log, tool="search_knowledge", query="the migration decision",
        chunks=[_Chunk("notes", "plan.md"), _Chunk("email", "t-9")],
        elapsed_ms=12.0, top_k=5,
    )

    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["query"] == "the migration decision"
    assert rec["top_k"] == 5
    assert rec["results"] == [
        {"source_type": "notes", "source_key": "plan.md"},
        {"source_type": "email", "source_key": "t-9"},
    ]


def test_results_can_be_withheld_while_still_recording_the_query(
    tmp_path: Path, serving
) -> None:
    log = tmp_path / "q.jsonl"

    record_query(log, tool="search_knowledge", query="salary review",
                 chunks=[_Chunk("email", "t-1")], include_results=False)

    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["query"] == "salary review"
    assert "results" not in rec


def test_the_timer_reports_elapsed_time() -> None:
    with QueryTimer() as t:
        sum(range(1000))

    assert t.elapsed_ms >= 0


def test_a_synthetic_probe_is_never_logged_even_when_a_test_opts_in(
    tmp_path, monkeypatch, serving
) -> None:
    """A health check is not a search, and that guard has no override.

    corpus-smoke drives the real server with a probe. Without this, an
    afternoon of smoke tests left three archives whose logged queries were
    almost entirely that probe -- in a log whose entire purpose is to hold
    real usage.
    """
    log = tmp_path / "queries.jsonl"
    monkeypatch.setenv("CORPUS_SYNTHETIC_QUERY", "1")

    record_query(log, tool="search_knowledge", query="notes from last year")
    assert not log.exists()

    monkeypatch.delenv("CORPUS_SYNTHETIC_QUERY")
    record_query(log, tool="search_knowledge", query="a real question")
    assert log.exists()


def test_a_test_run_writes_nothing_unless_it_asks(tmp_path, monkeypatch) -> None:
    """A unit test is not a search either.

    Deliberately takes no `serving` fixture: this checks the DEFAULT. A
    consumer's test suite pointed at its repo's real query log, and 276 logged
    queries turned out to be 180 copies of "q", 32 of "find me something" and
    30 of "nothing here" -- fixtures, not usage. That archive had been read as
    the most-used of five on the strength of the count, and the log was
    collecting evidence for a deferred architecture decision, so the pollution
    destroyed exactly what it existed to gather.

    pytest sets PYTEST_CURRENT_TEST for the duration of each test, so this
    covers any consumer whose tests exercise a handler without having thought
    about the log at all.
    """
    log = tmp_path / "queries.jsonl"
    assert os.environ.get("PYTEST_CURRENT_TEST")

    record_query(log, tool="search_knowledge", query="find me something")
    assert not log.exists()

    monkeypatch.setenv("CORPUS_ALLOW_QUERY_LOG", "1")
    record_query(log, tool="search_knowledge", query="a real question")
    assert log.exists()
