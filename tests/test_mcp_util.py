"""Shared MCP helpers: not leaking internals, and recording what was asked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


def test_a_served_query_records_what_it_returned(tmp_path: Path) -> None:
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


def test_results_can_be_withheld_while_still_recording_the_query(tmp_path: Path) -> None:
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
