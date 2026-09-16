"""Append-only local log of queries the MCP server actually served.

Why this exists. Every tuning question this engine raises -- how to weight
BM25 against vectors, whether reranking earns its latency, whether Contextual
Retrieval is worth its cost -- needs a set of realistic queries to answer.
Without one you build a gold set, and a built gold set measures its own
construction:

  * Queries mined from document TITLES share nearly every token with their
    target, so they measure lexical overlap. Sweeping BM25 weight over such a
    set produced a clean, monotone, entirely misleading curve that reversed on
    a paraphrased set.
  * Queries written by a model from a document's body still echo that body's
    vocabulary more than a person would.

Real served queries have neither bias. They are the only population that is
actually the one being optimised for.

PRIVACY. Queries are among the most revealing things a person produces -- they
say what someone is looking for and when. So:

  * OFF by default. Logging someone's searches has to be a decision they make,
    not a default they discover.
  * The file is local, written next to the store, and nothing in this engine
    transmits it anywhere.
  * `include_results` can be turned off to record only that a query happened,
    without the document keys it surfaced.
  * Delete the file and the record is gone; nothing else references it.

Failures here are swallowed. A search must never fail because its log could
not be written -- the log is an aid, not part of the contract.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# One process may serve several tool calls concurrently (the MCP server hands
# store work to `asyncio.to_thread`). A short lock keeps lines from
# interleaving; the write is small enough that contention is irrelevant.
_lock = threading.Lock()
_warned = False


def log_query(
    path: Path | str,
    *,
    tool: str,
    query: str,
    top_k: int | None = None,
    filters: Any = None,
    results: list[tuple[str, str]] | None = None,
    elapsed_ms: float | None = None,
) -> None:
    """Append one JSON line describing a served query.

    `results` is a list of (source_type, source_key) pairs -- enough to judge
    relevance later and build a gold set from real usage, without copying
    document text into a second place.
    """
    global _warned
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "tool": tool,
        "query": query,
    }
    if top_k is not None:
        record["top_k"] = top_k
    if filters:
        record["filters"] = filters
    if results is not None:
        record["results"] = [{"source_type": st, "source_key": sk} for st, sk in results]
    if elapsed_ms is not None:
        record["elapsed_ms"] = round(elapsed_ms, 1)

    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock, open(p, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        # Once, not per query: a broken path would otherwise fill the log it
        # is failing to write to.
        if not _warned:
            _warned = True
            logger.warning(
                "query logging is enabled but writing to %s failed (%s); "
                "searches are unaffected and this will not be repeated",
                path,
                exc,
            )


class QueryTimer:
    """Context manager measuring a query's wall time in milliseconds."""

    def __enter__(self) -> QueryTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000

    elapsed_ms: float = 0.0

def configured_log_path(config: object) -> Path | None:
    """Where a config says served queries are appended, or None when off.

    THE WRITER AND THE AUDITOR MUST AGREE. This was written twice: the MCP
    server resolved it to append, and `corpus-doctor` did not resolve it at
    all, so its query-log check was skipped on every run unless someone passed
    `--query-log` by hand. An auditor that guessed a different path would be
    worse still -- it would read a file nothing writes to and report a clean
    bill of health. One function, used by both.

    Defaults beside the store rather than the cwd: a server's working
    directory is whatever launched it, and a log that lands somewhere
    different on each launch is worse than no log.
    """
    if config is None or not getattr(config, "query_log", None):
        return None
    log = config.query_log  # type: ignore[attr-defined]
    if not log.enabled:
        return None
    if log.path:
        return Path(log.path)
    return Path(config.db_path).parent / "queries.jsonl"  # type: ignore[attr-defined]
