"""Deciding whether a query log can be trusted as evidence of real usage.

A query log is an instrument, and an instrument nobody checks drifts. This one
drifted twice without anyone noticing, and both times it inflated an archive
that had almost no real use:

* `corpus-smoke` drove every configured server with a synthetic probe, so an
  afternoon of health checks left three archives whose logged queries were
  almost entirely that probe.
* A consumer's TEST SUITE pointed at its repo's real log. 276 entries turned
  out to be 180 copies of `"q"`, 32 of `"find me something"` and 30 of
  `"nothing here"` — fixtures, not questions.

Both were silent. The count went up, which is what a log being used looks
like. In that deployment the log was gathering evidence for a deferred
architecture decision with an explicit threshold ("revisit at ~200 entries"),
so the pollution did not merely add noise — it would have triggered the
decision on data that contained no usage at all.

The checks here are deliberately blunt. They do not try to judge whether a
query was a GOOD one; they answer the narrower question of whether a human
plausibly typed it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Probes shipped by this project's own tooling. Anything here reaching a log
# means a guard regressed.
KNOWN_SYNTHETIC = frozenset({
    "notes from last year",   # corpus-smoke's default probe
    "invoice",                # the live-source probe used in its docs
})

# Strings a person does not type into a search box. Test suites are full of
# them because a handler test only needs SOME string.
KNOWN_FIXTURES = frozenset({
    "q", "x", "test", "query", "anything", "something", "nothing here",
    "find me something", "hello", "foo", "bar", "asdf",
})

# Below this, a "query" is a placeholder rather than a question.
MIN_REAL_QUERY_CHARS = 4

# One query making up this much of a log is the signature of something
# automated, not of a person who asks the same thing often.
DOMINANCE_SHARE = 0.25
DOMINANCE_MIN_COUNT = 10


@dataclass(frozen=True)
class LogFinding:
    kind: str
    detail: str
    severity: str  # "error" | "warning"

    def render(self) -> str:
        mark = "ERROR" if self.severity == "error" else "warn "
        return f"  [{mark}] {self.kind}: {self.detail}"


@dataclass
class LogReport:
    path: Path
    total: int = 0
    synthetic: int = 0
    fixture: int = 0
    unreadable: int = 0
    real_queries: list[str] = field(default_factory=list)
    findings: list[LogFinding] = field(default_factory=list)

    @property
    def distinct_real(self) -> int:
        return len(set(self.real_queries))

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)


def read_queries(path: Path | str) -> tuple[list[str], int]:
    """(query strings, unreadable line count) from a JSONL query log."""
    queries: list[str] = []
    unreadable = 0
    file = Path(path)
    if not file.is_file():
        return [], 0
    for line in file.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            unreadable += 1
            continue
        query = entry.get("query")
        if isinstance(query, str):
            queries.append(query)
        else:
            unreadable += 1
    return queries, unreadable


def _is_fixture(query: str) -> bool:
    stripped = query.strip().casefold()
    return stripped in KNOWN_FIXTURES or len(stripped) < MIN_REAL_QUERY_CHARS


def audit_query_log(
    path: Path | str,
    queries: Sequence[str] | None = None,
    *,
    unreadable: int = 0,
    min_real_for_conclusions: int = 30,
) -> LogReport:
    """Judge whether a log holds real usage.

    `queries` may be passed directly (for tests); otherwise the file is read.
    `min_real_for_conclusions` is the point below which the log is reported as
    too thin to draw anything from — a low count is only evidence of low usage
    once the log is known to be clean and known to have been exercised.
    """
    if queries is None:
        queries, unreadable = read_queries(path)

    report = LogReport(path=Path(path), total=len(queries), unreadable=unreadable)

    counts: dict[str, int] = {}
    for query in queries:
        key = query.strip().casefold()
        counts[key] = counts.get(key, 0) + 1
        if key in KNOWN_SYNTHETIC:
            report.synthetic += 1
        elif _is_fixture(query):
            report.fixture += 1
        else:
            report.real_queries.append(query.strip())

    if report.synthetic:
        report.findings.append(LogFinding(
            kind="synthetic-probe-in-log",
            detail=(
                f"{report.synthetic} entr(ies) are this project's own health-check "
                "probe. A guard has regressed -- a health check is not a search, "
                "and these inflate the count that decisions are read off"
            ),
            severity="error",
        ))

    if report.fixture:
        sample = ", ".join(
            sorted({q.strip() for q in queries if _is_fixture(q)})[:4]
        )
        report.findings.append(LogFinding(
            kind="fixture-queries-in-log",
            detail=(
                f"{report.fixture} entr(ies) look like test fixtures rather than "
                f"questions (e.g. {sample}). A test suite is probably writing to "
                "the real log"
            ),
            severity="error",
        ))

    # Even without a recognised fixture string, one query dominating a log is
    # the shape of something automated.
    for key, count in counts.items():
        if count >= DOMINANCE_MIN_COUNT and count / max(1, len(queries)) >= DOMINANCE_SHARE:
            if key in KNOWN_SYNTHETIC or key in KNOWN_FIXTURES:
                continue  # already reported, more specifically
            report.findings.append(LogFinding(
                kind="one-query-dominates",
                detail=(
                    f"{key!r} is {count} of {len(queries)} entries "
                    f"({count / len(queries):.0%}). A person asking the same thing "
                    "that often is possible; something automated is likelier"
                ),
                severity="warning",
            ))

    if report.unreadable:
        report.findings.append(LogFinding(
            kind="unreadable-entries",
            detail=f"{report.unreadable} line(s) could not be parsed as a logged query",
            severity="warning",
        ))

    if report.distinct_real < min_real_for_conclusions:
        report.findings.append(LogFinding(
            kind="too-thin-to-conclude",
            detail=(
                f"{report.distinct_real} distinct real quer(ies), below the "
                f"{min_real_for_conclusions} this check treats as a usable sample. "
                "A low count is evidence of low USAGE only once the log is known "
                "clean and known to have been exercised -- otherwise it is "
                "indistinguishable from logging that never worked"
            ),
            severity="warning",
        ))

    return report


def cross_archive_overlap(reports: Iterable[LogReport]) -> dict[str, list[str]]:
    """Real queries that appear in more than one archive's log.

    This is the measurement behind "should these be one server or several?".
    A query asked of two archives is one a single façade would have served in
    one call; if that almost never happens, separate servers are already right
    and nothing needs building.
    """
    seen: dict[str, list[str]] = {}
    for report in reports:
        for query in set(report.real_queries):
            seen.setdefault(query.casefold(), []).append(report.path.parent.parent.name)
    return {q: sorted(set(a)) for q, a in seen.items() if len(set(a)) > 1}
