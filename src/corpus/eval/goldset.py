"""Auditing the eval set itself, before trusting what it says.

A gold set is code that grades other code, and nothing grades it. Every check
here exists because a broken answer key was mistaken for a broken retriever --
which is the expensive failure, because the response is to go tune something
that was working correctly.

`expected_keys` is an OR-set: a query hits if ANY key in it ranks. That makes
an INCOMPLETE key actively misleading rather than merely imprecise. The
retriever returns a perfectly good answer, the key does not list it, and the
eval reports a miss.

Observed on three personal archives in one afternoon:

* An answer set built by a labelling query that matched nothing. Scored as a
  "negative" query -- silently excluded from the metrics rather than failing.
* An answer set capped at 400 keys when the correct answer had 1,456 members.
  Reported recall 0.500; the true figure was 0.625.
* An answer set built by prefix-matching a mail subject, which missed every
  "Re:" in the thread. A 93-message thread became one message.
* A query written from a FILENAME rather than from the document's text. The
  file was what its name implied, its contents were not, and the eval reported
  a retrieval failure against a correct result.

None of these is detectable from the metrics -- they all look exactly like a
retriever that cannot find things.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Sizes that look chosen by a person rather than produced by data. An answer
# set landing exactly on one is weak evidence on its own; several sets sharing
# one is strong evidence of a LIMIT in the labelling code.
ROUND_SIZES = frozenset({50, 100, 128, 200, 250, 256, 300, 400, 500, 512, 1000, 1024})

# Consecutive words from the query found verbatim in an expected document.
# Three is long enough that ordinary shared vocabulary does not trip it, and
# short enough to catch a query pasted out of the answer.
_ECHO_RUN_WORDS = 3

_WORD = re.compile(r"\w+", re.UNICODE)


class _KeyLookup(Protocol):
    def __call__(self, keys: Sequence[str]) -> set[str]: ...


@dataclass(frozen=True)
class GoldFinding:
    query: str
    kind: str
    detail: str
    severity: str  # "error" | "warning"

    def render(self) -> str:
        mark = "ERROR" if self.severity == "error" else "warn "
        return f"  [{mark}] {self.kind}: {self.detail}\n          query: {self.query!r}"


def _expected(query: Any) -> list[str]:
    return list(getattr(query, "expected_keys", None) or [])


def _text_of(query: Any) -> str:
    return str(getattr(query, "query", "") or "")


def _empty_answer_sets(queries: Sequence[Any]) -> Iterable[GoldFinding]:
    """A query with no expected keys is scored as a negative and excluded.

    That is a legitimate thing to want, and indistinguishable from a labelling
    function that silently matched nothing -- which is how a mail gold set
    shipped with a thread that had zero members.
    """
    for query in queries:
        if not _expected(query):
            yield GoldFinding(
                query=_text_of(query),
                kind="empty-answer-set",
                detail=(
                    "no expected keys, so this is scored as a negative query and "
                    "excluded from the metrics. If that is intended, ignore this; "
                    "if the keys come from a labelling function, it matched nothing"
                ),
                severity="warning",
            )


def _unindexed_keys(
    queries: Sequence[Any], lookup: _KeyLookup | None
) -> Iterable[GoldFinding]:
    """Expected keys that are not in the index cannot ever be retrieved."""
    if lookup is None:
        return
    for query in queries:
        expected = _expected(query)
        if not expected:
            continue
        present = lookup(expected)
        missing = [key for key in expected if key not in present]
        if not missing:
            continue
        sample = ", ".join(missing[:3])
        if len(missing) == len(expected):
            yield GoldFinding(
                query=_text_of(query),
                kind="answer-not-in-index",
                detail=(
                    f"none of the {len(expected)} expected key(s) exist in this "
                    f"index, so this query can only ever fail (e.g. {sample})"
                ),
                severity="error",
            )
        else:
            yield GoldFinding(
                query=_text_of(query),
                kind="answer-partly-not-in-index",
                detail=(
                    f"{len(missing)} of {len(expected)} expected keys are absent "
                    f"from this index (e.g. {sample})"
                ),
                severity="warning",
            )


def _capped_answer_sets(queries: Sequence[Any]) -> Iterable[GoldFinding]:
    """Several answer sets stopping at the same round number means a LIMIT.

    A labelling function with a cap in it produces answer sets that are not
    wrong so much as truncated, and truncation is invisible in the metrics: the
    retriever returns a correct document that the key does not happen to list.
    """
    sizes: dict[int, list[Any]] = {}
    for query in queries:
        expected = _expected(query)
        if expected:
            sizes.setdefault(len(expected), []).append(query)
    for size, sharing in sizes.items():
        if size in ROUND_SIZES and len(sharing) > 1:
            for query in sharing:
                yield GoldFinding(
                    query=_text_of(query),
                    kind="possible-cap",
                    detail=(
                        f"{len(sharing)} answer sets are exactly {size} keys. A "
                        "round number repeated across sets is usually a LIMIT in "
                        "the labelling code, which truncates correct answers"
                    ),
                    severity="warning",
                )


def _duplicate_queries(queries: Sequence[Any]) -> Iterable[GoldFinding]:
    seen: dict[str, int] = {}
    for query in queries:
        text = _text_of(query).strip().casefold()
        seen[text] = seen.get(text, 0) + 1
    for query in queries:
        text = _text_of(query).strip().casefold()
        if seen.get(text, 0) > 1:
            seen[text] = 0  # report each duplicate group once
            yield GoldFinding(
                query=_text_of(query),
                kind="duplicate-query",
                detail="the same query text appears more than once in the set",
                severity="warning",
            )


def _word_runs(text: str, size: int) -> set[tuple[str, ...]]:
    words = [w.casefold() for w in _WORD.findall(text)]
    if len(words) < size:
        return set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def _query_echoes_answer(
    queries: Sequence[Any], documents: dict[str, str] | None
) -> Iterable[GoldFinding]:
    """A query lifted from its own answer tests string matching, not retrieval.

    The whole point of paraphrasing a gold query away from the target's wording
    is that a query sharing the answer's phrasing is answered by BM25 whatever
    the embedder does, so the score says nothing about semantic retrieval.
    """
    if not documents:
        return
    for query in queries:
        runs = _word_runs(_text_of(query), _ECHO_RUN_WORDS)
        if not runs:
            continue
        for key in _expected(query):
            content = documents.get(key)
            if not content:
                continue
            shared = runs & _word_runs(content, _ECHO_RUN_WORDS)
            if shared:
                phrase = " ".join(next(iter(shared)))
                yield GoldFinding(
                    query=_text_of(query),
                    kind="query-echoes-answer",
                    detail=(
                        f"the phrase {phrase!r} appears verbatim in an expected "
                        "document, so BM25 answers this regardless of the "
                        "embedder -- paraphrase the query away from the source"
                    ),
                    severity="warning",
                )
                break


def audit_queries(
    queries: Sequence[Any],
    *,
    lookup: _KeyLookup | None = None,
    documents: dict[str, str] | None = None,
) -> list[GoldFinding]:
    """Every check, most severe first.

    `lookup` maps candidate keys to the subset present in the index; `documents`
    maps key to text. Both optional -- without them the structural checks still
    run, which is what makes this usable before an index exists.
    """
    findings = [
        *_unindexed_keys(queries, lookup),
        *_empty_answer_sets(queries),
        *_capped_answer_sets(queries),
        *_duplicate_queries(queries),
        *_query_echoes_answer(queries, documents),
    ]
    return sorted(findings, key=lambda f: 0 if f.severity == "error" else 1)


# --- reading an index, for the checks that need one -------------------------
# These live here rather than in a CLI so every deployment gets the same audit
# from the same code. The private archives run their own forked eval CLIs and
# import `corpus.eval.metrics` for exactly this reason; the audit travels the
# same way, and a check that only half the archives run is a check that only
# half the archives get.

# Enough keys per query to catch a query copied out of its own answer, without
# reading a 1,456-document answer set to do it.
ECHO_SAMPLE_PER_QUERY = 25

_KEY_BATCH = 500


def sqlite_lookup(db_path: Path | str) -> _KeyLookup:
    """A callable reporting which of `keys` exist in a corpus-schema index.

    Read-only and batched, so an answer set of any size costs one query per 500
    keys and cannot hold a long transaction against a database a server is
    serving. On any SQLite error it reports every key as present: an audit that
    cannot read the index must not manufacture findings about it.
    """

    def lookup(keys: Sequence[str]) -> set[str]:
        found: set[str] = set()
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error:
            return set(keys)
        try:
            for start in range(0, len(keys), _KEY_BATCH):
                batch = list(keys[start : start + _KEY_BATCH])
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT DISTINCT source_key FROM chunks WHERE source_key IN ({marks})",
                    batch,
                ).fetchall()
                found.update(row[0] for row in rows)
        except sqlite3.Error:
            return set(keys)
        finally:
            conn.close()
        return found

    return lookup


def sqlite_documents(
    db_path: Path | str,
    queries: Sequence[Any],
    *,
    sample_per_query: int = ECHO_SAMPLE_PER_QUERY,
) -> dict[str, str]:
    """Text for a sample of each query's expected keys, for the echo check."""
    wanted: list[str] = []
    for query in queries:
        keys = list(getattr(query, "expected_keys", None) or [])
        wanted.extend(keys[:sample_per_query])
    if not wanted:
        return {}
    out: dict[str, str] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        for start in range(0, len(wanted), _KEY_BATCH):
            batch = wanted[start : start + _KEY_BATCH]
            marks = ",".join("?" * len(batch))
            for key, content in conn.execute(
                f"SELECT source_key, content FROM chunks WHERE source_key IN ({marks})",
                batch,
            ):
                out.setdefault(key, content or "")
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return out


def report_findings(findings: Sequence[GoldFinding], *, stream: Any) -> bool:
    """Print an audit report. Returns True if any finding is an error.

    Shared so every deployment's eval says the same thing in the same shape.
    """
    if findings:
        print(f"\nGold set audit: {len(findings)} finding(s)", file=stream)
        for finding in findings:
            print(finding.render(), file=stream)
        print("", file=stream)
    return any(f.severity == "error" for f in findings)
