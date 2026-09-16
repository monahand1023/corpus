"""The same passage indexed from more than one document.

WHAT THIS FOUND. On one live archive, 12,076 chunks -- 6.5% of the index --
were passages that also appear under a different document. Tracing the
transcript source: 168 recordings were present twice under different paths,
byte-identical and the same duration. That is 17.8 GPU-hours of audio
transcribed twice, ~1,884 redundant chunks embedded and paid for, and a search
that can return the same passage twice from two files, quietly eating result
slots.

REPORTED, NEVER DEDUPLICATED, and the distinction is the whole design. Some
duplication is legitimate and must not be touched: an email thread quotes what
it replies to, a template repeats its boilerplate, a report restates a summary.
A second archive measured 1.2% and almost all of it was email quoting. Nothing
here can tell that apart from a file copied into two folders -- a person can,
given the evidence, so this produces evidence rather than deletions.

WHAT COUNTS AS DUPLICATION. The same content under DIFFERENT `source_key`s. A
passage repeated inside one document is that document's own business -- a
refrain, a repeated heading, a footer on every page -- and says nothing about
duplicate files.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from corpus.verify import Coverage


@dataclass
class DuplicateReport:
    total_chunks: int = 0
    shared_passages: int = 0
    redundant_chunks: int = 0
    # (document, other document, how many passages they share), worst first.
    duplicate_documents: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def coverage(self) -> Coverage:
        return Coverage(examined=self.total_chunks, unit="chunks")

    @property
    def percent(self) -> float:
        if not self.total_chunks:
            return 0.0
        return 100.0 * self.redundant_chunks / self.total_chunks


def find_duplicate_content(
    db_path: Path | str, *, top: int = 10, min_shared: int = 2
) -> DuplicateReport:
    """Passages appearing under more than one document, worst offenders first.

    `min_shared` keeps the document list actionable. Two documents sharing a
    single passage is usually a stock sentence; sharing many is usually the
    same file in two places, which is the case worth acting on.
    """
    report = DuplicateReport()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT content_hash, source_type, source_key FROM chunks"
        ).fetchall()
    finally:
        conn.close()

    report.total_chunks = len(rows)
    if not rows:
        return report

    # A passage is duplicated when its hash appears under more than one
    # document. Repeats WITHIN a document are deliberately invisible here.
    docs_by_hash: dict[str, set[str]] = defaultdict(set)
    for content_hash, source_type, source_key in rows:
        docs_by_hash[content_hash].add(f"{source_type}:{source_key}")

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for docs in docs_by_hash.values():
        if len(docs) < 2:
            continue
        report.shared_passages += 1
        report.redundant_chunks += len(docs) - 1
        ordered = sorted(docs)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                pair_counts[(left, right)] += 1

    report.duplicate_documents = [
        (left.split(":", 1)[1], right.split(":", 1)[1], n)
        for (left, right), n in sorted(
            pair_counts.items(), key=lambda kv: -kv[1]
        )
        if n >= min_shared
    ][:top]
    return report


__all__ = ["DuplicateReport", "find_duplicate_content"]
