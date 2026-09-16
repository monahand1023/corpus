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


# Path fragments that mark a copy as the DERIVATIVE one, so the other is the
# one to keep.
#
# These decide only which member of an ALREADY-IDENTICAL pair to suggest
# dropping. Getting it "wrong" swaps which of two documents holding the same
# passages survives, so the cost of a bad guess is nothing -- which is what
# makes guessing acceptable here and nowhere else in this module.
_DERIVATIVE_MARKERS = (
    "backup",
    "/.bak-",
    "archive - backups/",
    ".zip::",
    "copy of ",
    " (1).",
    " (2).",
    " (3).",
)


@dataclass(frozen=True)
class DuplicatePair:
    """Two documents with identical content, and which to drop.

    `keep` and `drop` are a SUGGESTION about paths, not a judgement about
    value: the two documents hold the same passages, so either would do.
    """

    keep: str
    drop: str
    shared: int


def _derivative_rank(key: str) -> int:
    """0 for a path with no derivative marker, 1 for one that has any.

    Lower sorts first and is kept. Ties break on path length, then name, so
    the result does not depend on dict ordering.
    """
    lowered = key.lower()
    return 1 if any(marker in lowered for marker in _DERIVATIVE_MARKERS) else 0


def duplicate_documents(
    db_path: Path | str, *, min_chunks: int = 2
) -> list[DuplicatePair]:
    """Documents whose EVERY chunk also appears elsewhere, paired for removal.

    The decidable subset of `find_duplicate_content`. A document all of whose
    passages exist under another document can be dropped losslessly -- unlike
    a partial overlap, where two versions share most passages and differ in
    the ones that matter.

    PAIRS, NOT A LIST, because both members of a duplicate pair are wholly
    contained in the other and both would otherwise appear. Acting on that
    list deletes the content. N identical copies yield N-1 removals and
    exactly one survivor.

    `min_chunks` skips single-chunk documents: one shared passage is a stock
    sentence, not a copied file.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT content_hash, source_key FROM chunks").fetchall()
    finally:
        conn.close()

    docs_by_hash: dict[str, set[str]] = defaultdict(set)
    hashes_by_doc: dict[str, set[str]] = defaultdict(set)
    for content_hash, source_key in rows:
        docs_by_hash[content_hash].add(source_key)
        hashes_by_doc[source_key].add(content_hash)

    # Group documents by their exact content set: identical sets are copies.
    by_content: dict[frozenset[str], list[str]] = defaultdict(list)
    for doc, hashes in hashes_by_doc.items():
        if len(hashes) < min_chunks:
            continue
        if all(len(docs_by_hash[h]) > 1 for h in hashes):
            by_content[frozenset(hashes)].append(doc)

    pairs: list[DuplicatePair] = []
    for content, docs in by_content.items():
        if len(docs) < 2:
            # Wholly duplicated, but not by a single other document -- its
            # passages are scattered across several. Dropping it could still
            # lose the document's own identity, so it is not offered.
            continue
        ordered = sorted(docs, key=lambda d: (_derivative_rank(d), len(d), d))
        keep = ordered[0]
        pairs.extend(
            DuplicatePair(keep=keep, drop=other, shared=len(content))
            for other in ordered[1:]
        )
    return sorted(pairs, key=lambda p: -p.shared)


__all__ = [
    "DuplicatePair",
    "DuplicateReport",
    "duplicate_documents",
    "find_duplicate_content",
]
