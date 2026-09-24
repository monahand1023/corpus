"""Have this source's stored chunks stopped matching what the chunker makes?

A source is ingested, the chunker changes, and nothing re-ingests that
source. Its stored chunks are then whatever an older version produced, and
NOTHING SAYS SO: search still works, the doctor is happy, the eval still
passes. The drift is invisible until someone re-ingests and sees the bill.

Measured on a live archive: 24% of one source's chunks no longer
matched what the current chunker produces. The re-ingest re-embedded 18,080
of them for 8.4M tokens, and the only warning anyone got was the invoice.

The content was not WRONG, exactly. It was two chunker versions old, which
means the boundaries it was embedded at are not the boundaries retrieval is
tuned for -- and that is invisible in every number the system reports.

WHY A SAMPLE. Chunking a million-chunk archive to answer "is it current?"
costs more than the answer is worth. A sample cannot prove a source is clean;
it can show that it is not, which is the direction that matters here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus.util.sqlite_ro import connect_ro

DEFAULT_SAMPLE = 200


@dataclass(frozen=True)
class DriftReport:
    source: str
    examined: int = 0
    drifted: int = 0
    unseen: int = 0
    total_documents: int = 0

    @property
    def percent(self) -> float:
        return 100.0 * self.drifted / self.examined if self.examined else 0.0

    @property
    def is_current(self) -> bool:
        """False when nothing was examined, not just when drift was found.

        "I examined nothing" and "I examined everything and it was fine" are
        the two facts this codebase keeps having to keep apart.
        """
        return self.examined > 0 and self.drifted == 0

    def describe(self) -> str:
        if not self.examined:
            return (
                f"{self.source}: examined nothing, so this proves nothing about "
                "whether the stored chunks are current"
            )
        if not self.drifted:
            return (
                f"{self.source}: {self.examined:,} document(s) sampled, all "
                "matching what the current chunker produces"
            )
        return (
            f"{self.source}: {self.drifted:,} of {self.examined:,} sampled "
            f"document(s) ({self.percent:.0f}%) no longer match what the current "
            "chunker produces -- their text was chunked by an older version, so "
            "they are embedded at boundaries retrieval is no longer tuned for. "
            "Re-ingest this source."
        )


def chunker_drift(
    db_path: Path | str,
    source: str,
    documents: Iterable[Any],
    chunker: Any,
    *,
    sample: int = DEFAULT_SAMPLE,
) -> DriftReport:
    """Compare freshly chunked documents against what is stored.

    A document the store has never seen is NOT drift -- new content is not
    stale content, and counting it as drift would make every growing archive
    look permanently out of date.
    """
    conn = connect_ro(db_path)
    try:
        stored: dict[str, str] = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT id, content_hash FROM chunks WHERE source_type = ?",
                (source,),
            )
        }
        total = conn.execute(
            "SELECT count(DISTINCT source_key) FROM chunks WHERE source_type = ?",
            (source,),
        ).fetchone()[0]
    finally:
        conn.close()

    examined = drifted = unseen = 0
    for doc in documents:
        if examined >= sample:
            break
        chunks: Sequence[Any] = chunker.chunk(doc)
        if not chunks:
            continue
        known = [c for c in chunks if c.id in stored]
        if not known:
            unseen += 1
            continue
        examined += 1
        if any(stored[c.id] != c.content_hash for c in known):
            drifted += 1

    return DriftReport(
        source=source,
        examined=examined,
        drifted=drifted,
        unseen=unseen,
        total_documents=total,
    )


__all__ = ["DEFAULT_SAMPLE", "DriftReport", "chunker_drift"]
