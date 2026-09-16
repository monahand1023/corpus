"""Auditing what actually landed in an index, after ingest.

The rest of `corpus.survey` answers "what is out there, and should we index
it?". This answers the question that only exists afterwards: "we indexed it --
is any of it junk?"

The junk it looks for is transcription artefacts. A speech-to-text model
trained on audio paired with scraped subtitles reproduces caption boilerplate
over silence, and that text is indistinguishable from real speech to every
downstream stage: it embeds, it ranks, and it comes back as an answer.

Two distinct defects, because they are fixed differently:

* A chunk that is ENTIRELY a sign-off should never have been indexed. Its
  presence means the connector is not filtering, and the fix is a filter.
* A chunk with a sign-off glued to the END of real speech must NOT be dropped
  -- doing so deletes real recordings, measured at 12.3% of one archive. The
  fix is to cut the tail and keep the speech.

Both are invisible from outside: the index reports a successful build, search
returns results, and nobody notices that some of the results are text no
person ever said. Measured on one 45,147-chunk transcript archive: 586 glued
tails plus 4 whole-chunk sign-offs, across 361 documents. Fixing it at
ingest and re-running left 1.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from corpus.transcripts.quality import strip_caption_tail, subtitle_boilerplate
from corpus.verify import Coverage, self_check

# Read in batches so a large index does not have to fit in memory, and so a
# scan cannot hold a long transaction against a database a server is serving.
_BATCH = 5_000


class NotACorpusIndexError(Exception):
    """The database exists but holds no corpus index."""


@dataclass
class QualityFinding:
    source_type: str
    source_key: str
    kind: str  # "whole-chunk-boilerplate" | "sign-off-tail"
    before: str
    after: str


@dataclass
class IndexQualityResult:
    total_chunks: int = 0
    scanned_chunks: int = 0
    whole_chunk: list[QualityFinding] = field(default_factory=list)
    tails: list[QualityFinding] = field(default_factory=list)
    by_source_type: Counter[str] = field(default_factory=Counter)
    documents_affected: set[str] = field(default_factory=set)

    @property
    def affected_chunks(self) -> int:
        return len(self.whole_chunk) + len(self.tails)

    @property
    def coverage(self) -> Coverage:
        """What this scan actually examined. Zero is not a pass."""
        return Coverage(examined=self.scanned_chunks, unit="chunks")

    @property
    def clean(self) -> bool:
        """No artefacts found AND something was actually examined.

        The second half is load-bearing. This was `affected_chunks == 0`,
        which is trivially true for an empty index -- so a scan pointed at the
        wrong database, or filtered to a source type that does not exist,
        printed "no transcription artefacts" having looked at nothing.
        """
        return bool(self.coverage) and self.affected_chunks == 0


def run_index_quality(
    db_path: Path | str,
    *,
    source_types: tuple[str, ...] = (),
    sample_per_kind: int = 8,
) -> IndexQualityResult:
    """Scan an index for transcription artefacts. Read-only.

    `source_types` limits the scan; empty means every type. `sample_per_kind`
    caps how many example findings are RETAINED -- counts are always exact,
    because the point is a number you can act on, not a wall of text.
    """
    # Prove the detectors can fire BEFORE trusting anything they do not find.
    # A phrase list that has been emptied, or normalisation that stops
    # matching, turns this scan into one that reports a clean index forever --
    # and that already happened here: 22 of 45 entries in an early list could
    # never match anything, while the tests passed.
    self_check(
        subtitle_boilerplate,
        positive="Thanks for watching!",
        label="whole-chunk boilerplate",
    )
    self_check(
        lambda text: strip_caption_tail(text) != text,
        positive="and then we drove home. Thanks for watching",
        label="sign-off tail",
    )

    result = IndexQualityResult()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # A consumer repo holds several SQLite files -- a caption cache, a
        # sidecar, an embeddings copy -- and only one of them is the index.
        # Pointing at the wrong one should say so, not raise OperationalError
        # from the middle of a scan.
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if table is None:
            raise NotACorpusIndexError(
                f"{db_path} has no 'chunks' table, so it is not a corpus index"
            )
        where, params = "", []
        if source_types:
            marks = ",".join("?" * len(source_types))
            where = f" WHERE source_type IN ({marks})"
            params = list(source_types)
        result.total_chunks = conn.execute(
            f"SELECT count(*) FROM chunks{where}", params
        ).fetchone()[0]

        cursor = conn.execute(
            f"SELECT source_type, source_key, content FROM chunks{where}", params
        )
        while rows := cursor.fetchmany(_BATCH):
            for source_type, source_key, content in rows:
                result.scanned_chunks += 1
                text = (content or "").strip()
                if not text:
                    continue
                if subtitle_boilerplate(text):
                    result.by_source_type[source_type] += 1
                    result.documents_affected.add(source_key)
                    if len(result.whole_chunk) < sample_per_kind:
                        result.whole_chunk.append(
                            QualityFinding(
                                source_type, source_key,
                                "whole-chunk-boilerplate", text[:160], "",
                            )
                        )
                    else:
                        result.whole_chunk.append(
                            QualityFinding(source_type, source_key,
                                           "whole-chunk-boilerplate", "", "")
                        )
                    continue
                cleaned = strip_caption_tail(text)
                if cleaned != text:
                    result.by_source_type[source_type] += 1
                    result.documents_affected.add(source_key)
                    if len(result.tails) < sample_per_kind:
                        result.tails.append(
                            QualityFinding(
                                source_type, source_key, "sign-off-tail",
                                text[-120:], cleaned[-90:] or "<empty>",
                            )
                        )
                    else:
                        result.tails.append(
                            QualityFinding(source_type, source_key,
                                           "sign-off-tail", "", "")
                        )
    finally:
        conn.close()
    return result
