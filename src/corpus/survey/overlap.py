"""Index overlap estimator: how much of a directory is already indexed in an
existing corpus database, before ingesting it somewhere and duplicating
thousands of documents.

Method (the technique that worked by hand, made repeatable): reservoir-
sample up to `sample_size` plain-text-decodable documents from the tree,
extract one distinctive phrase per document (the longest line meeting a
minimum word count — long enough to be unlikely by chance, short enough to
survive minor reformatting), and check each phrase against the target
database in two stages:

  1. **Recall** — `ChunkStore.fts_search` (BM25 over the existing
     `chunks_fts` index) to find candidate chunks that share vocabulary with
     the phrase. Cheap, and reuses the store's own public search API rather
     than hand-rolling SQL against `chunks_fts`/`chunks` directly.
  2. **Confirmation** — a literal, case-insensitive, whitespace-normalized
     substring check of the phrase against each candidate's `content`. This
     is the step that makes the estimate defensible rather than a bare
     "matched" count: FTS5's `porter` stemmer and OR-of-terms matching would
     otherwise call two chunks "matching" when they merely share common
     words, not the actual phrase.

The result is a sampled proportion with a stated Wilson-interval confidence
range — not a bare percentage — plus explicit caveats about what this
estimate does and doesn't cover (see `OverlapResult`'s docstring and the CLI
output). It is read-only in both directions: opens the target database via
`ChunkStore(read_only=True)`, and never writes to or otherwise touches the
directory being surveyed.
"""

from __future__ import annotations

import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from corpus.db.sqlite import ChunkStore
from corpus.survey.sampling import ReservoirSampler
from corpus.survey.stats import wilson_interval
from corpus.survey.walk import WalkStats, walk_files

# Extensions this module can extract a phrase from without any optional
# extra installed — plain UTF-8-decodable text. Deliberately excludes
# PDF/DOCX/XLSX/RTF/HTML: extracting THEIR text reuses those connectors'
# optional-extra-gated parsing, which would make this subcommand's dependency
# footprint depend on which extras happen to be installed. The gap is
# reported honestly in the CLI output rather than silently skipped — run
# `corpus-survey census` to see how much of the tree those formats cover.
PLAIN_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml"}
)

DEFAULT_SAMPLE_SIZE = 40
DEFAULT_MIN_WORDS = 8
DEFAULT_MAX_READ_BYTES = 65_536  # cap per-file read; a phrase from the first
# 64KB is as good as one from anywhere else in the file, and this keeps a
# single huge log file from dominating read time.
DEFAULT_TOP_K = 5  # FTS candidates pulled per phrase before substring confirmation

_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", text).strip().casefold()


def _extract_phrase(path: Path, min_words: int, max_read_bytes: int) -> str | None:
    """Return the longest line (by word count) with at least `min_words`
    words from the first `max_read_bytes` of `path`, or `None` if the file
    can't be decoded or has no qualifying line. A long line is more likely
    to be distinctive (low false-match rate) than a short one, and "longest
    among candidates" is a simple, deterministic tie-break."""
    try:
        with path.open("rb") as f:
            raw = f.read(max_read_bytes)
    except OSError:
        return None
    text = raw.decode("utf-8", errors="ignore")

    best: str | None = None
    best_words = 0
    for line in text.splitlines():
        stripped = line.strip()
        words = stripped.split()
        if len(words) < min_words:
            continue
        if len(words) > best_words:
            best = stripped
            best_words = len(words)
    return best


def _read_stored_embedding_dim(db_path: Path) -> int | None:
    """Read the embedding dim a database was actually created with, so
    `ChunkStore`'s open-time dim guard is fed its own value instead of a
    guess — this module never needs to know or care what embedder produced
    the target database, only that opening it read-only must not raise a
    dim-mismatch error over a value this module invented."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dim'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return int(row[0]) if row is not None else None


@dataclass
class SampledDocument:
    rel_path: str
    phrase: str
    matched: bool
    matched_source: str | None = None  # "<source_type>::<source_key>" of the confirming chunk


@dataclass
class OverlapResult:
    root: str
    db_path: str
    excludes: tuple[str, ...]
    use_default_excludes: bool
    eligible_document_count: int
    sample: list[SampledDocument] = field(default_factory=list)
    walk_stats: WalkStats = field(default_factory=WalkStats)

    @property
    def sample_size(self) -> int:
        return len(self.sample)

    @property
    def matched_count(self) -> int:
        return sum(1 for s in self.sample if s.matched)

    @property
    def estimated_overlap_fraction(self) -> float | None:
        return self.matched_count / self.sample_size if self.sample else None

    @property
    def confidence_interval_95(self) -> tuple[float, float] | None:
        return wilson_interval(self.matched_count, self.sample_size) if self.sample else None


def run_overlap_survey(
    root: Path,
    db_path: Path,
    excludes: tuple[str, ...] = (),
    use_default_excludes: bool = True,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_words: int = DEFAULT_MIN_WORDS,
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
) -> OverlapResult:
    rng = random.Random(seed)
    stats = WalkStats()
    sampler: ReservoirSampler[tuple[str, Path, str]] = ReservoirSampler(sample_size, rng)
    eligible = 0

    for wf in walk_files(root, excludes, use_default_excludes, stats=stats):
        if wf.path.suffix.lower() not in PLAIN_TEXT_EXTENSIONS:
            continue
        phrase = _extract_phrase(wf.path, min_words, max_read_bytes)
        if phrase is None:
            continue
        eligible += 1
        sampler.offer((wf.rel_path, wf.path, phrase))

    result = OverlapResult(
        root=str(root),
        db_path=str(db_path),
        excludes=excludes,
        use_default_excludes=use_default_excludes,
        eligible_document_count=eligible,
        walk_stats=stats,
    )
    if not sampler.sample:
        return result

    dim = _read_stored_embedding_dim(db_path) or 1
    store = ChunkStore(db_path, embedding_dim=dim, read_only=True)
    try:
        for rel_path, _path, phrase in sampler.sample:
            candidates = store.fts_search(phrase, top_k=top_k)
            target = _normalize(phrase)
            matched_source = None
            for c in candidates:
                if target in _normalize(c.content):
                    matched_source = f"{c.source_type}::{c.source_key}"
                    break
            result.sample.append(
                SampledDocument(
                    rel_path=rel_path,
                    phrase=phrase,
                    matched=matched_source is not None,
                    matched_source=matched_source,
                )
            )
    finally:
        store.close()

    return result
