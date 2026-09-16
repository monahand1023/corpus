"""Load and validate `corpus.toml`.

The config is the central knob: source paths, embedder choice, reference
patterns. Everything that needs to vary between deployments lives here so
the codebase itself stays generic.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from corpus.types import SOURCE_TYPE_PATTERN

DEFAULT_CONFIG_PATH = Path("corpus.toml")


class ConfigError(Exception):
    """A user-facing configuration problem: missing file, invalid TOML, or
    values that fail validation. Carries a readable message (no traceback)."""


class EmbedderConfig(BaseModel):
    provider: str = "voyage"
    model: str = "voyage-4-large"
    dim: int = Field(default=1024, gt=0)


class RetrieverConfig(BaseModel):
    top_k: int = 5
    max_per_source_type: int | None = 3
    hybrid: bool = True


class RerankerConfig(BaseModel):
    """Local cross-encoder re-ranker settings (only consulted when a caller
    passes `--rerank`; `rerank` itself stays a CLI/query-time flag, not a
    config toggle, so eval/judge can compare on vs. off in one run).

    `device` is explicit on purpose — see corpus/reranker/local.py for why we
    never let sentence-transformers auto-select MPS on Apple Silicon."""

    device: str = "cpu"


class PruningConfig(BaseModel):
    """Guards `Ingester.ingest`'s orphan sweep against a connector that
    under-yields documents while still honestly reporting zero failures — a
    connector-side bug the `failed_files` counter cannot see, because the
    connector isn't reporting any failure. If it happened once (a connector
    skipping unchanged files by fingerprint, which then yielded nothing at
    all on a later run) it can happen again, so the engine enforces a rail
    rather than relying on every connector author to get this right.

    `delete_orphans` refuses to prune a source when the chunks it would
    delete exceed `max_orphan_ratio` of that source's existing chunk count —
    UNLESS the source has fewer than `min_chunks_for_guard` chunks to begin
    with, since a small or freshly-created source can cross a 20% ratio on
    a handful of genuine deletions and the blast radius is too small to
    matter. `--prune-anyway` at the CLI (or `force=True` on `delete_orphans`)
    bypasses the guard entirely for a deliberate bulk deletion."""

    max_orphan_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    min_chunks_for_guard: int = Field(default=50, ge=0)


class PerformanceConfig(BaseModel):
    """SQLite memory-tuning pragmas, beyond the engine's fixed baseline
    (WAL journal mode, synchronous=NORMAL, busy_timeout=5000 -- see
    `ChunkStore._open_connection`). SQLite's own default page cache is
    tiny (~2MB), which is fine for a small corpus but not for the
    hundreds-of-MB-to-multi-GB indexes a real personal archive reaches.

    These defaults come from isolating each pragma's actual contribution
    on a synthetic 150k-chunk / 711MB store (2026-09-09; see
    .superpowers/sdd/2026-09-09-corpus-hardening/pragma-benchmark.md for
    the full methodology and raw numbers), not from bundling all three on
    faith:

    - `mmap_size` is responsible for effectively the ENTIRE speedup.
      sqlite-vec's `chunks_vec` (vec0) does an exhaustive scan per KNN
      query -- there's no ANN index, so every query touches ~all of it,
      meaning there is no "hot" subset for a bigger page cache to protect.
      Memory-mapped I/O still helps enormously because it turns each page
      touch into a direct memory read instead of a read() syscall + copy,
      a win that applies to literally every page regardless of caching.
      Measured 3.18x-3.29x on vector KNN (k=20) once the mmap window
      covered the whole file; only 1.37x at a window covering ~1/3 of it.
      **mmap_size must cover the store's actual size to get most of the
      benefit** -- a "few tens of MB" default (reasonable for cache_size)
      would barely move the needle here.
    - `cache_size` measured NO effect, even reusing a single persistent
      connection across six repeated rounds of the same queries (SQLite's
      own page cache had every chance to warm and never helped) --
      expected, given the point above: nothing to cache helps a query
      that touches everything anyway. Kept at a modest default regardless,
      as a safety net for access patterns this benchmark didn't exercise
      (point lookups: `get_by_id`, the `id IN (...)` dedup check ingest
      runs before every batch, etc.) which DO have real page locality.
    - `temp_store_memory` also measured no effect. `EXPLAIN QUERY PLAN`
      confirms `vector_search`'s `ORDER BY` genuinely does use a temp
      b-tree (`fts_search`'s rank-ordered `LIMIT` does not) -- but it
      only sorts the ~`top_k` result rows, trivially cheap on disk or in
      memory either way at that size. Left on anyway: free today, and a
      margin against some future query shape that sorts something larger.

    Why `mmap_size` gets a generous default while `cache_size` stays
    conservative, and both are fixed rather than scaled off detected
    physical memory: `mmap_size` is a CEILING on a lazily-paged-in, clean,
    evictable virtual mapping, not a memory reservation -- setting it
    larger than the store costs nothing until pages are actually touched,
    and a store bigger than the ceiling just falls back to ordinary I/O
    for the excess rather than erroring. `cache_size`'s memory, by
    contrast, is a real allocation proportional to what's configured, so
    it stays small. Scaling either off detected total RAM was the other
    option the brief for this offered; skipped because there's no portable
    stdlib way to read total physical memory (no `os.sysconf` on Windows),
    and corpus already keeps its base install dependency-minimal on
    purpose (see "Why embedders are optional extras" in
    docs/configuration.md) -- pulling in `psutil` just for this felt like
    the wrong tradeoff against a fixed, documented, one-line override in
    `corpus.toml`.
    """

    cache_size_mb: int = Field(default=64, gt=0)
    mmap_size_mb: int = Field(default=1024, gt=0)
    temp_store_memory: bool = True


class QueryLogConfig(BaseModel):
    """Append served queries to a local JSONL file. OFF by default.

    The point is to answer tuning questions with the queries actually being
    served rather than a synthesised gold set, because a synthesised set
    measures its own construction (see `corpus.query_log`).

    Queries are revealing, so this is opt-in, the file stays local, and
    `include_results` can be cleared to record that a query happened without
    the documents it surfaced.
    """

    enabled: bool = False
    # Defaults alongside the store when enabled and unset.
    path: Path | None = None
    include_results: bool = True


class ContextualConfig(BaseModel):
    """Settings for `corpus-contextualize`.

    Off by default: it calls a paid API over every eligible chunk, so opting
    in is a decision the operator makes rather than something a fresh config
    does on their behalf.
    """

    model: str = "claude-haiku-4-5-20251001"
    # Global floor; a source can raise or lower it via `context_min_tokens`.
    min_tokens: int = 50
    # Chunks per request. Each request re-sends the parent document, so a
    # larger window means fewer paid document copies — bounded by the model's
    # output limit, since every chunk in the window needs a context back.
    window_size: int = 40


class SourceConfig(BaseModel):
    name: str = Field(pattern=SOURCE_TYPE_PATTERN)
    type: str  # which built-in connector to use, e.g. "markdown"
    path: str
    glob: str | None = None
    # `zip` connector only (every other connector ignores this field, same as
    # they already ignore `glob`'s zip-specific meaning of "which archives to
    # find" vs. "which files to find"). When true (the default), members
    # under a vendored-dependency or build-output directory (`node_modules`,
    # `site-packages`, `.git`, etc.) or matching an obvious minified/compiled
    # leaf pattern (`*.min.js`, `*.map`, ...) are excluded before extraction —
    # see `corpus.connectors.zip._is_dependency_noise`. Set to false to index
    # a library's vendored docs on purpose.
    # Paths this source must not index, as fnmatch patterns matched against
    # the root-relative path AND the basename, so "Backup", "**/Backup/*" and
    # "Backup/report.md" all do what the reader expects.
    #
    # It exists so the documents `corpus-survey duplicates` reports as wholly
    # duplicated can actually be dropped. Naming them while offering no way to
    # act on them is the same defect as a setting documented in three places
    # and read in none.
    exclude: list[str] = Field(default_factory=list)
    exclude_dependencies: bool = True
    # `aup3` connector only -- every other connector ignores these two.
    # Audacity's `.aup3` project files store sample rate and channel count
    # only inside their unparseable binary-XML `project.doc` column (see
    # `corpus.connectors.aup3`'s module docstring), so neither is ever
    # detected -- these are the assumed values, reported as assumptions in
    # the generated document, overridable here when you know the real ones
    # for a given source. Defaults mirror
    # `corpus.connectors.aup3.DEFAULT_SAMPLE_RATE_HZ`/`DEFAULT_CHANNELS`
    # (not imported from there -- config.py stays free of any dependency on
    # a specific connector's module).
    sample_rate: int = 44100
    channels: int = 1
    # `olm` connector only -- every other connector ignores these three.
    # `olm_folders` is a list of fnmatch patterns against a message's
    # store-relative folder path (e.g. "archive PST/Inbox*"); None means
    # every folder. An `.olm` mixes years and employers in one file, so
    # scoping by folder is how you index part of a mailbox without paying to
    # embed all of it. `olm_skip_mirror_tree` drops the archive's duplicate
    # of itself under "Outlook for Mac Archive OLM/On My Computer/" -- ~50%
    # of a real archive's messages -- and `olm_trim_quotes` keeps only what
    # each reply added rather than re-embedding the quoted thread once per
    # message. See `corpus.connectors.olm`'s module docstring for the
    # measurements behind both defaults.
    # Contextual Retrieval, per source. `contextualize` opts this source in;
    # `context_min_tokens` overrides the global floor for it, because the
    # right floor is corpus-shaped: prose fragments benefit far below the
    # threshold that source code does. See `corpus.contextual`.
    contextualize: bool = False
    context_min_tokens: int | None = None
    olm_folders: list[str] | None = None
    olm_skip_mirror_tree: bool = True
    olm_trim_quotes: bool = True

    def resolved_path(self) -> Path:
        return Path(os.path.expanduser(self.path)).resolve()


class ReferencePattern(BaseModel):
    """Compiled lazily; we keep the raw pattern string for serialization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pattern: str
    source_type: str = Field(pattern=SOURCE_TYPE_PATTERN)
    description: str | None = None

    @field_validator("pattern")
    @classmethod
    def _must_be_a_valid_regex(cls, value: str) -> str:
        """Compile once at load, so a typo is a config error.

        `compile()` is otherwise called lazily by whichever command reaches
        `compiled_references()` first, which turned a bad pattern into a raw
        `re.error` traceback at CLI startup -- and in the MCP server into a
        process that dies before printing anything, which the client reports
        only as "server failed to start".
        """
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(
                f"[[references]] pattern {value!r} is not a valid regular "
                f"expression: {exc}"
            ) from exc
        return value

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


class CorpusConfig(BaseModel):
    db_path: Path = Path("./corpus.db")
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    contextual: ContextualConfig = ContextualConfig()
    query_log: QueryLogConfig = Field(default_factory=QueryLogConfig)
    sources: list[SourceConfig] = Field(default_factory=list)
    references: list[ReferencePattern] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None) -> CorpusConfig:
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            raise ConfigError(
                f"corpus.toml not found at {config_path}. "
                "Copy corpus.toml.example to corpus.toml and edit."
            )
        try:
            raw = tomllib.loads(config_path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"corpus.toml at {config_path} is not valid TOML: {e}") from e
        # Top-level [corpus] section maps to db_path etc.
        corpus_section = raw.get("corpus", {})
        merged = {
            "db_path": Path(corpus_section.get("db_path", "./corpus.db")),
            "embedder": raw.get("embedder", {}),
            "retriever": raw.get("retriever", {}),
            "reranker": raw.get("reranker", {}),
            "pruning": raw.get("pruning", {}),
            "performance": raw.get("performance", {}),
            # Every field declared on this class must appear here. `contextual`
            # was declared and never read, so `[contextual]` in a corpus.toml
            # parsed as valid TOML and was silently discarded -- min_tokens,
            # model and window_size could not be configured at all, and
            # nothing said so. `test_every_declared_section_is_read` pins it.
            "contextual": raw.get("contextual", {}),
            "query_log": raw.get("query_log", {}),
            "sources": raw.get("sources", []),
            "references": raw.get("references", []),
        }
        try:
            return cls.model_validate(merged)
        except ValidationError as e:
            raise ConfigError(f"corpus.toml at {config_path} has invalid values:\n{e}") from e

    def source_by_name(self, name: str) -> SourceConfig | None:
        return next((s for s in self.sources if s.name == name), None)

    def compiled_references(self) -> list[tuple[re.Pattern[str], str]]:
        """Return [(compiled_pattern, target_source_type), ...] for runtime use."""
        return [(r.compile(), r.source_type) for r in self.references]
