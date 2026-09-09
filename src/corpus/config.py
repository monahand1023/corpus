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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class SourceConfig(BaseModel):
    name: str = Field(pattern=SOURCE_TYPE_PATTERN)
    type: str  # which built-in connector to use, e.g. "markdown"
    path: str
    glob: str | None = None

    def resolved_path(self) -> Path:
        return Path(os.path.expanduser(self.path)).resolve()


class ReferencePattern(BaseModel):
    """Compiled lazily; we keep the raw pattern string for serialization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pattern: str
    source_type: str = Field(pattern=SOURCE_TYPE_PATTERN)
    description: str | None = None

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


class CorpusConfig(BaseModel):
    db_path: Path = Path("./corpus.db")
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
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
