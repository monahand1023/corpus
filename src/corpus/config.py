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

from pydantic import BaseModel, ConfigDict, Field

from corpus.types import SOURCE_TYPE_PATTERN

DEFAULT_CONFIG_PATH = Path("corpus.toml")


class EmbedderConfig(BaseModel):
    provider: str = "voyage"
    model: str = "voyage-3-large"
    dim: int = 1024


class RetrieverConfig(BaseModel):
    top_k: int = 5
    max_per_source_type: int | None = 3
    hybrid: bool = True


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
    sources: list[SourceConfig] = Field(default_factory=list)
    references: list[ReferencePattern] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None) -> CorpusConfig:
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            raise FileNotFoundError(
                f"corpus.toml not found at {config_path}. "
                "Copy corpus.toml.example to corpus.toml and edit."
            )
        raw = tomllib.loads(config_path.read_text())
        # Top-level [corpus] section maps to db_path etc.
        corpus_section = raw.get("corpus", {})
        merged = {
            "db_path": Path(corpus_section.get("db_path", "./corpus.db")),
            "embedder": raw.get("embedder", {}),
            "retriever": raw.get("retriever", {}),
            "sources": raw.get("sources", []),
            "references": raw.get("references", []),
        }
        return cls.model_validate(merged)

    def source_by_name(self, name: str) -> SourceConfig | None:
        return next((s for s in self.sources if s.name == name), None)

    def compiled_references(self) -> list[tuple[re.Pattern[str], str]]:
        """Return [(compiled_pattern, target_source_type), ...] for runtime use."""
        return [(r.compile(), r.source_type) for r in self.references]
