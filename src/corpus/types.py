"""Boundary types shared across the retrieval pipeline.

`source_type` is intentionally a free-form string, not an enum — each
deployment defines its own source names in `corpus.toml`. The validation
regex restricts to lowercase identifier-style labels so they're safe to
use as filenames, table values, and MCP schema enums.

`ChunkKind` stays an enum: it's a genuinely closed set of structural
categories that any connector's chunker emits.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SOURCE_TYPE_PATTERN = r"^[a-z][a-z0-9_]*$"


class ChunkKind(StrEnum):
    HEADER = "header"
    BODY = "body"
    COMMENTS = "comments"
    SECTION = "section"
    METADATA = "metadata"


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_type: str = Field(pattern=SOURCE_TYPE_PATTERN)
    source_key: str
    chunk_kind: ChunkKind
    chunk_index: int
    title: str
    url: str | None = None
    author: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    token_count: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    content: str
    content_hash: str
    metadata: ChunkMetadata


class SourceDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_type: str = Field(pattern=SOURCE_TYPE_PATTERN)
    source_key: str
    title: str
    url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    raw: Any
