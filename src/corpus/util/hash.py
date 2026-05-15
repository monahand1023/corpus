from __future__ import annotations

import hashlib

from corpus.types import ChunkKind


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(
    source_type: str,
    source_key: str,
    chunk_kind: ChunkKind,
    chunk_index: int,
) -> str:
    return f"{source_type}:{source_key}:{chunk_kind.value}:{chunk_index}"
