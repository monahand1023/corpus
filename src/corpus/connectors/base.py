from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from corpus.types import Chunk, SourceDocument


class Connector(Protocol):
    source_type: str

    def load(self) -> Iterable[SourceDocument]: ...


class Chunker(Protocol):
    source_type: str

    def chunk(self, doc: SourceDocument) -> list[Chunk]: ...
