"""Build Anthropic Message Batches requests for Contextual Retrieval.

One request per (document, chunk-window). Each request sends the full document
body once plus the window's chunks, and uses a forced tool call to return a
JSON array of per-chunk contexts. Windowing caps chunks-per-request so the
response fits Haiku's output limit.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from corpus.contextual.contextualizer import (
    DEFAULT_MODEL,
    MAX_CHUNK_CHARS,
    MAX_DOC_CHARS,
    SYSTEM_PROMPT,
)
from corpus.db.sqlite import StoredChunk

DEFAULT_WINDOW_SIZE = 40

CONTEXT_TOOL = {
    "name": "emit_chunk_contexts",
    "description": "Return a 1-2 sentence situating context for each numbered chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "contexts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The chunk's index in the list"},
                        "context": {"type": "string", "description": "1-2 sentence context, no chunk-content repetition"},
                    },
                    "required": ["index", "context"],
                },
            }
        },
        "required": ["contexts"],
    },
}


def _iter_windows(
    chunks: list[StoredChunk], window_size: int
) -> Iterator[tuple[str, list[StoredChunk]]]:
    """SINGLE SOURCE OF TRUTH for windowing: yield (custom_id, [chunk]) per window.

    Both the request builder and the persisted result-mapping derive from this, so
    a custom_id always maps to exactly the chunks that were sent in that request.
    custom_id = "{source_type}_{source_key}_{window_index}"."""
    by_key: dict[tuple[str, str], list[StoredChunk]] = defaultdict(list)
    for ch in chunks:
        by_key[(ch.source_type, ch.source_key)].append(ch)
    for (stype, skey), key_chunks in by_key.items():
        key_chunks.sort(key=lambda c: (c.metadata or {}).get("chunk_index", 0))
        for w, start in enumerate(range(0, len(key_chunks), window_size)):
            yield f"{stype}_{skey}_{w}", key_chunks[start : start + window_size]


def _chunks_block(window_chunks: list[StoredChunk]) -> str:
    lines = ["Chunks to contextualize (each needs a 1-2 sentence context situating it in the document):"]
    for i, ch in enumerate(window_chunks):
        text = ch.content.split("\n\n", 1)[-1]
        lines.append(f"\n<chunk index=\"{i}\">\n{text[:MAX_CHUNK_CHARS]}\n</chunk>")
    lines.append("\nCall emit_chunk_contexts with one entry per chunk index.")
    return "\n".join(lines)


def _window_user_message(doc_body: str, window_chunks: list[StoredChunk]) -> str:
    """Single-string user message (body + chunks) for the local Ollama path, which
    sends a plain string rather than Anthropic cache_control content blocks."""
    body = doc_body[:MAX_DOC_CHARS]
    return f"Document:\n<document>\n{body}\n</document>\n\n" + _chunks_block(window_chunks)


def build_batch_requests(
    chunks: list[StoredChunk],
    doc_bodies: dict[str, str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Group chunks by source_key, window them, and build one batch request per window.

    H1 prompt caching: the document body is its OWN content block carrying
    cache_control=ephemeral, so the cacheable prefix is [system + tools + body].
    Windows of the SAME document share that identical prefix → the body is cache-
    WRITTEN once and cache-READ (~10% price) for the doc's remaining windows. Big
    multi-window docs (some have 30-40 windows) stop re-paying full input price for
    an 80K-char body every window. The per-window chunk list is the uncached suffix.
    """
    requests: list[dict[str, Any]] = []
    for custom_id, window in _iter_windows(chunks, window_size):
        body = doc_bodies.get(window[0].source_key, "")[:MAX_DOC_CHARS]
        requests.append({
            # Anthropic custom_id must match ^[a-zA-Z0-9_-]{1,64}$ (no colons).
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": 4096,
                "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                "tools": [CONTEXT_TOOL],
                "tool_choice": {"type": "tool", "name": CONTEXT_TOOL["name"]},
                "messages": [{"role": "user", "content": [
                    {"type": "text",
                     "text": f"Document:\n<document>\n{body}\n</document>",
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": _chunks_block(window)},
                ]}],
            },
        })
    return requests


def build_window_mapping(chunks: list[StoredChunk], window_size: int = DEFAULT_WINDOW_SIZE) -> dict[str, list[str]]:
    """{custom_id: [chunk_id, …]} for every window — persisted at submit time so the
    apply step maps results to the EXACT chunks sent, independent of later DB state."""
    return {cid: [c.id for c in window] for cid, window in _iter_windows(chunks, window_size)}


def window_chunk_ids(chunks: list[StoredChunk], custom_id: str, window_size: int = DEFAULT_WINDOW_SIZE) -> list[str]:
    """LEGACY re-derivation of a custom_id's chunk IDs from a chunk list. Correct
    only when `chunks` matches the set used at build time — prefer the persisted
    build_window_mapping for cross-run applies. Kept for one-off batch recovery."""
    parts = custom_id.split("_")
    stype, skey, w = parts[0], "_".join(parts[1:-1]), int(parts[-1])
    key_chunks = sorted(
        [c for c in chunks if c.source_type == stype and c.source_key == skey],
        key=lambda c: (c.metadata or {}).get("chunk_index", 0),
    )
    start = w * window_size
    return [c.id for c in key_chunks[start : start + window_size]]


def parse_batch_result(tool_input: dict[str, Any], window_chunk_ids: list[str]) -> dict[str, str]:
    """Map a tool_use result's {contexts:[{index,context}]} onto chunk IDs by position.

    Tolerant of model-output drift:
      - `contexts` may be the whole JSON array serialized as a STRING — decode it
        first (otherwise we'd iterate the string's CHARACTERS and write "[", "{",
        '"' … as chunk contexts — a real corruption bug caught in batch recovery).
      - `contexts` may be a bare list of strings (no index/context objects) — map
        by position.
    Malformed entries are skipped, never raised, so one bad entry can't abort
    applying the rest of the batch."""
    contexts = tool_input.get("contexts", []) or []
    if isinstance(contexts, str):
        try:
            contexts = json.loads(contexts)
        except (ValueError, TypeError):
            return {}
    if not isinstance(contexts, list):
        return {}
    mapping: dict[str, str] = {}
    for pos, entry in enumerate(contexts):
        if isinstance(entry, dict):
            idx = entry.get("index", pos)
            ctx = entry.get("context")
        elif isinstance(entry, str):
            idx, ctx = pos, entry          # bare-string list → positional
        else:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = pos
        if not isinstance(ctx, str) or not ctx.strip():
            continue
        if 0 <= idx < len(window_chunk_ids):
            mapping[window_chunk_ids[idx]] = ctx
    return mapping
