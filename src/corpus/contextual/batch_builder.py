"""Build Anthropic Message Batches requests for Contextual Retrieval.

One request per (document, chunk-window). Each request sends the full document
body once plus the window's chunks, and uses a forced tool call to return a
JSON array of per-chunk contexts. Windowing caps chunks-per-request so the
response fits Haiku's output limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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

logger = logging.getLogger(__name__)

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


CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def window_custom_id(source_type: str, source_key: str, window_index: int) -> str:
    """A batch custom_id for one window: valid for any source_key, always.

    Anthropic rejects anything outside `^[a-zA-Z0-9_-]{1,64}$`, and rejects the
    whole batch rather than the offending request. Source keys are file paths,
    so they are hashed rather than sanitized — sanitizing would collapse
    `a/b.md` and `a_b.md` onto one id and silently apply one document's
    contexts to another's chunks.
    """
    digest = hashlib.sha256(f"{source_type}\x00{source_key}\x00{window_index}".encode()).hexdigest()
    return f"w{window_index}_{digest[:32]}"


def _iter_windows(
    chunks: list[StoredChunk], window_size: int
) -> Iterator[tuple[str, list[StoredChunk]]]:
    """SINGLE SOURCE OF TRUTH for windowing: yield (custom_id, [chunk]) per window.

    Both the request builder and the persisted result-mapping derive from this,
    so a custom_id always maps to exactly the chunks that were sent in that
    request.

    The id is a hash, NOT the source_key. Anthropic requires custom_id to match
    `^[a-zA-Z0-9_-]{1,64}$`, and corpus source_keys are file paths — slashes,
    dots, spaces, non-ASCII. Interpolating one produced a 400 that failed the
    ENTIRE batch, and only after the request had been built and sent. A sibling
    archive keyed by hex hashes never hit it, which is exactly why it survived
    review: the constraint was written in a comment next to the field and
    nothing enforced it.

    Hashing means the id carries no meaning, which is fine — the mapping from
    id to chunk ids is persisted at submit time and is what results are applied
    through. It is deterministic so the same window yields the same id across
    runs.
    """
    by_key: dict[tuple[str, str], list[StoredChunk]] = defaultdict(list)
    for ch in chunks:
        by_key[(ch.source_type, ch.source_key)].append(ch)
    for (stype, skey), key_chunks in by_key.items():
        key_chunks.sort(key=lambda c: (c.metadata or {}).get("chunk_index", 0))
        for w, start in enumerate(range(0, len(key_chunks), window_size)):
            yield window_custom_id(stype, skey, w), key_chunks[start : start + window_size]


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


def window_chunk_ids(
    chunks: list[StoredChunk], custom_id: str, window_size: int = DEFAULT_WINDOW_SIZE
) -> list[str]:
    """Chunk ids for a custom_id, re-derived from a chunk list.

    Correct only when `chunks` matches the set used at build time; prefer the
    mapping persisted at submit time (`build_window_mapping`) for cross-run
    applies. Kept for one-off batch recovery.

    Re-derives by rebuilding the windows and matching the id, rather than
    parsing the id apart. The id used to be `{source_type}_{source_key}_{w}`
    and was parsed with `split("_")` — which was already wrong for any
    source_key containing an underscore, and became wrong for every id once
    the id became a hash. Going through `_iter_windows` means this function
    cannot disagree with what was actually sent.
    """
    for candidate_id, window in _iter_windows(chunks, window_size):
        if candidate_id == custom_id:
            return [c.id for c in window]
    return []


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
    # Refuse to guess when the model omitted its indices AND returned a
    # different number of entries than were sent. Falling back to position
    # then attaches one chunk's context to another — silently, at scale, with
    # no error, which is the single worst outcome this module has. A refused
    # window costs one re-run; a misaligned one corrupts the index invisibly.
    indexed = [e for e in contexts if isinstance(e, dict) and "index" in e]
    if not indexed and len(contexts) != len(window_chunk_ids):
        logger.warning(
            "discarding a window: model returned %d context(s) for %d chunk(s) "
            "with no indices, so position cannot be trusted",
            len(contexts),
            len(window_chunk_ids),
        )
        return {}

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
