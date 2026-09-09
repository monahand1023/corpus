"""Contextual Retrieval: an LLM-written sentence prepended to each chunk
before embedding, so a fragment retrieves on what it is about rather than
only on the words it happens to contain.

The problem it solves is structural, not incidental. Chunking a long document
produces pieces that are individually ambiguous — "Yes, approved, let's go
with option B" names no project, no person, and no date, so no query for any
of those will ever find it. Prepending "From the June 2015 roadmap thread
where option B of the migration was approved" makes the same chunk findable
by all three. Anthropic's published result for the technique is a ~35%
reduction in retrieval failure rate, and ~49% combined with reranking (which
`corpus.reranker` already provides).

**The cost mechanism is prompt caching.** Each request sends the whole parent
document as a `cache_control=ephemeral` prefix, billed at roughly a tenth of
normal input tokens on a cache hit. A document therefore pays full price once
and every subsequent chunk of it is ~90% cheaper. This is why
`ChunkStore.chunks_missing_context` orders by `(source_key, chunk_index)` and
why `batch_builder` windows requests by document: break that adjacency and
the cost multiplies by roughly the number of chunks per document.

**Which chunks are worth contextualizing** is a real decision, not a
formality, and the answer is corpus-shaped rather than universal. Measured on
two real archives:

  - Prose whose chunks are fragments of a larger thread or page — email,
    tickets, meeting notes, long documents — benefits most. A middle chunk
    names almost nothing identifying.
  - Source code benefits least. A chunk of code already carries its function
    names, imports, and identifiers, and those are exactly what people search
    for. On one archive `code` was 65% of all chunks and would have consumed
    most of the budget for the least gain.
  - Chunks shorter than a few dozen tokens are actively made worse: the
    context sentence rivals the content in length, so the embedding ends up
    describing the context rather than the chunk.

Hence `should_contextualize`, and hence `min_tokens` being configurable per
source in `corpus.toml` rather than a constant here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Below this, the generated context rivals the chunk in length and the
# embedding describes the blurb rather than the content.
DEFAULT_MIN_TOKENS = 50

# How much of a parent document is sent as the cached prefix. A pathological
# input — a generated file, a giant export — would otherwise dominate the
# bill for the one document it belongs to.
MAX_DOC_CHARS = 80_000

# The model only needs enough of the chunk to place it within the document.
# It is not summarizing the chunk, so sending all of a long one buys nothing.
MAX_CHUNK_CHARS = 2_000

SYSTEM_PROMPT = """You write concise contextual descriptions for document chunks.
Given a full document and one chunk from it, output a single sentence (max 2)
that situates the chunk within the document for search retrieval purposes.

Rules:
- Be specific: name the document type, project, ticket, system, or date if present.
- Do NOT repeat the chunk content verbatim.
- Do NOT include filler like "This chunk discusses..." — just state the context.
- Output the context sentence only. No preamble, no explanation.

Example output: "From the 2022 warranty widget proposal, the section covering rollout timeline and success metrics."
"""

_USER_TEMPLATE = """Document:
<document>
{doc_body}
</document>

Chunk to contextualize:
<chunk>
{chunk_content}
</chunk>

Write a 1-2 sentence context for this chunk:"""


def build_context_prompt(doc_body: str, chunk_content: str) -> str:
    """The user message for one context-generation call."""
    if len(doc_body) > MAX_DOC_CHARS:
        doc_body = doc_body[:MAX_DOC_CHARS] + "\n...[truncated]"
    return _USER_TEMPLATE.format(
        doc_body=doc_body, chunk_content=chunk_content[:MAX_CHUNK_CHARS]
    )


def should_contextualize(token_count: int | None, min_tokens: int = DEFAULT_MIN_TOKENS) -> bool:
    """Whether a chunk earns a context sentence.

    A missing `token_count` counts as 0 and is therefore skipped. That is the
    safe direction: it declines to spend rather than spending blind on a chunk
    whose size is unknown.
    """
    return (token_count or 0) >= min_tokens


def estimate_chunk_cost_tokens(doc_chars: int, chunk_chars: int) -> tuple[int, int]:
    """Rough (uncached_input, output) token counts for one chunk's request.

    Used by the planner to price a run before it is submitted. Uses the same
    chars/4 heuristic as `corpus.util.tokens` — the real tokenizer shifts
    individual requests by ~10% and shifts non-English text considerably more,
    so a caller reporting this to a user should say so rather than presenting
    it as a quote.
    """
    doc = min(doc_chars, MAX_DOC_CHARS)
    chunk = min(chunk_chars, MAX_CHUNK_CHARS)
    return (doc + chunk) // 4, 42
