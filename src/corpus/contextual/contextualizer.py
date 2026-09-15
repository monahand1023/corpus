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
  - Short chunks are actively made WORSE, and the threshold is higher than it
    looks. The deciding quantity is the context's share of the embedded text,
    not the chunk's absolute length: measured on a real archive, 7-8% was
    neutral-to-positive and 28% was clearly negative (recall 0.683 vs 0.750
    on vector-only retrieval, Wilcoxon p=0.0002). See `DEFAULT_MIN_TOKENS`.

    CONFIRMED THE HARD WAY on a second archive, 2026-09-15. Transcribed audio
    chunks one ~30-second window at a time, and exactly 1 of 45,092 chunks
    cleared the 260-token floor -- so the floor was lowered to 10 to reach the
    documents a gold set showed were failing. That is a coverage argument, and
    coverage is the wrong quantity. At a floor of 10 a ~40-character chunk
    carries a ~141-character context: a share of roughly 78%, far past the 28%
    already measured as harmful. Result on 34,872 contextualised chunks:

        recall@5  0.625 -> 0.500    MRR 0.442 -> 0.406    nDCG 0.486 -> 0.429

    It fixed none of the failing queries and broke one that had been passing.
    The mechanism is the one predicted above: every window of a recording gets
    a near-identical context, which collapses the distinctions between them.

    The general rule this archive illustrates: contextual retrieval assumes a
    chunk is a FRAGMENT of something longer. Where the chunk is most of what
    exists, there is nothing for a context sentence to situate, and it only
    adds a shared prefix that makes siblings look alike. Check the context's
    share BEFORE lowering the floor to improve coverage.

Hence `should_contextualize`, and hence `min_tokens` being configurable per
source in `corpus.toml` rather than a constant here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# MEASURED, not guessed. A matched-pair experiment on 2026-09-10 (
# chunks, 360 balanced gold queries, arm A contextualized vs arm B identical
# chunks embedded from content alone) found that what decides whether context
# helps is not the chunk's absolute size but the context's SHARE of the
# embedded text:
#
#     ctx share  7%  ->  recall 0.833 vs 0.833   (neutral)
#     ctx share  8%  ->  recall 0.600 vs 0.583   (slightly positive)
#     ctx share 28%  ->  recall 0.733 vs 0.775   (clearly NEGATIVE;
#                        0.683 vs 0.750 vector-only, Wilcoxon p=0.0002)
#
# The generated context is a near-constant ~141 characters whatever the chunk
# size, and every one is written by the same model from the same prompt, so
# they are formulaic. At a small share that is harmless; at a quarter of the
# text it injects a large SHARED component into every vector and compresses
# the distinctions between them. Short-chunk sources lose.
#
# Holding the share at or under 12% needs content of at least
# 141/0.12 - 141 = 1,034 characters, which is ~258 tokens in the chars/4 units
# `token_count` is recorded in. Rounded to 260.
#
# The old default was 50, which passes a 200-character chunk whose context
# would be 41% of its embedding -- deep into the range measured as harmful.
DEFAULT_MIN_TOKENS = 260

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


# A model asked to situate an empty or fully-redacted chunk has nothing to
# work with, and says so. Measured on a real archive: 0.037% of generated
# contexts came back as "<UNKNOWN>" or a bare "Empty or redacted document
# section." Those are honest answers, not failures — but storing one
# prepends a meaningless token to the chunk's embedding AND marks the chunk
# done, so it is never reconsidered.
_PLACEHOLDER_CONTEXTS = frozenset(
    {"<unknown>", "unknown", "n/a", "none", "empty", "empty or redacted document section."}
)
MIN_USEFUL_CONTEXT_CHARS = 20


def is_useful_context(context: str) -> bool:
    """Whether a generated context is worth storing and embedding."""
    text = (context or "").strip()
    if len(text) < MIN_USEFUL_CONTEXT_CHARS:
        return False
    return text.lower() not in _PLACEHOLDER_CONTEXTS
