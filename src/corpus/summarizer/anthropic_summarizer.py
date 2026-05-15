"""Per-doc summarization via Claude Haiku.

Generic across source types — the system prompt uses the user's `source_type`
label as context ("This is a `tickets` document...") rather than baking in
specific knowledge of Jira/PR/Notion/etc. If your archive has structurally
different source types, customize the SYSTEM_PROMPT or override per-type
in the corpus.toml [summarizer] section (future v0.2 feature).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_INPUT_CHARS = 80_000

SYSTEM_PROMPT = """You write tight, factual summaries of documents from a personal archive.

Constraints:
- Target ~120 words. Hard cap 200.
- First sentence states what the document IS and what it's about.
- Surface concrete decisions, names, dates, and specific identifiers mentioned.
- No filler ("This document discusses...", "In conclusion..."). Lead with substance.
- No editorializing. Stick to what the document says.
- Use the document's own terminology (specific names, project codes, identifiers).
"""


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int


class AnthropicSummarizer:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing. Set in .env or pass api_key= to AnthropicSummarizer."
            )
        from anthropic import Anthropic

        self._client = Anthropic(api_key=key)
        self._model = model

    def summarize(
        self,
        source_type: str,
        title: str,
        content: str,
    ) -> SummaryResult:
        truncated = content[:MAX_INPUT_CHARS]
        guidance = (
            f"This is a document from your `{source_type}` source type. "
            "Capture: what it covers, any concrete decisions or facts it surfaces, "
            "and identifiers (people, references, dates) that anchor it in your archive."
        )
        # cache_control on the static base prompt — saves cost on bulk runs.
        system_blocks = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": guidance},
        ]
        user_msg = (
            f"Document title: {title}\n\n"
            f"Document content:\n---\n{truncated}\n---\n\n"
            f"Write the summary now. No preamble."
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            system=system_blocks,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        usage = response.usage
        return SummaryResult(
            summary=text,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
