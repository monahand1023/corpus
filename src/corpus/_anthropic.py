"""Shared Anthropic client construction, retry, and tool-output extraction.

Used by the per-doc summarizer and the eval generation/judge modules so the
client + backoff logic lives in exactly one place. `anthropic` is imported
lazily (only inside `make_client`, and under TYPE_CHECKING for annotations) so
importing this module never requires the optional `[summarizer]` extra.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import Anthropic

logger = logging.getLogger(__name__)

# Exponential-backoff retry around each API call. ~5 attempts absorbs a
# transient 429/503/network blip without crashing a run.
RETRY_ATTEMPTS = 5


def make_client(api_key: str | None = None) -> Anthropic:
    """Build an Anthropic client, resolving the key from `api_key` or the
    `ANTHROPIC_API_KEY` env var. Raises a clean RuntimeError if neither is set —
    this is the feature-flag: generation/judge/summaries no-op with no key."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing. Set it in .env (or the environment)."
        )
    from anthropic import Anthropic

    return Anthropic(api_key=key)


def retry(fn: Callable[[], Any], what: str) -> Any:
    """Retry a network call with exponential backoff so a transient blip
    (429/503/connection reset) doesn't crash an otherwise-resumable run."""
    delay = 2.0
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as e:  # SDK/network errors are heterogeneous
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.0fs",
                what, attempt + 1, RETRY_ATTEMPTS, e, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)


def extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    """Return the forced tool_use block's input dict from a Messages response.

    Raises RuntimeError if the expected tool call is absent or its input is not
    a JSON object — surfaces a broken forced-tool response clearly instead of a
    downstream KeyError.
    """
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            data = block.input
            if not isinstance(data, dict):
                raise RuntimeError(f"{tool_name} tool input was not a JSON object")
            return data
    raise RuntimeError(f"model did not call the forced {tool_name!r} tool")
