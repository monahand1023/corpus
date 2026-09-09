"""Shared Anthropic Batch API helpers — payload-size-aware sub-batching and
retry/backoff — used by both the CR runner (`batch_runner`) and the cloud labeler
(`labeling/cloud_labeler`). Extracted so neither has to borrow the other's
internals.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Anthropic caps a batch-create payload at 256 MB and 90k requests. Doc/attachment
# CR requests each carry a full document body, so a single source's requests can
# serialize to ~500 MB — we split by cumulative bytes, not just request count.
MAX_REQUESTS_PER_BATCH = 90_000
MAX_BATCH_BYTES = 200 * 1024 * 1024  # safety margin under the 256 MB hard limit
RETRY_ATTEMPTS = 6


def pack_batches(
    requests: list[dict[str, Any]],
    max_requests: int = MAX_REQUESTS_PER_BATCH,
    max_bytes: int = MAX_BATCH_BYTES,
) -> list[list[dict[str, Any]]]:
    """Split requests into sub-batches respecting BOTH the request-count and the
    payload-byte caps."""
    batches: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_bytes = 0
    for r in requests:
        rbytes = len(json.dumps(r))
        if cur and (len(cur) >= max_requests or cur_bytes + rbytes > max_bytes):
            batches.append(cur)
            cur, cur_bytes = [], 0
        cur.append(r)
        cur_bytes += rbytes
    if cur:
        batches.append(cur)
    return batches


def retry_with_backoff(fn: Callable[[], Any], what: str, attempts: int = RETRY_ATTEMPTS) -> Any:
    """Retry a network call with exponential backoff (2s→60s cap) so a transient
    blip can't crash a long, money-spending batch run."""
    delay = 2.0
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # network/SDK errors are heterogeneous; retry them all
            if attempt == attempts - 1:
                raise
            logger.warning("%s failed (attempt %d/%d): %s — retrying in %.0fs",
                           what, attempt + 1, attempts, e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
