"""Near-duplicate detection for source documents.

Strips noise (URLs, dates, case, whitespace) before hashing so re-exports
of the same doc collide even when timestamps differ. Real content differences
(English vs Spanish, different versions of a design doc) hash differently
because the underlying text differs.
"""

from __future__ import annotations

import hashlib
import re

_URL_RE = re.compile(r"https?://[^\s)]+|www\.[^\s)]+")
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    cleaned = _URL_RE.sub("", text)
    cleaned = _DATE_RE.sub("", cleaned)
    cleaned = cleaned.lower()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()[:32]
