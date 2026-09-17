"""Secret/credential redaction. Runs once over each chunk's content right
before embedding + storage.

Threat model: the `.db` file leaks. Defends against credential exfiltration,
not against hostile content in chunks. Tuned to avoid over-redaction —
emails, git SHAs, base64 thumbnails, etc. pass through because they carry
retrieval signal ("who did what") more often than they're secrets.

If your archive contains genuinely sensitive PII (medical records, customer
data), add patterns here before ingesting.
"""

from __future__ import annotations

import contextvars
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED:aws-access-key]"),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b"), "[REDACTED:github-token]"),
    ("stripe-live", re.compile(r"\b(?:sk|rk|pk)_live_[A-Za-z0-9]{20,}\b"), "[REDACTED:stripe-live-key]"),
    ("stripe-test", re.compile(r"\b(?:sk|rk|pk)_test_[A-Za-z0-9]{20,}\b"), "[REDACTED:stripe-test-key]"),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED:slack-token]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[REDACTED:jwt]"),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[REDACTED:openai-key]"),
    ("voyage-key", re.compile(r"\bpa-[A-Za-z0-9_-]{30,}\b"), "[REDACTED:voyage-key]"),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{30,}\b"), "[REDACTED:anthropic-key]"),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "[REDACTED:private-key]"),
    ("api-key-assignment", re.compile(r"""(?i)\b(api[-_]?key|secret|token|password)["']?\s*[:=]\s*["']?([A-Za-z0-9_\-./+=]{16,})["']?"""), r"\1=[REDACTED:credential]"),
]


# What the current ingest has redacted, when someone is counting.
#
# `scrub` removing a credential protects the INDEX. It does nothing about the
# credential, which is still live in whatever the archive was built from, and
# rotation is the only remedy once text has sat in a personal archive for
# years. `find_secrets` -- the function that reports rather than replaces --
# had zero production callers, so the one fact the owner needs was the one
# nothing produced.
_redactions: contextvars.ContextVar[Counter[str] | None] = contextvars.ContextVar(
    "corpus_redactions", default=None
)


@contextmanager
def count_redactions() -> Iterator[Counter[str]]:
    """Count what `scrub` redacts inside this block, per pattern.

    Per PATTERN rather than as a total: an AWS key and a Slack token send
    someone to two different consoles.
    """
    counter: Counter[str] = Counter()
    token = _redactions.set(counter)
    try:
        yield counter
    finally:
        _redactions.reset(token)


def scrub(text: str) -> str:
    # Counting is OBSERVATION and is never required for redaction: with no
    # counter active this is the same function it always was.
    counter = _redactions.get()
    for name, pattern, replacement in _PATTERNS:
        if counter is None:
            text = pattern.sub(replacement, text)
            continue
        text, hits = pattern.subn(replacement, text)
        if hits:
            counter[name] += hits
    return text


def find_secrets(text: str) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for name, pattern, _ in _PATTERNS:
        for m in pattern.finditer(text):
            hits.append((name, m.group(0)))
    return hits
