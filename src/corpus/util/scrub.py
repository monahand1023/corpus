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

import re

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


def scrub(text: str) -> str:
    for _name, pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def find_secrets(text: str) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for name, pattern, _ in _PATTERNS:
        for m in pattern.finditer(text):
            hits.append((name, m.group(0)))
    return hits
