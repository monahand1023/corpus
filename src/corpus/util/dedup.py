"""Near-duplicate detection for source documents.

Strips noise (URLs, dates, case, whitespace) before hashing so re-exports
of the same doc collide even when timestamps differ. Real content differences
(English vs Spanish, different versions of a design doc) hash differently
because the underlying text differs.

WHAT THAT TRADE-OFF ACTUALLY COSTS, measured 2026-09-16 by running five
connectors twice over one real archive -- once with this function and once
with an exact hash -- and comparing the document counts:

    word_docs      one distinct document merged
    papers         2,741 vs 2,743     2
    web_clippings    266 vs   267     1
    spreadsheets     133 vs   133     0
    notes            496 vs   496     0

0.07% of documents, and -- the part worth knowing -- the
stripping caught NOTHING an exact hash would have missed on that archive.
The re-exported-with-a-new-timestamp case this exists for did not occur in
it. The benefit is real but unobserved here; the cost is small and real.

**DO NOT USE THIS FOR MAIL, or anything else where a date or a URL is the
content rather than noise.** `connectors/olm.py` did, briefly, and it
collapsed 53 salon booking confirmations -- same template, same sender,
bookings across 2014 and 2015 with a different reservation URL each -- into
ONE, because the only things telling them apart were exactly the two fields
removed here. 52 real appointments would have left the index. That connector
now uses an exact subject+body hash and says why.

The rule: strip noise only where the stripped fields ARE noise. For a
document re-exported by a tool, a timestamp is noise. For a message, it is
the thing that makes it a different message.
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
