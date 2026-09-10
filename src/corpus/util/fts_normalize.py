"""Normalization shared by the FTS write path and the FTS query path.

Two problems this solves, and one it deliberately does NOT create:

  * `unicode61` cannot segment CJK — Japanese uses no spaces, so a whole
    sentence becomes one token and `東京` never matches inside it. CJK runs are
    rewritten as overlapping character bigrams so two-character words match.
  * The query path previously extracted terms with an ASCII-only regex, which
    dropped Japanese entirely and shredded accented Latin.

DO NOT add Unicode NFKD decomposition or combining-mark stripping here. It is
redundant — `unicode61` already folds diacritics on both the index and the query
side — and it is destructive: Japanese voiced kana decompose into a base plus a
combining dakuten, so stripping marks maps がっこう (school) onto かっこう
(cuckoo) and バス (bus) onto ハス (lotus). Applied symmetrically it is invisible
to round-trip tests. `tests/test_fts_normalize.py` guards this.

Known limitations, accepted deliberately:
  * A single-character CJK query does not match inside a longer indexed run,
    because the index holds bigrams. Adding unigrams would inflate the index for
    a rare and highly ambiguous query.
  * Bigram OR-joins rank imprecisely; tests assert ranked order, not just match.
"""

from __future__ import annotations

import re

# Hiragana, Katakana, Halfwidth Katakana, CJK Unified Ideographs + Extension A,
# and the CJK compatibility block. Hangul is deliberately excluded: Korean is
# not a target language and would need its own handling.
_CJK = (
    "\u3040-\u309f"  # Hiragana
    "\u30a0-\u30ff"  # Katakana
    "\uff66-\uff9f"  # Halfwidth Katakana
    "\u3400-\u4dbf"  # CJK Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
)
_CJK_RUN = re.compile(f"[{_CJK}]+")
_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+){0,3}", re.UNICODE)


def _bigrams(run: str) -> str:
    """`東京で会議` -> `東京 京で で会 会議`. A one-character run is returned as is."""
    if len(run) < 2:
        return run
    return " ".join(run[i : i + 2] for i in range(len(run) - 1))


def normalize_for_fts(text: str) -> str:
    """Prepare text for the FTS index and for MATCH expressions.

    Must be applied symmetrically to indexed content and to queries.
    """
    text = text.replace("ß", "ss").replace("ẞ", "SS")

    def _expand(m: re.Match[str]) -> str:
        run = _bigrams(m.group(0))
        start, end = m.span()
        # Separate the run from any adjacent Latin or digits. `unicode61`
        # classifies Han and kana as letters, so `Public会議室` would tokenize
        # as the single token `Public会議` -- the run's first bigram is
        # swallowed and no query for 会議 can ever reach it. Symmetrically,
        # `会議室Public` loses its last. Measured on a real archive: 21 of 591
        # chunks containing 会議 were unreachable for exactly this reason.
        if start > 0 and not text[start - 1].isspace():
            run = " " + run
        if end < len(text) and not text[end].isspace():
            run = run + " "
        return run

    return _CJK_RUN.sub(_expand, text)


MAX_FTS_TERMS = 64


def fts_terms(query: str) -> list[str]:
    """Quoted FTS5 MATCH terms for a query string.

    Every term is quoted so that user punctuation cannot produce an FTS5 syntax
    error, and so that bare `AND`/`OR`/`NOT` are treated as words rather than
    operators.

    De-duplicated (order preserved) and capped at `MAX_FTS_TERMS`: a CJK query
    emits one term per overlapping bigram with no natural length limit, so a
    long or repeated query would otherwise OR-join unboundedly many terms --
    measured at 40k chunks, a 74-character CJK query took 223ms. Repetition
    also double-counts a term for BM25 scoring (`fts_terms("東京の会議
    東京の会議")` previously produced 8 terms, only 4 of them unique), skewing
    rank toward accidentally-repeated queries.
    """
    normalized = normalize_for_fts(query)
    seen: set[str] = set()
    terms: list[str] = []
    for token in _TOKEN.findall(normalized):
        if token in seen:
            continue
        seen.add(token)
        escaped = token.replace('"', '""')
        terms.append(f'"{escaped}"')
        if len(terms) >= MAX_FTS_TERMS:
            break
    return terms
