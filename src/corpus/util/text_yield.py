"""Per-connector-type text-yield calibration for pre-ingest cost estimates.

Estimating embedding cost from raw file size alone (`bytes / 4`, treating
every byte as roughly one character of eventual text) is wrong by orders of
magnitude for compressed/binary container formats. `corpus-index` measured
this directly against a real, mixed large mixed corpus on 2026-09-09 (see
`.superpowers/sdd/` for the run) — chars of EXTRACTED text per byte of
SOURCE file, by connector type:

    type        chars/byte   source example
    pptx          0.0006     several sources (slide media dominates)
    olm           0.0039     a large Outlook archive (see caveat below)
    pdf           0.0016     several sources (pages, fonts, images dominate)
    zip           0.0107     several sources (compressed; mixed contents)
    docx          0.0287     word_docs (OOXML zip: styles/media overhead)
    rtf           0.0850     rich_text
    xlsx          0.3025     spreadsheets
    html          0.3532     web_clippings (markup overhead, but less than
                             the binary formats above)
    text          0.9951     texts (already plain text; ~1:1 as expected)
    markdown      1.0054     notes (already plain text; ~1:1 as expected)

`zip` and `pptx` were re-measured on 2026-09-09 against many real sources
after the first pass proved unrepresentative, and both moved by ~2 orders
of magnitude:

  - `zip` was 0.0001, taken from a single archive of mostly-binary content.
    Across several real zip sources it runs 0.00056 (that same archive, itself
    5.6x above the number derived from it) to 0.03208 (an archive of office
    documents) — a 57x spread, since a zip's yield is entirely a property of
    what someone put in it. The old value understated a large archive's cost
    by 107x, which is the direction this module explicitly calls the worse
    error.
  - `pptx` was 0.0287, inherited from `docx` on the reasoning that both are
    OOXML zip containers. Measured, it runs 0.00037 to 0.04371 across three
    sources — the container shape is shared but the CONTENT is not: a deck
    is mostly embedded media, a Word document mostly text. One 13-file,
    176 MB source was estimated at 1.26M tokens and cost 15,487.

`olm` is measured the same way but describes a connector that decides how
much of its own input to read, which no other row does. Its 0.0039 assumes
this module's defaults: the archive's duplicate of itself skipped (~50% of
messages) and quoted reply history trimmed (~78% of the remaining text).
Turning either off multiplies real yield by roughly 2x and 4.6x
respectively; scoping to a subset of folders divides it. `corpus-index`'s
plan cannot see any of those settings, so for `.olm` its number is a
whole-archive default-settings figure and nothing more.

Both replacements are corpus-wide aggregates, the same methodology as every
other row (the `pdf` row spans 0.00146 to 0.03050 across its six sources and
its aggregate predicted the total to within 1%). An aggregate is the right
estimator for a whole-source cost, and the wrong one for any single file.

The old flat assumption (ratio 1.0, i.e. `bytes / 4` tokens) overstated a
large PDF source's embedding cost by roughly 400x — a
$0.06 operation reported as if it might cost $23, exactly the kind of
number that makes someone decline a run they should just do. This module
exists so `corpus-index`'s plan (and anything else estimating cost/size
from raw bytes — `corpus.planner`, potentially `corpus-survey census`)
applies the same, single, calibrated table instead of each guessing bytes/4
independently.

**This is still an estimate, not a measurement**, and deliberately errs
toward NOT understating cost when uncertain — an underestimate is worse
than an overestimate, because the user only discovers it after being
billed. Two consequences of that:

  - Every ratio above is rounded UP to 4 significant figures from the raw
    measurement, and `estimate_tokens_from_bytes` rounds the final token
    count up (`math.ceil`), never down.
  - Connector types with no measured data (see `_UNMEASURED_DEFAULTS`
    below) default to `DEFAULT_TEXT_YIELD_RATIO` (1.0 — the same "roughly
    1 char per byte" assumption this module exists to correct for the
    *measured* formats) rather than a guessed-low ratio, UNLESS there is
    a specific, documented, structural reason a format's true yield is
    known to be far below that ceiling regardless of measurement (see
    `aup3` below) — in which case guessing high would reintroduce the
    exact "decision-changing overestimate" failure mode this module exists
    to fix, just for a different format.

PDF in particular still varies enormously within the single measured
number above: a text-layer PDF and a scanned-image PDF (no OCR) can differ
by an order of magnitude in real yield. The table is a corpus-wide average,
not a per-file oracle — `corpus-index`'s output says so explicitly rather
than presenting false precision.
"""

from __future__ import annotations

import math

# Measured 2026-09-09 against a real, mixed corpus (many files) -- see the
# module docstring's table. Keyed by connector `type` (the same string used
# in `corpus.connectors.registry.DEFAULT_GLOBS` / `SourceConfig.type` /
# `corpus.survey.classify`'s indexable `detail`), so any caller that already
# has one of those type strings and a byte count can look up a ratio here
# with no translation layer.
MEASURED_TEXT_YIELD_RATIOS: dict[str, float] = {
    "pptx": 0.0006,
    "olm": 0.0039,
    "pdf": 0.0016,
    "zip": 0.0107,
    "docx": 0.0287,
    "rtf": 0.0850,
    "xlsx": 0.3025,
    "html": 0.3532,
    "text": 0.9951,
    "markdown": 1.0054,
}

# Registered connector types NOT in the measured table above, with reasoned
# (not measured) defaults and why each was picked. Kept separate from
# `MEASURED_TEXT_YIELD_RATIOS` so it's never mistaken for real data — see
# `TEXT_YIELD_RATIOS` below, which merges both for lookup purposes.
_UNMEASURED_DEFAULTS: dict[str, float] = {
    # An `.abcdp` is a binary plist whose bytes are mostly UID strings, sync
    # hashes, and timestamps; the human-readable part (name, numbers, note) is
    # a small fraction. Not measured -- 379 files totalling well under a
    # megabyte is too little to aggregate -- so this takes the same order of
    # magnitude as the other binary-container formats rather than the
    # safe-high default, which would estimate a contacts backup at hundreds of
    # times its real cost.
    "abcdp": 0.05,
    # A music file's bytes are almost entirely encoded audio; this connector
    # extracts only tags, and emits ONE document per album rather than per
    # track, so the text produced is a few hundred characters against tens of
    # megabytes. The same structural argument as `aup3`: applying the
    # safe-high default would price a 40 GB library at billions of tokens for
    # what is actually a few thousand words of track listings.
    "music": 0.00002,
    # Legacy `.xls` (OLE2/BIFF) holds the same tabular content as `.xlsx`
    # but in a denser binary container with no zip compression, so its
    # chars-per-byte should land at or below xlsx's measured 0.3025. Not
    # measured directly (the archive here has 296 such files, too few and too
    # small to aggregate meaningfully), so it takes xlsx's ratio as the
    # nearest anchor -- the safe direction, since a denser container yields
    # less text per byte, not more.
    "xls": 0.3025,
    # `csv`/`tsv` source bytes are already plain text (no binary container
    # tax the way docx/xlsx/pptx have), so the safe ceiling (1.0) applies
    # for the common case this connector fully renders. The one thing that
    # would make real yield LOWER is this connector's own summarization for
    # files over its 100-row/50,000-char cap (see `csv_.py`) -- which only
    # ever REDUCES yield below this estimate (a safe-direction miss, not an
    # understatement) for exactly the large files where it kicks in.
    "csv": 1.0,
    "tsv": 1.0,
    # NOT a text container at all, unlike every format above -- an `.aup3`
    # project's file size is almost entirely binary audio sample data, and
    # this connector extracts a small, fixed-shape description (duration,
    # assumed sample rate/channels, a few sentences) whose length barely
    # varies with the project's actual size. Applying the safe-ceiling
    # default (1.0) here would reproduce the EXACT bug this module exists
    # to fix, just for this one format: a 500MB `.aup3` file would be
    # "estimated" at over a hundred million tokens for a document that's
    # actually a few hundred words. This is a structural fact about the
    # format, not genuine uncertainty calling for the conservative-high
    # default -- so it gets a low ratio. (This was originally described as
    # "the same order of magnitude as `zip`". It no longer is: `zip` was
    # re-measured from 0.0001 to 0.0107. The reasoning here never depended
    # on that comparison -- an `.aup3` description really is a fixed few
    # hundred words regardless of file size, which is not true of a zip --
    # so the value stands on its own and the comparison is dropped.)
    "aup3": 0.0001,
}

# Merged table: prefer measured data, fall back to the reasoned defaults
# above. Exposed as one dict because most callers just want "the ratio for
# this type" without caring which tier it came from.
TEXT_YIELD_RATIOS: dict[str, float] = {**_UNMEASURED_DEFAULTS, **MEASURED_TEXT_YIELD_RATIOS}

# Fallback for any connector type in neither table above -- a third-party
# registration in `CONNECTOR_REGISTRY`, or a new built-in connector added
# without updating this module. 1.0 (chars ≈ bytes) is the same "roughly
# plain text" assumption `MEASURED_TEXT_YIELD_RATIOS`'s own `text`/`markdown`
# entries confirm is directionally correct for genuinely textual formats,
# and it's the conservative (overestimate, not underestimate) choice for
# anything else per this module's stated bias -- see the module docstring.
DEFAULT_TEXT_YIELD_RATIO = 1.0

CHARS_PER_TOKEN_ESTIMATE = 4


def text_yield_ratio(connector_type: str) -> float:
    """Characters of extracted text per byte of source file for
    `connector_type`, or `DEFAULT_TEXT_YIELD_RATIO` if it isn't in the
    table (see the module docstring for what that default means and why)."""
    return TEXT_YIELD_RATIOS.get(connector_type, DEFAULT_TEXT_YIELD_RATIO)


def estimate_tokens_from_bytes(connector_type: str, total_bytes: int) -> int:
    """Estimated embedding-time token count for `total_bytes` of raw source
    file belonging to `connector_type`, via `bytes -> calibrated chars ->
    chars/4`. Always rounds up (`math.ceil`) -- see the module docstring's
    "erring toward NOT understating cost" rationale. Still a rough ceiling,
    not the embedder's real tokenizer count; callers should say so in their
    own output rather than presenting this as exact.
    """
    estimated_chars = total_bytes * text_yield_ratio(connector_type)
    return math.ceil(estimated_chars / CHARS_PER_TOKEN_ESTIMATE)
