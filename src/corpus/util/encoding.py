"""Shared text-decoding fallback for connectors that read a whole file as
text: `markdown.py`, `text.py`, `html.py`, `rtf.py`. (`csv_.py` has its own
UTF-8/latin-1-only fallback with its own justification in its module
docstring — CSV exports are overwhelmingly Western-locale in practice, so it
doesn't carry the Shift-JIS tier below.)

Before this existed, all four connectors called
`path.read_text(encoding="utf-8", errors="replace")` — silently substituting
U+FFFD for every byte sequence that isn't valid UTF-8. That's not a
theoretical concern: it was found live in a real index (~970 chunks with
U+FFFD) while chasing a related zip-filename mojibake bug (see
`corpus/connectors/zip.py`'s `_repair_filename_encoding`) — the same
Japanese-locale archives that had mis-decoded member *names* also had
Shift-JIS-encoded member *content*, which the old `errors="replace"` path
was silently turning into replacement-character soup instead of an error OR
correct text.

Tries, in order: UTF-8 (strict — the correct, common case, left completely
unaffected), then CP932 (Shift-JIS's IBM/Microsoft superset — genuinely
common for East-Asian-locale text files, matching the same tier used for
zip filename repair), then latin-1 as the final backstop, which — mapping
every byte 0-255 to a codepoint — cannot itself fail to decode, so *some*
text always comes back rather than raising.

Deliberately NOT `chardet`/`charset-normalizer`: those give better coverage
of the long tail of encodings, at the cost of a real dependency and
non-trivial per-file classification cost, for files that made it into a
personal archive at all — this fixes the two encodings this project has
actually measured a real archive tripping over (UTF-8, CP932) plus a
never-fails backstop, not every encoding that has ever existed.

Trade-off worth naming: CP932 is a strict multi-byte encoding (specific
lead/trail byte ranges), so a false-positive "successful" CP932 decode of
genuinely Windows-1252/Latin-1 bytes is possible in principle for a short
byte sequence, though increasingly unlikely as the file gets longer (every
double-byte pair has to land in-range). Accepted deliberately: the
alternative (skip CP932, go straight to latin-1) reliably decodes without
error but reliably produces mojibake for real Shift-JIS content, which is
the exact failure mode this module exists to fix.
"""

from __future__ import annotations

from pathlib import Path

# Order matters: first successful strict decode wins. latin-1 is not listed
# here — it's the unconditional final fallback below, since it can never
# raise and so could never be "reached" by the same try-in-order loop as a
# real alternative.
_STRICT_FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8", "cp932")


def decode_text_with_fallback(raw: bytes) -> str:
    """Decode `raw` as text, trying UTF-8, then CP932, then latin-1 (which
    always succeeds). See the module docstring for the reasoning.

    Also normalizes `\\r\\n`/`\\r` line endings to `\\n` — preserving the
    universal-newline translation that `Path.read_text()` (what every
    caller used before this existed) performs as a side effect of opening
    in text mode, so switching to explicit bytes+decode doesn't change
    output for `\\r\\n`-terminated files.
    """
    text: str | None = None
    for encoding in _STRICT_FALLBACK_ENCODINGS:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text_with_fallback(path: Path) -> str:
    """`decode_text_with_fallback` over a file's full contents. Raises
    `OSError` exactly like `Path.read_text()` did — every existing caller
    already wraps its read in a per-file `try/except OSError`, so this is a
    drop-in replacement, not a new error-handling case."""
    return decode_text_with_fallback(path.read_bytes())
