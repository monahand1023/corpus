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

A byte-order mark is checked FIRST, before any of that, because a BOM is not
a guess: it is the file declaring its own encoding, and the ladder below
cannot recover from one. Measured on a real archive, 7 of the 8 files the
ladder decoded into pure garbage were BOM-carrying UTF-16 — `cp932` and
`latin-1` both "succeed" on UTF-16 bytes and return one NUL byte per ASCII
character, which is worse than the `errors="replace"` behaviour this module
replaced: U+FFFD announces the damage, whereas `0S0n0\xe1` looks like text to
a search index and is silently unfindable. The BOM tier took that corpus from
1,811 stray NUL characters across 8 files to 65 in 1 (a 1995 telnet capture
with genuinely embedded NULs — not a decoding failure).

Deliberately NOT `chardet`/`charset-normalizer`: those give better coverage
of the long tail of encodings, at the cost of a real dependency and
non-trivial per-file classification cost, for files that made it into a
personal archive at all — this fixes the two encodings this project has
actually measured a real archive tripping over (UTF-8, CP932) plus a
never-fails backstop, not every encoding that has ever existed. That call was
re-tested when the BOM tier was added, by scoring `charset-normalizer` over
the same 66 mis-decoded files: on top of BOM sniffing it corrected exactly
zero additional files, so it stays out.

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

# Byte-order marks, mapped to the codec that both decodes the file AND strips
# the mark itself (`utf-16`/`utf-32` read the BOM to pick an endianness;
# `utf-16-le` and friends would leave a stray U+FEFF at the start of the text,
# which then rides into the first chunk of the index).
#
# Longest signature first: a UTF-32-LE mark (ff fe 00 00) starts with the
# whole UTF-16-LE mark (ff fe), so checking the short one first would decode
# every UTF-32-LE file as UTF-16-LE.
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16"),
    (b"\xff\xfe", "utf-16"),
)


def decode_text_with_fallback(raw: bytes) -> str:
    """Decode `raw` as text: honour a byte-order mark if one is present,
    otherwise try UTF-8, then CP932, then latin-1 (which always succeeds).
    See the module docstring for the reasoning.

    Also normalizes `\\r\\n`/`\\r` line endings to `\\n` — preserving the
    universal-newline translation that `Path.read_text()` (what every
    caller used before this existed) performs as a side effect of opening
    in text mode, so switching to explicit bytes+decode doesn't change
    output for `\\r\\n`-terminated files.
    """
    text = _decode_declared_by_bom(raw)
    if text is None:
        for encoding in _STRICT_FALLBACK_ENCODINGS:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
    if text is None:
        text = raw.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _decode_declared_by_bom(raw: bytes) -> str | None:
    """Decode `raw` using the encoding its byte-order mark declares, or return
    None if it carries no BOM — or carries one it then fails to honour, in
    which case the caller's ladder still has better odds than no text at all.
    A BOM names the intent; it doesn't promise the rest of the file keeps it.
    """
    for signature, bom_encoding in _BOM_ENCODINGS:
        if not raw.startswith(signature):
            continue
        try:
            return raw.decode(bom_encoding)
        except UnicodeDecodeError:
            return None
    return None


def read_text_with_fallback(path: Path) -> str:
    """`decode_text_with_fallback` over a file's full contents. Raises
    `OSError` exactly like `Path.read_text()` did — every existing caller
    already wraps its read in a per-file `try/except OSError`, so this is a
    drop-in replacement, not a new error-handling case."""
    return decode_text_with_fallback(path.read_bytes())
