"""Tiny TOML string-escaping helper shared by anything that hand-renders
TOML text (`corpus-init`'s wizard output, `corpus-index`'s `[[sources]]`
merge). No TOML *writer* library is used on purpose: the stdlib `tomllib`
is read-only, and pulling in a write-capable TOML library just for one
escaping rule is not worth a new dependency — corpus.toml's shape is simple
enough (flat tables, arrays of tables, string/bool/int/float values) that a
few lines of manual rendering cover every field this codebase writes.
"""

from __future__ import annotations

# TOML basic strings forbid literal control characters, with ONE exception:
# tab is allowed unescaped. These four have a shorthand; anything else in the
# range needs \uXXXX.
_CONTROL_ESCAPES = {
    "\b": "\\b",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def toml_str(value: str) -> str:
    """Escape a value for a TOML basic (double-quoted) string.

    Without this, a path containing a `"` or `\\` (both legal on APFS) would
    render invalid, unloadable TOML.

    CONTROL CHARACTERS MATTER FOR THE SAME REASON, and were missed until this
    module got its first test. A filename may contain a newline on APFS and on
    ext4 — every byte except `/` and NUL is legal — and an unescaped one ends
    the string mid-document. The result is a corpus.toml that will not parse,
    discovered on the NEXT run, after the working config it replaced is gone.

    Backslash is replaced first, or the backslashes introduced by the later
    replacements would themselves be escaped.
    """
    out = value.replace("\\", "\\\\").replace('"', '\\"')
    for raw, escaped in _CONTROL_ESCAPES.items():
        out = out.replace(raw, escaped)
    out = "".join(
        ch if ch == "\t" or (ord(ch) >= 0x20 and ord(ch) != 0x7F)
        else f"\\u{ord(ch):04X}"
        for ch in out
    )
    return '"' + out + '"'
