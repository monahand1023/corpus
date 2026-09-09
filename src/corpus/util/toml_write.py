"""Tiny TOML string-escaping helper shared by anything that hand-renders
TOML text (`corpus-init`'s wizard output, `corpus-index`'s `[[sources]]`
merge). No TOML *writer* library is used on purpose: the stdlib `tomllib`
is read-only, and pulling in a write-capable TOML library just for one
escaping rule is not worth a new dependency — corpus.toml's shape is simple
enough (flat tables, arrays of tables, string/bool/int/float values) that a
few lines of manual rendering cover every field this codebase writes.
"""

from __future__ import annotations


def toml_str(value: str) -> str:
    """Escape a value for a TOML basic (double-quoted) string.

    Without this, a path containing a `"` or `\\` (both legal on APFS)
    would render invalid, unloadable TOML.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
