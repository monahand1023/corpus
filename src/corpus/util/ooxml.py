"""Tell a permanently-unreadable Office file from a transiently-unreadable one.

The distinction matters more than it looks. A connector reports a file it
could not read as `failed_files`, which suppresses orphan pruning for the
whole source — the right behaviour for a file that was momentarily locked,
because pruning would otherwise delete that file's chunks as if it had been
deleted. But a file that will NEVER be readable fails identically on every
future run, so counting it as a failure suppresses pruning *forever*: the
source can never again delete a genuinely removed document.

Measured on a real archive, all five files a `.docx` source could not open
were permanent:

  - `~$25 taxes.docx` — a Word owner-lock file, which is not a document at
    all and is invisible in Finder.
  - three OLE2 files with a `.docx` extension — legacy `.doc` renamed rather
    than converted, which `python-docx` can never read.

Both kinds would have blocked pruning on that source indefinitely. (A fourth
was a zero-byte file; see below for why that one is deliberately NOT treated
as permanent.)

The check is by content, not extension: a legacy Office file is an OLE2
compound document with a fixed 8-byte signature, which a rename cannot hide.

**Only two conditions qualify as permanent, and the bar is deliberately
high.** Getting this wrong in the other direction loses data: a large file
being copied over a network is not a valid zip *yet*, and an interrupted
copy leaves a zero-byte file that will be complete a minute later. Marking
either "permanent" lets the next prune delete that document's chunks. A
stale suppressed prune costs an out-of-date index, recoverable with
`--prune-anyway`; a wrong skip costs content. So "not a zip" and "empty"
stay transient, and only a name Office reserves for lock files and an
unmistakable OLE2 signature are treated as final.
"""

from __future__ import annotations

from pathlib import Path

# OLE2 / Compound File Binary Format. Legacy `.doc`, `.xls`, `.ppt` all start
# with this, and so does any of them renamed to a modern extension.
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def is_office_lock_file(path: Path) -> bool:
    """True for a Word/Excel/PowerPoint owner-lock file (`~$name.docx`).

    Office writes these beside an open document and deletes them on close;
    an ungraceful quit leaves them behind. They are not documents, they are
    invisible in Finder, and a glob picks them up anyway.
    """
    return path.name.startswith("~$")


def permanent_read_failure_reason(path: Path) -> str | None:
    """Why this file can never be read as OOXML, or None if it might be fine.

    Returning None does NOT promise the file parses — a corrupt, truncated, or
    still-being-copied file still fails downstream, and SHOULD, because those
    are transient. This only identifies the two conditions that can never
    resolve on a later run, so a connector can count those as `skipped_files`
    rather than blocking its source's pruning forever.
    """
    if is_office_lock_file(path):
        return "Office owner-lock file (~$…), not a document"
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(_OLE2_MAGIC))
    except OSError:
        # A failed read is transient-shaped (a race, a permission blip); let
        # the caller's normal error path count it as a failure.
        return None
    if head.startswith(_OLE2_MAGIC):
        return (
            "legacy OLE2 Office file with a modern extension — convert it "
            "(e.g. `soffice --headless --convert-to docx`) and index the result"
        )
    return None
