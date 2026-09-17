"""This repo holds source code and documentation. Nothing else, ever.

THE RULE, stated as an ALLOWLIST on purpose. Every guard here until now was
a denylist: a list of private names to look for. A denylist only catches
what someone thought of in advance, and all three strings that reached this
public repo were ones nobody had thought of -- a ticket key, two document
filenames, a disk-volume name, each added before any pattern covered it.

An allowlist catches the case nobody anticipated, which is the only case
that has ever actually happened here.

THE ANALOGY THIS ENFORCES. corpus is the CLASS: the generic engine, the
rules, the mechanisms. The private archives are the INSTANCES: the data,
and every artefact derived from it. A class may describe what an instance
looks like. It must never contain one.

So: a file type not on this list does not get added quietly. It gets added
deliberately, here, with a reason -- and that edit is visible in review,
which is the point.

Measured when written: 303 tracked files -- 252 .py, 33 .md, and 18 pieces
of project scaffolding. There was nothing to clean up. This keeps it that
way.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Source code and documentation.
ALLOWED_SUFFIXES = {
    ".py",      # the engine
    ".md",      # documentation
    ".pyi",     # type stubs
    ".typed",   # PEP 561 marker
}

# Project scaffolding: configuration that describes how the code is built,
# tested, installed and guarded. Each is here because a Python project cannot
# exist without it -- not because it was convenient.
ALLOWED_SCAFFOLDING_SUFFIXES = {
    ".toml",     # pyproject, example configs
    ".lock",     # dependency pinning
    ".yml",      # CI, security policy
    ".yaml",
    ".json",     # example thresholds
    ".sh",       # hook installer
    ".example",  # .env.example, corpus.toml.example
    ".cfg",
    ".ini",
}

# Exact paths with no suffix, or whose suffix says nothing useful.
ALLOWED_EXACT = {
    "LICENSE",
    ".gitignore",
    ".gitattributes",
    ".githooks/commit-msg",
    ".githooks/pre-commit",
    ".githooks/pre-push",
}

# What must NEVER appear, whatever else changes. Redundant with the allowlist
# by construction -- stated anyway, because these are the shapes that carry an
# archive rather than describe one, and a reader deserves to see them named.
FORBIDDEN_SUFFIXES = {
    ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm",  # an index
    ".csv", ".tsv", ".parquet", ".jsonl",                # exports
    ".pdf", ".docx", ".xlsx", ".pptx", ".rtf",           # documents
    ".mp4", ".mov", ".m4a", ".mp3", ".wav", ".jpg",      # media
    ".jpeg", ".png", ".gif", ".heic", ".zip", ".tar",
    ".gz", ".pkl", ".npy", ".bin", ".ipynb",
}


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [name for name in out.split("\0") if name]


def test_every_tracked_file_is_code_scaffolding_or_documentation() -> None:
    allowed = ALLOWED_SUFFIXES | ALLOWED_SCAFFOLDING_SUFFIXES
    offenders = [
        name
        for name in _tracked()
        if name not in ALLOWED_EXACT and Path(name).suffix not in allowed
    ]
    assert not offenders, (
        "File(s) tracked in this PUBLIC repo that are neither source code, "
        "documentation, nor project scaffolding:\n"
        + "\n".join(f"  {n}" for n in sorted(offenders))
        + "\n\n"
        "corpus is the ENGINE. The archives it indexes -- and everything "
        "derived from them -- live in private consumer repos. A class may "
        "describe what an instance looks like; it must never contain one.\n\n"
        "If this file genuinely belongs here, add its suffix to "
        "ALLOWED_SCAFFOLDING_SUFFIXES in this test, with a comment saying "
        "why. That edit is the review step, and it is the point."
    )


def test_no_tracked_file_is_a_data_carrier() -> None:
    """Stated separately from the allowlist so the failure message names the
    actual problem. "Add .xlsx to the allowlist" must never look like a
    reasonable fix."""
    offenders = [
        name for name in _tracked() if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    assert not offenders, (
        "File(s) tracked in this PUBLIC repo whose format exists to carry "
        "content:\n"
        + "\n".join(f"  {n}" for n in sorted(offenders))
        + "\n\n"
        "A database, spreadsheet, document, archive or media file is an "
        "INSTANCE. It does not belong in the engine under any circumstances, "
        "including as a test fixture. Build fixtures in tmp_path at run time "
        "instead.\n\n"
        "Do NOT resolve this by adding the suffix to an allowlist."
    )


def test_no_binary_blob_is_tracked() -> None:
    """A binary file defeats every text guard in this repo at once.

    `git log -p` prints "Binary files differ" and nothing else, so the
    push-time diff scan and the pushed-history audit both report it clean
    whatever is inside it. Verified across the entire history when this was
    written: no binary has ever been committed, which is the only reason
    those two guards can be trusted retroactively.
    """
    empty_tree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    out = subprocess.run(
        ["git", "diff", "--numstat", empty_tree, "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    offenders = {
        line.split("\t", 2)[2]
        for line in out.splitlines()
        if line.startswith("-\t-\t")
    }
    # AND the files on disk. The history comparison above only sees what is
    # COMMITTED, so a staged binary passed this test until someone committed
    # it -- found by mutation-testing this very check, which reported clean
    # with a NUL-byte file staged. A guard that only fires after the mistake
    # is recorded is a guard that fires too late.
    for name in _tracked():
        path = REPO_ROOT / name
        try:
            if b"\x00" in path.read_bytes():
                offenders.add(name)
        except OSError:
            continue
    assert not offenders, (
        "Binary file(s) tracked in this PUBLIC repo:\n"
        + "\n".join(f"  {n}" for n in sorted(offenders))
        + "\n\n"
        "Every content guard here scans text. git shows a binary blob as "
        '"Binary files differ", so the push-time scan and the history audit '
        "both report it clean no matter what it contains."
    )
