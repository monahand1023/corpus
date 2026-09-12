"""Guards against the exact failure mode this project has already had once:
a real personal-document index and a real config sitting untracked in this
PUBLIC repo's own working directory, for weeks, kept safe only by
`.gitignore` correctness.

This deliberately inspects the filesystem, not `git ls-files` / `git status`.
The actual danger was never "will git commit this" -- `.gitignore` covered
it from day one and held. The danger was that an untracked file surviving
on `.gitignore` alone is exactly what `git clean -fdx` deletes, and exactly
what a rewritten `.gitignore` or an overlooked new file pattern would stop
protecting. A check that only looked at git-tracked or git-visible state
would trust the very thing that needs to be treated as fallible, so this
test walks the real directory tree instead.

This is intentionally paranoid and intentionally cheap to run: it costs a
handful of filesystem stats and runs on every `pytest` invocation, locally
and in CI, unprompted, forever -- the durable layer of this hardening
effort, independent of anyone remembering it exists.

Personal indexes belong in a private *consumer* repo -- its own directory,
its own config, no public remote -- not inside corpus itself. See the
"corpus is a generic engine" section of the README for the pattern.

Note on scope: this does NOT check for a stray `.env`. Unlike a database or
a real `corpus.toml`, a `.env` holding an API key is the project's own
documented, expected local setup (see `corpus-init` / the README quick
start) -- every real install needs one to run ingest or query at all, so a
presence check here would fail on every legitimately configured checkout,
including this one, not just an accidental one. `.env` is still covered:
`.gitignore` excludes it, and the `.githooks/pre-commit` hook (layer 3)
blocks it from ever being committed, even via `git add -f`.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that legitimately hold build/cache/tooling output, or a
# bundled *public* example fixture -- never the place personal data would
# land -- skipped so this test reflects the actual risk (personal data
# quietly sitting in the repo) rather than noise from tool caches.
#
# "examples" specifically: examples/sample_corpus/corpus.toml is a tracked,
# committed CI fixture that intentionally builds its (gitignored) database
# inside the repo tree, using the zero-dependency hash embedder over the
# bundled, already-public sample docs under examples/sample_corpus/ -- see
# that file's own header comment. That is a deliberate, content-free demo
# artifact, not an instance of the personal-data mistake this test guards
# against, and running the documented example locally must not turn a
# subsequent `pytest` run red.
_EXCLUDED_DIR_NAMES = {".git", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"}
_EXCLUDED_TOP_LEVEL_DIRS = {"dist", "build", "examples"}


def _is_excluded(path: Path) -> bool:
    relative_parts = path.relative_to(REPO_ROOT).parts
    if relative_parts and relative_parts[0] in _EXCLUDED_TOP_LEVEL_DIRS:
        return True
    return any(part in _EXCLUDED_DIR_NAMES or part.endswith(".egg-info") for part in relative_parts)


def _find(pattern: str) -> list[Path]:
    return [p for p in REPO_ROOT.rglob(pattern) if not _is_excluded(p)]


def test_no_database_files_in_the_repository() -> None:
    hits = sorted(
        {*_find("*.db"), *_find("*.db-wal"), *_find("*.db-shm")},
    )
    assert not hits, (
        "Found database file(s) inside this repository's working directory:\n"
        + "\n".join(f"  {p.relative_to(REPO_ROOT)}" for p in hits)
        + "\n\n"
        "corpus is a PUBLIC repo (github.com/monahand1023/corpus). A database "
        "file here is exactly the mistake this test exists to catch: a "
        "personal document index must never live inside the engine's own "
        "source tree, where an untracked file survives only as long as "
        ".gitignore stays correct -- and is exactly what `git clean -fdx` "
        "deletes outright. Move it into a private consumer repo's own "
        "directory instead (the a mail consumer / a media consumer / a document consumer pattern: "
        "its own data/, its own corpus.toml, no public remote) and point "
        "db_path at it there."
    )


def test_no_real_corpus_toml_in_the_repository_root() -> None:
    real_config = REPO_ROOT / "corpus.toml"
    assert not real_config.is_file(), (
        f"Found {real_config.relative_to(REPO_ROOT)} in this PUBLIC repo's root. "
        "corpus.toml.example is the tracked template; a real corpus.toml here "
        "means it is pointing at someone's actual data and sources, which is "
        "exactly the mistake this test exists to catch. Move the real config "
        "into a private consumer repo instead (the a mail consumer / a media consumer / "
        "a document consumer pattern), never inside corpus itself."
    )


# --- identifiers borrowed from private archives ------------------------------
#
# The checks above guard against private *files* landing here. They do not
# guard against a private identifier landing inside a tracked source file,
# which is how a real ticket key from a private consumer's archive reached
# this repo's test suite and sat there through 71 commits. It never reached
# origin, but nothing would have stopped it.
#
# The guard is PATTERN-based rather than a denylist of private project names,
# for the obvious reason: writing those names here to forbid them would put
# them in the public repo, which is the thing being prevented. So instead the
# rule is stated positively -- a ticket key in this repo must be a recognisable
# placeholder -- which leaks nothing and holds for any private archive, named
# or not, present or future.


# Hyphen-number identifiers that are technical vocabulary, not ticket keys.
_STANDARD_PREFIXES = {
    "UTF", "PDF", "CVE", "UUID", "MPEG", "ISO", "RFC", "SHA", "MD", "AES",
    "RSA", "HTTP", "JPEG", "PNG", "GIF", "MP", "H", "X", "ES", "PEP", "BGE",
}

# Placeholder prefixes a public test suite may legitimately use.
_PLACEHOLDER_PREFIXES = {"TKT", "PROJ", "TICKET", "DOC", "ABC", "FOO", "BAR", "TEST", "EXAMPLE"}

_TICKET_KEY = re.compile(r"\b([A-Z][A-Z0-9]{1,9})-[0-9]{1,6}\b")

_TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".txt", ".cfg", ".ini", ".yml", ".yaml",
    ".json", ".sh", ".rst", ".html",
}


def _tracked_text_files() -> list[Path]:
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [
        REPO_ROOT / name
        for name in out.split("\0")
        # This file is skipped because it DEFINES the pattern: the regex
        # source and the prefix allowlists are themselves ticket-key-shaped.
        if name and Path(name).suffix in _TEXT_SUFFIXES
        and Path(name).name != Path(__file__).name
    ]


def test_no_ticket_keys_from_a_private_archive() -> None:
    offenders: list[str] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in _TICKET_KEY.finditer(line):
                prefix = match.group(1)
                if prefix in _STANDARD_PREFIXES or prefix in _PLACEHOLDER_PREFIXES:
                    continue
                offenders.append(
                    f"  {path.relative_to(REPO_ROOT)}:{lineno}: {match.group(0)}"
                )

    assert not offenders, (
        "Found ticket-key-shaped identifier(s) in this PUBLIC repo that are "
        "neither a known technical standard nor a recognisable placeholder:\n"
        + "\n".join(offenders)
        + "\n\n"
        "corpus is a PUBLIC repo (github.com/monahand1023/corpus). A real key "
        "from a private consumer's archive must never appear here, not even as "
        "test data -- it discloses that the archive exists, what its "
        "identifiers look like, and roughly how large it is. Use a placeholder "
        f"prefix instead ({', '.join(sorted(_PLACEHOLDER_PREFIXES))}), or add "
        "a genuinely standard prefix to _STANDARD_PREFIXES if that is what "
        "this is."
    )
