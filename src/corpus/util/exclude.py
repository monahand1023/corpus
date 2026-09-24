"""Canonical "noise directory" definitions shared by every part of corpus
that walks a live filesystem tree and needs to skip build/dependency/VCS
output rather than treat it as personal content.

Two call sites need the identical judgment call, and disagreeing between
them is its own bug — which is exactly what motivated pulling this out of
`corpus.survey.walk` into its own module rather than leaving
`corpus.connectors.discovery` to reinvent (and inevitably drift from) it:

  - `corpus.survey.walk.walk_files` — powers `corpus-survey census` and
    `corpus-index`'s PLAN, i.e. what a user is told will happen
    ("Noise (excluded from the plan, not ingested)").
  - `corpus.connectors.discovery.discover_files` — what every file connector
    (`corpus-ingest`, and `corpus-index`'s actual ingest step) really reads.
    Before this module existed, `discover_files` had no exclusion mechanism
    at all: every connector globbed straight into `node_modules`, `.git`,
    and everything else named below, while the plan already excluded them
    from its counts and told the user so. That claim was false — this
    module, and `discover_files` using it, is what makes it true.

`corpus.connectors.zip` (archive members, not a live directory) uses
`DEPENDENCY_DIR_NAMES` and `CORROBORATED_BUILD_DIRS` from here, plus `vendor`,
which it can afford to exclude because an archive is not a raw personal
filesystem (see below).

Split into two tiers, matching the same reasoning `corpus.connectors.zip`
already uses for archive members:

  - `DEFAULT_EXCLUDED_DIR_NAMES` / `DEFAULT_EXCLUDED_DIR_SUFFIXES`:
    unconditional — matched purely on the directory's own basename, no
    corroboration needed or possible, because none of these names
    plausibly denotes a folder of personal documents (`node_modules`,
    `.git`, `__pycache__`, a `.photoslibrary` bundle, ...).
  - `CORROBORATED_BUILD_DIRS`: `dist`, `build`, and `target` are also
    ordinary English words — a construction "build" journal, a mailing
    "dist" list, a "target" folder of goals are all plausible personal
    folder names. Excluding them unconditionally risks silently discarding
    real documents with no error and no count, which is worse than the
    noise problem this module exists to fix. Each is excluded only when a
    well-known ecosystem manifest for the matching build tool is ALSO
    present in an ancestor directory (at or above the flagged folder's
    parent) — see `has_corroborating_manifest`.

`vendor` — which `corpus.connectors.zip` excludes unconditionally for
archive members — is deliberately left off `DEFAULT_EXCLUDED_DIR_NAMES`
here: a live personal filesystem is exactly the domain where that has a
higher false-positive cost than an already-archived tree (a real "Vendor
Invoices" or "Vendors" folder is a plausible thing to find on someone's
disk). There's also no single ecosystem manifest that corroborates "vendor"
the way there is for dist/build/target, so it can't be moved into the
corroborated tier either. Left as a known, deliberately conservative gap.
"""

from __future__ import annotations

from pathlib import Path

# Directory basenames that are near-universally build/cache/VCS output,
# never personal content, and that otherwise dominate a walk to the point of
# hiding the signal (a `node_modules` tree alone can be 100k+ files).
# Matched on the exact basename, case-insensitively.
# Dependency, package and VCS directories. Shared with
# `corpus.connectors.zip`, which adds `vendor` for archive members.
DEPENDENCY_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "bower_components",
        "site-packages",
        "__pycache__",
        ".tox",
        ".venv",
        "venv",
        ".next",
        ".nuxt",
    }
)

DEFAULT_EXCLUDED_DIR_NAMES: frozenset[str] = DEPENDENCY_DIR_NAMES | frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        "$recycle.bin",
        ".trash",
    }
)

# Directory basename SUFFIXES excluded regardless of exact name — a macOS
# Photos/iPhoto library is a directory bundle, not a plain folder, and its
# contents are Apple's internal SQLite/plist/derivative-image storage, not
# documents.
DEFAULT_EXCLUDED_DIR_SUFFIXES: tuple[str, ...] = (".photoslibrary", ".photoslibrary/")

# Directory name (lowercase) -> marker filenames (lowercase) whose presence
# in an ancestor directory corroborates it as build output rather than a
# same-named personal folder. See `has_corroborating_manifest`. Also used by
# `corpus.connectors.zip` for archive members.
CORROBORATED_BUILD_DIRS: dict[str, frozenset[str]] = {
    "dist": frozenset({"package.json"}),
    "build": frozenset({"package.json", "pyproject.toml", "setup.py"}),
    "target": frozenset({"cargo.toml", "pom.xml"}),
}


def is_unconditionally_excluded_dir_name(name_lower: str) -> bool:
    """True for a directory basename (already lowercased) that's excluded
    regardless of context — no corroboration needed or possible."""
    return name_lower in DEFAULT_EXCLUDED_DIR_NAMES or any(
        name_lower.endswith(suf) for suf in DEFAULT_EXCLUDED_DIR_SUFFIXES
    )


def has_corroborating_manifest(name_lower: str, parent: Path, root: Path) -> bool:
    """True if `name_lower` is one of `CORROBORATED_BUILD_DIRS` AND a
    matching marker file exists in `parent` or any of its ancestors up to
    (and including) `root`.

    `parent` is the directory that directly contains the candidate
    dist/build/target folder — e.g. for `repo/pkg/dist`, `parent` is
    `repo/pkg`. Checking `parent` and its ancestors (not the candidate
    folder's own contents) mirrors `corpus.connectors.zip._is_dependency_noise`:
    a flat layout has the manifest right next to the flagged folder, a
    monorepo-style layout has it several directories above. Unlike
    `zip.py` (which precomputes every archive member's path up front,
    because a `ZipFile` isn't a real filesystem it can `.exists()` against),
    a live directory tree can just be asked directly — no precomputation
    needed.

    Cheap even for a deep tree: at most one `exists()` call per marker per
    ancestor level, and this is only ever invoked for a directory literally
    named `dist`/`build`/`target` — everything else short-circuits on the
    dict lookup before `parent`/`root` are touched at all.
    """
    markers = CORROBORATED_BUILD_DIRS.get(name_lower)
    if not markers:
        return False
    current = parent
    while True:
        if any((current / marker).exists() for marker in markers):
            return True
        if current == root or current == current.parent:
            return False
        current = current.parent
