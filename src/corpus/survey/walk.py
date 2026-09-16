"""Streaming, symlink-safe directory walk shared by every survey subcommand.

Two properties every survey subcommand needs and none should reimplement:

  - **Never follows symlinks.** Matches `corpus.connectors.discovery`, which
    corpus's own ingester uses — a survey that walks a tree differently than
    the ingester would is worse than useless, since its counts would not
    describe what an actual `corpus-ingest` run would see. A symlinked
    directory is not descended into; a symlinked file is not stat'd or
    yielded. Both are counted, not silently dropped.
  - **Streams.** Callers iterate a generator and accumulate only bounded
    per-bucket counters, never a list of every path — a tree with hundreds
    of thousands of files must not exhaust RAM. `os.walk` itself is already
    streaming (one directory's entries at a time); this module preserves
    that rather than materializing anything.

Errors are swallowed into counters, never raised: a permission-denied
subdirectory, a broken symlink, or a file that vanishes between `os.walk`
listing it and this module `stat`-ing it are all normal conditions for a
survey tool pointed at someone's real, messy filesystem, not bugs to crash
on.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from corpus.util.exclude import (
    CORROBORATED_BUILD_DIRS,
    DEFAULT_EXCLUDED_DIR_NAMES,
    DEFAULT_EXCLUDED_DIR_SUFFIXES,
    has_corroborating_manifest,
    is_unconditionally_excluded_dir_name,
)

__all__ = [
    "CORROBORATED_BUILD_DIRS",
    "DEFAULT_EXCLUDED_DIR_NAMES",
    "DEFAULT_EXCLUDED_DIR_SUFFIXES",
    "WalkStats",
    "WalkedFile",
    "walk_files",
]

# `DEFAULT_EXCLUDED_DIR_NAMES` / `DEFAULT_EXCLUDED_DIR_SUFFIXES` / the
# dist/build/target corroboration rules below now live in
# `corpus.util.exclude`, shared with `corpus.connectors.discovery` (the
# actual ingest-time file discovery every connector uses) — the two used to
# be defined independently, which was exactly how a survey could report a
# directory as excluded while a real ingest walked straight into it anyway.
# Re-exported here (not just imported) because `tests/test_survey_walk.py`
# and external callers already do `from corpus.survey.walk import
# DEFAULT_EXCLUDED_DIR_NAMES`.
#
# `dist`, `build`, and `target` are also ordinary English words that could
# name a real folder of documents — previously left out of the unconditional
# set entirely, with this module's docstring explaining that "a
# directory-name-only prune here cannot" corroborate them the way
# `corpus.connectors.zip` does for archive members. That reasoning doesn't
# actually hold for a *live* filesystem walk (unlike a zip's member list, a
# real directory can just be asked "does `pyproject.toml` exist here?"), so
# `has_corroborating_manifest` below now does the same corroboration check
# zip.py does, and this walk is no longer stuck with a narrower default than
# discover_files needs to stay in sync with it.


@dataclass
class WalkStats:
    """Mutable counters updated in place while `walk_files` is iterated.
    Read these only after the generator is exhausted (or abandoned) — they
    fill in as a side effect of iteration, not up front."""

    files_scanned: int = 0
    dirs_visited: int = 0
    dirs_pruned: int = 0
    dir_symlinks_skipped: int = 0
    file_symlinks_skipped: int = 0
    permission_errors: int = 0
    stat_errors: int = 0
    files_excluded: int = 0
    # Pruned directories that are MEDIA BUNDLES -- a `.photoslibrary`, whose
    # `originals/` is home video. `dirs_pruned` is a bare count, and a count
    # cannot tell a caller that the thing it was looking for is inside the
    # thing it skipped. Named, because the remedy is to point the root AT one
    # of these, which is impossible to do without knowing where they are.
    # Bounded in practice by how many photo libraries a person owns.
    media_bundles_pruned: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WalkedFile:
    path: Path
    rel_path: str  # POSIX-style, relative to root — stable across platforms
    size: int


def _matches_any(rel_posix: str, name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, p) or fnmatch.fnmatch(name, p) for p in patterns)


def walk_files(
    root: Path,
    excludes: tuple[str, ...] = (),
    use_default_excludes: bool = True,
    stats: WalkStats | None = None,
) -> Iterator[WalkedFile]:
    """Stream every regular, non-excluded file under `root`.

    `excludes` are `fnmatch` patterns (e.g. `*.log`, `*/Caches/*`) checked
    against both the file/dir's basename and its `root`-relative POSIX path,
    so `--exclude node_modules` and `--exclude '**/node_modules/*'` both
    work without the caller needing to know which form matches. Directories
    are pruned (not descended into) the moment they match a default or
    user-supplied exclude — this is a performance requirement, not just a
    filtering one: without it, excluding `node_modules` would still mean
    walking every file inside it before discarding each.

    `stats`, if given, is filled in as a side effect; pass the same object
    back in to inspect counts after iteration.
    """
    stats = stats if stats is not None else WalkStats()
    root = root.resolve()

    def _on_error(exc: OSError) -> None:
        stats.permission_errors += 1

    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_error, followlinks=False):
        dirpath_p = Path(dirpath)
        stats.dirs_visited += 1

        kept_dirnames = []
        for d in dirnames:
            dir_full = dirpath_p / d
            if dir_full.is_symlink():
                stats.dir_symlinks_skipped += 1
                continue
            lower = d.lower()
            if use_default_excludes and (
                is_unconditionally_excluded_dir_name(lower)
                or has_corroborating_manifest(lower, dirpath_p, root)
            ):
                stats.dirs_pruned += 1
                if any(lower.endswith(suf) for suf in DEFAULT_EXCLUDED_DIR_SUFFIXES):
                    stats.media_bundles_pruned.append(str(dir_full))
                continue
            if excludes:
                rel = (dir_full.relative_to(root)).as_posix()
                if _matches_any(rel, d, excludes):
                    stats.dirs_pruned += 1
                    continue
            kept_dirnames.append(d)
        # Mutating dirnames in place is `os.walk`'s documented mechanism for
        # pruning traversal — replacing the list object would not work.
        dirnames[:] = kept_dirnames

        for fname in filenames:
            file_full = dirpath_p / fname
            if file_full.is_symlink():
                stats.file_symlinks_skipped += 1
                continue

            rel = file_full.relative_to(root).as_posix()
            if excludes and _matches_any(rel, fname, excludes):
                stats.files_excluded += 1
                continue

            try:
                st = file_full.stat()
            except OSError:
                stats.stat_errors += 1
                continue

            stats.files_scanned += 1
            yield WalkedFile(path=file_full, rel_path=rel, size=st.st_size)
