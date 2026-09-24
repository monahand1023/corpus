"""Safe file discovery for connectors.

Connectors point at a user-configured directory and glob for files. Without
containment checks, a symlink inside that directory (e.g. `notes.md ->
~/.ssh/id_rsa`) or a `..` in the glob would let ingestion read files outside
the configured root — which then surface via search results to the LLM. This
helper enforces: no symlinks, and every yielded path resolves inside `root`.

**Directory exclusion.** By default (`use_default_excludes=True`, every
existing caller's implicit behavior — no connector needed to change to get
this), files under a well-known noise directory — `node_modules`, `.git`,
build caches, a `.photoslibrary` bundle, or `dist`/`build`/`target` when a
corroborating ecosystem manifest confirms it — are skipped, using the exact
same definitions `corpus.survey.walk` uses for `corpus-survey census` and
`corpus-index`'s plan (see `corpus.util.exclude`). Before this existed,
`corpus-index` printed "Noise (excluded from the plan, not ingested)" while
every connector's `discover_files` call walked straight into those same
directories anyway — the plan's exclusion and the real ingest disagreed, and
the message was false. Every connector funnels through this one function, so
fixing it here fixes all of them at once, with no per-connector change.

Pass `use_default_excludes=False` for a source that genuinely wants a
vendored/build tree indexed — e.g. a deliberate archive of a dependency's
docs. There is currently no per-source knob for this in `corpus.toml` (every
connector's constructor takes only `path`/`glob`, and adding one would mean
threading a new field through `SourceConfig` and ten connector constructors
for the same result this default already gives everyone for free); the
practical workaround today is pointing a source's `path` directly at the
subdirectory you want indexed — exclusion only ever prunes a directory
encountered *during* the walk, never the configured root itself.

`corpus.connectors.zip` needs a different lever than that parameter for the
same problem: it composes other connectors' unmodified `.load()` methods
over a disposable extraction directory it has already filtered with its own
(more nuanced) logic, so there's no call site of its own where it could pass
`use_default_excludes=False` to those connectors' internal `discover_files`
calls. See `default_excludes_suppressed()` below for that case.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path, PurePath

from corpus.util.exclude import has_corroborating_manifest, is_unconditionally_excluded_dir_name

logger = logging.getLogger(__name__)

# Set only by `corpus.connectors.zip.ZipConnector._load_extracted`, via
# `default_excludes_suppressed()` below — see that context manager's
# docstring for why a disposable zip-extraction directory needs every
# `discover_files` call under it (including each per-type sub-connector's
# own internal one, which this module has no other way to reach) to ignore
# the default exclusion this module otherwise always applies.
_suppress_default_excludes: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "corpus_discovery_suppress_default_excludes", default=False
)


@contextmanager
def default_excludes_suppressed() -> Iterator[None]:
    """Within this context, every `discover_files` call — including ones
    made deep inside a connector this code doesn't call directly — behaves
    as if `use_default_excludes=False` were passed, regardless of what its
    caller actually passed.

    Exists for exactly one caller: `corpus.connectors.zip.ZipConnector._load_extracted`.
    A disposable zip-extraction directory has ALREADY been filtered by
    zip.py's own noise/dependency logic (`_is_archive_noise` /
    `_is_dependency_noise`, governed by `SourceConfig.exclude_dependencies`)
    before any file reaches it — that logic is more nuanced than this
    module's live-filesystem-oriented default (it also handles archive/OS
    packaging noise and corroborated dist/build/target dirs) and has
    already made the complete decision about what belongs there. Applying
    this module's default on top would be redundant when
    `exclude_dependencies=True` (nothing noise-named survives extraction to
    trip it) and actively wrong when `exclude_dependencies=False` (it would
    silently re-exclude content the archive was explicitly configured to
    include, defeating that setting — the regression this context manager
    exists to prevent; see `tests/test_zip_connector.py`).

    A plain function parameter can't reach this: `_load_extracted` composes
    each per-type connector (`MarkdownConnector`, `PdfConnector`, ...) by
    calling its unmodified, registry-built `.load()`, which calls
    `discover_files` internally with no way for the caller to override that
    one argument. A `contextvars.ContextVar` reaches every `discover_files`
    call in the dynamic scope below `with default_excludes_suppressed():`
    without changing any connector's constructor or `SourceConfig`.
    """
    token = _suppress_default_excludes.set(True)
    try:
        yield
    finally:
        _suppress_default_excludes.reset(token)


def _is_under_excluded_dir(resolved: Path, root: Path) -> bool:
    """True if any directory strictly between `root` and `resolved` (i.e.
    every path component of `resolved` other than `root` itself and the
    final filename) is noise per `corpus.util.exclude`.

    Checking every intermediate component — not just the immediate parent —
    matters: `node_modules/pkg/dist/readme.md` must be excluded because of
    `node_modules` even though `dist` (its own immediate parent) has no
    corroborating manifest sitting next to it.
    """
    current = root
    for part in resolved.relative_to(root).parts[:-1]:
        lower = part.lower()
        if is_unconditionally_excluded_dir_name(lower) or has_corroborating_manifest(
            lower, current, root
        ):
            return True
        current = current / part
    return False


# The SOURCE's own exclude list, set around a connector's `load()`.
#
# A ContextVar rather than a parameter on seventeen connector constructors:
# `discover_files` is the one seam they all share, and threading a new
# argument through every builder would mean a connector that forgot it
# silently ignores the setting -- which is the exact failure this feature
# exists to stop making. Same mechanism as `_suppress_default_excludes`
# directly above.
_source_excludes: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "corpus_source_excludes", default=()
)


@contextmanager
def source_excludes(patterns: Sequence[str]) -> Iterator[None]:
    """Apply a source's `exclude` list to every `discover_files` call within."""
    token = _source_excludes.set(tuple(patterns))
    try:
        yield
    finally:
        _source_excludes.reset(token)


def _excluded(rel: PurePath, name: str, patterns: Sequence[str]) -> bool:
    """fnmatch against the root-relative path AND the basename.

    Both, for the same reason `walk_files` does it: `exclude = ["Backup"]`
    and `exclude = ["**/Backup/*"]` are the two forms a reader writes without
    knowing which one the implementation wanted, and only matching one of
    them makes the setting look broken. A directory name matches when it is
    any component of the path, so excluding a folder excludes what is under
    it.
    """
    posix = rel.as_posix()
    parts = set(rel.parts)
    for pattern in patterns:
        # A leading `**/` is path-glob syntax, which fnmatch does not know:
        # fnmatch("Backup/report.md", "**/Backup/*") is False because `**/`
        # wants a segment before it. Stripping it is what the reader meant.
        bare = pattern.removeprefix("**/")
        if (
            fnmatch(posix, pattern)
            or fnmatch(posix, bare)
            or fnmatch(name, pattern)
            or fnmatch(name, bare)
            or pattern in parts
            or fnmatch(posix, f"{bare.rstrip('/')}/*")
            or any(fnmatch(part, bare) for part in rel.parts[:-1])
        ):
            return True
    return False


def discover_files(root: Path, glob: str) -> Iterator[Path]:
    """Yield regular files under `root` matching `glob`, excluding symlinks,
    any path that resolves outside `root`, and — by default — anything under
    a well-known noise directory (see the module docstring and
    `corpus.util.exclude`).

    The source's own `exclude` list (from `corpus.toml`) applies through
    `source_excludes`, and the default exclusion is lifted with
    `default_excludes_suppressed`: both reach every connector's call without
    a parameter on each constructor."""
    use_default_excludes = not _suppress_default_excludes.get()
    source_exclude = _source_excludes.get()
    root = root.resolve()
    for path in sorted(root.glob(glob)):
        if path.is_symlink():
            logger.warning("skipping symlink (not followed): %s", path)
            continue
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            logger.warning("skipping path outside source root: %s", path)
            continue
        if use_default_excludes and _is_under_excluded_dir(resolved, root):
            logger.debug("skipping file under excluded noise directory: %s", path)
            continue
        if source_exclude and _excluded(resolved.relative_to(root), path.name, source_exclude):
            logger.debug("skipping file excluded by this source: %s", path)
            continue
        yield path
