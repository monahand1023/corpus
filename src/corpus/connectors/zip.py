"""Zip archive connector — makes documents inside `.zip` archives searchable
without reimplementing PDF/DOCX/etc. extraction for archived copies of the
same file types.

Approach: extract one archive to a fresh temp directory, run the existing
per-file-type connectors from `registry.py` (`CONNECTOR_REGISTRY`) against
that directory, remap each resulting document's `source_key` to encode which
archive it came from, then delete the temp directory. **The archive on disk
is only ever opened for reading — never modified, moved, or deleted.**

Composition, not duplication: this module owns none of the PDF/DOCX/HTML/etc.
parsing logic. It builds a `SourceConfig` pointing at the extraction
directory for each file type present and calls straight into
`CONNECTOR_REGISTRY`, the same factories `corpus-ingest --path` and
hand-written `[[sources]]` blocks already use — so a bug fix or a new
extraction feature in `pdf.py` benefits archived PDFs for free, and a missing
optional extra (`pip install corpus-rag[pdf]`) produces the exact same
actionable error it would outside a zip.

Cleanup guarantee: extraction and reading happen SYNCHRONOUSLY per archive in
`_load_one_archive`, inside a `try/finally` that removes the temp directory
before that method returns — success, safety refusal, or exception. Documents
are only handed to the caller (materialized into a plain list) after the temp
directory is already gone; nothing is yielded while it's still suspended
mid-extraction. That means cleanup never depends on a generator being fully
drained or garbage-collected: a crash anywhere in extraction or in a
downstream connector's `load()` still runs the `finally` before the exception
leaves this module. See `tests/test_zip_connector.py`'s crash test.

Safety properties (each a known archive-extraction failure mode):

  - **Zip-slip.** Every member's extraction target is computed as
    `(extract_dir / member.filename).resolve()` and checked with
    `Path.is_relative_to(extract_dir)` — never a string-prefix check. A
    string check can be defeated by `Path` join semantics: joining an
    ABSOLUTE member name onto a base path *discards the base* per pathlib's
    documented behavior (`Path("/root") / "/etc/passwd" == Path("/etc/passwd")`),
    so a naive "starts with extract_dir" check on the unresolved path could
    look fine right up until you inspect what actually got joined. Resolving
    first and checking containment on the resolved result catches this,
    absolute paths, `../..` traversal, and (were we ever to create one — see
    below) a symlink pointing outside the root, uniformly. A refused member
    is skipped individually and counted in `skipped_files`; the rest of the
    archive still extracts.
  - **Symlink members.** `zipfile` never recreates a symlink from a "symlink"
    entry (Unix mode bits in `external_attr`) — extraction always writes
    member content as a plain file at a path we compute and validate
    ourselves. We do not special-case or honor those mode bits, which is what
    keeps this true; if that ever changes, the zip-slip check above still
    covers a symlink's *target path*, but a created symlink's target being
    followed later by something else reading the file back would not be.
  - **Zip bombs.** Declared total uncompressed size (`ZipInfo.file_size`,
    summed) and member count are checked against the archive's central
    directory *before* extracting a single byte. Actual bytes written are
    also tracked cumulatively *during* extraction and checked against the
    same cap, because a crafted header can declare a small size while the
    compressed stream inflates far larger — checking only the declared value
    would refuse an honestly-labeled bomb but extract a lying one. Either
    check tripping refuses the archive (declared-size trip refuses before
    extracting anything; actual-size trip refuses mid-stream and the partial
    output is discarded with the rest of the temp directory).
  - **Encrypted archives.** Detected via `ZipInfo.flag_bits & 0x1` (the
    standard ZIP "file is encrypted" flag), checked PER MEMBER, before any
    extraction is attempted for that member — never by attempting to read it
    and catching the resulting `RuntimeError`, and never by prompting for a
    password, since this runs unattended. An encrypted member is skipped on
    its own, the same way a zip-slip member is: an archive with 619 plain
    files and one encrypted one still yields the 619, rather than discarding
    the whole archive over one file. (A member encrypted with a scheme that
    doesn't set the standard bit would still surface as a `RuntimeError` from
    `zipfile` when opened; that path is handled the same way — skip, don't
    prompt.)
  - **Nested archives.** A member whose name ends in `.zip` is refused, not
    extracted and not recursed into — see `MAX_DEPTH` below for why depth is
    fixed at 1 rather than configurable.
  - **Same-path collisions during extraction.** Two zip members can land on
    the identical extraction-directory path without either being a security
    attack: a case-insensitive host filesystem (the macOS/Windows default)
    folds `notes.md` and `notes.MD` onto one inode even though the archive's
    own central directory treats them as distinct entries, and a malformed
    archive can simply declare a name twice. `_disambiguate` checks before
    every write and renames the second arrival (`name__1.ext`) rather than
    letting it silently truncate-and-overwrite the first — an overwrite here
    would delete content with no error and no count.

Mojibake member names (measured live in a real index: 77.3% of one archive
source's chunks, a few percent of another — roughly a few thousand chunks total). Many
real-world zip tools write non-ASCII filenames as raw UTF-8 (or, for older
Japanese-locale tools, Shift-JIS/CP932) bytes WITHOUT setting the standard
"filename is UTF-8" flag bit (0x800) in the member header. Per the ZIP spec,
`zipfile` decodes a member's filename as UTF-8 when that flag is set and as
CP437 otherwise — so for one of these archives, `info.filename` as `zipfile`
hands it to us is the CP437 misdecoding of what were really UTF-8/Shift-JIS
bytes (e.g. `'Finances/18∩╜₧20µ£ƒPL...'` instead of the correct Japanese
filename it was actually storing).
Because the member name becomes both the extraction path and (via
`_load_extracted`) the chunk `source_key` and document title, an unrepaired
mojibake name means the document is unfindable by its real name AND
re-ingests under a different key if the misdecoding ever changed. Every
member's name is repaired (if needed) once, up front in `_load_one_archive`
via `_repair_member_names`, before anything else touches `info.filename` —
zip-slip containment, noise/dependency filtering, extraction, and
`source_key` construction all then see the corrected name automatically,
with no further plumbing. The repair is never a guess: it re-encodes the
CP437 misdecoding back to its original bytes (CP437 round-trips any byte
0-255 losslessly) and only accepts a UTF-8 or CP932 (Shift-JIS) strict
decode of those bytes that both succeeds AND differs from the original — an
archive whose non-flagged name genuinely IS CP437/ASCII (the common case)
round-trips to itself and is left untouched. See `_repair_filename_encoding`.

Archive/OS packaging artifacts (macOS's `__MACOSX/` AppleDouble tree and
`.DS_Store`, Windows' `Thumbs.db`) are filtered out before any of the above —
before type matching, before extraction, before either counter. See
`_is_archive_noise`; this is not a courtesy, it closes a real failure mode:
an AppleDouble stub like `__MACOSX/reports/._q3.pdf` still matches the
`.pdf` glob and would otherwise reach the PDF connector, fail to parse, and
land in `failed_files` — which suppresses orphan pruning for the WHOLE
source, permanently, for files that were never documents.

Dependency and build-output noise (overridable, unlike the packaging noise
above): measured on a real large multi-archive / many-member sample, 33,644
members were vendored-dependency or build-output files, not personal
content -- one deployment-bundle archive's entire 243 "documents" turned out
to be third-party npm package READMEs (`node_modules/<pkg>/readme.md` and
the like). Those are real markdown, so unfiltered they match the markdown
connector, get chunked, embedded, and indexed -- costing embedding tokens
and actively degrading retrieval, since a search then has to compete against
hundreds of irrelevant library docs. `_is_dependency_noise` excludes a
member when any path component is a vendored-dependency/build/VCS directory,
or the member is itself an obvious minified/compiled leaf file. Given the
identical treatment as `_is_archive_noise`: not extracted, not type-matched,
not counted in either counter. UNLIKE the packaging noise above, this is a
judgment call about content the user might actually want (someone could
deliberately archive a library's docs), so it's a per-source toggle --
`SourceConfig.exclude_dependencies`, default on -- rather than an
unconditional exclusion. See `_UNAMBIGUOUS_DEPENDENCY_DIRS` and
`_CORROBORATED_BUILD_DIRS` for exactly what's matched and why `dist`,
`build`, and `target` need a corroborating signal that the other directory
names don't.

Extension matching, once safely extracted: a member is recognized by any
accepted spelling of its type, matched case-insensitively — `.htm` alongside
`.html`, `.markdown` alongside `.md`, and `.PDF`/`.DOCX`-style uppercase from
non-Unix tooling for every type. See `_extension_type_map` and
`_normalize_to_canonical_extension` — a non-canonical spelling is renamed
(within the disposable extraction directory only) to its type's canonical
extension so the existing per-type connector's own single-glob discovery
finds it unmodified. `.doc`/`.xls` are deliberately NOT treated as spelling
variants of `.docx`/`.xlsx` — they're different container formats that
python-docx/openpyxl cannot read.

Failure-counter policy for this connector specifically:

  - `skipped_files` (permanent, does not block orphan pruning): an encrypted
    member, a zip-slip refusal, or a nested archive (each counted per
    member); a file with no matching connector, or one whose type IS
    supported but the optional extra for it isn't installed here (counted
    per file); and a whole-archive refusal — too many members, or over
    either uncompressed-size cap — counted as **1** (the archive, not its
    member count, since none of it was processed).
  - `failed_files` (possibly transient, suppresses pruning for the source):
    an archive that fails to open (`BadZipFile`/`OSError` — could be mid-write
    or momentarily locked), counted as 1; and a member that raises while
    being decompressed, counted per member. Per-file failures inside an
    extracted directory (e.g. a genuinely corrupt PDF among otherwise-fine
    ones) are the delegated connector's own `failed_files`, summed in here.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

from corpus.config import SourceConfig
from corpus.connectors.discovery import default_excludes_suppressed, discover_files
from corpus.connectors.music import MUSIC_EXTENSIONS
from corpus.connectors.registry import CONNECTOR_REGISTRY, DEFAULT_GLOBS
from corpus.types import SourceDocument

logger = logging.getLogger(__name__)

# Enforced per archive, both against the archive's DECLARED sizes (fast,
# checked up front) and against ACTUAL bytes written during extraction (slow
# path, the backstop for a lying header — see the module docstring). Sized
# generously against the archives this connector shipped for (200-620
# ordinary office/text files per archive): a genuine archive that size is very
# unlikely to clear either number, so these exist to catch a hostile or
# corrupted archive, not to constrain normal use. Both are constructor
# parameters (not corpus.toml settings) so a deployment with unusual archives
# can pass different values without a config schema change; tests use small
# values to trigger the caps cheaply.
DEFAULT_MAX_UNCOMPRESSED_BYTES = 4 * 1024**3  # 4 GiB
DEFAULT_MAX_MEMBERS = 20_000

# This connector extracts exactly the archives it is pointed at and never
# recurses into a `.zip` found inside one — a zip-of-zips is refused outright
# (see `_extract_members`) rather than unpacked one more level. Two reasons,
# not one: recursing would need its own, compounding size budget at each
# level to keep the zip-bomb guard meaningful (a bomb can hide behind
# nesting, not just repetition), and it would make "how far did this
# actually unpack" harder to reason about for a connector whose entire job is
# extraction safety. If real-world archives turn out to nest, raise this
# deliberately — it documents a choice, not a limitation nobody noticed.
MAX_DEPTH = 1

# File-type connectors this module composes, keyed exactly as they are in
# `CONNECTOR_REGISTRY` (see `_load_extracted`). Derived from the registry
# itself (minus `_NON_MEMBER_CONNECTOR_TYPES` below) rather than hand-
# duplicated — this WAS a hand-maintained tuple, and it silently missed
# `pptx`, `csv`, and `tsv` for a release cycle after each was registered
# elsewhere (`tests/test_zip_connector.py::test_member_connector_types_...`
# pins the fix). A hand-maintained duplicate of a registry is exactly the
# kind of thing that drifts the moment someone adds a connector without
# knowing this list exists; deriving it means adding a connector is enough.
#
# `_NON_MEMBER_CONNECTOR_TYPES` is the one thing still hand-maintained, and
# deliberately so — these are registry entries that would be actively wrong
# to recurse into automatically, not merely "not added yet":
#   zip   — this module's own recursion guard (see `MAX_DEPTH` above). A
#           future second "zip" entry existing at all would only ever be
#           this one, so excluding it by name is unambiguous.
#   aup3  — `AupThreeConnector.load()` itself would be perfectly safe here
#           (it only reads, same as every other member connector); what's
#           NOT safe is that its whole value proposition — a companion
#           `extract_audio()` writing a playable file "adjacent to the
#           source" (see `corpus/connectors/aup3.py`) — has no sensible
#           "adjacent" inside a zip member's disposable, about-to-be-
#           deleted extraction directory. Rather than register a connector
#           whose main capability silently can't be used for anything
#           reached through this path, `.aup3` members are left in the
#           unsupported-extension bucket (`skipped_files`), same as any
#           other unregistered type — extract it from the archive first if
#           you want its audio. Revisit if that tradeoff stops making sense.
#   transcripts — `TranscriptConnector` does not read FILES. It reads a
#           sidecar DATABASE, and its `DEFAULT_GLOBS` entry is marked
#           "detection pattern only" in the registry for exactly that
#           reason. Handed a zip's extraction directory it raises
#           "transcript database not found", which `Ingester.ingest`
#           treats as a source that cannot be enumerated -- so the WHOLE
#           zip source is skipped with nothing indexed. Found on a live
#           archive whose `archives` source had been failing that way,
#           reported once as an ERROR line and then never again because
#           every later run failed identically.
#
#           This is the cost of deriving the list from the registry: the
#           derivation stops a connector being FORGOTTEN, and in exchange
#           a registry entry that must not be recursed into is included
#           the moment someone adds it. That trade is still the right
#           one -- forgetting is silent, this was loud -- but it is the
#           second entry to need adding here.
_NON_MEMBER_CONNECTOR_TYPES = frozenset({"zip", "aup3", "transcripts"})
_MEMBER_CONNECTOR_TYPES = tuple(
    t for t in CONNECTOR_REGISTRY if t not in _NON_MEMBER_CONNECTOR_TYPES
)

# Secondary spellings a real archive can contain that its type's connector
# already documents accepting via an explicit `corpus.toml` `glob` override
# (markdown.py: ".md / .markdown"; html.py + README: ".html / .htm") but
# which DEFAULT_GLOBS -- one glob per type -- doesn't reach on its own, and a
# zip's contents can't be reconfigured per file the way a `[[sources]]` block
# can. Extension lookup is case-insensitive (see `_extension_type_map`), so
# this also covers `.HTM`, `.MARKDOWN`, etc. Legacy formats that are NOT
# spelling variants are deliberately absent -- `.doc` and `.xls` are
# different container formats entirely (see docx.py's and xlsx.py's own
# docstrings) that python-docx/openpyxl cannot read, so aliasing them would
# silently produce "cannot open" failures instead of an honest unsupported-
# extension skip.
_EXTENSION_ALIASES: dict[str, str] = {
    ".htm": "html",
    ".markdown": "markdown",
}

# Types whose connector reads several formats natively, not one format under
# several spellings. Each keeps its own extension (lowercased) instead of
# being renamed to the canonical one: a FLAC renamed to `.mp3` is a file
# whose name lies about its container.
_NATIVE_EXTENSIONS: dict[str, tuple[str, ...]] = {"music": MUSIC_EXTENSIONS}


def _extension_type_map(default_globs: dict[str, str]) -> dict[str, str]:
    """Extension (lowercase, with leading dot) -> connector type name,
    covering each type's canonical `DEFAULT_GLOBS` extension plus the
    aliases above. Derived from `DEFAULT_GLOBS` rather than hand-duplicated
    so the canonical spellings can't drift out of sync with the registry."""
    mapping = {
        Path(default_globs[type_name]).suffix: type_name for type_name in _MEMBER_CONNECTOR_TYPES
    }
    mapping.update(_EXTENSION_ALIASES)
    for type_name, extensions in _NATIVE_EXTENSIONS.items():
        if type_name in _MEMBER_CONNECTOR_TYPES:
            mapping.update(dict.fromkeys(extensions, type_name))
    return mapping


# Archive/OS packaging artifacts, never user content, seen constantly in
# real-world zips: macOS Finder writes a parallel `__MACOSX/` tree of
# `._filename` AppleDouble resource-fork stubs when it creates a zip (one
# real archive from this connector's own target user had 5,202 of these
# across ~8,400 total members), plus `.DS_Store` per directory; Windows
# Explorer writes `Thumbs.db` thumbnail caches. None of these are documents,
# and treating them as such is actively harmful, not just noisy: a stub like
# `__MACOSX/reports/._q3.pdf` still matches the `.pdf` glob, gets handed to
# the PDF connector, and fails to parse (it's a resource fork, not a PDF) --
# counted in `failed_files`, which suppresses orphan pruning for the WHOLE
# source, permanently, on every future run, for files that were never
# documents to begin with. Filtered out here, before type matching and
# before either counter, rather than "not documents" being downgraded to a
# `skipped_files` count -- they're not almost-content, they're not content
# at all. Deliberately a small explicit list, not a general glob/heuristic:
# add another entry only for another artifact this specific, not "anything
# that looks unimportant."
_ARCHIVE_NOISE_BASENAMES = frozenset({".ds_store", "thumbs.db"})


# Tried in order, strict decode only — see `_repair_filename_encoding` and
# the module docstring's "Mojibake member names" section. Not `chardet`/
# `charset-normalizer`: this fixes the two encodings actually measured in a
# real archive (UTF-8, CP932/Shift-JIS), not every encoding that has ever
# existed. `corpus.util.encoding` runs the same UTF-8 -> CP932 -> latin-1
# tiering for member *content*; there's no shared helper between the two
# because this one has a different final step (give up and keep the
# CP437-decoded original, since a raw ZIP filename field has no
# "impossible to fail" encoding the way bytes-to-text does) and a different
# acceptance rule (must differ from the input, not just decode).
_FILENAME_REPAIR_ENCODINGS: tuple[str, ...] = ("utf-8", "cp932")


def _repair_filename_encoding(info: zipfile.ZipInfo) -> str:
    """Return `info.filename`, repaired if it looks like UTF-8 (or
    Shift-JIS) misdecoded as CP437 — see the module docstring's "Mojibake
    member names" section for the full mechanism and the real numbers that
    motivated it.

    Never guesses: only returns a repaired name when re-encoding the
    CP437-decoded name back to bytes and strict-decoding those bytes as
    UTF-8 or CP932 both SUCCEEDS and DIFFERS from the original. A name that
    already had the UTF-8 flag set is returned unchanged (nothing to
    repair); a non-flagged name that genuinely is CP437/ASCII — the common
    case, most filenames are ASCII — round-trips to itself and is returned
    unchanged too, rather than accepted as its own "repair".
    """
    if info.flag_bits & 0x800:
        return info.filename  # UTF-8 flag was set; zipfile already decoded it correctly

    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        # zipfile decoded this filename as cp437 in the first place, so
        # re-encoding it as cp437 cannot fail in practice — this guard is
        # defense in depth against a future zipfile behavior change, not a
        # path this module's own tests can trigger honestly.
        return info.filename

    for encoding in _FILENAME_REPAIR_ENCODINGS:
        try:
            repaired = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if repaired != info.filename:
            return repaired
    return info.filename


def _path_components(member_name: str) -> list[str]:
    """Split a raw archive member name into path components, tolerant of
    both `/` (the zip spec) and `\\` (some Windows-authored tools write it
    anyway) as separators, with empty components (leading/trailing/doubled
    separators) dropped."""
    return [p for p in member_name.replace("\\", "/").split("/") if p]


def _is_archive_noise(member_name: str) -> bool:
    """True for a member that is archive/OS packaging, not user content —
    see `_ARCHIVE_NOISE_BASENAMES` above. Checked on the raw member name
    (before any path-containment resolution), since these are filtered out
    entirely rather than merely refused."""
    parts = _path_components(member_name)
    if not parts:
        return False
    if "__MACOSX" in parts:
        return True
    basename = parts[-1]
    if basename.startswith("._"):
        return True
    return basename.lower() in _ARCHIVE_NOISE_BASENAMES


# ---------------------------------------------------------------------------
# Dependency / build-output noise — overridable via
# `SourceConfig.exclude_dependencies` (default True), unlike the packaging
# noise above. See the module docstring's "Dependency and build-output
# noise" section for the real-world numbers that motivated this.
# ---------------------------------------------------------------------------
#
# Split into two tiers by false-positive risk. Matching is always on an
# EXACT path component, never a substring — `my-node_modules-notes/` and
# `distribution/` do not match `node_modules`/`dist` — and always
# case-insensitive: real tooling always lowercases these names, so matching
# case-insensitively only widens what's caught (a Windows tool, or a manual
# rezip that changed case, shouldn't defeat the filter) at essentially no
# false-positive cost, since none of these names is a plausible *intentional*
# different-case folder name either.
#
# Unconditional — excluded regardless of what else is in the archive. None
# of these plausibly names a real folder in a personal document archive:
#   node_modules, bower_components  — JS/npm dependency trees
#   site-packages                   — Python dependency tree
#   vendor                          — Go/PHP/Ruby dependency-tree convention.
#     Considered requiring corroboration for this one too — "vendor" alone
#     is also an ordinary business word ("Vendors/" of invoices is
#     plausible). Kept unconditional anyway: a *bare lowercase* "vendor"
#     path component is overwhelmingly the dependency-tree convention in
#     practice (a real "Vendors" folder is usually capitalized or
#     multi-word — "vendor-contracts", "Vendor Invoices"), and this filter
#     is aimed at exactly the archives that skew toward the former.
#   .git, .svn, .hg                 — VCS metadata, never content
#   __pycache__, .tox, .venv, venv  — Python build/tooling artifacts
#   .next, .nuxt                    — framework build-output dirs; the name
#     alone identifies the framework, not plausible as a personal folder
#
# Corroborated-only — `dist`, `build`, and `target` are deliberately NOT on
# the unconditional list. Each is also a common English word that could
# plausibly name a real folder in someone's documents (a construction
# "build" journal, a "target" list, a mailing "dist" list); excluding on the
# bare name risks silently discarding real documents with no error and no
# count, which is worse than the noise problem this exists to fix. Each is
# excluded only when a well-known ecosystem manifest for the matching build
# tool is ALSO present somewhere in the same archive, at or above the
# directory containing the flagged folder — a corroborating signal that this
# really is that ecosystem's build output, not a same-named personal folder
# that happens to sit alone in the archive. See `_is_dependency_noise`.
_UNAMBIGUOUS_DEPENDENCY_DIRS = frozenset(
    {
        "node_modules",
        "bower_components",
        "site-packages",
        "vendor",
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".tox",
        ".venv",
        "venv",
        ".next",
        ".nuxt",
    }
)

# Directory name (lowercase) -> marker filenames (lowercase) whose presence
# anywhere at or above that directory in the SAME archive corroborates it as
# build output rather than a same-named personal folder.
_CORROBORATED_BUILD_DIRS: dict[str, frozenset[str]] = {
    "dist": frozenset({"package.json"}),
    "build": frozenset({"package.json", "pyproject.toml", "setup.py"}),
    "target": frozenset({"cargo.toml", "pom.xml"}),
}

# Minified/compiled leaf artifacts: never hand-authored content, matched on
# the basename regardless of which directory they're in — a `dist/app.min.js`
# inside an uncorroborated `dist/` is still obviously build output by its own
# name, and `libs/jquery.min.js` vendored directly next to source with no
# recognizable dependency directory at all is common too. None of these
# extensions are matched by any connector in `_MEMBER_CONNECTOR_TYPES` today,
# so filtering them mainly avoids pointless extraction and `skipped_files`
# inflation for files nothing was ever going to index — cheap insurance
# against a future connector for one of these types picking them up
# unfiltered.
_MINIFIED_LEAF_SUFFIXES = (".min.js", ".min.css", ".map")


def _dependency_marker_paths(infos: list[zipfile.ZipInfo]) -> frozenset[str]:
    """Lowercase, `/`-joined path of every member in the archive (files and
    directory entries alike) — used only to look up whether a known
    ecosystem manifest sits at or above a `dist`/`build`/`target` directory
    before treating it as noise (see `_CORROBORATED_BUILD_DIRS`). Computed
    once per archive up front, not per member, since every candidate member
    consults the same lookup set."""
    return frozenset("/".join(_path_components(info.filename)).lower() for info in infos)


def _is_dependency_noise(member_name: str, marker_paths: frozenset[str]) -> bool:
    """True for a member under a vendored-dependency/build-output directory
    (`_UNAMBIGUOUS_DEPENDENCY_DIRS`, or `_CORROBORATED_BUILD_DIRS` with a
    matching marker present in `marker_paths`), or itself an obviously
    minified/compiled leaf file (`_MINIFIED_LEAF_SUFFIXES`). Only consulted
    when `ZipConnector`'s `exclude_dependencies` option is on (the default);
    see the module docstring's "Dependency and build-output noise" section."""
    parts = _path_components(member_name)
    if not parts:
        return False

    lower_parts = [p.lower() for p in parts]
    for i, part in enumerate(lower_parts):
        if part in _UNAMBIGUOUS_DEPENDENCY_DIRS:
            return True
        markers = _CORROBORATED_BUILD_DIRS.get(part)
        if markers is None:
            continue
        # Any ancestor directory from the archive root down to (and
        # including) this component's immediate parent, checked for one of
        # the marker files — covers both a flat layout (manifest sits right
        # next to dist/build/target) and a monorepo-style layout (manifest
        # sits at the project root, several directories above).
        for depth in range(i + 1):
            prefix = "/".join(lower_parts[:depth])
            if any(
                (f"{prefix}/{marker}" if prefix else marker) in marker_paths
                for marker in markers
            ):
                return True

    basename = lower_parts[-1]
    return basename.endswith(_MINIFIED_LEAF_SUFFIXES)


def _disambiguate(target: Path, *, moving: Path | None = None) -> Path:
    """Return `target` unchanged if nothing occupies that path yet, otherwise
    a numbered variant (`name__1.ext`, `name__2.ext`, ...) that doesn't
    collide. Used everywhere two distinct sources could otherwise land on
    the same extraction-directory path and silently overwrite one another:
    two zip members whose names differ only by case, folded together by a
    case-insensitive host filesystem; a malformed archive declaring the same
    name twice; or two accepted spellings of one type (`readme.md` and
    `readme.markdown`) that only collide once BOTH are normalized to the
    same canonical extension. A silent overwrite here would mean whichever
    write happened second wins with no error and no count — the first
    member's content simply vanishes.

    `moving`, when given, is the file about to be placed at `target` (e.g.
    via `Path.rename`). If `target` already exists but IS `moving` itself —
    the same inode, reached when a case-insensitive filesystem folds two
    differently-cased spellings of one path together, such as renaming
    `REPORT.TXT` to `REPORT.txt` — that is not a collision at all, just the
    rename correcting the casing in place; without this check, `.exists()`
    alone would treat every case-only rename as colliding with itself.
    """
    if not target.exists():
        return target
    if moving is not None and target.samefile(moving):
        return target
    n = 0
    candidate = target
    while candidate.exists():
        n += 1
        candidate = target.with_name(f"{target.stem}__{n}{target.suffix}")
    return candidate


def _normalize_to_canonical_extension(
    paths: list[Path], canonical_ext: str, native: tuple[str, ...] = ()
) -> None:
    """Rename any path (in place, within the disposable extraction
    directory) whose suffix isn't exactly `canonical_ext` -- an alias
    (`.htm`), a non-canonical casing (`.PDF`, `.Markdown`), or both -- so the
    type's connector, which only globs its own single `DEFAULT_GLOBS`
    pattern, finds every accepted spelling. The archive itself is never
    touched; only the temp copy is renamed. Collisions from normalizing two
    different original names onto the same canonical spelling are resolved
    via `_disambiguate` rather than one silently overwriting the other.

    The reported chunk `source_key` reflects the canonical spelling after
    this, not necessarily the archive member's exact original spelling — an
    accepted, purely cosmetic tradeoff for reusing each connector's existing
    discovery unmodified.
    """
    for path in paths:
        target_ext = path.suffix.lower() if path.suffix.lower() in native else canonical_ext
        if path.suffix == target_ext:
            continue
        target = _disambiguate(path.with_suffix(target_ext), moving=path)
        path.rename(target)


_COPY_CHUNK_BYTES = 1024 * 1024


def _int_attr(obj: object, name: str) -> int:
    """Defensively read an optional non-negative int counter off a connector.

    Mirrors `Ingester._reported_int_attr`: a sub-connector reached through
    `CONNECTOR_REGISTRY` could, in principle, be a test double or a
    third-party registration that overwrote a built-in key, not one of the
    real connectors in this package. An absent or malformed attribute reads
    as 0 rather than raising or silently mis-comparing.
    """
    raw = getattr(obj, name, 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return raw


class ZipConnector:
    """Walks `path` for `.zip` files and yields one `SourceDocument` per
    supported file found inside each, extracted to a temp directory that is
    guaranteed removed (per archive) before this connector hands back that
    archive's documents. See the module docstring for the full safety
    contract."""

    def __init__(
        self,
        source_type: str,
        path: Path | str,
        glob: str = "**/*.zip",
        max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
        max_members: int = DEFAULT_MAX_MEMBERS,
        exclude_dependencies: bool = True,
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob
        self._max_uncompressed_bytes = max_uncompressed_bytes
        self._max_members = max_members
        self._exclude_dependencies = exclude_dependencies
        self.failed_files = 0
        self.skipped_files = 0

    def load(self) -> Iterable[SourceDocument]:
        # Reset per run — see the identical note in markdown.py/pdf.py/etc.:
        # a reused instance must not suppress pruning forever on the strength
        # of a failure from an earlier run.
        self.failed_files = 0
        self.skipped_files = 0
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Zip source '{self.source_type}': directory not found: {self._root}"
            )
        for archive_path in discover_files(self._root, self._glob):
            archive_key = str(archive_path.relative_to(self._root))
            # `_load_one_archive` is a plain function returning a list, not a
            # generator: extraction, reading, and temp-directory cleanup for
            # this archive are fully complete by the time it returns (or
            # raises), before any of its documents are yielded onward. See
            # the module docstring's "Cleanup guarantee".
            yield from self._load_one_archive(archive_path, archive_key)

    def _repair_member_names(self, infos: list[zipfile.ZipInfo], archive_key: str) -> None:
        """Repair mojibake member names IN PLACE, once, before anything else
        in `_load_one_archive`/`_extract_members`/`_load_extracted` reads
        `info.filename` — every downstream use (zip-slip target, noise and
        dependency-marker matching, disambiguation, nested-archive suffix
        check, extracted `source_key`) then sees the corrected name for
        free. See `_repair_filename_encoding` and the module docstring."""
        for info in infos:
            repaired = _repair_filename_encoding(info)
            if repaired != info.filename:
                logger.info(
                    "%s: repaired mojibake member name in '%s': %r -> %r",
                    self.source_type,
                    archive_key,
                    info.filename,
                    repaired,
                )
                info.filename = repaired

    def _load_one_archive(self, archive_path: Path, archive_key: str) -> list[SourceDocument]:
        try:
            zf = zipfile.ZipFile(archive_path)
        except (zipfile.BadZipFile, OSError) as e:
            # Could be genuinely corrupt, or a file mid-write / momentarily
            # locked — treat as transient, like every other connector's
            # "cannot open" case.
            logger.warning(
                "Zip source '%s': cannot open '%s': %s", self.source_type, archive_key, e
            )
            self.failed_files += 1
            return []

        with zf:
            infos = zf.infolist()
            self._repair_member_names(infos, archive_key)

            if len(infos) > self._max_members:
                logger.warning(
                    "%s: refusing '%s' — %d members exceeds the cap of %d",
                    self.source_type,
                    archive_key,
                    len(infos),
                    self._max_members,
                )
                self.skipped_files += 1
                return []

            declared_total = sum(info.file_size for info in infos)
            if declared_total > self._max_uncompressed_bytes:
                logger.warning(
                    "%s: refusing '%s' — declared uncompressed size %d bytes exceeds "
                    "the cap of %d",
                    self.source_type,
                    archive_key,
                    declared_total,
                    self._max_uncompressed_bytes,
                )
                self.skipped_files += 1
                return []

            # Built once per archive (empty, and never consulted, when the
            # filter is off) rather than per member — every candidate member
            # below needs the same lookup set. See `_dependency_marker_paths`.
            marker_paths = (
                _dependency_marker_paths(infos) if self._exclude_dependencies else frozenset()
            )

            extract_dir = Path(tempfile.mkdtemp(prefix="corpus-zip-")).resolve()
            try:
                bomb_free = self._extract_members(
                    zf, infos, extract_dir, archive_key, marker_paths
                )
                if not bomb_free:
                    return []
                return self._load_extracted(extract_dir, archive_key)
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)

    def _extract_members(
        self,
        zf: zipfile.ZipFile,
        infos: list[zipfile.ZipInfo],
        extract_dir: Path,
        archive_key: str,
        marker_paths: frozenset[str],
    ) -> bool:
        """Write every safe, supported member under `extract_dir`.

        Returns False only when the actual-bytes-written cap trips mid-stream
        — that refuses the WHOLE archive (declared sizes already passed, so
        this means the header lied). An individual zip-slip or nested-archive
        member is refused on its own and does not stop the rest of the
        archive; returns True in that case.
        """
        total_written = 0
        noise_ignored = 0
        dependency_ignored = 0
        for info in infos:
            if info.is_dir():
                # Nothing is written for a directory-only entry (we create
                # parent directories implicitly, below, from file targets),
                # so a directory entry poses no extraction risk even if its
                # name would otherwise fail the containment check.
                continue

            if _is_archive_noise(info.filename):
                # Checked first, before encryption/zip-slip/type matching:
                # archive packaging, not content, not extracted, not counted
                # in either counter. See `_is_archive_noise`'s docstring for
                # why this matters more than it looks like it should.
                noise_ignored += 1
                continue

            if self._exclude_dependencies and _is_dependency_noise(info.filename, marker_paths):
                # Same treatment as archive-packaging noise above, just an
                # overridable judgment call rather than an unconditional one
                # — see `_is_dependency_noise` and `SourceConfig.exclude_dependencies`.
                dependency_ignored += 1
                continue

            if info.flag_bits & 0x1:
                # Standard ZIP "file is encrypted" flag, checked per member
                # (not per archive): an archive with 619 plain files and one
                # encrypted one should still yield the 619, the same way a
                # zip-slip member is refused individually below rather than
                # discarding the rest of the archive. Never attempted: a
                # read-then-catch-password-error approach, or a password
                # prompt — this runs unattended.
                logger.info(
                    "%s: skipping encrypted member '%s' in '%s' — this runs "
                    "unattended and never prompts for a password",
                    self.source_type,
                    info.filename,
                    archive_key,
                )
                self.skipped_files += 1
                continue

            target = (extract_dir / info.filename).resolve()
            if not target.is_relative_to(extract_dir):
                logger.warning(
                    "%s: refusing member '%s' in '%s' — resolves outside the "
                    "extraction root (zip-slip)",
                    self.source_type,
                    info.filename,
                    archive_key,
                )
                self.skipped_files += 1
                continue

            if target.suffix.lower() == ".zip":
                logger.info(
                    "%s: skipping nested archive '%s' in '%s' — depth limit is %d",
                    self.source_type,
                    info.filename,
                    archive_key,
                    MAX_DEPTH,
                )
                self.skipped_files += 1
                continue

            # Two DIFFERENT member names can still land on the same real
            # file: a case-insensitive host filesystem (macOS/Windows
            # defaults) folds "notes.md" and "notes.MD" onto one inode even
            # though the zip's own central directory treats them as distinct
            # entries, and a malformed/adversarial archive can simply
            # declare the same name twice. Without this check, the second
            # member's `.open("wb")` would silently truncate and overwrite
            # the first's content -- no error, no count, content just gone.
            resolved_target = _disambiguate(target)
            if resolved_target != target:
                logger.info(
                    "%s: member '%s' in '%s' collided on-disk with an "
                    "earlier member (case-insensitive filesystem or a "
                    "duplicate name in the archive) — extracted as '%s' instead",
                    self.source_type,
                    info.filename,
                    archive_key,
                    resolved_target.name,
                )
            target = resolved_target

            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, target.open("wb") as dst:
                    while True:
                        block = src.read(_COPY_CHUNK_BYTES)
                        if not block:
                            break
                        total_written += len(block)
                        if total_written > self._max_uncompressed_bytes:
                            logger.warning(
                                "%s: refusing '%s' — actual extracted size exceeded "
                                "the %d byte cap during extraction (declared sizes "
                                "cannot be trusted)",
                                self.source_type,
                                archive_key,
                                self._max_uncompressed_bytes,
                            )
                            self.skipped_files += 1
                            return False
                        dst.write(block)
            except RuntimeError as e:
                # zipfile raises RuntimeError for a per-member password
                # requirement not caught by the flag_bits check above (e.g. a
                # non-standard encryption scheme). Same policy as that check:
                # skip this member, don't prompt.
                logger.info(
                    "%s: skipping encrypted member '%s' in '%s': %s",
                    self.source_type,
                    info.filename,
                    archive_key,
                    e,
                )
                self.skipped_files += 1
                continue
            except Exception as e:  # zipfile/zlib raise many types for corrupt/bad data
                logger.warning(
                    "%s: failed to extract member '%s' from '%s': %s",
                    self.source_type,
                    info.filename,
                    archive_key,
                    e,
                )
                self.failed_files += 1
                continue

        if noise_ignored:
            logger.info(
                "%s: ignored %d archive-packaging member(s) in '%s' "
                "(__MACOSX/AppleDouble, .DS_Store, Thumbs.db) — not documents, "
                "not counted in failed_files or skipped_files",
                self.source_type,
                noise_ignored,
                archive_key,
            )
        if dependency_ignored:
            logger.info(
                "%s: ignored %d dependency/build-output member(s) in '%s' "
                "(node_modules, site-packages, VCS/build-tool directories, "
                "minified/map leaf files, etc.) — not documents, not counted "
                "in failed_files or skipped_files; set exclude_dependencies = "
                "false on this source to index them anyway",
                self.source_type,
                dependency_ignored,
                archive_key,
            )
        return True

    def _load_extracted(self, extract_dir: Path, archive_key: str) -> list[SourceDocument]:
        """Run each already-registered file-type connector against the
        extracted tree and remap every yielded document's `source_key` to
        `<archive_key>::<inner_key>` — so two archives that each contain,
        say, `report.pdf` can never collide, and a search hit stays traceable
        back to exactly which archive it came from.

        Classification is a single walk of `extract_dir`, matching each
        file's extension case-insensitively against every accepted type
        (canonical spelling or alias — see `_extension_type_map`), rather
        than calling each connector's own single-glob discovery directly:
        a connector only knows its one `DEFAULT_GLOBS` pattern, which can't
        express "`.htm` or `.html`, any case" as one glob string. Matched
        files are renamed to their type's canonical extension inside the
        (disposable) extraction directory so each connector's existing
        discovery then finds them unmodified.

        Both this method's own `discover_files` call AND each sub-connector's
        internal one (inside `connector.load()`) run with
        `corpus.connectors.discovery`'s default directory exclusion
        suppressed — `extract_dir` has already been filtered by this
        connector's OWN noise/dependency logic during extraction (governed
        by `exclude_dependencies`), which is more nuanced than that generic
        default and has already made the complete decision about what
        belongs here. Without suppressing it, `exclude_dependencies=False`
        (an explicit request to index a vendored tree) would be silently
        defeated the moment a sub-connector re-globbed a `node_modules`-named
        path inside `extract_dir`. See `default_excludes_suppressed`'s
        docstring.
        """
        with default_excludes_suppressed():
            extension_type = _extension_type_map(DEFAULT_GLOBS)
            all_files = list(discover_files(extract_dir, "**/*"))

            by_type: dict[str, list[Path]] = {}
            for path in all_files:
                type_name = extension_type.get(path.suffix.lower())
                if type_name is not None:
                    by_type.setdefault(type_name, []).append(path)

            docs: list[SourceDocument] = []
            for type_name in _MEMBER_CONNECTOR_TYPES:
                matches = by_type.get(type_name)
                if not matches:
                    continue
                _normalize_to_canonical_extension(
                    matches,
                    Path(DEFAULT_GLOBS[type_name]).suffix,
                    _NATIVE_EXTENSIONS.get(type_name, ()),
                )

                sub_cfg = SourceConfig(
                    name=self.source_type, type=type_name, path=str(extract_dir)
                )
                try:
                    connector, _chunker = CONNECTOR_REGISTRY[type_name](sub_cfg)
                except ImportError as e:
                    # The type IS supported by corpus, but this environment
                    # doesn't have the optional extra installed. Every run will
                    # skip these same files until the extra is installed, so
                    # this is a permanent-for-now skip, not a transient failure.
                    logger.warning(
                        "%s: %d '%s' file(s) inside '%s' skipped — %s",
                        self.source_type,
                        len(matches),
                        type_name,
                        archive_key,
                        e,
                    )
                    self.skipped_files += len(matches)
                    continue

                for doc in connector.load():
                    docs.append(
                        doc.model_copy(update={"source_key": f"{archive_key}::{doc.source_key}"})
                    )
                self.failed_files += _int_attr(connector, "failed_files")
                self.skipped_files += _int_attr(connector, "skipped_files")

            matched = {p for paths in by_type.values() for p in paths}
            unclaimed = [p for p in all_files if p not in matched]
            if unclaimed:
                logger.info(
                    "%s: %d file(s) inside '%s' have no matching connector (unsupported "
                    "extension) — not indexed",
                    self.source_type,
                    len(unclaimed),
                    archive_key,
                )
                self.skipped_files += len(unclaimed)

            return docs
