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
    standard ZIP "file is encrypted" flag) on every member before any
    extraction is attempted for that archive — never by attempting to read a
    member and catching the resulting `RuntimeError`, and never by prompting
    for a password, since this runs unattended. (A member encrypted with a
    scheme that doesn't set the standard bit would still surface as a
    `RuntimeError` from `zipfile` when opened; that path is also handled,
    per-member, the same way — skip, don't prompt.)
  - **Nested archives.** A member whose name ends in `.zip` is refused, not
    extracted and not recursed into — see `MAX_DEPTH` below for why depth is
    fixed at 1 rather than configurable.

Failure-counter policy for this connector specifically:

  - `skipped_files` (permanent, does not block orphan pruning): a zip-slip
    refusal or a nested archive (counted per member); a file with no
    matching connector, or one whose type IS supported but the optional
    extra for it isn't installed here (counted per file); and a whole-archive
    refusal — encrypted, too many members, or over either uncompressed-size
    cap — counted as **1** (the archive, not its member count, since none of
    it was processed).
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
from corpus.connectors.discovery import discover_files
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
# `CONNECTOR_REGISTRY` (see `_load_extracted`). Deliberately not "every key in
# DEFAULT_GLOBS" so this list can't accidentally pick up a future `zip` (or
# other non-file-type) entry and recurse.
_MEMBER_CONNECTOR_TYPES = ("markdown", "text", "pdf", "html", "docx", "xlsx", "rtf")

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
    ):
        self.source_type = source_type
        self._root = Path(os.path.expanduser(str(path))).resolve()
        self._glob = glob
        self._max_uncompressed_bytes = max_uncompressed_bytes
        self._max_members = max_members
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

            if any(info.flag_bits & 0x1 for info in infos):
                logger.info(
                    "%s: skipping encrypted archive '%s' — this runs unattended and "
                    "never prompts for a password",
                    self.source_type,
                    archive_key,
                )
                self.skipped_files += 1
                return []

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

            extract_dir = Path(tempfile.mkdtemp(prefix="corpus-zip-")).resolve()
            try:
                bomb_free = self._extract_members(zf, infos, extract_dir, archive_key)
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
    ) -> bool:
        """Write every safe, supported member under `extract_dir`.

        Returns False only when the actual-bytes-written cap trips mid-stream
        — that refuses the WHOLE archive (declared sizes already passed, so
        this means the header lied). An individual zip-slip or nested-archive
        member is refused on its own and does not stop the rest of the
        archive; returns True in that case.
        """
        total_written = 0
        for info in infos:
            if info.is_dir():
                # Nothing is written for a directory-only entry (we create
                # parent directories implicitly, below, from file targets),
                # so a directory entry poses no extraction risk even if its
                # name would otherwise fail the containment check.
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
                # non-standard encryption scheme). Same policy as archive-level
                # encryption: skip, don't prompt.
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
        return True

    def _load_extracted(self, extract_dir: Path, archive_key: str) -> list[SourceDocument]:
        """Run each already-registered file-type connector against the
        extracted tree and remap every yielded document's `source_key` to
        `<archive_key>::<inner_key>` — so two archives that each contain,
        say, `report.pdf` can never collide, and a search hit stays traceable
        back to exactly which archive it came from."""
        from corpus.connectors.registry import CONNECTOR_REGISTRY, DEFAULT_GLOBS

        claimed: set[Path] = set()
        docs: list[SourceDocument] = []
        for type_name in _MEMBER_CONNECTOR_TYPES:
            matches = list(discover_files(extract_dir, DEFAULT_GLOBS[type_name]))
            if not matches:
                continue
            claimed.update(matches)

            sub_cfg = SourceConfig(name=self.source_type, type=type_name, path=str(extract_dir))
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

        unclaimed = [p for p in extract_dir.rglob("*") if p.is_file() and p not in claimed]
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
