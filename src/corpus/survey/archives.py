"""Archive inspection: what's really inside a set of `.zip` files, without
ever extracting a single byte to disk.

`zipfile.ZipFile(path)` parses only the archive's central directory — member
names, sizes, flags — which is exactly what this needs (member count, noise
breakdown, indexable-by-type breakdown) and never decompresses or writes
anything. This is what makes "without extracting to a permanent location"
trivial here: there is no extraction at all, temporary or otherwise, unlike
the zip *connector*, which must actually extract to read document content.

Classification reuses the zip connector's own noise judgment calls
(`_is_archive_noise`, `_is_dependency_noise`, `_dependency_marker_paths`,
`_extension_type_map`) rather than re-deciding them here — the whole point
of this subcommand is to predict what `corpus-ingest` would actually do with
these archives as a `zip` source, so drift between "what the survey counted"
and "what the connector would do" would make the tool actively misleading.
Nested archives (a `.zip` inside a `.zip`) are reported as their own bucket
rather than folded into "gap", matching the connector's own refusal at
`MAX_DEPTH = 1`.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corpus.connectors.registry import DEFAULT_GLOBS
from corpus.connectors.zip import (
    _dependency_marker_paths,
    _extension_type_map,
    _is_archive_noise,
    _is_dependency_noise,
    _path_components,
)
from corpus.survey.walk import WalkStats, walk_files


@dataclass
class ArchiveInfo:
    path: str  # root-relative, POSIX
    readable: bool = True
    error: str | None = None
    total_members: int = 0
    total_declared_bytes: int = 0
    indexable_by_type: dict[str, int] = field(default_factory=dict)
    packaging_noise: int = 0
    dependency_noise: int = 0
    encrypted: int = 0
    nested_archive_refused: int = 0
    gap: int = 0  # matched no connector by extension

    @property
    def indexable_total(self) -> int:
        return sum(self.indexable_by_type.values())

    @property
    def noise_ratio(self) -> float:
        """Fraction of members that are dependency/build-output or OS
        packaging noise — the number that distinguishes an archive worth
        ingesting from one that's mostly junk. Excludes `gap` (unsupported
        but not noise) and `encrypted`/`nested_archive_refused` (refused for
        reasons unrelated to being noise) from the denominator's numerator
        by design; `total_members` is still the full denominator."""
        if self.total_members == 0:
            return 0.0
        return (self.packaging_noise + self.dependency_noise) / self.total_members


@dataclass
class ArchiveSurveyResult:
    root: str
    excludes: tuple[str, ...]
    use_default_excludes: bool
    archives: list[ArchiveInfo]
    walk_stats: WalkStats

    @property
    def unreadable(self) -> list[ArchiveInfo]:
        return [a for a in self.archives if not a.readable]

    @property
    def totals(self) -> dict[str, Any]:
        readable = [a for a in self.archives if a.readable]
        total_members = sum(a.total_members for a in readable)
        packaging = sum(a.packaging_noise for a in readable)
        dependency = sum(a.dependency_noise for a in readable)
        return {
            "archive_count": len(self.archives),
            "unreadable_count": len(self.archives) - len(readable),
            "total_members": total_members,
            "total_declared_bytes": sum(a.total_declared_bytes for a in readable),
            "indexable_total": sum(a.indexable_total for a in readable),
            "packaging_noise": packaging,
            "dependency_noise": dependency,
            "encrypted": sum(a.encrypted for a in readable),
            "nested_archive_refused": sum(a.nested_archive_refused for a in readable),
            "gap": sum(a.gap for a in readable),
            "overall_noise_ratio": ((packaging + dependency) / total_members)
            if total_members
            else 0.0,
        }


def _inspect_one_archive(path: Path, rel_path: str) -> ArchiveInfo:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        return ArchiveInfo(path=rel_path, readable=False, error=str(e))

    with zf:
        try:
            infos = zf.infolist()
        except (zipfile.BadZipFile, OSError, EOFError) as e:
            return ArchiveInfo(path=rel_path, readable=False, error=str(e))

        extension_type = _extension_type_map(DEFAULT_GLOBS)
        marker_paths = _dependency_marker_paths(infos)

        info_result = ArchiveInfo(path=rel_path)
        for zi in infos:
            if zi.is_dir():
                continue
            info_result.total_members += 1
            info_result.total_declared_bytes += zi.file_size

            if _is_archive_noise(zi.filename):
                info_result.packaging_noise += 1
                continue
            if _is_dependency_noise(zi.filename, marker_paths):
                info_result.dependency_noise += 1
                continue
            if zi.flag_bits & 0x1:
                info_result.encrypted += 1
                continue

            parts = _path_components(zi.filename)
            basename = parts[-1] if parts else zi.filename
            ext = Path(basename).suffix.lower()
            if ext == ".zip":
                info_result.nested_archive_refused += 1
                continue

            conn_type = extension_type.get(ext)
            if conn_type is None:
                info_result.gap += 1
            else:
                info_result.indexable_by_type[conn_type] = (
                    info_result.indexable_by_type.get(conn_type, 0) + 1
                )

        return info_result


def run_archive_survey(
    root: Path,
    excludes: tuple[str, ...] = (),
    use_default_excludes: bool = True,
) -> ArchiveSurveyResult:
    stats = WalkStats()
    archives: list[ArchiveInfo] = []
    for wf in walk_files(root, excludes, use_default_excludes, stats=stats):
        if wf.path.suffix.lower() != ".zip":
            continue
        archives.append(_inspect_one_archive(wf.path, wf.rel_path))

    archives.sort(key=lambda a: (-a.total_members, a.path))
    return ArchiveSurveyResult(
        root=str(root),
        excludes=excludes,
        use_default_excludes=use_default_excludes,
        archives=archives,
        walk_stats=stats,
    )
