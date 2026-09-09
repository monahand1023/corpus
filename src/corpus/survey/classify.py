"""Classify a filename as indexable / a known-noise artifact / a gap.

Deliberately reuses the zip connector's own noise judgment calls
(`_ARCHIVE_NOISE_BASENAMES`, `_MINIFIED_LEAF_SUFFIXES`) instead of
re-deciding "is `.DS_Store` noise" a second time in a second place — those
constants describe real-world OS/tooling artifacts that show up identically
whether they're sitting loose in a directory or inside a zip archive, and
importing them means a future edit to the zip connector's judgment can't
silently drift out of sync with what this tool reports. See
`corpus/connectors/zip.py`'s module docstring for the reasoning behind each.

The indexable/not split is derived from `corpus.connectors.registry.DEFAULT_GLOBS`
— the same table `corpus-ingest --path` uses to autodetect connectors — plus
the zip connector's extension-alias table, for the same no-drift reason: a
new connector registration (say, `pptx` or `csv`) is picked up here
automatically, with no edit required in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from corpus.connectors.registry import DEFAULT_GLOBS
from corpus.connectors.zip import (
    _ARCHIVE_NOISE_BASENAMES,
    _EXTENSION_ALIASES,
    _MINIFIED_LEAF_SUFFIXES,
)

Category = Literal["indexable", "noise", "gap"]

# Compiled/build/cache artifacts that are never documents regardless of which
# directory they sit in. Distinct from `_MINIFIED_LEAF_SUFFIXES` (JS/CSS
# build output) and `_ARCHIVE_NOISE_BASENAMES` (OS packaging) — this is the
# generic "binary tooling byproduct" bucket, kept short and specific on
# purpose (see zip.py's own comment on why these lists stay narrow: a false
# positive here silently drops something that might be a real document).
_COMPILED_ARTIFACT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".o",
        ".obj",
        ".class",
        ".so",
        ".dylib",
        ".dll",
        ".a",
        ".lock",
    }
)


def indexable_extension_map() -> dict[str, str]:
    """Extension (lowercase, leading dot) -> connector type name.

    Includes `.zip` itself (a lone archive at the top level of a tree is
    fully indexable via the zip connector) — unlike
    `corpus.connectors.zip._extension_type_map`, which deliberately excludes
    "zip" because *members inside an already-open archive* are never
    recursed into. Both derive from the same `DEFAULT_GLOBS` + alias source,
    so they can't disagree about anything other than that one, deliberate
    top-level-vs-nested distinction.
    """
    mapping = {Path(glob).suffix.lower(): conn_type for conn_type, glob in DEFAULT_GLOBS.items()}
    mapping.update(_EXTENSION_ALIASES)
    return mapping


@dataclass(frozen=True)
class Classification:
    category: Category
    bucket: str  # display key: extension, or a basename/pattern for noise
    detail: str  # connector name (indexable) or a short reason (noise/gap)


def classify_file(name: str, indexable: dict[str, str] | None = None) -> Classification:
    """Classify one filename (basename only — no path context needed).

    Checked in this order: OS/archive packaging artifacts and minified/build
    leaves are recognized by basename/suffix pattern regardless of
    extension, then a generic compiled-artifact extension list, then the
    indexable map, else it's a gap — present in the tree, not ignorable, and
    corpus has no connector for it today.
    """
    indexable = indexable if indexable is not None else indexable_extension_map()
    lower = name.lower()

    if lower in _ARCHIVE_NOISE_BASENAMES:
        return Classification("noise", lower, "OS packaging artifact")
    if lower.startswith("._"):
        return Classification("noise", "._* (AppleDouble)", "OS packaging artifact")
    if lower.endswith(_MINIFIED_LEAF_SUFFIXES):
        for suf in _MINIFIED_LEAF_SUFFIXES:
            if lower.endswith(suf):
                return Classification("noise", suf, "minified/build artifact")

    ext = Path(name).suffix.lower()
    if ext in _COMPILED_ARTIFACT_EXTENSIONS:
        return Classification("noise", ext, "compiled/build artifact")

    conn_type = indexable.get(ext)
    if conn_type is not None:
        return Classification("indexable", ext or "(no extension)", conn_type)

    return Classification("gap", ext or "(no extension)", "no connector")
