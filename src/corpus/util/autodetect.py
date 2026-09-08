"""Work out which connectors apply to a folder, so `corpus-ingest --path DIR`
can ingest whatever is there without a hand-written `[[sources]]` block.

Detection is deliberately shallow: for each registered connector type, glob the
folder with that type's default pattern and keep the type if anything matches.
No content sniffing, no magic bytes — the extension is the signal, exactly as it
is for a hand-written source.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from corpus.config import SourceConfig
from corpus.connectors.discovery import discover_files
from corpus.connectors.registry import DEFAULT_GLOBS


def normalize_source_name(name: str) -> str:
    """Map a folder name onto a valid source_type identifier.

    `SourceConfig.name` is constrained to `^[a-z][a-z0-9_]*$` because it is used
    as a filename component, a table value, and an MCP schema enum. A folder
    called "Documents - Local" therefore becomes `documents_local`.
    """
    norm = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    norm = re.sub(r"^[^a-z]+", "", norm)
    norm = re.sub(r"_+", "_", norm).strip("_")
    return norm or "folder"


def detect_sources(path: Path | str) -> list[SourceConfig]:
    """Return one SourceConfig per connector type that has files under `path`.

    Source names are namespaced by folder — `documents_local_pdf`, not `pdf`.
    This is not cosmetic. `delete_orphans` is scoped by `source_type` alone with
    no per-document granularity, so two folders sharing the name `pdf` in one
    database would make ingesting the second delete every chunk from the first.
    Namespacing keeps folders independent.

    Types with no matching files are omitted rather than yielding empty sources:
    an empty source contributes nothing but still claims a slice of the
    retriever's per-source candidate budget.
    """
    root = Path(os.path.expanduser(str(path))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--path: directory not found: {root}")

    prefix = normalize_source_name(root.name)
    detected: list[SourceConfig] = []
    for source_type, glob in DEFAULT_GLOBS.items():
        if next(discover_files(root, glob), None) is None:
            continue
        detected.append(
            SourceConfig(
                name=f"{prefix}_{source_type}",
                type=source_type,
                path=str(root),
            )
        )
    return detected
