"""Work out which connectors apply to a folder, so `corpus-ingest --path DIR`
can ingest whatever is there without a hand-written `[[sources]]` block.

Detection is deliberately shallow: for each registered connector type, glob the
folder with that type's default pattern and keep the type if anything matches.
No content sniffing, no magic bytes — the extension is the signal, exactly as it
is for a hand-written source.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from corpus.config import SourceConfig
from corpus.connectors.discovery import discover_files
from corpus.connectors.registry import DEFAULT_GLOBS


def normalize_source_name(name: str) -> str:
    """Map a folder name onto a valid source_type identifier.

    `SourceConfig.name` is constrained to `^[a-z][a-z0-9_]*$` because it is
    used as a filename component, a table value, and an MCP schema enum. A
    folder called "Field Notes - 2024" therefore becomes `field_notes_2024`.

    **Two distinct folders must not normalize to the same name**, and this
    function used to let them. It stripped every leading non-letter, so
    "2023 Taxes" and "2024 Taxes" both became `taxes`; and it fell back to a
    constant `"folder"` for anything with no ASCII letters, so every
    Japanese- or Chinese-named folder became `folder`.

    That is data loss, not a cosmetic clash. Chunk ids are derived from
    (source_type, source_key, kind, index), so two folders sharing a name AND
    a filename produce the same chunk id and the second ingest silently
    OVERWRITES the first — no deletion, no orphan sweep, nothing for the
    blast-radius guard or the yield-drop warning to notice. Both runs report
    one document and look perfectly healthy.

    So: a leading digit is prefixed rather than stripped, and a name with no
    usable ASCII gets a short digest of the original instead of a constant.
    """
    norm = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    norm = re.sub(r"_+", "_", norm).strip("_")
    if not norm or not re.match(r"^[a-z]", norm):
        # Digest the ORIGINAL, not the normalized form: "日本語" and "中文"
        # both normalize to empty, and only the original tells them apart.
        digest = hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:8]
        # A leading digit is illegal, so `s` prefixes it rather than the digit
        # being dropped — "2024_taxes" keeps its year as `s2024_taxes`.
        return f"s{norm}_{digest}" if norm else f"folder_{digest}"
    return norm


def detect_sources(path: Path | str) -> list[SourceConfig]:
    """Return one SourceConfig per connector type that has files under `path`.

    Source names are namespaced by folder — `field_notes_2024_pdf`, not `pdf`.
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
        if source_type == "transcripts":
            # The connector reads a sidecar DATABASE, not a folder: one source
            # per sidecar found, pointing at the file itself.
            for i, db in enumerate(discover_files(root, glob)):
                suffix = "" if i == 0 else f"_{i + 1}"
                detected.append(
                    SourceConfig(
                        name=f"{prefix}_transcripts{suffix}", type="transcripts", path=str(db)
                    )
                )
            continue
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

