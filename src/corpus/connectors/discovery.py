"""Safe file discovery for connectors.

Connectors point at a user-configured directory and glob for files. Without
containment checks, a symlink inside that directory (e.g. `notes.md ->
~/.ssh/id_rsa`) or a `..` in the glob would let ingestion read files outside
the configured root — which then surface via search results to the LLM. This
helper enforces: no symlinks, and every yielded path resolves inside `root`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_files(root: Path, glob: str) -> Iterator[Path]:
    """Yield regular files under `root` matching `glob`, excluding symlinks and
    any path that resolves outside `root`."""
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
        yield path
