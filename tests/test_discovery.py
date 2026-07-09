from __future__ import annotations

import os
from pathlib import Path


def test_discovery_finds_regular_files(tmp_path: Path) -> None:
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("a")
    (root / "sub").mkdir()
    (root / "sub" / "b.md").write_text("b")
    found = {p.name for p in discover_files(root, "**/*.md")}
    assert found == {"a.md", "b.md"}


def test_discovery_skips_symlink_escaping_root(tmp_path: Path) -> None:
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "notes"
    root.mkdir()
    (root / "real.md").write_text("hello")
    secret = tmp_path / "secret.md"
    secret.write_text("SECRET")
    os.symlink(secret, root / "link.md")  # symlink pointing outside root
    found = {p.name for p in discover_files(root, "**/*.md")}
    assert "real.md" in found
    assert "link.md" not in found


def test_discovery_rejects_dotdot_glob(tmp_path: Path) -> None:
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "notes"
    root.mkdir()
    (tmp_path / "outside.md").write_text("nope")
    (root / "in.md").write_text("yes")
    found = {p.name for p in discover_files(root, "../*.md")}
    assert "outside.md" not in found
