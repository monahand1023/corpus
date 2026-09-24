from __future__ import annotations

import os
from pathlib import Path

from corpus.connectors.discovery import default_excludes_suppressed, discover_files


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


# ---------------------------------------------------------------------------
# Default directory exclusion — see `corpus.util.exclude`. `corpus-index`
# prints "Noise (excluded from the plan, not ingested)"; these tests are what
# makes "not ingested" actually true, since every connector calls
# `discover_files` to find its files.
# ---------------------------------------------------------------------------


def _touch(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_discovery_excludes_noise_dir_by_default_mixed_tree(tmp_path: Path) -> None:
    """A type with matches both inside and outside a noise directory yields
    ONLY the file outside it — the exact scenario from the bug report: a
    real `docs/real.md` alongside a `node_modules/pkg/README.md`."""
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "project"
    _touch(root, "docs/real.md")
    _touch(root, "node_modules/pkg/readme.md")

    found = {str(p.relative_to(root)) for p in discover_files(root, "**/*.md")}

    assert found == {"docs/real.md"}


def test_discovery_purely_noise_tree_yields_nothing(tmp_path: Path) -> None:
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "project"
    _touch(root, "node_modules/a/readme.md")
    _touch(root, ".git/COMMIT_EDITMSG.md")  # contrived, but a .git subpath either way

    found = list(discover_files(root, "**/*.md"))

    assert found == []


def test_discovery_use_default_excludes_false_yields_both(tmp_path: Path) -> None:

    root = tmp_path / "project"
    _touch(root, "docs/real.md")
    _touch(root, "node_modules/pkg/readme.md")

    found = {
        str(p.relative_to(root))
        for p in _without_default_excludes(root)
    }

    assert found == {"docs/real.md", "node_modules/pkg/readme.md"}


def test_discovery_dir_merely_containing_noise_name_is_not_excluded(tmp_path: Path) -> None:
    """A directory whose name merely CONTAINS a noise basename (rather than
    matching it exactly) must not be pruned — `my-node_modules-notes/` is a
    real folder someone could plausibly have, not a dependency tree."""
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "project"
    _touch(root, "my-node_modules-notes/plan.md")
    _touch(root, "distribution/notes.md")  # contains "dist" but isn't "dist"

    found = {str(p.relative_to(root)) for p in discover_files(root, "**/*.md")}

    assert found == {"my-node_modules-notes/plan.md", "distribution/notes.md"}


def test_discovery_dist_without_manifest_is_not_excluded(tmp_path: Path) -> None:
    """`dist` is an ordinary English word too — with no corroborating
    ecosystem manifest anywhere in the tree, it must be treated as a real
    folder, not build output."""
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "project"
    _touch(root, "dist/mailing-list-notes.md")

    found = {str(p.relative_to(root)) for p in discover_files(root, "**/*.md")}

    assert found == {"dist/mailing-list-notes.md"}


def test_discovery_dist_with_manifest_is_excluded(tmp_path: Path) -> None:
    """The same `dist/` folder, but this time `package.json` sits next to it
    — now it's corroborated as npm build output and is excluded, the same
    judgment call `corpus.connectors.zip` already makes for archive members."""
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "project"
    _touch(root, "package.json", "{}")
    _touch(root, "dist/bundle.md")
    _touch(root, "src/real.md")

    found = {str(p.relative_to(root)) for p in discover_files(root, "**/*.md")}

    assert found == {"src/real.md"}


def test_discovery_dist_with_ancestor_manifest_is_excluded(tmp_path: Path) -> None:
    """Monorepo-style layout: the manifest sits at the project root, several
    directories above the flagged `dist/` — still corroborated."""
    from corpus.connectors.discovery import discover_files

    root = tmp_path / "project"
    _touch(root, "package.json", "{}")
    _touch(root, "packages/app/dist/bundle.md")

    found = list(discover_files(root, "**/*.md"))

    assert found == []


def _without_default_excludes(root):
    with default_excludes_suppressed():
        return list(discover_files(root, "**/*.md"))
