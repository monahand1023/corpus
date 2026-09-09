"""Tests for `corpus.survey.walk` — the streaming, symlink-safe directory
walk shared by every survey subcommand. All fixture trees are synthetic and
built under `tmp_path`, per the project's public-repo constraints."""

from __future__ import annotations

import os
from pathlib import Path

from corpus.survey.walk import DEFAULT_EXCLUDED_DIR_NAMES, WalkStats, walk_files


def _touch(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_yields_every_file_with_correct_size(tmp_path: Path) -> None:
    _touch(tmp_path, "a.txt", "hello")
    _touch(tmp_path, "sub/b.md", "hi there")

    found = {wf.rel_path: wf.size for wf in walk_files(tmp_path)}

    assert found == {"a.txt": 5, "sub/b.md": 8}


def test_default_excludes_prune_noise_dirs(tmp_path: Path) -> None:
    _touch(tmp_path, "real/notes.txt")
    _touch(tmp_path, "node_modules/pkg/index.js")
    _touch(tmp_path, ".git/HEAD")

    found = {wf.rel_path for wf in walk_files(tmp_path)}

    assert found == {"real/notes.txt"}


def test_no_default_excludes_flag_disables_pruning(tmp_path: Path) -> None:
    _touch(tmp_path, "node_modules/pkg/index.js")

    found = {wf.rel_path for wf in walk_files(tmp_path, use_default_excludes=False)}

    assert found == {"node_modules/pkg/index.js"}


def test_user_exclude_pattern_prunes_directory(tmp_path: Path) -> None:
    _touch(tmp_path, "keep/a.txt")
    _touch(tmp_path, "Caches/b.txt")

    found = {wf.rel_path for wf in walk_files(tmp_path, excludes=("Caches",))}

    assert found == {"keep/a.txt"}


def test_user_exclude_pattern_matches_file_glob(tmp_path: Path) -> None:
    _touch(tmp_path, "a.txt")
    _touch(tmp_path, "b.log")

    found = {wf.rel_path for wf in walk_files(tmp_path, excludes=("*.log",))}

    assert found == {"a.txt"}


def test_symlinked_file_is_not_followed_and_is_counted(tmp_path: Path) -> None:
    target = _touch(tmp_path, "real.txt", "content")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    stats = WalkStats()
    found = {wf.rel_path for wf in walk_files(tmp_path, stats=stats)}

    assert found == {"real.txt"}
    assert stats.file_symlinks_skipped == 1


def test_symlinked_directory_is_not_descended_into(tmp_path: Path) -> None:
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    _touch(real_dir, "inside.txt")
    (tmp_path / "link_dir").symlink_to(real_dir, target_is_directory=True)

    stats = WalkStats()
    found = {wf.rel_path for wf in walk_files(tmp_path, stats=stats)}

    assert found == {"real_dir/inside.txt"}
    assert stats.dir_symlinks_skipped == 1


def test_broken_symlink_is_counted_not_raised(tmp_path: Path) -> None:
    link = tmp_path / "broken.txt"
    link.symlink_to(tmp_path / "does_not_exist.txt")

    stats = WalkStats()
    found = list(walk_files(tmp_path, stats=stats))

    assert found == []
    assert stats.file_symlinks_skipped == 1


def test_unreadable_directory_is_counted_not_raised(tmp_path: Path) -> None:
    _touch(tmp_path, "visible/a.txt")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    _touch(blocked, "hidden.txt")
    os.chmod(blocked, 0o000)
    try:
        stats = WalkStats()
        found = {wf.rel_path for wf in walk_files(tmp_path, stats=stats)}
        # Either the directory's contents are unreachable (permission_errors
        # bumped) or the test is running as a user that ignores 0o000 (e.g.
        # root in some CI containers) and sees the file anyway — either way
        # this must not raise, and the always-readable file must appear.
        assert "visible/a.txt" in found
    finally:
        os.chmod(blocked, 0o755)


def test_photoslibrary_bundle_is_pruned_by_default(tmp_path: Path) -> None:
    _touch(tmp_path, "Photos Library.photoslibrary/database/photos.db")
    _touch(tmp_path, "real.txt")

    found = {wf.rel_path for wf in walk_files(tmp_path)}

    assert found == {"real.txt"}


def test_default_excluded_dir_names_is_lowercase() -> None:
    # classify_file/walk_files both lowercase before comparing; a mixed-case
    # entry here would silently never match.
    assert all(name == name.lower() for name in DEFAULT_EXCLUDED_DIR_NAMES)


def test_empty_directory_yields_nothing(tmp_path: Path) -> None:
    assert list(walk_files(tmp_path)) == []


def test_missing_root_yields_nothing_without_raising(tmp_path: Path) -> None:
    # Matches `corpus.connectors.discovery.discover_files`'s contract (no
    # exception on a missing root) — callers that want a clean CLI error
    # check `Path.is_dir()` themselves before walking, same as the CLI does.
    stats = WalkStats()
    found = list(walk_files(tmp_path / "nope", stats=stats))
    assert found == []


# ---------------------------------------------------------------------------
# dist/build/target corroboration — shared with `corpus.connectors.discovery`
# via `corpus.util.exclude`, so the plan (this module) and the real ingest
# can't drift apart on this judgment call. See `tests/test_discovery.py` for
# the identical scenarios exercised against `discover_files` directly.
# ---------------------------------------------------------------------------


def test_dist_without_manifest_is_not_pruned(tmp_path: Path) -> None:
    _touch(tmp_path, "dist/mailing-list-notes.txt")

    found = {wf.rel_path for wf in walk_files(tmp_path)}

    assert found == {"dist/mailing-list-notes.txt"}


def test_dist_with_corroborating_manifest_is_pruned(tmp_path: Path) -> None:
    _touch(tmp_path, "package.json", "{}")
    _touch(tmp_path, "dist/bundle.txt")
    _touch(tmp_path, "src/real.txt")

    found = {wf.rel_path for wf in walk_files(tmp_path)}

    assert found == {"src/real.txt", "package.json"}


def test_dist_with_ancestor_manifest_is_pruned(tmp_path: Path) -> None:
    _touch(tmp_path, "package.json", "{}")
    _touch(tmp_path, "packages/app/dist/bundle.txt")

    found = {wf.rel_path for wf in walk_files(tmp_path)}

    assert found == {"package.json"}


def test_bare_vendor_and_dist_dirlike_names_are_not_false_positives(tmp_path: Path) -> None:
    # A directory that merely CONTAINS a noise basename is not the same
    # directory — "distribution" isn't "dist", "my-node_modules-notes" isn't
    # "node_modules".
    _touch(tmp_path, "distribution/notes.txt")
    _touch(tmp_path, "my-node_modules-notes/plan.txt")

    found = {wf.rel_path for wf in walk_files(tmp_path)}

    assert found == {"distribution/notes.txt", "my-node_modules-notes/plan.txt"}
