""""git could not run" was returned as "this history has no commits".

`reachable_messages` returns `[]` when `git log` exits non-zero -- not a
repo, a bad HEAD, git missing, a permission problem. An empty list is also
what a genuinely empty history returns, so the caller cannot tell them
apart.

The in-repo caller happens to be safe: it checks `scan.coverage.vacuous`
and fails. But this is an EXPORTED function in the module whose entire
thesis is keeping "I could not look" apart from "I looked and it was
clean", and the next caller gets no such protection. Raising makes the
distinction unlosable rather than conventional.
"""

from __future__ import annotations

import subprocess

import pytest


def test_a_directory_that_is_not_a_repo_raises(tmp_path):
    from corpus.publish_check import GitUnavailable, reachable_messages

    with pytest.raises(GitUnavailable):
        reachable_messages(tmp_path)


def test_an_empty_repo_is_not_confused_with_a_broken_one(tmp_path):
    """A repo with no commits has nothing to scan, and that IS a clean
    answer -- distinct from git failing."""
    from corpus.publish_check import reachable_messages

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert reachable_messages(tmp_path) == []


def test_a_real_history_still_returns_its_messages(tmp_path):
    from corpus.publish_check import reachable_messages

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "feat: a thing", "--no-verify"],
        cwd=tmp_path, check=True,
    )
    assert any("feat: a thing" in m for m in reachable_messages(tmp_path))
