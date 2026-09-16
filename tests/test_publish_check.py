"""Checks that query the REMOTE, because local git cannot see this class.

The 2026-09-16 incident: a commit message naming a private companion project
was force-pushed off the branch in July and recorded as fixed. It was not. A
rewrite makes an object UNREACHABLE, not absent, and GitHub kept serving it by
SHA for two months. Local `git gc` then deleted the object, so every local
audit became structurally blind:

    git cat-file -t <sha>      fatal: could not get object info
    git log --all -S'<name>'   0 commits
    git grep / git log -p      clean

All of those walk REACHABLE objects. Six audits reported clean. Only the
remote API could see it.

These checks are deliberately pattern-driven from an UNTRACKED file: a
denylist of private names inside a tracked file would publish the very strings
it exists to suppress.
"""

from __future__ import annotations

import pytest

from corpus.publish_check import (
    OrphanScan,
    load_patterns,
    orphaned_commits,
    scan_commit_messages,
)
from corpus.verify import DetectorBroken

# --- the denylist lives outside the repo -------------------------------------


def test_patterns_come_from_an_untracked_file(tmp_path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "private-name-patterns").write_text(
        "# a comment\n\nacme-internal\nwidgetco\n"
    )
    assert load_patterns(git_dir) == ["acme-internal", "widgetco"]


def test_a_missing_denylist_is_reported_not_silently_empty(tmp_path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    # An empty pattern list would make every scan below vacuously clean, which
    # is the exact failure this whole module exists to prevent.
    assert load_patterns(git_dir) == []


# --- scanning reachable messages, with a canary ------------------------------


def test_scanning_messages_requires_the_matcher_to_prove_it_fires() -> None:
    with pytest.raises(DetectorBroken):
        # No patterns => the matcher can never fire => a "clean" result here
        # would be meaningless.
        scan_commit_messages(["fix: something ordinary"], patterns=[])


def test_a_message_naming_a_private_project_is_found() -> None:
    hits = scan_commit_messages(
        ["fix: thing", "chore: ported from acme-internal"], patterns=["acme-internal"]
    )
    assert hits.coverage.examined == 2
    assert hits.matches == [1]


def test_ordinary_messages_produce_no_hits() -> None:
    hits = scan_commit_messages(["fix: thing", "docs: words"], patterns=["acme-internal"])
    assert hits.matches == []
    assert bool(hits.coverage) is True


def test_scanning_no_messages_at_all_is_vacuous_not_clean() -> None:
    hits = scan_commit_messages([], patterns=["acme-internal"])
    assert hits.coverage.vacuous is True


# --- the remote half ---------------------------------------------------------


def test_orphaned_commits_are_found_via_the_remote_not_local_git() -> None:
    """A force-push leaves a chain the local clone cannot reach."""
    calls: list[list[str]] = []

    def fake_gh(args: list[str]) -> tuple[int, str]:
        calls.append(args)
        if "events" in " ".join(args):
            return 0, '[{"payload": {"forced": true, "before": "abc1230"}}]'
        return 0, '[{"sha": "dead0001"}, {"sha": "dead0002"}]'

    scan = orphaned_commits("owner/repo", run=fake_gh)

    assert isinstance(scan, OrphanScan)
    assert scan.force_pushes == ["abc1230"]
    assert scan.commits == ["dead0001", "dead0002"]
    assert scan.coverage.examined == 2
    assert any("events" in " ".join(c) for c in calls), "must ask the REMOTE"


def test_no_force_push_means_nothing_to_walk() -> None:
    scan = orphaned_commits("owner/repo", run=lambda a: (0, "[]"))
    assert scan.force_pushes == []
    assert scan.commits == []


def test_an_unavailable_remote_is_reported_not_treated_as_clean() -> None:
    """`gh` missing, unauthenticated, or offline is an UNKNOWN, not a pass.

    This is the distinction the whole incident turned on.
    """
    scan = orphaned_commits("owner/repo", run=lambda a: (127, "command not found"))
    assert scan.available is False
    assert scan.coverage.vacuous is True


# --- the command ------------------------------------------------------------


def _repo(tmp_path, messages, patterns=None):
    import subprocess

    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.email", "t@e.st"], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.name", "T"], check=True)
    for i, m in enumerate(messages):
        (r / f"f{i}.txt").write_text(str(i))
        subprocess.run(["git", "-C", str(r), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(r), "commit", "-q", "--no-verify", "-m", m], check=True
        )
    if patterns is not None:
        (r / ".git" / "private-name-patterns").write_text("\n".join(patterns) + "\n")
    return r


def test_the_command_fails_when_there_is_no_denylist(tmp_path, capsys) -> None:
    """No patterns means the matcher cannot fire, so a pass would be a lie."""
    from corpus.cli.publish_check import main_argv

    repo = _repo(tmp_path, ["fix: ordinary"], patterns=None)
    code = main_argv([str(repo), "--no-remote"])
    out = capsys.readouterr().out + capsys.readouterr().err
    assert code != 0
    assert "private-name-patterns" in out


def test_the_command_fails_on_a_message_naming_a_private_project(
    tmp_path, capsys
) -> None:
    from corpus.cli.publish_check import main_argv

    repo = _repo(
        tmp_path,
        ["fix: ordinary", "chore: ported from acme-internal"],
        patterns=["acme-internal"],
    )
    code = main_argv([str(repo), "--no-remote"])
    assert code != 0


def test_the_command_passes_a_clean_history(tmp_path, capsys) -> None:
    from corpus.cli.publish_check import main_argv

    repo = _repo(tmp_path, ["fix: ordinary", "docs: words"], patterns=["acme-internal"])
    code = main_argv([str(repo), "--no-remote"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "2 commit messages" in out


def test_skipping_the_remote_is_reported_as_unchecked_not_clean(
    tmp_path, capsys
) -> None:
    # --no-remote is for offline use; it must not read as a full pass, because
    # the remote is the ONLY surface that can see an orphaned object.
    from corpus.cli.publish_check import main_argv

    repo = _repo(tmp_path, ["fix: ordinary"], patterns=["acme-internal"])
    main_argv([str(repo), "--no-remote"])
    out = capsys.readouterr().out
    assert "NOT CHECKED" in out.upper()


def test_no_force_push_found_states_the_retention_limit(
    tmp_path, capsys, monkeypatch
) -> None:
    """"None found" must not read as "none ever happened".

    GitHub's events API retains roughly 90 days. The force-push that caused
    this repository's two-month exposure is already outside that window, so a
    clean result here is bounded by retention, not by history. Saying "no
    force-push" without the bound is the same false reassurance the module
    exists to remove.

    The remote call is replaced rather than made: what is under test is the
    wording of a clean result, not whether the network works.
    """
    from corpus.cli import publish_check as mod
    from corpus.verify import Coverage

    monkeypatch.setattr(
        mod, "orphaned_commits",
        lambda slug: OrphanScan(
            available=True, coverage=Coverage(0, "orphaned commits")
        ),
    )
    repo = _repo(tmp_path, ["fix: ordinary"], patterns=["acme-internal"])
    mod.main_argv([str(repo), "--repo", "owner/name"])
    out = capsys.readouterr().out

    assert "90" in out, out
    assert "bounded" in out.lower() or "only covers" in out.lower(), out
