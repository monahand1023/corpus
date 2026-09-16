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


def test_a_force_push_whose_commits_cannot_be_listed_is_not_reported_as_walked() -> None:
    """Two `continue`s in the walk skipped a force-push whose commit listing
    the API refused (rate limit, permissions) or returned unparseably -- and
    the scan still claimed to have walked it.

    That is the exact distinction this module exists for, inside the module
    itself: "I could not look" reported as "I looked and it was clean".
    """
    def fake_gh(args: list[str]) -> tuple[int, str]:
        joined = " ".join(args)
        if "events" in joined:
            return 0, ('[{"payload": {"forced": true, "before": "aaa1111"}},'
                       ' {"payload": {"forced": true, "before": "bbb2222"}}]')
        if "aaa1111" in joined:
            return 0, '[{"sha": "dead0001"}]'
        return 1, "HTTP 403: rate limit exceeded"

    scan = orphaned_commits("owner/repo", run=fake_gh)

    assert scan.force_pushes == ["aaa1111", "bbb2222"]
    assert scan.unreadable == ["bbb2222"], "a skipped force-push was not reported"
    assert scan.commits == ["dead0001"]


def test_an_unreadable_commit_listing_is_not_silently_dropped() -> None:
    def fake_gh(args: list[str]) -> tuple[int, str]:
        if "events" in " ".join(args):
            return 0, '[{"payload": {"forced": true, "before": "aaa1111"}}]'
        return 0, "<html>an error page, not JSON</html>"

    scan = orphaned_commits("owner/repo", run=fake_gh)
    assert scan.unreadable == ["aaa1111"]


def test_a_fully_walked_scan_reports_nothing_unreadable() -> None:
    """The negative control: `unreadable` must not be a field that is always
    populated, or the check above would pass on any implementation."""
    def fake_gh(args: list[str]) -> tuple[int, str]:
        if "events" in " ".join(args):
            return 0, '[{"payload": {"forced": true, "before": "aaa1111"}}]'
        return 0, '[{"sha": "dead0001"}]'

    assert orphaned_commits("owner/repo", run=fake_gh).unreadable == []


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


def test_the_command_fails_when_a_force_push_could_not_be_walked(
    tmp_path, capsys, monkeypatch
) -> None:
    """NOT CHECKED is not a pass. A rewrite whose commits the remote would
    not list must count as a failure, or the exit code says "safe to
    publish" about a surface nobody read."""
    import corpus.cli.publish_check as mod
    from corpus.publish_check import OrphanScan
    from corpus.verify import Coverage

    repo = _repo(tmp_path, ["fix: something ordinary"], patterns=["zzsecretzz"])
    monkeypatch.setattr(
        mod, "orphaned_commits",
        lambda slug, **kw: OrphanScan(
            available=True,
            coverage=Coverage(0, "orphaned commits"),
            force_pushes=["aaa1111", "bbb2222"],
            unreadable=["bbb2222"],
        ),
    )
    code = mod.main_argv([str(repo), "--repo", "owner/name"])
    out = capsys.readouterr().out
    assert "NOT CHECKED" in out or "FAIL" in out
    assert "bbb2222" in out, "the operator was not told which rewrite went unread"
    assert code != 0


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


# --- surfaces that were checked by hand during the incident ------------------


def test_actions_logs_are_scanned_for_private_names() -> None:
    """CI logs are public on a public repo and retained ~90 days.

    Checked by hand during the incident; automated here so it is not
    rediscovered under pressure next time.
    """
    from corpus.publish_check import scan_actions_logs

    def fake_gh(args: list[str]) -> tuple[int, str]:
        if args[:2] == ["run", "list"]:
            return 0, "111\n222\n"
        if "111" in args:
            return 0, "ordinary build output"
        return 0, "cloning from acme-internal ..."

    scan = scan_actions_logs("owner/repo", patterns=["acme-internal"], run=fake_gh)

    assert scan.coverage.examined == 2
    assert scan.hits == ["222"]


def test_actions_logs_with_no_runs_is_vacuous_not_clean() -> None:
    from corpus.publish_check import scan_actions_logs

    scan = scan_actions_logs("owner/repo", patterns=["x"], run=lambda a: (0, ""))
    assert scan.coverage.vacuous is True


def test_an_unavailable_actions_api_is_reported_not_treated_as_clean() -> None:
    from corpus.publish_check import scan_actions_logs

    scan = scan_actions_logs("o/r", patterns=["x"], run=lambda a: (127, "no gh"))
    assert scan.available is False
    assert scan.coverage.vacuous is True


def test_published_artifacts_are_scanned() -> None:
    """A commit message cannot reach PyPI, but a generated CHANGELOG can."""
    from corpus.publish_check import scan_published_artifacts

    files = {
        "corpus_rag-0.1.0.tar.gz": b"ordinary source",
        "corpus_rag-0.2.0.tar.gz": b"CHANGELOG: ported from acme-internal",
    }
    scan = scan_published_artifacts(
        "corpus-rag", patterns=["acme-internal"],
        fetch=lambda project: files,
    )

    assert scan.coverage.examined == 2
    assert scan.hits == ["corpus_rag-0.2.0.tar.gz"]


def test_artifact_scan_requires_patterns_like_every_other_matcher() -> None:
    from corpus.publish_check import scan_published_artifacts

    with pytest.raises(DetectorBroken):
        scan_published_artifacts("p", patterns=[], fetch=lambda p: {"a": b"x"})
