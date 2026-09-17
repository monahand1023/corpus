"""The privacy hooks, actually run.

`test_repo_hygiene.py` checks that these exist, are executable and resolve
their patterns from outside the worktree. Nothing ever RAN one. The gap
mattered, because a shell guard has a failure mode that reading it does not
reveal:

    if grep -qiE "$pat" "$MSG_FILE"; then ...

`grep` exits 0 on a match, 1 on no match, and **2 on an error** -- an invalid
regular expression among the patterns, for instance. An `if` treats 1 and 2
alike, so a broken pattern reads as "clean" and the commit goes through. In
the `-f PATTERNFILE` form the whole file fails at once, so one bad line
disables every pattern.

That is this session's recurring defect -- a check that cannot fire is
indistinguishable from a check that passed -- sitting inside the guard built
after a private name reached a public commit message.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parent.parent / ".githooks"
SECRET = "zzprivatenamezz"


def _repo(tmp_path: Path, patterns: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / ".git" / "private-name-patterns").write_text(patterns)
    return repo


def _commit_msg(repo: Path, message: str) -> subprocess.CompletedProcess[str]:
    msg = repo / "MSG"
    msg.write_text(message)
    return subprocess.run(
        [str(HOOKS / "commit-msg"), str(msg)],
        cwd=repo, capture_output=True, text=True,
    )


# --- commit-msg ---------------------------------------------------------------


def test_commit_msg_blocks_a_message_naming_a_private_project(tmp_path):
    """The positive control. Without this the two tests below prove nothing."""
    repo = _repo(tmp_path, f"{SECRET}\n")
    result = _commit_msg(repo, f"port the eval harness from {SECRET}")
    assert result.returncode == 1
    assert "BLOCKED" in result.stderr
    assert SECRET not in result.stderr, "the hook printed the string it suppresses"


def test_commit_msg_allows_an_ordinary_message(tmp_path):
    repo = _repo(tmp_path, f"{SECRET}\n")
    assert _commit_msg(repo, "fix(db): close the connection on a failed open").returncode == 0


def test_commit_msg_comments_and_blank_lines_are_ignored(tmp_path):
    repo = _repo(tmp_path, f"# a comment\n\n{SECRET}\n")
    assert _commit_msg(repo, "mentions " + SECRET).returncode == 1


def test_commit_msg_refuses_to_run_on_a_pattern_it_cannot_compile(tmp_path):
    """A typo'd regex made `grep` exit 2, which the `if` read as "no match".
    The pattern silently stopped checking -- with stderr redirected to
    /dev/null, with no output at all -- and commits naming the private
    project went through."""
    repo = _repo(tmp_path, f"{SECRET}[0-9\n")
    result = _commit_msg(repo, f"port the eval harness from {SECRET}42")
    assert result.returncode != 0, "a broken pattern let the commit through"
    assert "pattern" in (result.stderr + result.stdout).lower()


def test_commit_msg_one_broken_pattern_does_not_disable_the_others(tmp_path):
    """Fail closed as a whole: the operator must not be left believing the
    remaining patterns still cover them."""
    repo = _repo(tmp_path, f"good-[0-9\n{SECRET}\n")
    assert _commit_msg(repo, "an entirely ordinary message").returncode != 0


def test_commit_msg_without_a_pattern_file_says_so_and_does_not_block(tmp_path):
    """A fresh clone has no denylist. Blocking every commit would be wrong;
    saying nothing would be worse."""
    repo = tmp_path / "bare"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    result = _commit_msg(repo, "anything")
    assert result.returncode == 0
    assert "NOT being" in result.stderr


# --- pre-push -----------------------------------------------------------------


def _push_attempt(repo: Path, patterns: str, message: str) -> subprocess.CompletedProcess[str]:
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", message, "--no-verify"], cwd=repo, check=True
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    zeros = "0" * 40
    return subprocess.run(
        [str(HOOKS / "pre-push"), "origin", "https://example.invalid/x.git"],
        cwd=repo, capture_output=True, text=True,
        input=f"refs/heads/main {sha} refs/heads/main {zeros}\n",
    )


def test_pre_push_blocks_a_commit_whose_message_names_a_private_project(tmp_path):
    repo = _repo(tmp_path, f"{SECRET}\n")
    result = _push_attempt(repo, f"{SECRET}\n", f"ported from {SECRET}")
    assert result.returncode == 1
    assert "BLOCKED" in result.stderr
    assert SECRET not in result.stderr


def test_pre_push_allows_a_clean_history(tmp_path):
    repo = _repo(tmp_path, f"{SECRET}\n")
    result = _push_attempt(repo, f"{SECRET}\n", "fix(db): close the connection")
    assert result.returncode == 0


def test_pre_push_refuses_to_run_on_a_pattern_it_cannot_compile(tmp_path):
    """`grep -f` fails for the WHOLE file on one bad line, so every pattern
    stops matching at once -- and the push succeeded."""
    repo = _repo(tmp_path, f"{SECRET}[0-9\n")
    result = _push_attempt(repo, f"{SECRET}[0-9\n", f"ported from {SECRET}42")
    assert result.returncode != 0, "a broken pattern let the push through"
    assert "pattern" in (result.stderr + result.stdout).lower()


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell hooks")
def test_the_hooks_are_runnable_at_all():
    for name in ("commit-msg", "pre-push"):
        assert os.access(HOOKS / name, os.X_OK)


# --- pre-commit ---------------------------------------------------------------


def _staged_commit(tmp_path: Path, filename: str) -> subprocess.CompletedProcess[str]:
    repo = tmp_path / "pc"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    target = repo / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    subprocess.run(["git", "add", "-f", filename], cwd=repo, check=True)
    return subprocess.run(
        [str(HOOKS / "pre-commit")], cwd=repo, capture_output=True, text=True
    )


def test_pre_commit_blocks_a_staged_database(tmp_path):
    """The positive control for the hook that keeps real data out of a
    public repo."""
    assert _staged_commit(tmp_path, "data/corpus.db").returncode == 1


def test_pre_commit_blocks_a_staged_dotenv(tmp_path):
    assert _staged_commit(tmp_path, ".env").returncode == 1


def test_pre_commit_allows_ordinary_source(tmp_path):
    assert _staged_commit(tmp_path, "src/corpus/thing.py").returncode == 0


def test_pre_commit_allows_the_tracked_example_config(tmp_path):
    """corpus.toml.example is the template and must stay committable."""
    assert _staged_commit(tmp_path, "corpus.toml.example").returncode == 0


def test_pre_commit_blocks_a_root_config(tmp_path):
    assert _staged_commit(tmp_path, "corpus.toml").returncode == 1


# --- the positive controls themselves -----------------------------------------
#
# A canary that is never exercised is the very thing it guards against: a
# check nobody has seen fire. These simulate the condition it exists for --
# a `grep` that answers "no match" to everything -- and assert the hook
# refuses rather than approving.


def _blind_grep(tmp_path: Path) -> dict[str, str]:
    """A PATH where `grep` always reports no match, whatever it is asked."""
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "grep"
    shim.write_text("#!/bin/sh\nexit 1\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    return env


def test_pre_commit_refuses_when_its_matcher_cannot_match(tmp_path):
    repo = tmp_path / "pc"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "corpus.db").write_text("x")
    subprocess.run(["git", "add", "-f", "corpus.db"], cwd=repo, check=True)

    result = subprocess.run(
        [str(HOOKS / "pre-commit")], cwd=repo, capture_output=True, text=True,
        env=_blind_grep(tmp_path),
    )
    assert result.returncode == 1, "a blind matcher approved a staged database"
    assert "cannot be trusted" in result.stderr


def test_commit_msg_refuses_when_its_matcher_cannot_match(tmp_path):
    repo = _repo(tmp_path, f"{SECRET}\n")
    msg = repo / "MSG"
    msg.write_text(f"ported from {SECRET}")
    result = subprocess.run(
        [str(HOOKS / "commit-msg"), str(msg)], cwd=repo,
        capture_output=True, text=True, env=_blind_grep(tmp_path),
    )
    assert result.returncode == 1, "a blind matcher approved a leaking message"
    assert "cannot be trusted" in result.stderr


# --- content, not just messages -----------------------------------------------


def _push_attempt_with_content(
    repo: Path, filename: str, body: str
) -> subprocess.CompletedProcess[str]:
    """Commit a file whose CONTENT carries the private name, message clean."""
    (repo / filename).write_text(body)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test: add a fixture", "--no-verify"],
        cwd=repo,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    zeros = "0" * 40
    return subprocess.run(
        [str(HOOKS / "pre-push"), "origin", "https://example.invalid/x.git"],
        cwd=repo, capture_output=True, text=True,
        input=f"refs/heads/main {sha} refs/heads/main {zeros}\n",
    )


def test_pre_push_blocks_a_private_name_in_the_CONTENT_of_a_commit(tmp_path):
    """The hook read commit MESSAGES only, and the leak was in a file.

    This repo pushed two real filenames from a personal work archive inside a
    test fixture, under the clean message "test(contextual): use invented
    paths, not real archive filenames" -- the commit that REMOVED them. The
    message check had nothing to match, the working-tree check went green the
    moment they were deleted, and the strings are still in the public
    history, because a force-push makes an object unreachable, not absent.

    A private name in a diff is the same disclosure as one in a subject line.
    """
    repo = _repo(tmp_path, f"{SECRET}\n")
    result = _push_attempt_with_content(
        repo, "fixture.py", f'SOURCE = "Work Files/{SECRET}/deck.pptx"\n'
    )

    assert result.returncode == 1, (
        "a private name in file content was pushed with no complaint"
    )
    assert "BLOCKED" in result.stderr
    assert SECRET not in result.stderr, (
        "the hook printed the very string it exists to suppress"
    )


def test_pre_push_allows_content_with_no_private_name(tmp_path):
    """The other half: it must not block ordinary commits, or it gets
    bypassed with --no-verify and protects nothing."""
    repo = _repo(tmp_path, f"{SECRET}\n")
    result = _push_attempt_with_content(
        repo, "fixture.py", 'SOURCE = "Work Files/Acme Corp/deck.pptx"\n'
    )

    assert result.returncode == 0, result.stderr


# --- surfaces a content scan cannot see --------------------------------------


def _push_range(repo: Path, sha: str, ref: str = "refs/heads/main"):
    zeros = "0" * 40
    return subprocess.run(
        [str(HOOKS / "pre-push"), "origin", "https://example.invalid/x.git"],
        cwd=repo, capture_output=True, text=True,
        input=f"{ref} {sha} {ref} {zeros}\n",
    )


def _commit(repo: Path, relpath: str, body: bytes, message: str = "chore: add") -> str:
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", message, "--no-verify"], cwd=repo, check=True
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_pre_push_blocks_a_private_name_in_a_FILE_PATH(tmp_path):
    """The hook reads messages and diffs. A file NAMED after private material
    discloses it in the tree listing, where neither scan looks -- and a repo's
    file list is the first thing anyone browses."""
    repo = _repo(tmp_path, f"{SECRET}\n")
    sha = _commit(repo, f"tests/fixtures/{SECRET}-sample.txt", b"harmless text\n")
    result = _push_range(repo, sha)

    assert result.returncode == 1, "a private name in a path was pushed silently"
    assert "BLOCKED" in result.stderr
    assert SECRET not in result.stderr


def test_pre_push_blocks_a_private_name_in_the_BRANCH_being_pushed(tmp_path):
    """Branch names are published as refs and appear in the remote's UI."""
    repo = _repo(tmp_path, f"{SECRET}\n")
    sha = _commit(repo, "ok.txt", b"fine\n")
    result = _push_range(repo, sha, ref=f"refs/heads/{SECRET}-experiment")

    assert result.returncode == 1, "a private name in a ref was pushed silently"
    assert "BLOCKED" in result.stderr
    assert SECRET not in result.stderr


def test_pre_push_refuses_a_binary_file_it_cannot_scan(tmp_path):
    """`git log -p` prints "Binary files differ" and nothing else, so every
    text guard reports a binary blob clean no matter what is inside it.

    No binary has ever been committed to this repo -- verified across the
    whole history -- which is the only reason the diff scan can be trusted
    retroactively. Refusing them keeps that property true rather than
    assuming it."""
    repo = _repo(tmp_path, f"{SECRET}\n")
    sha = _commit(repo, "assets/logo.png", b"\x89PNG\r\n\x1a\n\x00\x00binary\x00data")
    result = _push_range(repo, sha)

    assert result.returncode == 1, "an unscannable blob was pushed"
    assert "BLOCKED" in result.stderr
    assert "binary" in result.stderr.lower()
    assert "assets/logo.png" in result.stderr, "the file was not named"


def test_pre_push_allows_ordinary_text_on_an_ordinary_branch(tmp_path):
    """The clean path must stay clean, or the hook gets bypassed and guards
    nothing."""
    repo = _repo(tmp_path, f"{SECRET}\n")
    sha = _commit(repo, "src/module.py", b"def f():\n    return 1\n")
    result = _push_range(repo, sha, ref="refs/heads/feature-work")

    assert result.returncode == 0, result.stderr


def test_pre_push_blocks_when_it_cannot_list_the_commits(tmp_path):
    """`git rev-list "$RANGE"` with stderr discarded and status unread.

    When `$remote_sha` is not a local object -- a stale remote-tracking ref,
    someone else's force-push, a gc'd object, a shallow clone -- rev-list
    exits 128 with empty stdout. The loop body never runs, FOUND stays 0,
    and the hook falls through to `exit 0` printing NOTHING. Zero commits
    scanned is indistinguishable from zero commits matching.

    The same file already fails CLOSED on grep exit >= 2. This is the one
    place it did not.
    """
    repo = _repo(tmp_path, f"{SECRET}\n")
    sha = _commit(repo, "ok.txt", b"fine\n")
    missing = "0" * 39 + "1"  # a well-formed sha that is not in this repo
    result = subprocess.run(
        [str(HOOKS / "pre-push"), "origin", "https://example.invalid/x.git"],
        cwd=repo, capture_output=True, text=True,
        input=f"refs/heads/main {sha} refs/heads/main {missing}\n",
    )

    assert result.returncode == 1, (
        "the hook could not list what it was about to push and allowed it"
    )
    assert "BLOCKED" in result.stderr


def test_pre_push_blocks_when_the_pattern_file_is_unreadable(tmp_path):
    """Three outcomes collapsed into one silent pass.

    A failed `mktemp`, an unreadable pattern file, and a pattern file that
    legitimately holds only comments all reached the same `exit 0`. Only the
    last is correct. The MISSING-file case at least prints a NOTE; the
    unreadable one printed nothing at all.
    """
    repo = _repo(tmp_path, f"{SECRET}\n")
    sha = _commit(repo, "ok.txt", b"fine\n")
    patterns = repo / ".git" / "private-name-patterns"
    patterns.chmod(0o000)
    try:
        result = subprocess.run(
            [str(HOOKS / "pre-push"), "origin", "https://example.invalid/x.git"],
            cwd=repo, capture_output=True, text=True,
            input=f"refs/heads/main {sha} refs/heads/main {'0' * 40}\n",
        )
    finally:
        patterns.chmod(0o600)

    assert result.returncode == 1, (
        "an unreadable pattern file read as 'nothing to check' and the push "
        "was allowed"
    )
    assert "BLOCKED" in result.stderr


def test_a_comments_only_pattern_file_still_allows_the_push(tmp_path):
    """The one case that legitimately exits 0. It must stay distinguishable
    from the two failures above, or the fix is just a stricter hook that
    people disable."""
    repo = _repo(tmp_path, "# only a comment\n\n")
    sha = _commit(repo, "ok.txt", b"fine\n")
    result = subprocess.run(
        [str(HOOKS / "pre-push"), "origin", "https://example.invalid/x.git"],
        cwd=repo, capture_output=True, text=True,
        input=f"refs/heads/main {sha} refs/heads/main {'0' * 40}\n",
    )

    assert result.returncode == 0, result.stderr
