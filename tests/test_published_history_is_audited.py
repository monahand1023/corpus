"""Is what is ALREADY on the remote clean? Nothing ever asked.

Two GitHub Support tickets in two days, two different holes, one root
cause: every guard here checks what is about to happen. `commit-msg` reads
the message being written. `pre-push` reads the commits being pushed.
`test_repo_hygiene.py` reads the working tree. All three are correct and
all three are blind to the same thing -- a commit that is ALREADY public.

  leak 1  a private ticket key, in a commit MESSAGE
          -> built message-scanning hooks, after the fact
  leak 2  two real archive filenames, in FILE CONTENT
          -> the message hooks had nothing to match, and the tree test
             went green the moment the file was fixed, which was the same
             commit whose diff published them

Both predated the guard that would have caught them, and nothing in the
repo could see that, because no check has ever looked backwards. Both were
found by a human grepping history by hand, months later.

This is the backwards-looking one. It scans EVERY commit reachable from
the pushed ref -- message and diff -- against the same untracked pattern
file the hooks use, and it is a test, so it runs unprompted on every
pytest invocation rather than when someone remembers.

WHY IT SKIPS RATHER THAN FAILS WITHOUT THE PATTERN FILE: the patterns are
private names and live in .git/, untracked by design, so CI genuinely does
not have them. A skip says "not checked here", which is the honest thing
and the distinction this codebase keeps having to make. The check that
matters runs on the machine that has the names -- the same machine that
does the pushing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )


def _pattern_file() -> Path:
    git_dir = _git("rev-parse", "--git-dir").stdout.strip()
    return (REPO_ROOT / git_dir) / "private-name-patterns"


def _active_patterns() -> list[str]:
    raw = _pattern_file().read_text(encoding="utf-8", errors="replace")
    return [
        line for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _active_pattern_file(tmp: Path) -> Path:
    """The patterns with comments and blanks stripped, as the hooks do.

    NOT the raw file. `grep -f` treats EVERY line as a regex, including the
    `#` comments -- so passing the file directly makes "# why this name is
    private" match any source line containing that text, and the audit
    reports a leak that is a code comment. Found immediately: the first run
    of this test flagged `# id=2 appears in both, ranks first`.

    A guard that cries wolf is a guard that gets switched off, and this one
    only has value if a red result is believed.
    """
    active = tmp / "active-patterns"
    active.write_text("\n".join(_active_patterns()) + "\n", encoding="utf-8")
    return active


def _published_ref() -> str | None:
    """What the remote actually has, not what is local.

    `origin/main`, deliberately: auditing local `main` would flag a commit
    that has not left the machine, which is what `pre-push` is for and is a
    different, fixable-in-place situation.
    """
    for ref in ("refs/remotes/origin/main", "refs/remotes/origin/HEAD"):
        if _git("rev-parse", "--verify", "--quiet", ref).returncode == 0:
            return ref
    return None


def test_the_pattern_file_still_compiles(tmp_path) -> None:
    """One bad line makes `grep -f` reject the WHOLE file, so every pattern
    stops matching at once and every history reads clean. Fail closed."""
    if not _pattern_file().is_file():
        pytest.skip("no pattern file on this machine (untracked by design)")
    patterns = _active_patterns()
    if not patterns:
        pytest.skip("pattern file has no active patterns")

    proc = subprocess.run(
        ["grep", "-qiEf", str(_active_pattern_file(tmp_path))],
        input="", capture_output=True, text=True,
    )
    assert proc.returncode < 2, (
        "the private-name pattern file is not a valid POSIX extended regex, "
        "so grep rejects all of it and NOTHING is being checked -- not the "
        "broken pattern and not the others. The names are not printed here; "
        "run:  printf '' | grep -qiEf \"$(git rev-parse --git-dir)/private-name-patterns\""
    )


def test_no_private_name_is_in_the_history_already_pushed(tmp_path) -> None:
    if not _pattern_file().is_file():
        pytest.skip("no pattern file on this machine (untracked by design)")
    patterns = _active_patterns()
    if not patterns:
        pytest.skip("pattern file has no active patterns")
    ref = _published_ref()
    if ref is None:
        pytest.skip("no origin/main to audit -- nothing has been pushed here")

    # Positive control. A scan that cannot match reports every history clean
    # forever, which is the failure this whole file exists to remove.
    active = _active_pattern_file(tmp_path)
    control = subprocess.run(
        ["grep", "-qiEf", str(active)],
        input="\n".join(patterns[:1]).replace("[ -]?", " "),
        capture_output=True, text=True,
    )
    assert control.returncode == 0, (
        "the matcher did not match a string built from a pattern itself, so "
        "a clean result here would mean nothing"
    )

    log = _git("log", "--format=%H%n%B", "-p", ref)
    # Only the INTEGER survives. pytest explains a failed assert by
    # rendering its operands -- `assert proc.returncode == 1` prints
    # `where 0 = CompletedProcess(..., stdout='...').returncode`, and that
    # stdout is the matching lines. The first version of this test leaked a
    # private string into its own failure output, which is a remarkable way
    # for a privacy check to fail. Nothing but ints crosses into an assert
    # from here on, and `-c` asks grep to count rather than print.
    log_rc = log.returncode
    history = log.stdout
    del log
    assert log_rc == 0, "could not read the pushed history, so it was not checked"

    proc = subprocess.run(
        ["grep", "-ciEf", str(active)],
        input=history, capture_output=True, text=True,
    )
    rc = proc.returncode
    hits = proc.stdout.strip()
    del proc, history
    assert rc < 2, "the scan ERRORED, so the pushed history was not checked at all"
    assert rc == 1, (
        f"{hits} matching line(s) found.\n"
        f"A private name is in the history already pushed to {ref}.\n\n"
        "The matching text is NOT printed here -- printing it would put the "
        "string into terminal scrollback and CI logs, which is the thing "
        "being prevented. To see it locally, where it already is:\n\n"
        "    P=\"$(git rev-parse --git-dir)/private-name-patterns\"\n"
        "    grep -v '^[[:space:]]*#' \"$P\" | grep -v '^[[:space:]]*$' > /tmp/p\n"
        "    git log -p origin/main | grep -niEf /tmp/p; rm /tmp/p\n\n"
        "(The comments are stripped first: `grep -f` treats every line as a\n"
        "regex, so the `#` lines match ordinary code comments.)\n\n"
        "This cannot be fixed by a later commit. The diff is public: the "
        "history has to be rewritten and force-pushed, and because GitHub "
        "keeps serving unreachable objects by SHA, GitHub Support has to "
        "purge them afterwards."
    )
