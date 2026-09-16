"""Pre-publish checks that ask the REMOTE, because local git cannot see this.

WHAT HAPPENED. A commit message naming a private companion project was
force-pushed off this repository's branch and recorded as fixed. It was not. A
history rewrite makes an object UNREACHABLE, not absent: GitHub keeps
unreachable objects and serves them by SHA indefinitely, and it did so for two
months.

WHY SIX AUDITS MISSED IT. Local `git gc` deleted the object, so every tool
anyone reached for walks a graph the commit is no longer in:

    git cat-file -t <sha>        fatal: could not get object info
    git log --all -S'<name>'     0 commits
    git grep <name> HEAD         clean
    git log -p --all             clean

Each answer was true. None answered the question. Repeating them bought
confidence rather than coverage, because they all shared one blind spot, and
the only surface that could see the object was the remote API.

WHY THE NAMES ARE NOT IN THIS FILE. It is tracked, and this repository is
public. A denylist of private names here would publish the very strings it
exists to suppress -- the same reasoning as the pattern-based check in
`tests/test_repo_hygiene.py` and the `commit-msg` hook. They live in
`.git/private-name-patterns`, which is inside `.git/` and so cannot be
committed at all.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from corpus.verify import Coverage, DetectorBroken, self_check

Runner = Callable[[list[str]], tuple[int, str]]

PATTERN_FILE = "private-name-patterns"


def load_patterns(git_dir: Path | str) -> list[str]:
    """Private-name patterns, from the untracked file inside `.git/`.

    Returns [] when the file is absent -- a fresh clone has no denylist. The
    CALLER must treat that as an unknown rather than a pass, which is what
    `scan_commit_messages` enforces by refusing to run without patterns.
    """
    path = Path(git_dir) / PATTERN_FILE
    if not path.is_file():
        return []
    out: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out


@dataclass(frozen=True)
class MessageScan:
    coverage: Coverage
    matches: list[int] = field(default_factory=list)


def scan_commit_messages(
    messages: Sequence[str], *, patterns: Sequence[str]
) -> MessageScan:
    """Indices of messages matching any pattern.

    Refuses to run on an empty pattern list. A matcher that cannot fire would
    report every history clean forever, which is precisely the shape of defect
    this module exists for.
    """
    if not patterns:
        raise DetectorBroken(
            "commit-message scan: no private-name patterns loaded, so the "
            "matcher cannot fire -- this scan proves nothing. Create "
            f".git/{PATTERN_FILE} (one regex per line)."
        )
    combined = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    # Prove the matcher fires before believing anything it does not find.
    self_check(
        lambda text: bool(combined.search(text)),
        positive=patterns[0],
        label="commit-message matcher",
    )
    return MessageScan(
        coverage=Coverage(len(messages), "commit messages"),
        matches=[i for i, m in enumerate(messages) if combined.search(m)],
    )


@dataclass(frozen=True)
class OrphanScan:
    """Commits the remote still serves that no branch or tag reaches."""

    available: bool
    coverage: Coverage
    force_pushes: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    detail: str = ""


def _gh(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout


def orphaned_commits(repo: str, *, run: Runner = _gh) -> OrphanScan:
    """Walk what the remote still serves after each force-push.

    `run` is injected so this is testable without a network or a GitHub
    account -- the logic being tested is which questions get asked, not
    whether `gh` works.

    An unavailable remote returns `available=False` with vacuous coverage. It
    is an UNKNOWN, not a pass: "I could not look" and "I looked and it was
    clean" are the two facts this whole module exists to keep apart.
    """
    code, out = run(["api", f"repos/{repo}/events", "--paginate"])
    if code != 0:
        return OrphanScan(
            available=False,
            coverage=Coverage(0, "orphaned commits"),
            detail=f"could not reach the remote (exit {code}): {out.strip()[:120]}",
        )

    try:
        events = json.loads(out or "[]")
    except json.JSONDecodeError as exc:
        return OrphanScan(
            available=False,
            coverage=Coverage(0, "orphaned commits"),
            detail=f"unreadable response from the remote: {exc}",
        )

    before: list[str] = []
    for event in events if isinstance(events, list) else []:
        payload = (event or {}).get("payload") or {}
        if payload.get("forced") and payload.get("before"):
            before.append(str(payload["before"]))

    commits: list[str] = []
    for tip in before:
        code, body = run(["api", f"repos/{repo}/commits?sha={tip}&per_page=100"])
        if code != 0:
            continue
        try:
            rows = json.loads(body or "[]")
        except json.JSONDecodeError:
            continue
        commits.extend(str(r["sha"]) for r in rows if isinstance(r, dict) and "sha" in r)

    return OrphanScan(
        available=True,
        coverage=Coverage(len(commits), "orphaned commits"),
        force_pushes=before,
        commits=commits,
    )


def reachable_messages(repo_path: Path | str = ".") -> list[str]:
    """Every commit message reachable from HEAD. The local half of the check."""
    proc = subprocess.run(
        ["git", "-C", str(repo_path), "log", "--format=%B%x00", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return []
    return [m.strip() for m in proc.stdout.split("\0") if m.strip()]


def surfaces_to_check() -> Iterable[str]:
    """The surfaces a leak can survive on, enumerated so they are not rediscovered.

    Found ad hoc during the incident, at the worst possible time. Written down
    here because the expensive part was not checking them -- it was not knowing
    they existed.
    """
    return (
        "remote: unreachable objects after a force-push (only the API sees these)",
        "remote: commit messages on every reachable commit",
        "remote: release assets, repository description, topics, homepage",
        "remote: wiki, pages, discussions, issues",
        "remote: GitHub Actions run logs (public, retained ~90 days)",
        "account: gists, organizations, forks, other repositories by name",
        "published artifacts: PyPI sdists and wheels, and any generated CHANGELOG",
        "third parties: public event archives, source-code archives, code search",
    )


__all__ = [
    "MessageScan",
    "OrphanScan",
    "load_patterns",
    "orphaned_commits",
    "reachable_messages",
    "scan_commit_messages",
    "surfaces_to_check",
]
