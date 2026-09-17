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
    # Force-push tips whose commit listing the remote would not give us.
    # NOT the same as a tip with no orphaned commits: this is "I could not
    # look", and a caller that folds it into the clean case reproduces the
    # very defect this module was written for.
    unreadable: list[str] = field(default_factory=list)
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
    unreadable: list[str] = []
    for tip in before:
        code, body = run(["api", f"repos/{repo}/commits?sha={tip}&per_page=100"])
        if code != 0:
            # A rate limit, a permissions error, a network blip. This walk
            # skipped it silently until a test asked what happens then --
            # the force-push was still counted as examined, so a rewrite
            # nobody could read reported the same as a rewrite with nothing
            # in it.
            unreadable.append(tip)
            continue
        try:
            rows = json.loads(body or "[]")
        except json.JSONDecodeError:
            unreadable.append(tip)
            continue
        commits.extend(str(r["sha"]) for r in rows if isinstance(r, dict) and "sha" in r)

    return OrphanScan(
        available=True,
        coverage=Coverage(len(commits), "orphaned commits"),
        force_pushes=before,
        commits=commits,
        unreadable=unreadable,
    )


@dataclass(frozen=True)
class SurfaceScan:
    """One surface examined, with what was found and whether it could be read."""

    available: bool
    coverage: Coverage
    hits: list[str] = field(default_factory=list)
    detail: str = ""


def _matcher(patterns: Sequence[str], label: str) -> re.Pattern[str]:
    """A compiled matcher that has PROVED it can fire."""
    if not patterns:
        raise DetectorBroken(
            f"{label}: no private-name patterns loaded, so the matcher cannot "
            f"fire -- this scan proves nothing. Create .git/{PATTERN_FILE}."
        )
    combined = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    self_check(
        lambda text: bool(combined.search(text)),
        positive=patterns[0],
        label=label,
    )
    return combined


def scan_actions_logs(
    repo: str, *, patterns: Sequence[str], run: Runner = _gh, limit: int = 50
) -> SurfaceScan:
    """Scan CI run logs, which are public on a public repo and retained ~90 days.

    Checked by hand during the incident that motivated this module, which is
    the problem: the expensive part was not checking it but not knowing it was
    a surface at all.
    """
    matcher = _matcher(patterns, "actions-log matcher")
    code, out = run(
        ["run", "list", "--repo", repo, "--limit", str(limit),
         "--json", "databaseId", "--jq", ".[].databaseId"]
    )
    if code != 0:
        return SurfaceScan(
            available=False,
            coverage=Coverage(0, "run logs"),
            detail=f"could not list runs (exit {code})",
        )

    ids = [line.strip() for line in (out or "").splitlines() if line.strip()]
    hits: list[str] = []
    examined = 0
    for run_id in ids:
        code, body = run(["run", "view", run_id, "--repo", repo, "--log"])
        if code != 0:
            continue
        examined += 1
        if matcher.search(body or ""):
            hits.append(run_id)
    return SurfaceScan(
        available=True, coverage=Coverage(examined, "run logs"), hits=hits
    )


def _fetch_pypi_files(project: str) -> dict[str, bytes]:  # pragma: no cover - network
    import urllib.request

    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{project}/json", timeout=60
    ) as resp:
        meta = json.loads(resp.read())
    out: dict[str, bytes] = {}
    for files in meta.get("releases", {}).values():
        for entry in files:
            with urllib.request.urlopen(entry["url"], timeout=120) as artifact:
                out[entry["filename"]] = artifact.read()
    return out


def scan_published_artifacts(
    project: str,
    *,
    patterns: Sequence[str],
    fetch: Callable[[str], dict[str, bytes]] = _fetch_pypi_files,
) -> SurfaceScan:
    """Scan everything already published to an index.

    A commit MESSAGE cannot reach an sdist, but a generated CHANGELOG can, and
    a published artifact is the one surface no amount of repository cleanup
    can retract.
    """
    matcher = _matcher(patterns, "artifact matcher")
    try:
        files = fetch(project)
    except Exception as exc:
        return SurfaceScan(
            available=False,
            coverage=Coverage(0, "published files"),
            detail=f"could not fetch: {type(exc).__name__}: {exc}",
        )
    hits = [
        name
        for name, blob in files.items()
        if matcher.search(blob.decode("utf-8", errors="replace"))
    ]
    return SurfaceScan(
        available=True, coverage=Coverage(len(files), "published files"), hits=hits
    )


class GitUnavailable(RuntimeError):
    """git could not be asked. NOT the same as a clean history."""


def reachable_messages(repo_path: Path | str = ".") -> list[str]:
    """Every commit message reachable from HEAD. The local half of the check."""
    # THREE outcomes, not two. `git log HEAD` fails both when this is not a
    # repository AND when it is a repository whose HEAD is unborn -- a fresh
    # `git init` with no commits. The first is "I could not look"; the second
    # is a real, clean answer of "nothing to scan". Asked separately so they
    # cannot collapse into each other.
    is_repo = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--git-dir"],
        capture_output=True, text=True, check=False,
    )
    if is_repo.returncode != 0:
        raise GitUnavailable(
            f"not a git repository, or git could not run, at {repo_path}: "
            f"{is_repo.stderr.strip() or 'no stderr'}"
        )
    has_head = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--verify", "--quiet", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if has_head.returncode != 0:
        return []  # a repository with no commits: nothing to scan, and clean

    proc = subprocess.run(
        ["git", "-C", str(repo_path), "log", "--format=%B%x00", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        # RAISE, do not return []. An empty list is what a genuinely empty
        # history returns, so returning it here made "git could not run" --
        # not a repo, a bad HEAD, git missing, a permission problem --
        # indistinguishable from "there is nothing to scan".
        #
        # This module exists to keep "I could not look" apart from "I looked
        # and it was clean". The one in-repo caller happened to guard against
        # it; an exported function should not depend on that.
        raise GitUnavailable(
            f"git log failed in {repo_path} (exit {proc.returncode}): "
            f"{proc.stderr.strip() or 'no stderr'}"
        )
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
    "GitUnavailable",
    "MessageScan",
    "OrphanScan",
    "SurfaceScan",
    "load_patterns",
    "orphaned_commits",
    "reachable_messages",
    "scan_actions_logs",
    "scan_commit_messages",
    "scan_published_artifacts",
    "surfaces_to_check",
]
