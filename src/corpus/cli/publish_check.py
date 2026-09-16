"""corpus-publish-check: what to verify BEFORE a repository goes public.

    corpus-publish-check                  # this repo, local + remote
    corpus-publish-check --no-remote      # offline; reports the remote UNCHECKED
    corpus-publish-check --repo owner/name

The check that matters here cannot be done locally. A history rewrite makes a
commit UNREACHABLE, not absent; GitHub keeps unreachable objects and serves
them by SHA. Once local `git gc` removes the object, `git log --all`, `-S`,
`git grep` and `git log -p --all` are all structurally blind to it -- they
walk reachable objects, and it is no longer one. Six local audits of this
repository returned clean while such an object was live for two months.

So this command asks the remote, and it is careful about the difference
between "I looked and it was clean" and "I could not look". The second is
never reported as the first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from corpus.publish_check import (
    load_patterns,
    orphaned_commits,
    reachable_messages,
    scan_commit_messages,
    surfaces_to_check,
)
from corpus.verify import DetectorBroken


def _remote_slug(repo_path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return None
    url = proc.stdout.strip()
    for prefix in ("https://github.com/", "git@github.com:"):
        if url.startswith(prefix):
            return url[len(prefix):].removesuffix(".git")
    return None


def main_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="corpus-publish-check",
        description="Verify a repository is safe to make public, including "
                    "surfaces only the remote can see",
    )
    parser.add_argument("path", nargs="?", default=".", help="Repository to check")
    parser.add_argument("--repo", default=None, metavar="OWNER/NAME",
                        help="Remote slug (default: inferred from origin)")
    parser.add_argument("--no-remote", action="store_true",
                        help="Skip remote checks. Reported as UNCHECKED, not clean.")
    args = parser.parse_args(argv)

    repo_path = Path(args.path).expanduser().resolve()
    print(f"=== corpus-publish-check: {repo_path} ===\n")

    failures = 0

    # --- local: reachable commit messages ---------------------------------
    patterns = load_patterns(repo_path / ".git")
    messages = reachable_messages(repo_path)
    try:
        scan = scan_commit_messages(messages, patterns=patterns)
    except DetectorBroken as exc:
        print(f"commit messages   [ FAIL ] {exc}")
        failures += 1
    else:
        if scan.matches:
            print(
                f"commit messages   [ FAIL ] {len(scan.matches)} of "
                f"{scan.coverage.describe()} name a private project"
            )
            for i in scan.matches[:5]:
                # Deliberately NOT printing the matched text: it would put the
                # private string into terminal scrollback and CI logs.
                first = messages[i].splitlines()[0][:60]
                print(f"                  #{i}: {first}")
            failures += 1
        elif scan.coverage.vacuous:
            print(f"commit messages   [ FAIL ] {scan.coverage.describe()}")
            failures += 1
        else:
            print(f"commit messages   [  ok  ] {scan.coverage.describe()}, clean")

    # --- remote: the part no local command can do -------------------------
    if args.no_remote:
        print("orphaned objects  [NOT CHECKED] --no-remote was passed")
        print("                  A force-push leaves objects only the remote "
              "can see; this run did not look.")
    else:
        slug = args.repo or _remote_slug(repo_path)
        if not slug:
            print("orphaned objects  [NOT CHECKED] no GitHub origin found "
                  "(pass --repo OWNER/NAME)")
        else:
            orphans = orphaned_commits(slug)
            if not orphans.available:
                print(f"orphaned objects  [NOT CHECKED] {orphans.detail}")
            elif not orphans.force_pushes:
                # "None found" is NOT "none ever happened". The events API
                # retains roughly 90 days; the force-push that caused this
                # repository's two-month exposure is already outside it. A
                # bound left unstated is how a clean result misleads.
                print("orphaned objects  [ warn ] no force-push in the "
                      "retained event history")
                print("                  BOUNDED: the events API only covers "
                      "~90 days, so this cannot")
                print("                  see an older rewrite. For those, ask "
                      "Support or check known SHAs.")
            else:
                print(
                    f"orphaned objects  [ warn ] {orphans.coverage.describe()} "
                    f"still served after {len(orphans.force_pushes)} force-push(es)"
                )
                print("                  Check their MESSAGES; a rewrite does "
                      "not delete them.")

    print("\nSurfaces a leak survives on (check each before publishing):")
    for surface in surfaces_to_check():
        print(f"  - {surface}")

    print(
        f"\n{'PROBLEMS FOUND' if failures else 'OK'}: {failures} failing check(s)."
        "\n  NOT CHECKED is not a pass -- it is the state every audit of this "
        "repository was in\n  while an unreachable object was live on the remote."
    )
    return 1 if failures else 0


def main() -> int:
    return main_argv(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
