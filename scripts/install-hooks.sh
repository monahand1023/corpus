#!/bin/sh
# Bootstrap this checkout's git hooks. core.hooksPath lives in .git/config,
# which is per-checkout and never committed -- a fresh `git clone` of this
# repo has NO hooks wired up until this script (or the equivalent manual
# `git config core.hooksPath .githooks`) has been run in it. Safe to re-run.
#
# This is a convenience for THIS checkout, not a guarantee: it does nothing
# for anyone else's clone, a CI runner, or a checkout where this script was
# never run. Layer 2 (the pytest guard in tests/test_repo_hygiene.py) is
# the layer that actually runs unprompted, everywhere, forever -- this hook
# is a second, local, opt-in line of defense on top of it.
set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# Verify before configuring: if a hook is missing or not executable, leave
# core.hooksPath untouched rather than pointing git at a directory that
# won't actually run anything.
#
# EVERY hook, not just pre-commit. Setting core.hooksPath enables the whole
# directory, so checking one and reporting "hooks installed" covered two
# that had never been looked at -- including pre-push, the private-name
# guard, whose failure mode is a string reaching a public remote where a
# force-push cannot undo it. A hook without its executable bit is skipped
# by git in silence.
for hook in .githooks/commit-msg .githooks/pre-commit .githooks/pre-push; do
  if [ ! -x "$hook" ]; then
    echo "FAILED: $hook is missing or not executable — core.hooksPath left unchanged" >&2
    exit 1
  fi
  echo "$hook is executable"
done

git config core.hooksPath .githooks

configured="$(git config core.hooksPath)"
if [ "$configured" != ".githooks" ]; then
  echo "FAILED: core.hooksPath is '$configured', expected '.githooks'" >&2
  exit 1
fi
echo "core.hooksPath = $configured"

echo "corpus hooks installed:"
echo "  pre-commit  blocks committing database files, a root corpus.toml and"
echo "              .env, even via 'git add -f'"
echo "  commit-msg  blocks a private name in a commit message"
echo "  pre-push    scans outgoing commit messages AND diffs, ref names, and"
echo "              refuses binaries it cannot read"
echo "Re-run after every fresh clone -- core.hooksPath does not survive one."
