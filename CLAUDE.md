# corpus

corpus is the generic RAG **engine** (the class). Separate per-domain repos
elsewhere on this machine are **instances** of it: they install corpus
editable, add their own config/connectors, and hold their own data.

## HARD RULE: this repository is PUBLIC. No private data, ever.

No exceptions, for any reason, including "just a test" or "it's only in a
commit message".

Never put anything from a private instance into this repo — not in files,
tests, fixtures, docs, CHANGELOG, commit messages, branch or tag names, or
PR/issue text. That includes:

- the **names** of private instance repos, or of the organisations whose data
  they hold
- document content, titles, subjects, source keys, file paths, ticket keys,
  people's names, hostnames
- eval queries, golden sets, scores, chunk counts, or any per-source stats
  measured on a private archive
- anything about the owner's personal reasons for this work

Use neutral placeholders instead (`TICKET-123`, `example.com`, `Alice`).
Engineering lessons may be described generically, where they explain code.

**How it is enforced — keep these working, never bypass them:**

- `.githooks/` (pre-commit, commit-msg, pre-push; install with
  `scripts/install-hooks.sh`). The pre-push hook reads its denylist from
  `.git/private-name-patterns`, which lives inside `.git/` so the denylist
  itself is never published. If it is missing, pushes are blocked on purpose.
- `tests/test_repo_hygiene.py` — no database files, no real `corpus.toml`,
  no private ticket-key shapes, no names embedded in the hooks. Its checks are
  pattern-based on purpose: a literal denylist here would itself be the leak.
- Never use `--no-verify`. Never open GitHub PRs for this repo (PR refs cannot
  be deleted); work on `main` or local branches.

A rewrite + force-push does NOT remove a leaked commit from GitHub — it stays
fetchable by SHA until GitHub Support purges it. Prevention is the only cheap
fix.

Instance-specific notes (which repos, where they live) belong in the
gitignored `CLAUDE.local.md`, never in this file.
