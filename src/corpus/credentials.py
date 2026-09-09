"""Explicit, predictable `.env` resolution for CLI entry points and the MCP
server.

The problem this exists to solve: `load_dotenv()` called with no arguments
resolves via `find_dotenv()`, which walks upward from the *calling frame's
file* — i.e. from wherever this library happens to be installed — not from
the consumer's own working directory or config. Whether a consumer's own
`.env` was honoured therefore depended on invocation style (`import
corpus.cli.query` vs. running the `corpus-query` console script), which is
not something a user can reason about. Worse, a naive fallback to a path
inside corpus's own source tree turns a *public* library checkout into a
shared secret store for every private consumer.

Precedence, most to least specific. An environment variable that is already
set is never overwritten by anything found here — an operator's real
shell/CI environment always wins over any `.env` file:

  1. An environment variable already set in the process. Untouched.
  2. A `.env` file next to the config file passed via `--config`, if one
     exists. This is the common real-world case: an archive repo keeps its
     `corpus.toml` and `.env` together, and the CLI is often run from
     elsewhere.
  3. A `.env` file found by walking up from the current working directory
     (`find_dotenv(usecwd=True)`).
  4. Nothing found. Callers must fail clearly when a required variable turns
     out to still be unset; `describe_search` builds the "where corpus
     looked" half of that message.

Call `resolve_dotenv` once per process, as early as practical in an entry
point's `main()` — right after argument parsing, since `--config` isn't
known before that.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class MissingCredentialError(RuntimeError):
    """Raised when none of the accepted environment variables are set after
    `.env` resolution has run."""


def _config_sibling_env(config_path: str | Path | None) -> Path | None:
    """The `.env` next to `config_path`, if `config_path` is given and such a
    file actually exists there."""
    if config_path is None:
        return None
    candidate = Path(config_path).expanduser().resolve().parent / ".env"
    return candidate if candidate.is_file() else None


def resolve_dotenv(config_path: str | Path | None = None) -> list[Path]:
    """Load `.env` file(s) into the environment, per the precedence above.

    Returns the paths actually loaded (0, 1, or 2). Safe to call more than
    once per process: `load_dotenv`'s `override=False` means a variable
    already present — whether set by the real environment or by an earlier
    call here — is never clobbered by a later `.env` file.
    """
    loaded: list[Path] = []

    sibling = _config_sibling_env(config_path)
    if sibling is not None:
        load_dotenv(dotenv_path=sibling, override=False)
        loaded.append(sibling)

    cwd_env = find_dotenv(usecwd=True)
    if cwd_env:
        cwd_path = Path(cwd_env)
        if cwd_path not in loaded:
            load_dotenv(dotenv_path=cwd_path, override=False)
            loaded.append(cwd_path)

    return loaded


def describe_search(config_path: str | Path | None = None) -> str:
    """Human-readable summary of where `resolve_dotenv` looked (or would
    look) for a `.env`, for building an actionable "variable X is missing"
    error message."""
    parts: list[str] = []

    sibling = _config_sibling_env(config_path)
    if sibling is not None:
        parts.append(f"beside --config ({sibling})")
    elif config_path is not None:
        parent = Path(config_path).expanduser().resolve().parent
        parts.append(f"beside --config (no .env in {parent})")

    cwd_env = find_dotenv(usecwd=True)
    parts.append(f"current directory ({cwd_env})" if cwd_env else "current directory (no .env found)")

    return "; ".join(parts)


def require_env(*names: str, config_path: str | Path | None = None) -> str:
    """Resolve `.env` file(s), then return the value of the first set
    variable among `names`.

    Raises `MissingCredentialError` naming every variable checked and where
    corpus looked for a `.env`, if none of them are set.
    """
    resolve_dotenv(config_path)
    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    checked = " or ".join(names)
    raise MissingCredentialError(
        f"{checked} not set. corpus looked for a .env: {describe_search(config_path)}. "
        "Set the variable in your shell environment, or add it to one of those files."
    )
