"""corpus-init: interactive setup wizard.

Walks the user through 5 questions and writes corpus.toml + .env. Refuses
to overwrite existing files unless --force is passed.

Usage:
  corpus-init                    # interactive
  corpus-init --force            # overwrite existing corpus.toml / .env
  corpus-init --quiet ...        # non-interactive (uses defaults; for CI/tests)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# Built-in source types we know how to wire up. Keep in sync with
# connectors/registry.py.
KNOWN_SOURCE_TYPES = ["markdown", "text", "pdf", "html"]


class _AbortWizard(Exception):
    """Raised when stdin closes (EOF) so the wizard aborts instead of looping."""


class _ProviderInfo(TypedDict):
    default_model: str
    default_dim: int
    env_var: str
    signup_url: str
    extra_install: str | None


# Providers we know how to dispatch via the embedder factory.
KNOWN_PROVIDERS: dict[str, _ProviderInfo] = {
    "voyage": {
        "default_model": "voyage-3-large",
        "default_dim": 1024,
        "env_var": "VOYAGE_API_KEY",
        "signup_url": "https://dash.voyageai.com/",
        "extra_install": "corpus-rag[voyage]",
    },
    "gemini": {
        "default_model": "gemini-embedding-001",
        "default_dim": 1536,  # Matryoshka — also 768 / 3072 valid
        "env_var": "GEMINI_API_KEY",
        "signup_url": "https://aistudio.google.com/apikey",
        "extra_install": "corpus-rag[gemini]",
    },
}


@dataclass
class WizardAnswers:
    db_path: Path
    source_name: str
    source_type: str
    source_path: Path
    provider: str
    model: str
    dim: int


def _ask(prompt: str, default: str | None = None, choices: list[str] | None = None) -> str:
    """Prompt the user with a default and an optional fixed choice list."""
    suffix = f" [{default}]" if default else ""
    if choices:
        suffix = f" ({'/'.join(choices)})" + (f" [{default}]" if default else "")
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError as e:
            # stdin closed (piped input exhausted, or a stray Ctrl-D). Abort
            # rather than returning the default forever — the path-validation
            # loop would otherwise spin infinitely. Use --quiet for defaults.
            raise _AbortWizard() from e
        value = raw or default or ""
        if choices and value not in choices:
            print(f"  Please choose one of: {', '.join(choices)}")
            continue
        if value or not default:
            return value


def _normalize_source_name(name: str) -> str:
    """Map a friendly name to a valid source_type identifier."""
    norm = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    norm = re.sub(r"^[^a-z]+", "", norm)
    return norm or "notes"


def _run_wizard() -> WizardAnswers:
    print()
    print("=== corpus init wizard ===")
    print()

    # 1. Where does the DB live?
    db_path = Path(_ask("Where should the corpus database live?", default="./corpus.db"))

    # 2. What's this source called?
    print()
    raw_name = _ask("What do you want to call your first source? (e.g. notes, papers, bookmarks)", default="notes")
    source_name = _normalize_source_name(raw_name)
    if source_name != raw_name:
        print(f"  → normalized to '{source_name}' (lowercase identifier required)")

    # 3. What format?
    print()
    print("Built-in source types:")
    print("  markdown — .md files with optional YAML frontmatter")
    print("  text     — .txt files")
    print("  pdf      — PDF files via pypdf (needs `pip install corpus-rag[pdf]`)")
    print("  html     — HTML via trafilatura (needs `pip install corpus-rag[html]`)")
    source_type = _ask("Which format?", default="markdown", choices=KNOWN_SOURCE_TYPES)

    # 4. Where is the data?
    print()
    while True:
        raw_path = _ask("Path to the data (use ~ for home dir)", default=f"~/Documents/{source_name}")
        source_path = Path(os.path.expanduser(raw_path))
        if source_path.exists():
            break
        print(f"  Warning: {source_path} does not exist yet.")
        if _ask("  Continue anyway?", default="no", choices=["yes", "no"]) == "yes":
            break

    # 5. Which embedder?
    print()
    print("Embedder providers:")
    print("  voyage — Voyage AI (default). Best retrieval quality. Free tier:")
    print("           ~200M tokens. Card-on-file required to lift rate limits.")
    print("  gemini — Google Gemini. Free tier via AI Studio (no card needed):")
    print("           ~1500 requests/day. Comparable quality.")
    provider = _ask("Which provider?", default="voyage", choices=list(KNOWN_PROVIDERS.keys()))
    info = KNOWN_PROVIDERS[provider]
    model = _ask("Model name", default=info["default_model"])
    dim_input = _ask(f"Embedding dimension ({info['default_dim']} recommended)", default=str(info["default_dim"]))
    try:
        dim = int(dim_input)
    except ValueError:
        dim = info["default_dim"]
        print(f"  Invalid integer; using {dim}.")

    return WizardAnswers(
        db_path=db_path,
        source_name=source_name,
        source_type=source_type,
        source_path=source_path,
        provider=provider,
        model=model,
        dim=dim,
    )


def _toml_str(value: str) -> str:
    """Escape a value for a TOML basic (double-quoted) string.

    Without this, a db_path/source_path containing a `"` or `\\` (both legal
    on APFS) would render invalid, unloadable TOML.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_corpus_toml(a: WizardAnswers) -> str:
    return f"""# corpus.toml — generated by `corpus-init`.

[corpus]
db_path = {_toml_str(str(a.db_path))}

[embedder]
provider = {_toml_str(a.provider)}
model = {_toml_str(a.model)}
dim = {a.dim}

[retriever]
top_k = 5
max_per_source_type = 3
hybrid = true

[[sources]]
name = {_toml_str(a.source_name)}
type = {_toml_str(a.source_type)}
path = {_toml_str(str(a.source_path))}

# Add more sources by repeating the [[sources]] block.
# Configure cross-document reference patterns by adding [[references]] blocks.
# See corpus.toml.example for the full annotated template.
"""


def _render_env(a: WizardAnswers) -> str:
    info = KNOWN_PROVIDERS[a.provider]
    return f"""# .env — generated by `corpus-init`. Fill in the value below.

# Required for embedder provider '{a.provider}'.
# Get a key at: {info['signup_url']}
{info['env_var']}=

# Optional: only needed if you run `corpus-summarize` for per-doc Claude summaries.
# Sign up: https://console.anthropic.com/
ANTHROPIC_API_KEY=
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive setup for corpus")
    parser.add_argument("--force", action="store_true", help="Overwrite existing corpus.toml / .env")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Non-interactive: accept all defaults without prompting (CI/tests)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Where to write corpus.toml and .env (default: current dir)",
    )
    return parser


def _default_answers() -> WizardAnswers:
    """The values the wizard would produce by hitting Enter at every prompt."""
    info = KNOWN_PROVIDERS["voyage"]
    return WizardAnswers(
        db_path=Path("./corpus.db"),
        source_name="notes",
        source_type="markdown",
        source_path=Path(os.path.expanduser("~/Documents/notes")),
        provider="voyage",
        model=info["default_model"],
        dim=info["default_dim"],
    )


def _print_next_steps(a: WizardAnswers, toml_path: Path, env_path: Path) -> None:
    info = KNOWN_PROVIDERS[a.provider]
    print()
    print(f"✓ Wrote {toml_path}")
    print(f"✓ Wrote {env_path}")
    print()
    print("Next steps:")
    if info.get("extra_install"):
        print(f"  1. pip install {info['extra_install']}   # if not already installed")
        next_step = 2
    else:
        next_step = 1
    print(f"  {next_step}. Edit {env_path.name} and paste your {info['env_var']}")
    print(f"     (sign up: {info['signup_url']})")
    print(f"  {next_step + 1}. corpus-ingest --source {a.source_name} -v")
    print(f"  {next_step + 2}. corpus-query \"your first question\"")
    print()


def main_with_args(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = out_dir / "corpus.toml"
    env_path = out_dir / ".env"

    for p in (toml_path, env_path):
        if p.exists() and not args.force:
            print(f"Refusing to overwrite existing {p}. Pass --force to replace.")
            return 1

    try:
        answers = _default_answers() if args.quiet else _run_wizard()
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1
    except _AbortWizard:
        print("\nAborted (stdin closed). Re-run interactively, or use --quiet for defaults.")
        return 1

    toml_path.write_text(_render_corpus_toml(answers))
    env_path.write_text(_render_env(answers))
    _print_next_steps(answers, toml_path, env_path)
    return 0


def main() -> int:
    return main_with_args()


if __name__ == "__main__":
    sys.exit(main())
