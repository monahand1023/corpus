"""A command that occupies the machine for minutes must lower its priority.

`be_nice` and `DEFAULT_NICE = 15` exist and are unit-tested, but nothing
checked WHICH commands call them -- so a CLI that grew into long work simply
never got the call, and there was no failing test to say so. The two found
this way both walk and `ffprobe` whole media trees.

The exemptions are as important as the rule. A benchmark must NOT be nice:
lowering the priority of the thing being timed corrupts the measurement it
exists to produce. Interactive commands are not nice either, because someone
is sitting there waiting for the answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CLI_DIR = Path(__file__).resolve().parent.parent / "src" / "corpus" / "cli"

#: Commands that occupy the machine long enough that something interactive is
#: probably sharing it.
LONG_RUNNING = [
    "contextualize",
    "eval",
    "index",
    "ingest",
    "reembed",
    "summarize",
    "survey",
    "transcribe",
]

#: Everything else, with the reason. Listing them explicitly is what makes a
#: NEW command a deliberate decision rather than an omission.
EXEMPT = {
    "__init__": "not a command",
    "_common": "shared helpers, not a command",
    "benchmark": "nicing the job under measurement corrupts the measurement",
    "doctor": "diagnostic, seconds, and a person is waiting on it",
    "init": "writes a config file",
    "judge": "waits on a remote API; it is not what is occupying the machine",
    "list_sources": "reads metadata",
    "migrate_fts": "one-shot schema change",
    "publish_check": "reads git",
    "query": "interactive; someone is waiting for the answer",
    "rename": "renames ids in place",
    "reset": "deletes rows",
    "smoke": "a handful of queries",
}


def _module_names() -> set[str]:
    return {p.stem for p in CLI_DIR.glob("*.py")}


def test_every_command_is_either_long_running_or_explicitly_exempt() -> None:
    """A new CLI must be classified, not silently default to 'not my problem'."""
    unclassified = _module_names() - set(LONG_RUNNING) - set(EXEMPT)
    assert not unclassified, (
        f"new command(s) {sorted(unclassified)}: add to LONG_RUNNING (and call "
        "be_nice), or to EXEMPT with the reason"
    )


@pytest.mark.parametrize("name", LONG_RUNNING)
def test_a_long_running_command_lowers_its_priority(name: str) -> None:
    source = (CLI_DIR / f"{name}.py").read_text(encoding="utf-8")
    assert "be_nice(" in source, (
        f"corpus-{name} runs for minutes at normal priority -- it never calls "
        "be_nice(), so it competes with whatever the person is actually doing"
    )


@pytest.mark.parametrize("name", LONG_RUNNING)
def test_a_long_running_command_lets_the_default_be_overridden(name: str) -> None:
    source = (CLI_DIR / f"{name}.py").read_text(encoding="utf-8")
    assert '"--nice"' in source, f"corpus-{name} hard-codes its priority"


def test_a_benchmark_is_never_niced() -> None:
    """Pinned as a rule, not left to whoever edits benchmark.py next."""
    source = (CLI_DIR / "benchmark.py").read_text(encoding="utf-8")
    assert "be_nice(" not in source, (
        "a niced benchmark measures the scheduler, not the code"
    )
