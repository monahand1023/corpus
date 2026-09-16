"""Long jobs should get out of the way of everything else on the machine.

Transcription runs for hours and ingestion for minutes, both pinning CPU.
They are background work by nature: nobody is waiting on them, and an
interactive process on the same machine IS being waited on. Measured on the
machine this was written for, one unrelated virtualization service was
holding 213,407 open file descriptors while a transcribe ran -- the box is
not idle, and corpus should not behave as though it were.

`nice(15)` is nearly the lowest priority Unix offers. It costs these jobs
almost nothing when the machine is free -- the scheduler still gives them
every idle cycle -- and yields immediately when anything else wants to run.

NOT REVERSIBLE, and that is why it is applied deliberately rather than
guessed at: a process cannot lower its own niceness again without
privileges. So it is a flag with a default, not a hardcoded call.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from corpus.util.priority import be_nice


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """`be_nice` applies once per PROCESS, and pytest is one process.

    Without this, any earlier test that ran a CLI `main()` has already set the
    flag and every test here reads as "already lowered" -- which is exactly
    how the ordering failure that prompted the idempotence fix showed up.
    """
    import corpus.util.priority as mod

    monkeypatch.setattr(mod, "_applied", False)


def test_it_lowers_this_process_priority(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(os, "nice", lambda n: calls.append(n) or 0)
    assert be_nice(15) is True
    assert calls == [15]


def test_zero_is_a_no_op_rather_than_a_call(monkeypatch):
    """`--nice 0` must mean "leave my priority alone", not "nice(0)", so a
    caller can opt out without the function having to know why."""
    calls: list[int] = []
    monkeypatch.setattr(os, "nice", lambda n: calls.append(n) or 0)
    assert be_nice(0) is False
    assert calls == []


def test_a_negative_value_is_refused(monkeypatch):
    """Raising priority needs privileges and would be the opposite of the
    point. Refused loudly rather than attempted and silently failing."""
    monkeypatch.setattr(os, "nice", lambda n: 0)
    with pytest.raises(ValueError):
        be_nice(-5)


def test_a_platform_without_nice_is_not_an_error(monkeypatch):
    """Windows has no os.nice. A long job must still run there; it just does
    not get to be polite."""
    monkeypatch.delattr(os, "nice", raising=False)
    assert be_nice(15) is False


def test_a_refusal_by_the_os_is_not_an_error(monkeypatch):
    """Some sandboxes deny it. Failing the whole run over a courtesy would
    be worse than not being courteous."""

    def denied(_n):
        raise PermissionError("not permitted")

    monkeypatch.setattr(os, "nice", denied)
    assert be_nice(15) is False


def test_it_really_works_on_this_platform(monkeypatch):
    """Not a mock. The others prove the wrapper's logic; this proves the
    syscall is actually reachable, which is the only part that can quietly
    stop being true."""
    if not hasattr(os, "nice"):
        pytest.skip("no os.nice on this platform")
    before = os.nice(0)
    if before >= 19:
        pytest.skip("already at the niceness cap; nothing left to lower")
    assert be_nice(1) is True
    assert os.nice(0) == before + 1


def test_it_only_applies_once_per_process(monkeypatch):
    """`main()` must be safe to call. It is a function, and a test suite or an
    embedding application calls several of them in one process -- each one
    raising niceness by 15 again, so two commands land at 30 and the caller
    never asked for either.

    Found by the suite: running a CLI test before the real-syscall test above
    pushed the pytest process far enough that it could not go higher.
    """
    calls: list[int] = []
    monkeypatch.setattr(os, "nice", lambda n: calls.append(n) or 0)

    assert be_nice(15) is True
    assert be_nice(15) is False, "a second call lowered priority again"
    assert calls == [15], f"niceness was applied more than once: {calls}"


# --- every long job must actually use it --------------------------------------

LONG_RUNNING = [
    "transcribe",     # hours of GPU
    "ingest",         # minutes of CPU + network
    "index",          # same, the wrapper command
    "contextualize",  # long, paid
    "summarize",      # long, paid
    "reembed",        # long, paid
]


@pytest.mark.parametrize("command", LONG_RUNNING)
def test_the_long_commands_offer_a_nice_flag(command):
    """A helper nothing calls is the defect this codebase keeps finding --
    `file_timeout` was imported, tested and wired to nothing for the life of
    the transcription pipeline. This asserts the wiring, not the helper."""
    import importlib

    module = importlib.import_module(f"corpus.cli.{command}")
    assert module.__file__ is not None
    text = pathlib.Path(module.__file__).read_text()
    assert "be_nice" in text, f"corpus-{command} never lowers its priority"
    assert "--nice" in text, f"corpus-{command} gives no way to opt out"


@pytest.mark.parametrize("command", LONG_RUNNING)
def test_the_default_is_to_be_nice(command):
    """Default-on. These are background jobs; being polite should not be
    something you have to remember."""
    import importlib

    module = importlib.import_module(f"corpus.cli.{command}")
    assert module.__file__ is not None
    text = pathlib.Path(module.__file__).read_text()
    assert "DEFAULT_NICE" in text, (
        f"corpus-{command} hardcodes its own value instead of the shared default"
    )


def test_children_really_do_inherit_it(tmp_path):
    """The docstring claims a worker subprocess and the ffmpeg it spawns start
    at the parent's niceness, which is what makes setting it once at the top
    of a command enough. Asserted rather than assumed."""
    import subprocess
    import sys
    import textwrap

    if not hasattr(os, "nice"):
        pytest.skip("no os.nice on this platform")

    script = textwrap.dedent(
        """
        import os, subprocess, sys
        os.nice(7)
        out = subprocess.run(
            [sys.executable, "-c", "import os; print(os.nice(0))"],
            capture_output=True, text=True, check=True,
        )
        print(out.stdout.strip())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    child_nice = int(result.stdout.strip())
    assert child_nice >= 7, (
        f"a child started at nice {child_nice}, so setting it on the parent "
        "does NOT cover the processes doing the work"
    )
