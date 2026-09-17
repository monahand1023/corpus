"""A worker must not outlive the parent that enforces its deadline.

`TranscribeWorker` bounds a file by polling the pipe from the PARENT and
replacing the child when it misses. That bound is only as alive as the parent
holding it. A parent that exits normally takes its daemon children with it,
and a child waiting on `conn.recv()` between files sees EOF and returns -- so
both of those cases were already safe.

The case that was not: a parent killed hard (SIGKILL, crash, OOM) while a
child is INSIDE a file. The child never returns to `recv()`, nothing polls the
deadline any more, and if Whisper is in an unbounded decode loop it never
comes back at all. Measured, on this machine: one such orphan held the GPU at
93% for 20 hours and 15 minutes, burned 3h23m of CPU, and wrote nothing to the
transcript database for the last 12h37m of it -- its results had nowhere to
go, because the process meant to receive them was gone.

So the child watches for its own parent disappearing. This is the same bound,
enforced from the side that still exists.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generous: the point is "it exits on its own", not "it exits in under a
# second". A failure here means it never exits at all.
ORPHAN_EXIT_DEADLINE = 20.0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until_gone(pid: int, deadline: float) -> float | None:
    """Seconds taken to exit, or None if it outlived the deadline."""
    started = time.monotonic()
    while time.monotonic() - started < deadline:
        if not _alive(pid):
            return time.monotonic() - started
        time.sleep(0.1)
    return None


# A parent that starts a worker stuck mid-file, reports the child's pid, then
# waits to be killed. `hang` stands in for an unbounded decode loop: it takes
# the work item and never produces a result.
_PARENT_SCRIPT = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo!r})
    from corpus.transcripts.worker import TranscribeWorker

    def hang(conn):
        conn.recv()
        while True:
            time.sleep(0.05)

    if __name__ == "__main__":
        worker = TranscribeWorker(target=hang)
        worker._conn.send(("some-file.m4a", None))
        print(worker.pid, flush=True)
        while True:
            time.sleep(0.05)
    """
)


def test_a_worker_stuck_mid_file_exits_when_its_parent_is_killed(tmp_path: Path) -> None:
    script = tmp_path / "parent.py"
    script.write_text(_PARENT_SCRIPT.format(repo=str(REPO_ROOT / "src")), encoding="utf-8")

    parent = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    try:
        assert parent.stdout is not None
        line = parent.stdout.readline().strip()
        assert line.isdigit(), f"parent never reported a worker pid; stderr:\n{parent.stderr.read() if parent.stderr else ''}"
        child_pid = int(line)

        # Precondition: the worker is running and stuck in the file.
        time.sleep(1.0)
        assert _alive(child_pid), "worker died on its own before the parent was killed"

        # The failure mode: parent dies hard, so its atexit hooks -- the thing
        # that normally reaps daemon children -- never run.
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=10)

        took = _wait_until_gone(child_pid, ORPHAN_EXIT_DEADLINE)
        assert took is not None, (
            f"worker {child_pid} outlived its killed parent by more than "
            f"{ORPHAN_EXIT_DEADLINE}s -- it is now an orphan holding the GPU "
            f"with nowhere to send a result"
        )
    finally:
        if child_pid is not None and _alive(child_pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)


def _echo(conn: object) -> None:  # pragma: no cover - runs in a child
    """A worker body that answers and keeps going, like a healthy one."""
    while True:
        item = conn.recv()  # type: ignore[attr-defined]
        if item is None:
            return
        conn.send(("ok", f"did {item[0]}"))  # type: ignore[attr-defined]


def test_a_worker_with_a_living_parent_is_left_alone() -> None:
    """The dangerous failure of a watchdog is the false positive.

    Killing healthy workers mid-file would be worse than the orphan it
    prevents: every transcription would die a second after starting, and the
    pipeline would report it as a per-file error rather than as a bug in the
    guard. So: survive several poll intervals, and keep answering afterwards.
    """
    from corpus.transcripts.worker import PARENT_POLL_SECONDS, TranscribeWorker

    worker = TranscribeWorker(target=_echo)
    try:
        assert worker.run("first.m4a", timeout=30) == ("ok", "did first.m4a")

        pid = worker.pid
        assert pid is not None
        time.sleep(PARENT_POLL_SECONDS * 3)

        assert _alive(pid), "the watchdog killed a worker whose parent is alive"
        assert worker.run("second.m4a", timeout=30) == ("ok", "did second.m4a")
        assert worker.pid == pid, "worker was replaced despite never missing a deadline"
    finally:
        worker.close()
