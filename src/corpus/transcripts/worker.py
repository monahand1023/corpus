"""A worker process for transcription, replaced whenever it stops responding.

WHY THIS HAS TO BE A PROCESS. Whisper can enter an unbounded decode loop on
non-speech audio. Observed on a real archive: a 170-second clip held the
pipeline for over 75 minutes, GPU busy, producing nothing, with no way to
notice or recover.

Signals cannot fix it -- they are delivered at Python bytecode boundaries and
the loop is inside Metal kernels, so they never arrive. Threads cannot be
killed. Only a separate process can be terminated, which is why one exists.

WHY IT DID NOT EXIST HERE UNTIL NOW. `corpus.transcripts.segment.file_timeout`
computes how long a file may take, is unit-tested, and was called by NOTHING.
The value was ported into corpus from the script this mechanism comes from;
the mechanism was left behind. A bound with no way to enforce it reads exactly
like a bound that works -- which is how a 17-minute stall on a work list whose
longest recording should take ten went unnoticed until someone watched the
log stop moving.

THE MODEL IS LOADED ONCE PER WORKER, not once per file, so the cost of a
replacement is paid only when a file actually hangs.

THE DEADLINE IS ONLY AS ALIVE AS THE PARENT HOLDING IT. Polling happens here,
in the parent, so a parent killed hard -- SIGKILL, a crash, the OOM killer --
takes the enforcement with it while the child is still inside a file. Its
daemon flag does not help: that is honoured by an atexit hook, which a hard
kill never runs. Between files the child is waiting on `recv()` and sees EOF,
so it exits on its own; INSIDE a file it never returns to `recv()` to find
out. Measured on this machine: one such orphan held the GPU at 93% for 20h15m
and wrote nothing for the last 12h37m of it, because the process meant to
receive its results no longer existed. So the child watches for the parent
disappearing too -- the same bound, enforced from the side that still exists.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: How often a child checks that its parent is still there. One syscall at
#: this interval is nothing beside a transcription, and it bounds how long an
#: orphan can hold the GPU.
PARENT_POLL_SECONDS = 1.0

#: Distinct from 0 and 1 so an orphaned exit is identifiable in a crash log.
EXIT_ORPHANED = 3


class WorkerTimeout(Exception):
    """A file exceeded its deadline and its worker was replaced."""


def _exit_when_orphaned(
    parent_pid: int, interval: float = PARENT_POLL_SECONDS
) -> None:  # pragma: no cover - runs in a child thread
    """Hard-exit this process once `parent_pid` is no longer our parent.

    `os._exit`, not `sys.exit`: raising in a daemon thread ends the thread and
    leaves the main thread inside Metal exactly where it was. There is also
    nothing left to clean up for -- the pipe's far end died with the parent,
    so no result can be delivered and no shutdown needs to be orderly.
    """
    while True:
        time.sleep(interval)
        if os.getppid() != parent_pid:
            os._exit(EXIT_ORPHANED)


def _child_bootstrap(
    target: Callable[[Any], None], conn: Any, parent_pid: int
) -> None:  # pragma: no cover - runs in a child
    """Start the orphan watch, then hand over to the real worker body.

    This wraps EVERY target rather than living inside `_worker_main`, because
    the guarantee belongs to running as this class's child -- not to one
    particular payload. A test double that hangs gets it too, which is the
    only way the property can be tested at all.
    """
    watcher = threading.Thread(
        target=_exit_when_orphaned, args=(parent_pid,), daemon=True
    )
    watcher.start()
    target(conn)


def _worker_main(conn: Any) -> None:  # pragma: no cover - runs in a child
    """Transcribe paths sent down the pipe until told to stop.

    Imports inside the function: this runs in a `spawn`ed interpreter, and the
    backend costs seconds to load. Doing it here keeps the cost in the child
    where it belongs, and keeps a parent that never transcribes from paying it
    at all.
    """
    from corpus.transcripts.backends import default_backend
    from corpus.transcripts.pipeline import Settings, transcribe_file

    backend = None
    while True:
        try:
            item = conn.recv()
        except (EOFError, OSError):
            return
        if item is None:
            return
        path, settings = item
        try:
            if backend is None:
                backend = default_backend()
            outcome = transcribe_file(
                Path(path), backend, settings=settings or Settings()
            )
            conn.send(("ok", outcome))
        except Exception as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class TranscribeWorker:
    """One transcription subprocess, restarted when it misses a deadline."""

    def __init__(self, target: Callable[[Any], None] | None = None) -> None:
        # `target` is injectable so the timeout path can be tested with a
        # worker that hangs on purpose. Nothing else should pass it.
        self._target = target or _worker_main
        self._start()

    def _start(self) -> None:
        # "spawn", not "fork": a forked child inherits the parent's Metal and
        # threading state, which is not safe to use after fork on macOS.
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_child_bootstrap,
            args=(self._target, child, os.getpid()),
            daemon=True,
        )
        self._proc.start()
        child.close()

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    def run(
        self, path: Path | str, timeout: float, settings: Any = None
    ) -> tuple[str, Any]:
        """Transcribe one file under a deadline.

        Returns ("ok", Outcome) | ("error", detail) | ("timeout", None). A
        timeout is a RESULT, not an exception, because the caller's job is to
        record it against this file and move on to the next -- a hang must
        cost one file, not the rest of the archive.
        """
        try:
            self._conn.send((str(path), settings))
        except (BrokenPipeError, OSError):
            self._replace()
            return ("error", "worker pipe closed before send")
        if not self._conn.poll(timeout):
            self._replace()
            return ("timeout", None)
        try:
            result: tuple[str, Any] = self._conn.recv()
            return result
        except (EOFError, OSError):
            self._replace()
            return ("error", "worker died mid-file")

    def _replace(self) -> None:
        # Every step suppressed: teardown must not mask the timeout that
        # caused it. A worker that is already dead, or a pipe already closed,
        # is the normal case here rather than an error.
        with contextlib.suppress(Exception):
            self._proc.kill()
            self._proc.join(30)
        with contextlib.suppress(Exception):
            self._conn.close()
        self._start()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.send(None)
            self._proc.join(10)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(10)
        with contextlib.suppress(Exception):
            self._conn.close()


__all__ = ["EXIT_ORPHANED", "PARENT_POLL_SECONDS", "TranscribeWorker", "WorkerTimeout"]
