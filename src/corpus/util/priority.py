"""Get out of the way of everything else on the machine.

Transcription runs for hours and ingestion for minutes, both pinning CPU.
They are background work by nature: nobody is waiting on them, and an
interactive process on the same machine IS being waited on.

`nice(15)` is nearly the lowest priority Unix offers, and it is close to free
for the job itself: the scheduler still hands it every idle cycle, and it
yields immediately when anything else wants to run. On a busy machine that is
the difference between a long job being invisible and it being the reason
someone's editor stutters.

CHILDREN INHERIT IT. A transcription worker subprocess, and every `ffmpeg` it
spawns, start at the parent's niceness -- so setting it once at the top of a
command covers the processes that do the actual work.

NOT REVERSIBLE. A process cannot lower its own niceness again without
privileges, which is why this is a flag with a default rather than a
hardcoded call: the caller decides, once, and can opt out with 0.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# High enough to yield to anything interactive, low enough that the job still
# gets every otherwise-idle cycle. 19 is the maximum and buys nothing more.
DEFAULT_NICE = 15

# Whether this process has already been lowered.
#
# `main()` is a FUNCTION, and a test suite or an embedding application calls
# several of them in one process. Without this, each call lowers priority by
# the full increment again -- two commands land at nice 30, and the caller
# asked for neither. Found by the test suite: running a CLI test before the
# real-syscall test pushed the pytest process to the cap.
#
# Once per process is also the honest semantics: this is a property of the
# process, not an operation you accumulate.
_applied = False


def be_nice(increment: int = DEFAULT_NICE) -> bool:
    """Lower this process's scheduling priority. True if it took effect.

    `0` means "leave my priority alone" rather than `nice(0)`, so a caller can
    opt out without this function needing to know why.

    Applied at most ONCE per process. Calling it again is a no-op that returns
    False, because `main()` is a function an embedding application may call
    several times and each call would otherwise lower priority by the full
    increment again.

    Returns False rather than raising when the platform has no `os.nice`
    (Windows) or the OS refuses (some sandboxes). Failing a multi-hour run
    over a courtesy would be worse than not being courteous -- but it IS
    logged, because a job that silently kept full priority on a shared machine
    is the kind of thing nobody notices until it matters.
    """
    if increment < 0:
        raise ValueError(
            f"be_nice({increment}) would RAISE priority, which needs privileges "
            "and is the opposite of the point. Use 0 to leave it unchanged."
        )
    if increment == 0:
        return False
    global _applied
    if _applied:
        logger.debug("priority already lowered in this process; not lowering again")
        return False
    nice = getattr(os, "nice", None)
    if nice is None:
        logger.debug("os.nice is unavailable on this platform; priority unchanged")
        return False
    try:
        nice(increment)
    except (OSError, PermissionError) as exc:
        logger.info("could not lower priority (%s); continuing at normal", exc)
        return False
    _applied = True
    return True


__all__ = ["DEFAULT_NICE", "be_nice"]
