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


def be_nice(increment: int = DEFAULT_NICE) -> bool:
    """Lower this process's scheduling priority. True if it took effect.

    `0` means "leave my priority alone" rather than `nice(0)`, so a caller can
    opt out without this function needing to know why.

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
    nice = getattr(os, "nice", None)
    if nice is None:
        logger.debug("os.nice is unavailable on this platform; priority unchanged")
        return False
    try:
        nice(increment)
    except (OSError, PermissionError) as exc:
        logger.info("could not lower priority (%s); continuing at normal", exc)
        return False
    return True


__all__ = ["DEFAULT_NICE", "be_nice"]
