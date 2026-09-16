"""Transcribing a folder of recordings into a sidecar, resumably.

`pipeline.transcribe_file` handles one file. This handles the run: which files
to visit, which to skip because a previous run already answered for them, and
writing every outcome down -- including the negative ones.

RESUMPTION IS THE POINT. A real archive is hours to days of compute, and it
WILL be interrupted: a crash, a reboot, a laptop lid. A run that cannot resume
is a run that never finishes, and the expensive mistake is subtler than losing
transcripts. Files that produced NO usable text are the ones a naive restart
re-does, because they leave no row behind to find. One interrupted pass
re-decoded and re-transcribed 889 already-examined silent clips before the
`no_text` table existed.

WHAT A POLICY CHANGE DOES. Every stored negative carries the fingerprint of
the rules that produced it. Change a threshold and those rows stop matching,
so the files are retried rather than inheriting a verdict made under different
rules. That is why `Settings.as_policy` has to list everything -- a setting
omitted from it silently keeps stale answers alive.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from corpus.survey.media import MEDIA_EXTENSIONS
from corpus.survey.walk import walk_files
from corpus.transcripts import store
from corpus.transcripts.pipeline import Outcome, Settings, transcribe_file

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    considered: int = 0
    skipped_done: int = 0
    transcribed: int = 0
    empty: int = 0
    failed: int = 0
    seconds_of_audio: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.transcribed + self.empty + self.failed


def find_media(
    root: Path | str,
    *,
    extensions: Iterable[str] = MEDIA_EXTENSIONS,
    excludes: Sequence[str] = (),
) -> Iterator[Path]:
    """Every media file under `root`, in a stable order.

    Sorted so an interrupted run resumes in the same sequence rather than
    re-walking in whatever order the filesystem returns, which makes progress
    legible across restarts.
    """
    wanted = {e.lower() for e in extensions}
    found = [
        walked.path
        for walked in walk_files(Path(root), excludes=tuple(excludes))
        if walked.path.suffix.lower() in wanted
    ]
    yield from sorted(found)


def transcribe_directory(
    root: Path | str,
    db_path: Path | str,
    backend: object,
    *,
    settings: Settings | None = None,
    excludes: Sequence[str] = (),
    limit: int | None = None,
    on_progress: Callable[[int, int, Path, Outcome | None], None] | None = None,
    transcribe: Callable[..., Outcome] = transcribe_file,
) -> RunStats:
    """Transcribe everything under `root` into the sidecar at `db_path`.

    Safe to interrupt and re-run: anything already answered for under the same
    policy is skipped. `transcribe` is injectable so the run loop can be
    tested without a model.
    """
    settings = settings or Settings()
    model_name = getattr(backend, "model_name", "unknown")
    policy = store.policy_fingerprint(settings.as_policy(model_name))
    stats = RunStats()

    files = list(find_media(root, excludes=excludes))
    stats.considered = len(files)

    with store.open_store(db_path) as conn:
        done = store.already_done(conn, policy=policy)
        todo = [p for p in files if str(p) not in done]
        stats.skipped_done = len(files) - len(todo)
        if limit is not None:
            todo = todo[:limit]

        for index, path in enumerate(todo, start=1):
            try:
                outcome = transcribe(path, backend, settings=settings)
            except Exception as exc:  # one bad file must not end the run
                # A failure is recorded and NOT skipped on the next run: it
                # usually means a broken decode or a transient resource
                # problem, not a settled verdict about the audio.
                stats.failed += 1
                stats.errors.append((str(path), f"{type(exc).__name__}: {exc}"))
                store.save_failure(conn, str(path), f"{type(exc).__name__}: {exc}")
                logger.warning("transcription failed for %s: %s", path, exc)
                if on_progress:
                    on_progress(index, len(todo), path, None)
                continue

            stats.seconds_of_audio += outcome.duration_s
            if outcome.dropped:
                store.save_dropped_windows(
                    conn,
                    str(path),
                    [
                        {
                            "window_start": d.window_start,
                            "no_speech": d.no_speech,
                            "avg_logprob": d.avg_logprob,
                            "text": d.text,
                        }
                        for d in outcome.dropped
                    ],
                    policy=policy,
                )

            if outcome.transcript is not None:
                outcome.transcript.policy = policy
                store.save_transcript(conn, outcome.transcript)
                stats.transcribed += 1
            else:
                # The row that makes a restart cheap. Without it this file is
                # simply absent, and the next run pays for it again to reach
                # the same answer.
                store.save_no_text(
                    conn,
                    str(path),
                    duration_s=outcome.duration_s,
                    policy=policy,
                    reason=outcome.empty_reason or "no_text",
                    rejected_text=outcome.rejected_text,
                )
                stats.empty += 1

            if on_progress:
                on_progress(index, len(todo), path, outcome)

    return stats
