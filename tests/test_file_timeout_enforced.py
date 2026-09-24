"""A hung file must not take the run with it.

`file_timeout` has been in corpus since the transcription pipeline landed:
imported, re-exported, unit-tested, and CALLED BY NOTHING. Its own docstring
says why it exists -- "One file stalled for 81 minutes before this existed" --
and the guard does not run. The value was ported; the mechanism that enforces
it was left behind in the script it came from.

WHY A SUBPROCESS AND NOT A SIGNAL OR A THREAD. Whisper can enter an unbounded
decode loop on non-speech audio: the original archive observed a 170-second
clip holding the pipeline for over 75 minutes, GPU busy, producing nothing.
Signals are delivered at Python bytecode boundaries and the loop is inside
Metal kernels, so they never arrive. Threads cannot be killed. Only a separate
process can be terminated.

Observed live while re-transcribing that same archive with corpus: one file
ran 17+ minutes against a work list whose LONGEST remaining recording should
take about 10. The guard that would have caught it was sitting right there,
tested, unwired.
"""

from __future__ import annotations

import time
from pathlib import Path

from corpus.transcripts.worker import TranscribeWorker, WorkerTimeout

# Module level, all of them: `spawn` pickles the target by qualified name, so
# a function defined inside a test cannot be a worker.


def _hang(conn):  # pragma: no cover - runs in a child process
    conn.recv()
    time.sleep(3600)


def _echo(conn):  # pragma: no cover - runs in a child process
    while True:
        item = conn.recv()
        if item is None:
            return
        path, _settings = item
        conn.send(("ok", f"transcribed {Path(path).name}"))


def _explode(conn):  # pragma: no cover - runs in a child process
    conn.recv()
    conn.send(("error", "backend blew up"))


def _hang_on_stuck(conn):  # pragma: no cover - runs in a child process
    while True:
        item = conn.recv()
        if item is None:
            return
        path, _settings = item
        if "stuck" in str(path):
            time.sleep(3600)
        conn.send(("ok", "fine"))


def _die(conn):  # pragma: no cover - runs in a child process
    conn.recv()
    raise SystemExit(1)


def test_a_healthy_file_comes_back(tmp_path):
    """The positive control. Without it every timeout test below would pass
    on a worker that never works at all."""
    w = TranscribeWorker(target=_echo)
    try:
        kind, payload = w.run(tmp_path / "clip.mov", timeout=10.0)
        assert kind == "ok"
        assert "clip.mov" in str(payload)
    finally:
        w.close()


def test_a_hung_file_is_cut_off_at_the_deadline(tmp_path):
    w = TranscribeWorker(target=_hang)
    try:
        started = time.monotonic()
        kind, _ = w.run(tmp_path / "stuck.mov", timeout=1.0)
        elapsed = time.monotonic() - started
        assert kind == "timeout"
        assert elapsed < 20, f"the deadline did not hold: {elapsed:.1f}s"
    finally:
        w.close()


def test_the_worker_is_replaced_so_the_run_continues(tmp_path):
    """A hang must cost ONE file, not the rest of the archive. The model is
    loaded once per worker, so a replacement is paid for only when a file
    actually hangs."""
    w = TranscribeWorker(target=_hang_on_stuck)
    try:
        # Captured BEFORE the hang. Reading it afterwards reads the
        # REPLACEMENT's pid, and the comparison then passes whether or not
        # anything was replaced -- which is what the first version of this
        # test did.
        original_pid = w.pid
        assert w.run(tmp_path / "stuck.mov", timeout=1.0)[0] == "timeout"
        assert w.pid != original_pid, "the hung worker was not replaced"

        kind, _ = w.run(tmp_path / "next.mov", timeout=10.0)
        assert kind == "ok", "the run did not recover from a hang"
    finally:
        w.close()


def test_an_error_inside_the_worker_is_returned_not_raised(tmp_path):
    w = TranscribeWorker(target=_explode)
    try:
        kind, detail = w.run(tmp_path / "bad.mov", timeout=10.0)
        assert kind == "error"
        assert "blew up" in str(detail)
    finally:
        w.close()


def test_a_dead_worker_is_reported_not_hung(tmp_path):
    w = TranscribeWorker(target=_die)
    try:
        kind, _ = w.run(tmp_path / "x.mov", timeout=10.0)
        assert kind == "error"
    finally:
        w.close()


def test_the_timeout_scales_with_the_recording(tmp_path):
    """A flat bound is wrong at both ends: too short for a two-hour
    recording, and long enough for a thirty-second clip to hang for the whole
    of it."""
    from corpus.transcripts.segment import file_timeout

    assert file_timeout(30.0) < file_timeout(7200.0)
    assert file_timeout(170.0) < 75 * 60, (
        "the observed 75-minute hang on a 170s clip would still not be caught"
    )


def test_worker_timeout_is_an_exception_callers_can_catch():
    assert issubclass(WorkerTimeout, Exception)


# --- the run loop must actually apply it -------------------------------------


def test_a_hung_file_is_recorded_and_the_run_moves_on(tmp_path):
    """The guard is only real if `transcribe_directory` enforces it. corpus
    had the timeout VALUE, tested, for the whole life of the pipeline and
    nothing called it."""
    from corpus.transcripts import store
    from corpus.transcripts.pipeline import Settings
    from corpus.transcripts.run import transcribe_directory

    for name in ("a.mov", "hangs.mov", "c.mov"):
        (tmp_path / name).write_bytes(b"x")
    db = tmp_path / "t.db"
    seen: list[str] = []

    def fake(path, backend, *, settings=None):
        seen.append(Path(path).name)
        if "hangs" in str(path):
            time.sleep(60)
        outcome = type("O", (), {})()
        outcome.duration_s = 1.0
        outcome.dropped = []
        outcome.transcript = None
        outcome.empty_reason = "silence"
        outcome.rejected_text = ""
        outcome.elapsed_s = 0.1
        return outcome

    class _Backend:
        model_name = "test-model"

    stats = transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(),
        transcribe=fake, file_timeout_s=1.0,
    )
    assert len(seen) == 3, f"the run stopped at the hang: {seen}"
    assert stats.timed_out == 1
    with store.open_store(db, read_only=True) as conn:
        row = conn.execute(
            "SELECT error FROM failures WHERE path LIKE '%hangs.mov'"
        ).fetchone()
        assert row is not None, "the hang was not recorded"
        assert "timeout" in row[0].lower() or "timed out" in row[0].lower()


def test_a_timeout_is_a_FAILURE_so_it_is_retried(tmp_path):
    """Not a settled verdict. A hang is usually about this run -- thermal
    state, memory pressure, a transient decode loop -- so the file must come
    back on the next pass rather than being written off."""
    from corpus.transcripts import store
    from corpus.transcripts.pipeline import Settings
    from corpus.transcripts.run import transcribe_directory

    (tmp_path / "hangs.mov").write_bytes(b"x")
    db = tmp_path / "t.db"

    def fake(path, backend, *, settings=None):
        time.sleep(60)

    class _Backend:
        model_name = "test-model"

    transcribe_directory(
        tmp_path, db, _Backend(), settings=Settings(),
        transcribe=fake, file_timeout_s=1.0,
    )
    with store.open_store(db, read_only=True) as conn:
        policy = store.policy_fingerprint(Settings().as_policy("test-model"))
        assert str(tmp_path / "hangs.mov") not in store.already_done(conn, policy=policy)


def test_the_cli_supplies_the_SUBPROCESS_runner_not_the_thread_one(monkeypatch, tmp_path):
    """The distinction the whole fix turns on. A thread-bounded deadline
    bounds how long the run WAITS; only a subprocess bounds how long the work
    takes, and Whisper's loop is inside Metal kernels where nothing else
    reaches. A CLI that quietly used the weak form would look identical in
    every test and fail in exactly the situation this exists for.
    """
    import corpus.cli.transcribe as mod

    captured = {}

    def fake_directory(*args, **kwargs):
        captured.update(kwargs)
        from corpus.transcripts.run import RunStats

        return RunStats()

    (tmp_path / "a.mov").write_bytes(b"x")
    monkeypatch.setattr(mod, "transcribe_directory", fake_directory)
    monkeypatch.setattr(mod, "ffmpeg_available", lambda: True, raising=False)

    class _Backend:
        model_name = "m"

    monkeypatch.setattr(
        "corpus.transcripts.backends.default_backend", lambda: _Backend()
    )
    monkeypatch.setattr(mod, "find_media", lambda *a, **k: [tmp_path / "a.mov"])
    mod.main_argv([str(tmp_path), "--db", str(tmp_path / "t.db")])

    assert captured.get("run_one") is not None, (
        "the CLI did not supply a subprocess runner, so the deadline cannot "
        "be enforced against a real backend"
    )


# --- a file seen for the first time ---------------------------------------------


def _deadlines_for(tmp_path, **kwargs) -> dict[str, float]:
    """Run the loop over one never-seen file and report the deadline it got."""
    from corpus.transcripts.pipeline import Settings
    from corpus.transcripts.run import transcribe_directory

    (tmp_path / "long.mov").write_bytes(b"x")
    given: dict[str, float] = {}

    def run_one(path, timeout):
        given[Path(path).name] = timeout
        return ("error", "stop here")

    class _Backend:
        model_name = "test-model"

    transcribe_directory(
        tmp_path, tmp_path / "t.db", _Backend(), settings=Settings(),
        run_one=run_one, **kwargs,
    )
    return given


def test_a_new_file_gets_a_deadline_scaled_to_its_length(tmp_path):
    """The deadline came only from durations already in the sidecar, and a
    file seen for the first time has none -- so every new recording got the
    flat 120s base. A two-hour recording cannot finish in that; it timed out,
    was retried with the same 120s, and after three passes was settled as
    `repeatedly_timed_out`. The longest recordings were written off first.
    """
    from corpus.transcripts.segment import file_timeout

    given = _deadlines_for(tmp_path, duration_of=lambda _path: 7200.0)
    assert given["long.mov"] == file_timeout(7200.0)


def test_an_unreadable_duration_falls_back_to_the_base_deadline(tmp_path):
    from corpus.transcripts.segment import file_timeout

    given = _deadlines_for(tmp_path, duration_of=lambda _path: None)
    assert given["long.mov"] == file_timeout(0.0)


def test_the_cli_supplies_a_duration_lookup(monkeypatch, tmp_path):
    """Without one, the run loop has nothing to scale a new file's deadline
    by -- the defect above, reintroduced by a caller that forgets it."""
    import corpus.cli.transcribe as mod

    captured = {}

    def fake_directory(*args, **kwargs):
        captured.update(kwargs)
        from corpus.transcripts.run import RunStats

        return RunStats()

    (tmp_path / "a.mov").write_bytes(b"x")
    monkeypatch.setattr(mod, "transcribe_directory", fake_directory)

    class _Backend:
        model_name = "m"

    monkeypatch.setattr(
        "corpus.transcripts.backends.default_backend", lambda: _Backend()
    )
    monkeypatch.setattr(mod, "find_media", lambda *a, **k: [tmp_path / "a.mov"])
    mod.main_argv([str(tmp_path), "--db", str(tmp_path / "t.db")])

    assert callable(captured.get("duration_of"))
