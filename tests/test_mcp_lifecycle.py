"""Guards for MCP servers on transports that can outlive their caller.

A stdio server needs none of this; these cover the HTTP/SSE case, where
nothing owns the process and so nothing would otherwise reap it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from corpus.mcp_util import (
    claim_single_instance,
    exit_when_orphaned,
    port_holder,
)


def test_the_first_claim_succeeds(tmp_path: Path) -> None:
    assert claim_single_instance("first", runtime_dir=tmp_path) is None


def test_a_second_claim_reports_the_holders_pid(tmp_path: Path) -> None:
    assert claim_single_instance("dupe", runtime_dir=tmp_path) is None
    # Idempotent startup: the second ask is not an error, it is a no-op that
    # can say who is already doing the job.
    assert claim_single_instance("dupe", runtime_dir=tmp_path) == os.getpid()


def test_different_names_do_not_contend(tmp_path: Path) -> None:
    assert claim_single_instance("alpha", runtime_dir=tmp_path) is None
    assert claim_single_instance("beta", runtime_dir=tmp_path) is None


def test_the_lock_survives_nothing_when_the_holder_is_killed(tmp_path: Path) -> None:
    """SIGKILL leaves no stale lock -- the kernel drops it with the process.

    This is the property a PID file does not have, and the reason a crashed
    server never blocks the one replacing it.
    """
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(Path("src").resolve())!r})
        from corpus.mcp_util import claim_single_instance
        assert claim_single_instance("killed", runtime_dir=__import__("pathlib").Path({str(tmp_path)!r})) is None
        print("locked", flush=True)
        time.sleep(60)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "locked"
        # Held while that process lives.
        assert claim_single_instance("killed", runtime_dir=tmp_path) == proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)

    assert claim_single_instance("killed", runtime_dir=tmp_path) is None


def test_a_deliberately_daemonised_process_is_left_alone(monkeypatch) -> None:
    """PPID 1 at startup means nohup/launchd, not abandonment."""
    monkeypatch.setattr(os, "getppid", lambda: 1)
    assert exit_when_orphaned() is False


def test_a_server_with_a_live_parent_arms_the_watchdog() -> None:
    assert exit_when_orphaned(interval=3600) is True


def test_an_orphaned_server_exits_on_its_own(tmp_path: Path) -> None:
    """The actual failure this exists to prevent: a listener nobody owns.

    A grandchild is orphaned by killing its parent, then must terminate
    without anyone signalling it.

    The grandchild announces itself through a file rather than the inherited
    pipe, and the test refuses to proceed until that file appears. Without the
    handshake this passes whether or not the guard works: the parent prints
    the grandchild's PID the instant it spawns, so a grandchild that died on
    import would look exactly like one the watchdog shut down.
    """
    ready = tmp_path / "armed"
    grandchild = textwrap.dedent(
        f"""
        import sys, time, pathlib
        sys.path.insert(0, {str(Path("src").resolve())!r})
        from corpus.mcp_util import exit_when_orphaned
        assert exit_when_orphaned(interval=0.2, grace=2.0) is True
        pathlib.Path({str(ready)!r}).write_text("ok")
        time.sleep(120)
        """
    )
    parent = textwrap.dedent(
        f"""
        import subprocess, sys, time
        p = subprocess.Popen([sys.executable, "-c", {grandchild!r}])
        print(p.pid, flush=True)
        time.sleep(120)
        """
    )
    outer = subprocess.Popen(
        [sys.executable, "-c", parent], stdout=subprocess.PIPE, text=True
    )
    assert outer.stdout is not None
    grandchild_pid = int(outer.stdout.readline().strip())

    deadline = time.monotonic() + 30
    while not ready.exists():
        if time.monotonic() > deadline:
            outer.kill()
            pytest.fail("grandchild never armed its watchdog")
        time.sleep(0.05)

    os.kill(grandchild_pid, 0)  # alive and armed before we orphan it
    outer.kill()
    outer.wait(timeout=10)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except OSError:
            return  # exited on its own, which is the whole point
        time.sleep(0.05)

    os.kill(grandchild_pid, 9)
    pytest.fail("orphaned process was still alive after 30s")


def test_a_free_port_reports_nothing() -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    # Reused immediately after close, which is the realistic case.
    assert port_holder(free) is None


def test_a_taken_port_is_reported_not_raised() -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        held = port_holder(taken)
    assert held is not None and str(taken) in held
