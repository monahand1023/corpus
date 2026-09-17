"""corpus-smoke asserted a literal `True` and called it an assertion.

    # ... so the exit is worth asserting rather than assuming.
    report.add("exited on stdin EOF", True, "client closed cleanly")

Nothing is polled, no returncode is read, no process is waited on. The line
also sits OUTSIDE the try, after both `except TimeoutError` and
`except Exception`, so it prints a tick even when the handshake never
happened:

    handshake            FAIL  timed out after 30s
    exited on stdin EOF  ok    client closed cleanly

A server that never exits on EOF -- the exact leak class this line names --
reports the same tick as one that exits cleanly. Delete the behaviour under
test and the output is byte-identical.

WHY IT IS NOT SIMPLY MEASURED INSTEAD. `stdio_client` owns the subprocess
and never exposes its exit status, so from here that fact is genuinely not
observable. The honest report is that it was not checked, and where it IS
checked -- which is a real test, not a claim.
"""

from __future__ import annotations


def _report():
    from corpus.cli.smoke import ServerReport

    return ServerReport(server="x")


def test_the_exit_is_not_reported_as_a_pass_it_did_not_observe():
    from corpus.cli.smoke import note_stdin_eof_unverified

    report = _report()
    note_stdin_eof_unverified(report)
    check = report.checks[-1]

    assert check.ok is None, (
        "still claiming a pass for something nothing observed"
    )
    assert "not checked" in check.detail.lower()


def test_it_points_at_where_the_behaviour_IS_covered():
    """An unverified line that says only 'unverified' trains people to ignore
    it. This one has a real test behind it and should say so."""
    from corpus.cli.smoke import note_stdin_eof_unverified

    report = _report()
    note_stdin_eof_unverified(report)
    assert "lifecycle" in report.checks[-1].detail.lower()


def test_an_unverified_check_does_not_make_the_run_fail():
    """It is not a failure, it is an absence. Failing on it would make smoke
    red on every healthy server and the whole report would get ignored."""
    from corpus.cli.smoke import ServerReport, note_stdin_eof_unverified

    report = ServerReport(server="x")
    report.add("handshake", True)
    note_stdin_eof_unverified(report)

    assert report.ok is True


def test_a_real_failure_still_fails():
    from corpus.cli.smoke import ServerReport, note_stdin_eof_unverified

    report = ServerReport(server="x")
    report.add("handshake", False, "timed out")
    note_stdin_eof_unverified(report)

    assert report.ok is False
