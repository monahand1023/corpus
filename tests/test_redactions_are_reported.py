"""A secret was redacted and nobody was told.

`scrub()` rewrites credentials out of chunk content before embedding and
storage, which is right. `find_secrets()` -- the function that REPORTS what
was found rather than silently replacing it -- has zero production callers;
only tests reference it.

So the one fact the archive's owner needs is the one nothing produces. A
credential in an indexed document is a credential that should be ROTATED,
and rotation is the only remedy that exists once text has been sitting in a
personal archive for years. Silently removing it from the index protects
the index and leaves the credential live.

Counted per PATTERN, not as a total: "an AWS key" and "a Slack token" send
someone to two different consoles.
"""

from __future__ import annotations


def test_scrubbing_counts_what_it_redacted():
    from corpus.util.scrub import count_redactions, scrub

    with count_redactions() as found:
        scrub("key AKIAIOSFODNN7EXAMPLE here")
        scrub("and ghp_abcdefghij0123456789klmnopqrstuvwx here")
        scrub("nothing sensitive in this one")

    assert found["aws-access-key"] == 1
    assert found["github-token"] == 1
    assert sum(found.values()) == 2


def test_the_counter_does_not_leak_out_of_its_block():
    """A ContextVar left set makes the NEXT ingest report the previous run's
    secrets, which is worse than reporting none."""
    from corpus.util.scrub import count_redactions, scrub

    with count_redactions() as first:
        scrub("AKIAIOSFODNN7EXAMPLE")
    assert sum(first.values()) == 1

    scrub("AKIAIOSFODNN7EXAMPLE")  # outside any block
    assert sum(first.values()) == 1, "a later scrub was added to a closed count"


def test_scrubbing_still_works_with_no_counter_active():
    """Counting is observation. It must never be required for redaction."""
    from corpus.util.scrub import scrub

    out = scrub("token AKIAIOSFODNN7EXAMPLE end")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "REDACTED" in out


def test_the_same_secret_twice_counts_twice():
    """Two occurrences are two things to rotate, even if identical."""
    from corpus.util.scrub import count_redactions, scrub

    with count_redactions() as found:
        scrub("AKIAIOSFODNN7EXAMPLE and AKIAIOSFODNN7EXAMPLE")
    assert found["aws-access-key"] == 2


def test_the_ingest_reports_what_it_redacted(capsys):
    """A counter nobody reads is the same defect one layer along.

    The number has to reach the person running the ingest, and it has to name
    the PATTERN, because the action it implies -- rotate this -- differs by
    credential type.
    """
    from collections import Counter

    from corpus.cli.ingest import report_redactions

    report_redactions(Counter({"aws-access-key": 2, "slack-token": 1}))
    out = capsys.readouterr().out

    assert "3" in out, out
    assert "aws-access-key" in out and "slack-token" in out
    assert "rotate" in out.lower(), (
        "it reported a redaction without saying the credential is still live"
    )


def test_a_clean_ingest_says_nothing_about_redactions(capsys):
    """Silence on the clean path. Most ingests redact nothing, and a line
    printed every run stops being read on the run that matters."""
    from collections import Counter

    from corpus.cli.ingest import report_redactions

    report_redactions(Counter())
    assert capsys.readouterr().out == ""
