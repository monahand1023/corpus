"""A token count of zero must not mean "this provider does not count".

`GeminiEmbedder` sets `total_tokens_used = 0` and never increments it --
deliberately, because the Gemini API returns no per-call usage and the
comment says "we don't pretend". But the pipeline reads that counter
unconditionally and both ingest CLIs print:

    tokens billed:    0

So a Gemini-backed archive reports the same 0 whether it embedded five
chunks or five hundred thousand, and it reads as "this run was free". Not
pretending inside the embedder became pretending at the only place a user
looks.

This is the session's recurring shape once more: "not measured" rendered
identically to "measured, and it was nothing".
"""

from __future__ import annotations

from corpus.embedder.hash import HashEmbedder


def test_an_embedder_that_counts_says_so():
    assert HashEmbedder(dim=64).counts_tokens is True


def test_the_gemini_embedder_declares_that_it_cannot_count(monkeypatch):
    """Constructed without touching the network: only the flag is under test."""
    import sys
    import types
    from unittest.mock import MagicMock

    fake = types.ModuleType("google.genai")
    fake.Client = MagicMock()
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", fake)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from corpus.embedder.gemini import GeminiEmbedder

    assert GeminiEmbedder().counts_tokens is False


def test_an_ingest_result_carries_whether_the_count_is_real():
    from corpus.ingester import IngestResult

    counted = IngestResult(
        source_name="s", documents=1, chunks_seen=1, chunks_upserted=1,
        chunks_skipped=0, orphans_deleted=0, tokens_used=0, elapsed_seconds=0.1,
    )
    assert counted.tokens_counted is True, "the default must stay the honest case"


def test_the_cli_prints_not_reported_rather_than_zero(capsys):
    from corpus.cli.ingest import _print_tokens

    _print_tokens(tokens_used=0, counted=False)
    out = capsys.readouterr().out
    assert "0" not in out.split(":")[-1], "an uncounted run still showed a number"
    assert "not reported" in out.lower()


def test_the_cli_still_prints_a_real_zero_as_zero(capsys):
    """A run that genuinely embedded nothing -- every chunk unchanged -- must
    still say 0, or resumption looks broken."""
    from corpus.cli.ingest import _print_tokens

    _print_tokens(tokens_used=0, counted=True)
    assert "0" in capsys.readouterr().out
