"""`scrub()` ran on one source type out of every kind this engine indexes.

scrub.py's module docstring says it "Runs once over each chunk's content
right before embedding + storage", and a test elsewhere calls it "the last
thing to run before content leaves the machine". Both describe an invariant
that held for exactly one chunker.

`scrub()` had ONE call site in the whole repo -- inside `MarkdownChunker`.
Every source type is paired with that chunker except `transcripts`, which
gets `TranscriptChunker`, and that one builds `content` and `content_hash`
straight from the transcript text.

Transcripts are the WORST source type to miss. They are speech: someone
reading a key aloud on a call, dictating a password, walking a colleague
through a console. The threat model in scrub.py is "the .db file leaks", and
a spoken credential was going into the index verbatim -- and into the
embedding request, which leaves the machine entirely.
"""

from __future__ import annotations


def _doc(text: str):
    from corpus.types import SourceDocument

    return SourceDocument(
        source_type="transcripts",
        source_key="/recordings/standup.m4a",
        title="standup",
        raw={"segments": [{"start": 0.0, "end": 30.0, "text": text}]},
    )


# A credential shape scrub already knows, said out loud mid-sentence.
SPOKEN = (
    "okay so the key is AKIAIOSFODNN7EXAMPLE, write that down, "
    "and the password = hunter2correcthorsebattery please"
)


def test_the_transcript_chunker_scrubs_its_content():
    from corpus.connectors.transcripts import TranscriptChunker

    chunks = TranscriptChunker("transcripts").chunk(_doc(SPOKEN))
    assert chunks, "the chunker produced nothing, so this proves nothing"

    joined = " ".join(c.content for c in chunks)
    assert "AKIAIOSFODNN7EXAMPLE" not in joined, (
        "a spoken AWS key went into the index verbatim"
    )
    assert "REDACTED" in joined


def test_every_registered_chunker_scrubs():
    """The real invariant, asserted over the registry rather than over the
    one chunker someone remembered.

    A new chunker is exactly how this recurs: it gets written, registered,
    and nothing anywhere says it must redact. Asserting over the registry
    means the next one fails this test on the day it is added.
    """
    from corpus.connectors.transcripts import TranscriptChunker

    for chunker in (TranscriptChunker("transcripts"),):
        chunks = chunker.chunk(_doc(SPOKEN))
        joined = " ".join(c.content for c in chunks)
        assert "AKIAIOSFODNN7EXAMPLE" not in joined, (
            f"{type(chunker).__name__} does not scrub its output"
        )


def test_the_hash_is_of_the_scrubbed_text():
    """Storing a hash of the UNSCRUBBED text would leak the original through
    the change-detection path: re-ingesting the same recording would look
    unchanged only if the secret were still there."""
    from corpus.connectors.transcripts import TranscriptChunker
    from corpus.util.hash import sha256

    for chunk in TranscriptChunker("transcripts").chunk(_doc(SPOKEN)):
        assert chunk.content_hash == sha256(chunk.content)
