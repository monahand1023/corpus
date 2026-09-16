"""A transcription sign-off only got stripped by the TRANSCRIPT connector.

`strip_caption_tail` removes "Transcribed by https://otter.ai", "Thanks for
watching", "Subscribe to my channel" and the rest of a closed vocabulary from
the end of transcribed text. It was applied in exactly one place --
`connectors/transcripts.py` -- so the cleanup was attached to the CONNECTOR
rather than to the CONTENT.

Transcribed audio does not only arrive through that connector. A voicemail
transcript inside an email, a meeting transcript pasted into a note, an
Otter export saved as a .docx: all go through `MarkdownChunker`, and all kept
their sign-off. `corpus-survey index-quality` detected them across every
source type and then told the operator to go and edit a connector by hand --
which is the shape of fix this project exists to stop making.

Found on a live mail archive: one chunk ending
`"...Okay, bye, bye. Sounds good. Bye.\\n\\nTranscribed by https://otter.ai"`.
One chunk is a small number. The mechanism is not: it scales with however
much transcribed material an archive holds outside the transcript connector.
"""

from __future__ import annotations

from corpus.connectors.markdown import MarkdownChunker
from corpus.types import SourceDocument


def _chunk_one(body: str, title: str = "Voicemail"):
    doc = SourceDocument(
        source_type="notes",
        source_key="k",
        title=title,
        url=None,
        raw={"body": body},
    )
    return MarkdownChunker(source_type="notes").chunk(doc)


def test_a_transcription_credit_is_stripped_whatever_connector_carried_it():
    chunks = _chunk_one(
        "All right, take care. Look for Lincoln. We'll be in touch. "
        "Okay, bye, bye. Sounds good. Bye.\n\nTranscribed by https://otter.ai"
    )
    assert chunks
    joined = "\n".join(c.content for c in chunks)
    assert "otter.ai" not in joined
    assert "Look for Lincoln" in joined, "the real speech went with it"


def test_a_subscribe_signoff_is_stripped():
    chunks = _chunk_one(
        "The meeting agreed to move the launch to March.\n\n"
        "Thanks for watching! Please subscribe to my channel."
    )
    joined = "\n".join(c.content for c in chunks)
    assert "subscribe" not in joined.lower()
    assert "move the launch to March" in joined


def test_ordinary_prose_is_untouched():
    """The strip is tail-only and its vocabulary is closed, but a document
    that merely discusses captioning must survive intact."""
    body = (
        "# Notes on captioning\n\n"
        "Otter and Whisper both append a credit line, which we strip at "
        "index time. See the transcript quality doc for the full list.\n\n"
        "The next review is on Tuesday."
    )
    joined = "\n".join(c.content for c in _chunk_one(body, title="Captioning"))
    assert "The next review is on Tuesday." in joined
    assert "Otter and Whisper both append a credit line" in joined


def test_a_document_that_is_nothing_but_a_signoff_yields_no_chunk():
    """Indexing a chunk whose entire content was boilerplate gives search a
    result that answers nothing."""
    assert _chunk_one("Thanks for watching!", title="Clip") == []
