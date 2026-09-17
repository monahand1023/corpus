"""`truncation=True` discards overlong input and returns a normal vector.

The Voyage call passes `truncation=True`, so a chunk past the model's
context is embedded from its FIRST N TOKENS and the response looks
identical to a complete one. Retrieval quality degrades for that chunk --
it is indexed by its opening only -- with no error and no signal anywhere.

Nothing compared the tokens we sent against the tokens Voyage billed, and
that comparison is free: `token_counts` uses Voyage's LOCAL tokenizer, and
the response already carries `total_tokens`. If the server billed
materially fewer tokens than the local tokenizer counted, the difference
is text that was thrown away.

Model-agnostic on purpose: no context-limit constant to hardcode, keep
current, or get wrong for a model someone swaps in.

Only when the local tokenizer is TRUSTWORTHY. It falls back to a char
estimate when unavailable, and comparing a rough estimate against a real
count would cry wolf on every batch -- which is how a warning gets ignored.
"""

from __future__ import annotations

import logging


class _Resp:
    def __init__(self, n_texts: int, total_tokens: int) -> None:
        self.embeddings = [[0.1, 0.2] for _ in range(n_texts)]
        self.total_tokens = total_tokens


def _embedder(billed: int):
    from corpus.embedder.voyage import VoyageEmbedder

    emb = VoyageEmbedder.__new__(VoyageEmbedder)
    emb._model = "voyage-3-large"
    emb._tokenizer_ok = True
    emb.total_tokens_used = 0
    emb._token_window = []
    emb._throttle = lambda *_a, **_k: None
    emb.token_counts = lambda texts: [1000 for _ in texts]

    class _Client:
        def embed(self, **kw):
            return _Resp(len(kw["texts"]), billed)

    emb._client = _Client()
    return emb


def test_truncation_is_reported(caplog):
    emb = _embedder(billed=1200)  # local said 2000

    with caplog.at_level(logging.WARNING):
        emb._embed_batch(["a", "b"], "document")

    assert any("truncat" in r.message.lower() for r in caplog.records), (
        "Voyage billed far fewer tokens than were sent and nothing said so"
    )


def test_a_normal_batch_is_silent(caplog):
    emb = _embedder(billed=2000)

    with caplog.at_level(logging.WARNING):
        emb._embed_batch(["a", "b"], "document")

    assert not [r for r in caplog.records if "truncat" in r.message.lower()]


def test_small_disagreement_does_not_warn(caplog):
    """The local tokenizer and the server differ slightly on normal text.
    Warning on that would fire constantly and be tuned out."""
    emb = _embedder(billed=1990)

    with caplog.at_level(logging.WARNING):
        emb._embed_batch(["a", "b"], "document")

    assert not [r for r in caplog.records if "truncat" in r.message.lower()]


def test_no_warning_when_the_tokenizer_is_untrusted(caplog):
    emb = _embedder(billed=1200)
    emb._tokenizer_ok = False

    with caplog.at_level(logging.WARNING):
        emb._embed_batch(["a", "b"], "document")

    assert not [r for r in caplog.records if "truncat" in r.message.lower()], (
        "compared a char ESTIMATE against a real count and called it truncation"
    )
