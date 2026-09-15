"""The re-ranker is an optional extra, so its absence must explain itself.

~2GB of torch and sentence-transformers is deliberately opt-in, which means a
consumer that never asked for it WILL hit this path. Before, it surfaced as
ModuleNotFoundError from the middle of an eval run, five frames deep, with no
indication that an extra existed or what it was called.
"""

from __future__ import annotations

import builtins

import pytest

from corpus.reranker.local import BGEReranker, RerankerUnavailableError


def test_a_missing_extra_names_the_extra_and_how_to_install_it(monkeypatch) -> None:
    real_import = builtins.__import__

    def without_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sentence_transformers)

    with pytest.raises(RerankerUnavailableError) as caught:
        BGEReranker()._ensure_loaded()

    message = str(caught.value)
    assert "reranker" in message, "must name the extra"
    assert "corpus-rag[reranker]" in message, "must give the install command"


def test_the_error_is_an_importerror_so_existing_handlers_still_catch_it() -> None:
    # Callers that already catch ImportError around an optional dependency
    # should keep working; this only adds a better message.
    assert issubclass(RerankerUnavailableError, ImportError)
