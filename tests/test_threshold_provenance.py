"""A threshold file records numbers and not what they were measured from.

one consumer's floors were measured against an 8-query, transcripts-only gold
set. The set grew to 24 and the floors stayed, so the recall gate ended up
six queries below its measurement -- a gate that cannot fail. Nothing was
wrong with the floors when they were written; nothing connected them to the
thing they described, so nobody could see they had gone stale.

The drift is always in the same direction. A gold set grows, the
measurement rises, the floor stays -- and a gate quietly stops gating. It
never drifts toward failing, which is precisely why nobody notices.

So the file may record what it was measured against, and `--check` compares
that to the gold set in front of it. Optional: an unannotated file keeps
working exactly as before, because a hard requirement here would break every
existing thresholds file and get the feature reverted.
"""

from __future__ import annotations

import json

import pytest


def test_a_thresholds_file_may_record_the_gold_set_it_came_from(tmp_path):
    from corpus.cli.eval import _load_thresholds

    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "recall_at_k": 0.708,
        "measured": {"n_queries": 24, "date": "2026-09-16"},
    }))

    thresholds = _load_thresholds(path)
    assert thresholds == {"recall_at_k": 0.708}, (
        "provenance leaked into the metric dict and would be gated on"
    )


def test_provenance_is_readable_separately(tmp_path):
    from corpus.cli.eval import _load_threshold_provenance

    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "recall_at_k": 0.708,
        "measured": {"n_queries": 24, "date": "2026-09-16"},
    }))
    assert _load_threshold_provenance(path) == {"n_queries": 24, "date": "2026-09-16"}


def test_an_unannotated_file_still_loads(tmp_path):
    """Every thresholds file in existence is unannotated. Requiring the key
    would break them all, and a feature that breaks the working case gets
    reverted rather than adopted."""
    from corpus.cli.eval import _load_threshold_provenance, _load_thresholds

    path = tmp_path / "t.json"
    path.write_text(json.dumps({"recall_at_k": 0.5, "mrr": 0.4}))

    assert _load_thresholds(path) == {"recall_at_k": 0.5, "mrr": 0.4}
    assert _load_threshold_provenance(path) == {}


def test_a_gold_set_that_grew_since_the_floors_were_set_is_reported(capsys):
    from corpus.cli.eval import _warn_on_stale_thresholds

    _warn_on_stale_thresholds({"n_queries": 8, "date": "2026-09-15"}, n_queries=24)
    out = capsys.readouterr().out

    assert "8" in out and "24" in out, out
    assert "grew" in out.lower() or "changed" in out.lower(), out


def test_a_matching_gold_set_says_nothing(capsys):
    from corpus.cli.eval import _warn_on_stale_thresholds

    _warn_on_stale_thresholds({"n_queries": 24}, n_queries=24)
    assert capsys.readouterr().out == ""


def test_an_unannotated_file_says_nothing(capsys):
    """Silence, not a nag. Most files have no provenance and telling their
    owners off on every run is how a warning stops being read."""
    from corpus.cli.eval import _warn_on_stale_thresholds

    _warn_on_stale_thresholds({}, n_queries=24)
    assert capsys.readouterr().out == ""


def test_a_malformed_provenance_block_is_rejected(tmp_path):
    from corpus.cli.eval import _load_thresholds

    path = tmp_path / "t.json"
    path.write_text(json.dumps({"recall_at_k": 0.5, "measured": 24}))
    with pytest.raises(ValueError, match="measured"):
        _load_thresholds(path)
