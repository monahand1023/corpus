"""recall@k moved MORE than the rank-weighted metrics, not less.

`corpus-eval --repeat` printed this under its own numbers:

    "result ORDER moves while set membership usually does not -- which is
     why recall@k sits still and the rank-weighted metrics do not."

Measured on a real archive, 24 queries, three identical runs:

    recall@5   0.764  min 0.750  max 0.792  spread 0.042   <- LARGEST
    mrr        0.581  min 0.580  max 0.585  spread 0.005
    ndcg@5     0.604  min 0.599  max 0.613  spread 0.013

The premise is wrong, and the tool was printing it as the explanation of
the numbers directly above it -- which is worse than printing nothing,
because it tells the reader to trust the metric that moved most.

WHY IT IS WRONG. Order changes only matter to set membership when a hit
sits on the k BOUNDARY, and then they matter completely: rank 5 to rank 6
flips that query from 1 to 0. Observed directly -- one query's first four
results were byte-identical across two runs and the fifth changed, taking
a hit out of the top 5.

And recall@k is QUANTISED to 1/n, so a single boundary flip moves the
whole metric by 1/n -- here 1/24 = 0.042, which is exactly the spread
measured. MRR and nDCG absorb the same flip as a small continuous change,
which is why they move LESS. Quantisation does not make a metric stable;
it makes its noise arrive in one lump.
"""

from __future__ import annotations


def test_the_spread_note_does_not_claim_recall_is_stable(capsys):
    from corpus.cli.eval import _print_spread
    from corpus.eval.noise import MetricSpread

    spreads = {
        "recall_at_k": MetricSpread("recall_at_k", [0.750, 0.792, 0.750]),
        "mrr": MetricSpread("mrr", [0.580, 0.585, 0.580]),
        "ndcg_at_k": MetricSpread("ndcg_at_k", [0.599, 0.613, 0.599]),
    }
    _print_spread(spreads, 5)
    out = capsys.readouterr().out

    assert "sits still" not in out, (
        "still claiming recall@k does not move, under numbers showing it "
        f"moving most:\n{out}"
    )
    assert "boundary" in out.lower(), (
        f"the note does not explain WHEN order changes membership:\n{out}"
    )


def test_the_note_names_the_metric_that_actually_moved_most(capsys):
    """Generic prose about noise is not usable. The reader needs to know
    which of the three numbers on screen to distrust, and that differs by
    archive -- so it has to be computed, not written down."""
    from corpus.cli.eval import _print_spread
    from corpus.eval.noise import MetricSpread

    spreads = {
        "recall_at_k": MetricSpread("recall_at_k", [0.750, 0.792, 0.750]),
        "mrr": MetricSpread("mrr", [0.580, 0.585, 0.580]),
        "ndcg_at_k": MetricSpread("ndcg_at_k", [0.599, 0.613, 0.599]),
    }
    _print_spread(spreads, 5)
    out = capsys.readouterr().out

    assert "recall@5" in out
    assert "moved most" in out or "widest" in out, out


def test_a_stable_recall_is_reported_as_such(capsys):
    """The opposite archive must still read correctly: where recall really
    does sit still, saying it moved most would be the same error mirrored."""
    from corpus.cli.eval import _print_spread
    from corpus.eval.noise import MetricSpread

    spreads = {
        "recall_at_k": MetricSpread("recall_at_k", [0.700, 0.700, 0.700]),
        "mrr": MetricSpread("mrr", [0.500, 0.540, 0.520]),
        "ndcg_at_k": MetricSpread("ndcg_at_k", [0.560, 0.590, 0.575]),
    }
    _print_spread(spreads, 5)
    out = capsys.readouterr().out

    assert "mrr" in out.lower()
    assert "recall@5 moved most" not in out, out
