"""Smoke tests for UD parsing + stub fixtures (no network, no model)."""

import pytest

from _stub import tiny_acts
from _udparse import parse_conllu


_SAMPLE = """\
# sent_id = 1
# text = The dogs run.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tdogs\tdog\tNOUN\tNNS\tNumber=Plur\t3\tnsubj\t_\t_
3\trun\trun\tVERB\tVBP\t_\t0\troot\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_
"""


@pytest.mark.smoke
def test_parse_conllu_extracts_fields_and_spaceafter():
    """Parse a minimal CoNLL-U block and verify words, upos, number, space_after, text."""
    sents = parse_conllu(_SAMPLE)
    assert len(sents) == 1
    s = sents[0]
    assert s["words"] == ["The", "dogs", "run", "."]
    assert s["upos"] == ["DET", "NOUN", "VERB", "PUNCT"]
    assert s["number"] == ["", "Plur", "", ""]
    assert s["space_after"] == [True, True, False, True]  # "run" has SpaceAfter=No
    assert s["text"] == "The dogs run."


@pytest.mark.smoke
def test_tiny_acts_shapes():
    """Verify tiny_acts returns activation matrices whose row count matches metadata length."""
    acts, meta = tiny_acts()
    assert set(acts) >= {"embedding", "block_0", "block_1"}
    assert acts["embedding"].shape[0] == len(meta["upos"])
