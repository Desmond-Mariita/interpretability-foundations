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


@pytest.mark.smoke
def test_rows_to_table_roundtrip_preserves_empty_strings():
    """Verify rows_to_table roundtrip keeps empty strings (not null) via explicit schema."""
    import importlib

    mod = importlib.import_module("00_data")
    sents = [{"sent_id": "1", "text": "a b", "words": ["a", "b"], "upos": ["DET", "NOUN"],
              "number": ["", "Sing"], "space_after": [True, True]}]
    tbl = mod.rows_to_table(sents)
    df = tbl.to_pandas()
    assert list(df.loc[0, "number"]) == ["", "Sing"]      # empty string survived (not null)
    assert list(df.loc[0, "upos"]) == ["DET", "NOUN"]


@pytest.mark.smoke
def test_align_last_subword_uses_overlap_not_containment():
    """Verify align_words_to_tokens uses overlap predicate so BPE-prefixed tokens are matched."""
    from _models import align_words_to_tokens

    # offsets mimic byte-level BPE: leading space attached to the token (start one before word)
    word_spans = [(0, 5), (6, 11)]              # "hello world", space at idx 5
    token_offsets = [(0, 5), (5, 11)]            # token 1 = "hello", token 2 = " world"
    # last overlapping token for each word: word0 -> tok0; word1 -> tok1 (overlap, not containment)
    assert align_words_to_tokens(word_spans, token_offsets) == [0, 1]
