"""Smoke tests for the v1.1 causal stimulus/eval pipeline (no network, no model).

Covers the brief's required tests 10-12, 14, 15: subject token alignment, split
disjointness, deterministic stimulus generation, snapshot schema, and
report/snapshot consistency -- plus the competence-gate and donor-scoring wiring.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from awake.eval.causal import canonical_stimulus_key, stimulus_hash, table_hash


def _mod(name):
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# 10. subject token index / alignment
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_subject_last_subword_position_in_prompt():
    """Subject in the middle of a prompt: last overlapping subword is the subject token."""
    from _models import align_words_to_tokens

    # "The dogs near the park" tokenized at byte level: leading spaces attach to tokens
    prompt = "The dogs near the park"
    subject_start = prompt.index("dogs")
    spans = [(subject_start, subject_start + len("dogs"))]
    # offsets mimic byte-level BPE ("The"|0-3, " dogs"|3-8, " near"|8-13, ...)
    offsets = [(0, 3), (3, 8), (8, 13), (13, 17), (17, 22)]
    assert align_words_to_tokens(spans, offsets) == [1]
    # the subject span itself is exactly one token: subject_first == subject_last
    tokenize = _mod("40_stimuli").tokenize_stimulus

    class FakeTok:
        def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
            return {"input_ids": [1, 2, 3, 4, 5], "offset_mapping": offsets}

    out = tokenize(FakeTok(), prompt, "dogs")
    assert out["subject_first"] == 1
    assert out["subject_last"] == 1
    assert out["subject_token_count"] == 1
    assert out["final_token_pos"] == 4


# ---------------------------------------------------------------------------
# 11. lemma/template split disjointness + template-family holdout
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_splits_are_lemma_disjoint_and_attractor_held_out():
    """Dev/test lemmas disjoint; pilot excluded; attractor family appears only in test."""
    mod = _mod("40_stimuli")
    lex = [
        {
            "lemma": f"lemma{i:03d}",
            "sing_surface": f"noun{i:03d}",
            "plur_surface": f"noun{i:03d}s",
            "sing_freq": 5,
            "plur_freq": 5,
            "stratum": "suffix_transparent",
        }
        for i in range(20)
    ]
    cfg = {"pilot_n": 3, "dev_frac": 0.7, "max_dev_lemmas": 10, "max_test_lemmas": 7, "seed": 0}
    splits = mod.assign_splits(lex, cfg)
    assert mod.assert_splits_disjoint(splits)
    assert len(splits["pilot"]) == 3
    assert len(splits["dev"]) == 10
    assert len(splits["test"]) == 7
    rows = mod.build_stimulus_rows(
        splits,
        templates={
            "simple": "The {subject}",
            "near": "The {subject} near the park",
            "attractor": "The {subject} near the {attractor}",
        },
        families={"dev": ["simple", "near"], "test": ["simple", "near", "attractor"]},
        attractors={"park": "Sing", "parks": "Plur"},
    )
    dev_templates = set(rows.loc[rows["split"] == "dev", "template_id"])
    test_templates = set(rows.loc[rows["split"] == "test", "template_id"])
    assert dev_templates == {"simple", "near"}
    assert "attractor" in test_templates
    assert "attractor" not in dev_templates


# ---------------------------------------------------------------------------
# 12. deterministic stimulus generation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_stimulus_generation_is_deterministic_and_hashed():
    """Two identical builds produce identical stim ids and table hash; donors matched."""
    mod = _mod("40_stimuli")
    lex = [
        {
            "lemma": "dog",
            "sing_surface": "dog",
            "plur_surface": "dogs",
            "sing_freq": 9,
            "plur_freq": 9,
            "stratum": "suffix_transparent",
        },
        {
            "lemma": "child",
            "sing_surface": "child",
            "plur_surface": "children",
            "sing_freq": 8,
            "plur_freq": 8,
            "stratum": "nontransparent",
        },
    ]
    splits = {"pilot": [], "dev": lex, "test": []}
    templates = {
        "simple": "The {subject}",
        "near": "The {subject} near the park",
        "attractor": "The {subject} near the {attractor}",
    }
    families = {"dev": ["simple", "near"], "test": ["simple", "near", "attractor"]}
    attractors = {"park": "Sing", "parks": "Plur"}

    def build():
        return mod.build_stimulus_rows(splits, templates, families, attractors)

    r1, r2 = build(), build()
    assert list(r1["stim_id"]) == list(r2["stim_id"])
    keys = [
        canonical_stimulus_key(
            split=r["split"],
            lemma=r["lemma"],
            subject_surface=r["subject_surface"],
            subject_number=r["subject_number"],
            template_id=r["template_id"],
            attractor_surface=r["attractor_surface"],
            attractor_number=1
            if r["attractor_number_name"] == "Plur"
            else (-1 if r["attractor_number_name"] == "Sing" else 0),
            prompt_text=r["prompt_text"],
        )
        for _, r in r1.iterrows()
    ]
    assert table_hash(keys) == table_hash(keys)
    assert stimulus_hash(keys[0]) == r1.iloc[0]["stim_id"]

    # donors: opposite-number donor is the matched counterpart
    d = mod.assign_donors(r1, pairing_seed=0)
    for _, r in d.iterrows():
        assert r["donor_stim_id"] != r["stim_id"]
        donor = d.loc[d["stim_id"] == r["donor_stim_id"]].iloc[0]
        assert donor["subject_number"] == -r["subject_number"]
        assert donor["lemma"] == r["lemma"]
        assert donor["template_id"] == r["template_id"]
        # C1 same-number donor: same number, same template, different lemma
        c1 = d.loc[d["stim_id"] == r["donor_c1_stim_id"]].iloc[0]
        assert c1["subject_number"] == r["subject_number"]
        assert c1["lemma"] != r["lemma"]


# ---------------------------------------------------------------------------
# competence gate + donor scoring wiring (80_causal_eval)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_gate_passes_for_strong_agreement_and_fails_for_none():
    """Strong plural->M signal passes both gates; a null model fails them."""
    mod = _mod("80_causal_eval")
    lemmas = [f"l{i}" for i in range(10)]

    def items_with_shift(shift: float) -> pd.DataFrame:
        rows = []
        for lemma in lemmas:
            for verb in ("is", "was"):
                for num, y in ((-1, 0.0), (1, shift)):
                    for template in ("simple", "near"):
                        rows.append(
                            {
                                "lemma": lemma,
                                "verb_sing": verb,
                                "verb_plur": "are" if verb == "is" else "were",
                                "template_id": template,
                                "subject_number": num,
                                "m": y,
                            }
                        )
        return pd.DataFrame(rows)

    gate_pass = mod.gate_statistics(items_with_shift(2.0), n_resamples=200, seed=0)
    assert gate_pass["passed"]
    assert gate_pass["gate1"]["pass"] and gate_pass["gate2"]["pass"]
    gate_fail = mod.gate_statistics(items_with_shift(0.0), n_resamples=200, seed=0)
    assert not gate_fail["passed"]


@pytest.mark.smoke
def test_scoring_label_convention_and_build_items():
    """Donor-scoring convention: number/full use donor label; same_number uses recipient's.

    Random conditions use the opposite of the recipient's (the specificity yardstick).
    """
    mod = _mod("80_causal_eval")
    assert mod.scoring_label("number", +1, -1) == -1
    assert mod.scoring_label("full_residual", -1, +1) == +1
    assert mod.scoring_label("same_number", +1, +1) == +1
    assert mod.scoring_label("random:0", +1, None) == -1
    assert mod.scoring_label("random:1", -1, None) == +1
    with pytest.raises(ValueError):
        mod.scoring_label("number", +1, None)

    stimuli = pd.DataFrame(
        [
            {
                "stim_id": "s1",
                "lemma": "dog",
                "stratum": "suffix_transparent",
                "template_id": "simple",
                "attractor_relation": "na",
                "subject_number": -1,
            },
            {
                "stim_id": "s2",
                "lemma": "dog",
                "stratum": "suffix_transparent",
                "template_id": "simple",
                "attractor_relation": "na",
                "subject_number": +1,
            },
        ]
    )
    baseline = pd.DataFrame(
        [
            {
                "stim_id": "s1",
                "logit_is": 5.0,
                "logit_are": 2.0,
                "logit_was": 4.0,
                "logit_were": 3.0,
            },
            {
                "stim_id": "s2",
                "logit_is": 1.0,
                "logit_are": 6.0,
                "logit_was": 2.0,
                "logit_were": 5.0,
            },
        ]
    )
    interv = pd.DataFrame(
        [
            {
                "stim_id": "s1",
                "point": "embedding",
                "condition": "number",
                "donor_stim_id": "s2",
                "norm_delta": 1.0,
                "logit_is": 4.0,
                "logit_are": 3.0,
                "logit_was": 4.0,
                "logit_were": 3.0,
            },
        ]
    )
    items = mod.build_items(stimuli, baseline, interv, [["is", "are"], ["was", "were"]])
    assert len(items) == 2
    row_is = items[(items["verb_sing"] == "is")].iloc[0]
    # recipient singular, donor plural: M_baseline = -3 (is-are), M_patch = -1 -> shift +2
    assert row_is["m_baseline"] == pytest.approx(-3.0)
    assert row_is["m_patch"] == pytest.approx(-1.0)
    assert row_is["e"] == pytest.approx(2.0)  # y_d = +1 (plural donor)


# ---------------------------------------------------------------------------
# 14/15. snapshot schema + report/snapshot consistency
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_snapshot_schema_and_report_consistency():
    """causal_metrics.json schema is validated; report numbers must match the snapshot."""
    mod = _mod("80_causal_eval")
    snap = {k: {} for k in mod.SNAPSHOT_KEYS}
    snap["model"] = "EleutherAI/pythia-160m"
    snap["primary"] = {"points": {"embedding": {"mean_e": 0.12, "ci": [0.02, 0.22]}}}
    assert mod.validate_snapshot_schema(snap)
    assert not mod.validate_snapshot_schema({})
    # consistent report numbers pass
    assert mod.assert_report_snapshot_consistency({"primary.points.embedding.mean_e": 0.12}, snap)
    # a drift between report and snapshot fails
    assert not mod.assert_report_snapshot_consistency(
        {"primary.points.embedding.mean_e": 0.99}, snap
    )
    # a missing path fails
    assert not mod.assert_report_snapshot_consistency({"primary.points.block_0.mean_e": 0.12}, snap)


# ---------------------------------------------------------------------------
# 13. lemma-cluster bootstrap smoke (project-level wiring)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_per_point_effect_bootstrap_shape():
    """per_point_effect returns per-point means, CIs, and item counts."""
    mod = _mod("80_causal_eval")
    rng = np.random.default_rng(0)
    rows = []
    for g in range(8):
        for _ in range(3):
            rows.append(
                {
                    "lemma": f"l{g}",
                    "point": "embedding",
                    "e": 0.5 + 0.1 * rng.standard_normal(),
                }
            )
    out = mod.per_point_effect(pd.DataFrame(rows), n_resamples=100, seed=0, points=["embedding"])
    assert set(out) == {"embedding"}
    pt = out["embedding"]
    assert pt["n_items"] == 24
    assert pt["ci"][0] <= pt["mean_e"] <= pt["ci"][1]
