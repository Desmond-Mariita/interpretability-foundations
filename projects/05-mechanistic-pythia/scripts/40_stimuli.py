"""Build the v1.1 controlled subject-verb agreement stimulus set (deterministic, versioned).

Pure parts (lexicon rules, split assignment, row construction, donor pairing) are unit/smoke
tested; ``main`` wires the UD train conllu + the pinned tokenizer and writes
``outputs/stimuli/`` (stimuli parquet, tokenizer validation, split manifest). See ADR 006
Decision 4 and ``configs/causal.yaml``.
"""

from __future__ import annotations

import json
import re

import pandas as pd
from _models import align_words_to_tokens
from _paths import OUTPUTS, ensure_dirs, load_config
from _udparse import parse_conllu_lexicon

from awake.eval.causal import canonical_stimulus_key, stimulus_hash, table_hash

STIM_DIR = OUTPUTS / "stimuli"


# ---------------------------------------------------------------------------
# Pure: noun lexicon from UD train tokens
# ---------------------------------------------------------------------------


def _choose_surface(surface_counts: dict[str, int], lemma: str) -> str:
    """Most frequent surface; tie-break: prefer surface == lemma, then lexicographic."""
    ranked = sorted(
        surface_counts.items(),
        key=lambda kv: (-kv[1], kv[0] != lemma, kv[0]),
    )
    return ranked[0][0]


def build_lexicon(tokens: list[dict], min_surface_freq: int, surface_regex: str) -> list[dict]:
    """Derive noun lemma pairs (singular + plural surface) from per-token UD annotations.

    Per lemma, from NOUN tokens with ``Number=Sing`` / ``Number=Plur``:
    letter-only lowercase surfaces matching ``surface_regex`` are collected; the most
    frequent surface per number is chosen (tie-break: surface == lemma, then lexicographic);
    both chosen surfaces must reach ``min_surface_freq``; degenerate pairs
    (plural == singular, e.g. zero-plurals) are excluded. Stratum: ``suffix_transparent``
    iff ``plural == singular + 's'`` or ``plural == singular + 'es'``, else
    ``nontransparent`` (a surface heuristic, not a linguistic morphology claim).

    Args:
        tokens: Per-token dicts with ``lemma``, ``surface``, ``upos``, ``number``.
        min_surface_freq: Minimum train frequency for each chosen surface.
        surface_regex: Regex that surfaces must fully match.

    Returns:
        List of ``{lemma, sing_surface, plur_surface, sing_freq, plur_freq, stratum}``
        sorted by lemma.
    """
    pattern = re.compile(surface_regex)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for t in tokens:
        if t["upos"] != "NOUN" or t["number"] not in ("Sing", "Plur"):
            continue
        if not pattern.fullmatch(t["surface"]):
            continue
        key = t["number"].lower()  # sing / plur
        entry = counts.setdefault(t["lemma"], {"sing": {}, "plur": {}})
        entry[key][t["surface"]] = entry[key].get(t["surface"], 0) + 1

    out = []
    for lemma, by_num in counts.items():
        if not by_num["sing"] or not by_num["plur"]:
            continue
        sing = _choose_surface(by_num["sing"], lemma)
        plur = _choose_surface(by_num["plur"], lemma)
        if sing == plur:
            continue
        if by_num["sing"][sing] < min_surface_freq or by_num["plur"][plur] < min_surface_freq:
            continue
        stratum = (
            "suffix_transparent" if plur == sing + "s" or plur == sing + "es" else "nontransparent"
        )
        out.append(
            {
                "lemma": lemma,
                "sing_surface": sing,
                "plur_surface": plur,
                "sing_freq": by_num["sing"][sing],
                "plur_freq": by_num["plur"][plur],
                "stratum": stratum,
            }
        )
    return sorted(out, key=lambda r: r["lemma"])


# ---------------------------------------------------------------------------
# Pure: splits
# ---------------------------------------------------------------------------


def assign_splits(lexicon: list[dict], cfg: dict) -> dict[str, list[dict]]:
    """Assign lemmas to pilot/dev/test (lemma-disjoint; seeded over the sorted lemma list).

    Pilot = first ``pilot_n`` lemmas (plumbing only). The remainder is shuffled with
    ``seed``; dev takes ``dev_frac`` capped at ``max_dev_lemmas``; test takes the rest
    capped at ``max_test_lemmas``. Returns ``{split: [lexicon rows]}``.
    """
    import numpy as np

    pilot = lexicon[: cfg["pilot_n"]]
    rest = lexicon[cfg["pilot_n"] :]
    order = np.random.default_rng(cfg["seed"]).permutation(len(rest)).tolist()
    shuffled = [rest[i] for i in order]
    n_dev = min(cfg["max_dev_lemmas"], round(cfg["dev_frac"] * len(shuffled)))
    n_test = min(cfg["max_test_lemmas"], len(shuffled) - n_dev)
    return {"pilot": pilot, "dev": shuffled[:n_dev], "test": shuffled[n_dev : n_dev + n_test]}


def assert_splits_disjoint(split_rows: dict[str, list[dict]]) -> bool:
    """True iff the lemma sets of the three splits are pairwise disjoint."""
    sets = {k: {r["lemma"] for r in rows} for k, rows in split_rows.items()}
    return not (
        sets["pilot"] & sets["dev"] or sets["pilot"] & sets["test"] or sets["dev"] & sets["test"]
    )


# ---------------------------------------------------------------------------
# Pure: stimulus rows (pre-tokenizer)
# ---------------------------------------------------------------------------


def build_stimulus_rows(
    split_rows: dict[str, list[dict]],
    templates: dict[str, str],
    families: dict[str, list[str]],
    attractors: dict[str, str],
) -> pd.DataFrame:
    """Expand lexicon x templates x attractors x subject number into stimulus rows.

    Columns: split, lemma, stratum, subject_surface, subject_number (+1 Plur / -1 Sing),
    subject_number_name, template_id, attractor_surface, attractor_number_name
    (Sing/Plur/""), attractor_relation (same/opposite/na), prompt_text, pair_id, stim_id
    (SHA-256 over the canonical key). Rows are sorted for a stable reading order; the
    canonical key and hashes do not depend on row order.
    """
    rows = []
    all_families = sorted(set(families["dev"]) | set(families["test"]))
    for split, lex_rows in split_rows.items():
        for lex in lex_rows:
            # pilot covers every template family (plumbing validation); dev/test per config
            for template_id in families.get(split, all_families):
                template = templates[template_id]
                attractor_surfaces = list(attractors) if "{attractor}" in template else [""]
                for attr_surface in attractor_surfaces:
                    attr_number = attractors.get(attr_surface, "")
                    for name, label in (("Sing", -1), ("Plur", +1)):
                        surface = lex["sing_surface"] if name == "Sing" else lex["plur_surface"]
                        prompt = template.replace("{subject}", surface).replace(
                            "{attractor}", attr_surface
                        )
                        relation = "na"
                        if attr_surface:
                            relation = "same" if attr_number == name else "opposite"
                        pair_id = f"{split}|{lex['lemma']}|{template_id}|{attr_surface}"
                        key = canonical_stimulus_key(
                            split=split,
                            lemma=lex["lemma"],
                            subject_surface=surface,
                            subject_number=label,
                            template_id=template_id,
                            attractor_surface=attr_surface,
                            attractor_number=1
                            if attr_number == "Plur"
                            else (-1 if attr_number == "Sing" else 0),
                            prompt_text=prompt,
                        )
                        rows.append(
                            {
                                "split": split,
                                "lemma": lex["lemma"],
                                "stratum": lex["stratum"],
                                "subject_surface": surface,
                                "subject_number": label,
                                "subject_number_name": name,
                                "template_id": template_id,
                                "attractor_surface": attr_surface,
                                "attractor_number_name": attr_number,
                                "attractor_relation": relation,
                                "prompt_text": prompt,
                                "pair_id": pair_id,
                                "stim_id": stimulus_hash(key),
                            }
                        )
    df = pd.DataFrame(rows)
    order = ["split", "lemma", "template_id", "attractor_surface", "subject_number_name"]
    return df.sort_values(order).reset_index(drop=True)


def assign_donors(rows: pd.DataFrame, pairing_seed: int) -> pd.DataFrame:
    """Add ``donor_stim_id`` (opposite-number counterpart) and ``donor_c1_stim_id``.

    Opposite-number donors come from the matched singular/plural pair (same pair_id).
    C1 same-number donors use deterministic cyclic pairing within each
    (split, template_id, attractor_surface, subject_number) group ordered by stim_id
    (pairing_seed selects a cyclic offset). A one-member group degrades to self-donation
    (a no-op), flagged via ``c1_degenerate``.
    """
    rows = rows.copy()
    by_pair: dict[str, str] = {}
    for _, r in rows.iterrows():
        by_pair[(r["pair_id"], r["subject_number"])] = r["stim_id"]
    donors = [by_pair[(r["pair_id"], -r["subject_number"])] for _, r in rows.iterrows()]
    rows["donor_stim_id"] = donors

    c1: dict[str, str] = {}
    degenerate: dict[str, bool] = {}
    for group_index, (_group_key, grp) in enumerate(
        rows.groupby(["split", "template_id", "attractor_surface", "subject_number"], sort=False)
    ):
        ids = grp.sort_values("stim_id")["stim_id"].tolist()
        # deterministic cyclic offset (1..n-1); no self-donation for n >= 2
        offset = (group_index + pairing_seed) % (len(ids) - 1) + 1 if len(ids) >= 2 else 0
        for i, sid in enumerate(ids):
            c1[sid] = ids[(i + offset) % len(ids)]
        degenerate.update({sid: len(ids) < 2 for sid in ids})
    rows["donor_c1_stim_id"] = [c1[s] for s in rows["stim_id"]]
    rows["c1_degenerate"] = [degenerate[s] for s in rows["stim_id"]]
    return rows


# ---------------------------------------------------------------------------
# Slow path helpers (tokenizer-dependent)
# ---------------------------------------------------------------------------


def validate_verb_pairs(tok, verb_pairs: list[list[str]]) -> dict:
    """Check each verb form is exactly one tokenizer token (leading-space convention).

    Returns ``{pairs: [{"pair": [sing, plur], "ok": bool, "tokens": {form: [ids]}}],
    "kept_pairs": [...]}``.
    """
    out = {"pairs": [], "kept_pairs": []}
    for sing, plur in verb_pairs:
        info = {"pair": [sing, plur], "ok": True, "tokens": {}}
        for form in (sing, plur):
            ids = tok.encode(" " + form, add_special_tokens=False)
            info["tokens"][form] = ids
            if len(ids) != 1:
                info["ok"] = False
        out["pairs"].append(info)
        if info["ok"]:
            out["kept_pairs"].append([sing, plur])
    return out


def tokenize_stimulus(tok, prompt_text: str, subject_surface: str) -> dict:
    """Tokenize one stimulus and locate the subject's token span.

    The subject span is the first occurrence of ``subject_surface`` in the prompt (the
    template's subject slot); the intervention target is its LAST overlapping subword
    (v1.0 alignment convention).

    Returns:
        Dict with token ids, subject_first, subject_last, subject_token_count,
        prompt_token_count, final_token_pos, aligned (bool).
    """
    enc = tok(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    start = prompt_text.index(subject_surface)
    span = [(start, start + len(subject_surface))]
    matched = align_words_to_tokens(span, [tuple(o) for o in offsets])
    last = matched[0]
    first = next(
        (
            i
            for i, (ts, te) in enumerate(offsets)
            if te > start and ts < start + len(subject_surface)
        ),
        None,
    )
    return {
        "input_ids": ids,
        "subject_first": first,
        "subject_last": last,
        "subject_token_count": (last - first + 1)
        if (last is not None and first is not None)
        else 0,
        "prompt_token_count": len(ids),
        "final_token_pos": len(ids) - 1,
        "aligned": last is not None,
    }


def main() -> None:  # pragma: no cover - slow path (network-free; tokenizer cached)
    """Build lexicon + splits + rows, tokenize, validate verbs, write outputs/stimuli/."""
    import hashlib

    from _models import load_tokenizer
    from _paths import CONLLU

    cfg = load_config("causal")
    ensure_dirs(STIM_DIR)

    tokens = parse_conllu_lexicon((CONLLU / "en_ewt-ud-train.conllu").read_text(encoding="utf-8"))
    lexicon = build_lexicon(
        tokens, cfg["lexicon"]["min_surface_freq"], cfg["lexicon"]["surface_regex"]
    )
    splits = assign_splits(lexicon, cfg["splits"])
    assert assert_splits_disjoint(splits)

    rows = build_stimulus_rows(splits, cfg["templates"], cfg["families"], cfg["attractors"])
    rows = assign_donors(rows, cfg["intervention"]["donor_pairing_seed"])

    tok = load_tokenizer(cfg["model_id"], cfg["model_revision"])
    verb_val = validate_verb_pairs(tok, cfg["verb_pairs"])

    tok_cols = [
        tokenize_stimulus(tok, r["prompt_text"], r["subject_surface"]) for _, r in rows.iterrows()
    ]
    for col in (
        "input_ids",
        "subject_first",
        "subject_last",
        "subject_token_count",
        "prompt_token_count",
        "final_token_pos",
        "aligned",
    ):
        rows[col] = [c[col] for c in tok_cols]
    if not rows["aligned"].all():
        bad = rows.loc[~rows["aligned"], ["split", "lemma", "prompt_text"]]
        raise ValueError(f"subject alignment failed for rows:\n{bad}")

    # C3 eligibility: pairs whose sing/plur variants have equal subject token counts/positions
    pair_counts = rows.groupby("pair_id")["subject_token_count"].nunique()
    eligible = {p for p in pair_counts.index if pair_counts[p] == 1}
    rows["c3_eligible"] = rows["pair_id"].isin(eligible)

    rows.to_parquet(STIM_DIR / "stimuli.parquet", index=False)
    (STIM_DIR / "tokenizer_validation.json").write_text(json.dumps(verb_val, indent=2))

    manifest = {
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "ud_tag": cfg["ud_tag"],
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "lexicon": {
            "n_lemmas": len(lexicon),
            "by_stratum": {
                s: sum(1 for r in lexicon if r["stratum"] == s)
                for s in sorted({r["stratum"] for r in lexicon})
            },
        },
        "splits": {
            s: {"n_lemmas": len(rows_s), "lemmas": [r["lemma"] for r in rows_s]}
            for s, rows_s in splits.items()
        },
        "stimuli": {
            "n_rows": len(rows),
            "n_by_split": rows.groupby("split").size().to_dict(),
            "table_sha256": "",  # computed below from the canonical keys
            "row_hash_sha256": hashlib.sha256(
                "\n".join(rows["stim_id"].tolist()).encode("utf-8")
            ).hexdigest(),
        },
        "verb_pairs": {
            "kept": verb_val["kept_pairs"],
            "excluded": [p["pair"] for p in verb_val["pairs"] if not p["ok"]],
        },
    }
    # recompute the canonical table hash from the stored keys
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
        for _, r in rows.iterrows()
    ]
    manifest["stimuli"]["table_sha256"] = table_hash(keys)
    (STIM_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"lexicon: {len(lexicon)} lemmas "
        f"({manifest['lexicon']['by_stratum']}); rows: {len(rows)} "
        f"{rows.groupby('split').size().to_dict()}"
    )
    print(f"verb pairs kept: {verb_val['kept_pairs']}")
    print(f"verb pairs excluded: {[p['pair'] for p in verb_val['pairs'] if not p['ok']]}")
    print(f"table sha256: {manifest['stimuli']['table_sha256']}")


if __name__ == "__main__":  # pragma: no cover
    main()
