"""Pure CoNLL-U parsing for UD English-EWT (no network, no I/O)."""

from __future__ import annotations


def parse_conllu(text: str) -> list[dict]:
    """Parse CoNLL-U text into per-sentence dicts.

    Skips comments (except ``# text =``) and multiword/empty rows (ID containing ``-``/``.``).

    Returns:
        List of ``{sent_id, text, words, upos, number, space_after}``. ``number`` is the
        ``Number=`` FEAT value (``""`` if absent); ``space_after`` is ``False`` iff MISC has
        ``SpaceAfter=No``.
    """
    sents: list[dict] = []
    cur = _blank()
    for line in text.splitlines():
        if not line.strip():
            if cur["words"]:
                sents.append(cur)
            cur = _blank()
            continue
        if line.startswith("#"):
            if line.startswith("# text ="):
                cur["text"] = line.split("=", 1)[1].strip()
            elif line.startswith("# sent_id ="):
                cur["sent_id"] = line.split("=", 1)[1].strip()
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        feats, misc = cols[5], cols[9]
        number = ""
        for f in feats.split("|"):
            if f.startswith("Number="):
                number = f.split("=", 1)[1]
        cur["words"].append(cols[1])
        cur["upos"].append(cols[3])
        cur["number"].append(number)
        cur["space_after"].append("SpaceAfter=No" not in misc.split("|"))
    if cur["words"]:
        sents.append(cur)
    return sents


def _blank() -> dict:
    return {"sent_id": "", "text": "", "words": [], "upos": [], "number": [], "space_after": []}
