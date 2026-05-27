"""Lazy Pythia-160M loader + residual-stream hook extraction + word alignment (P5).

Heavy imports (torch/transformers) live inside functions so importing this module downloads
nothing. The pure ``align_words_to_tokens`` is smoke-tested; extraction is slow-only.
"""

from __future__ import annotations


def align_words_to_tokens(
    word_spans: list[tuple[int, int]],
    token_offsets: list[tuple[int, int]],
) -> list[int | None]:
    """For each word char-span, the index of its LAST overlapping subword token.

    Overlap (not containment): ``tok_end > ws and tok_start < we`` -- byte-level BPE attaches the
    leading space to the token, so a containment test would drop every non-first word.

    Returns:
        Per-word last-overlapping token index, or ``None`` if no token overlaps (dropped).
    """
    out: list[int | None] = []
    for ws, we in word_spans:
        last = None
        for ti, (ts, te) in enumerate(token_offsets):
            if te > ws and ts < we:
                last = ti
        out.append(last)
    return out


def load_pythia(model_id: str, revision: str, device: str = "cpu"):  # pragma: no cover - slow
    """Load a frozen Pythia (GPT-NeoX) model + Fast tokenizer (eval, no grad)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model, tok


def extract_points(model, tok, sentence_words, space_after, n_blocks, device="cpu"):  # pragma: no cover - slow
    """Return {point: (n_words, d_model) np.float16} for one sentence via forward hooks.

    Points: 'embedding' (embed_in output) + 'block_0..N-1' (each GPTNeoXLayer output, resid_post)
    + 'ln_f' (final_layer_norm output). Words are aligned to their LAST overlapping subword.
    """
    import numpy as np
    import torch

    # Reconstruct surface string + per-word char spans from FORM + SpaceAfter.
    text, spans, pos = "", [], 0
    for w, sa in zip(sentence_words, space_after, strict=True):
        spans.append((pos, pos + len(w)))
        text += w + (" " if sa else "")
        pos += len(w) + (1 if sa else 0)
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    last_tok = align_words_to_tokens(spans, offsets)

    captured: dict[str, torch.Tensor] = {}
    handles = []
    base = model.gpt_neox

    def mk(name):
        def hook(_m, _i, out):
            captured[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles.append(base.embed_in.register_forward_hook(mk("embedding")))
    for i in range(n_blocks):
        handles.append(base.layers[i].register_forward_hook(mk(f"block_{i}")))
    handles.append(base.final_layer_norm.register_forward_hook(mk("ln_f")))
    try:
        model(**{k: v.to(device) for k, v in enc.items()})
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name, t in captured.items():
        seq = t[0].to(torch.float64).cpu().numpy()  # (seq, d)
        rows = [seq[ti] for ti in last_tok if ti is not None]
        out[name] = np.asarray(rows, dtype=np.float16)
    keep = [ti is not None for ti in last_tok]
    return out, keep
