"""Lazy BLIP-2 / Qwen LM / Qwen-VL loaders and (prompt[, image]) -> text generators.

All heavy imports are inside functions so importing this module triggers no model
download. Exercised only by slow tests and the real run.
"""

from __future__ import annotations

from collections.abc import Callable


def _dtype(name: str):
    import torch

    return getattr(torch, name)


def load_blip2_captioner(model_id: str, gen_cfg: dict) -> Callable[[str], str]:
    """Return ``caption_fn(image_path) -> caption`` backed by BLIP-2."""
    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    proc = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=_dtype(gen_cfg["torch_dtype"]), device_map=gen_cfg["device_map"]
    ).eval()

    @torch.no_grad()
    def caption_fn(image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = proc(images=image, return_tensors="pt").to(model.device, _dtype(gen_cfg["torch_dtype"]))
        out = model.generate(**inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"])
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()

    return caption_fn


def load_qwen_lm(model_id: str, gen_cfg: dict) -> Callable[[str], str]:
    """Return ``generate(prompt) -> text`` backed by a Qwen2.5 instruct LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=_dtype(gen_cfg["torch_dtype"]), device_map=gen_cfg["device_map"]
    ).eval()

    @torch.no_grad()
    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"])
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()

    return generate


def load_qwen_vl(model_id: str, gen_cfg: dict) -> Callable[[str, str | None], str]:
    """Return ``generate(prompt, image_path|None) -> text`` backed by Qwen2.5-VL.

    When ``image_path`` is ``None`` the model is given a black tile (the vision
    ablation): same prompt structure, no real visual evidence.
    """
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=_dtype(gen_cfg["torch_dtype"]), device_map=gen_cfg["device_map"]
    ).eval()

    @torch.no_grad()
    def generate(prompt: str, image_path: str | None) -> str:
        image = Image.open(image_path).convert("RGB") if image_path else Image.new("RGB", (224, 224))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"])
        gen = out[0][inputs["input_ids"].shape[1]:]
        return proc.decode(gen, skip_special_tokens=True).strip()

    return generate


def model_revisions(model_ids: dict[str, str]) -> dict[str, str]:
    """Resolve each model id to its current HF commit hash (best-effort)."""
    revisions = {}
    for key, mid in model_ids.items():
        try:
            from huggingface_hub import HfApi

            revisions[key] = HfApi().model_info(mid).sha
        except Exception:
            revisions[key] = "unknown"
    return revisions
