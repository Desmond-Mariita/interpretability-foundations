"""Build explainer instances from config over a model + tokenizer."""

from __future__ import annotations

from awake.eval.explainers.gradient_x_input import GradientXInputExplainer
from awake.eval.explainers.integrated_gradients import IntegratedGradientsExplainer
from awake.eval.explainers.lime_text import LimeExplainer
from awake.eval.explainers.random_baseline import RandomExplainer


def build_explainer(name: str, model, tok, cfg: dict, device: str):
    """Return an explainer instance by name using ``cfg`` for hyperparameters."""
    if name == "random":
        return RandomExplainer(seed=cfg["bootstrap"]["seed"])
    if name == "grad_x_input":
        return GradientXInputExplainer(model, tok, device=device)
    if name == "integrated_gradients":
        return IntegratedGradientsExplainer(model, tok, device=device, n_steps=cfg["ig"]["n_steps"])
    if name == "lime":
        return LimeExplainer(model, tok, device=device, num_samples=cfg["lime"]["num_samples"])
    if name == "shap_partition":
        from awake.eval.explainers.shap_partition import ShapPartitionExplainer

        return ShapPartitionExplainer(model, tok, device=device, max_evals=cfg["shap"]["max_evals"])
    raise ValueError(f"unknown explainer: {name}")
