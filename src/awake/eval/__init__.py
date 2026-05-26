"""Shared interpretability evaluation utilities."""

from awake.eval.attribution import Explainer, ModelAdapter, TokenAttribution
from awake.eval.bootstrap import bootstrap_ci, paired_diff_test
from awake.eval.erasure import erase, top_k_mask
from awake.eval.faithfulness import aopc_comprehensiveness, comprehensiveness, sufficiency
from awake.eval.modality_shapley import interventional_values, modality_shapley, shapley_2player
from awake.eval.plausibility import (
    aggregate_subwords_to_words,
    clip_gold_mask_to_window,
    token_auprc,
    token_iou,
    token_prf1_at_k,
)
from awake.eval.text_occlusion import occlusion_importance
from awake.eval.vqa_consistency import (
    accuracy,
    consistency_rate,
    explanation_leaks_answer,
    extract_choice,
    normalize_text,
    parse_rate,
    pipeline_divergence,
    rationale_leaks_answer,
)

__all__ = [
    "Explainer",
    "ModelAdapter",
    "TokenAttribution",
    "accuracy",
    "aggregate_subwords_to_words",
    "aopc_comprehensiveness",
    "bootstrap_ci",
    "clip_gold_mask_to_window",
    "comprehensiveness",
    "consistency_rate",
    "erase",
    "explanation_leaks_answer",
    "extract_choice",
    "interventional_values",
    "modality_shapley",
    "normalize_text",
    "occlusion_importance",
    "paired_diff_test",
    "parse_rate",
    "pipeline_divergence",
    "rationale_leaks_answer",
    "shapley_2player",
    "sufficiency",
    "token_auprc",
    "token_iou",
    "token_prf1_at_k",
    "top_k_mask",
]
