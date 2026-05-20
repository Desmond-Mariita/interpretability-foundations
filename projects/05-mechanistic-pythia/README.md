# 05 — Where in a small language model does a property emerge?

**Question.** At which layer of a small open transformer does a given binary property become linearly decodable from the residual stream?

**Answer.** _Coming soon — this project ships in week 6._

**Why it matters.** Per-layer linear probing is the standard first step in mechanistic interpretability, and a small open model is the right place to demonstrate the method end-to-end.

<!-- ![hero](assets/hero.png) -->

## Method

Pythia-160M (EleutherAI). Per-layer linear probes for three binary properties drawn from existing public benchmarks. **Control tasks (Hewitt & Liang 2019)** are reported alongside, so the reader can tell whether each probe is reading model state or just memorising frequent labels. One figure per property: probe accuracy and control accuracy as a function of depth.

## Reproduce

```
just setup
just data
just train    # train probes only; the LM is frozen
just eval
```

## Scope (v1.0)

Probing only. **Deferred to v1.1:** activation patching on GPT-2-small (where the IOI literature is native; Wang et al. 2022), and pretrained-SAE feature inspection via `sae-lens` (GPT-2-small residual SAEs across all layers are available, per the `sae-lens` registry).

## Limitations

- Probing tells us what is linearly decodable; it does not show what the model uses.
- Three properties are a sample, not a survey.
