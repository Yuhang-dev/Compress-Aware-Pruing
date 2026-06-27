# Experiment Record — 2026-06-27

This document freezes the main results and conclusions before Step 0 restore-s oracle. It is a narrative
record only; raw artifacts remain under `results/`.

## Metric Glossary

- ASR: attack success rate, here the fraction of harmful prompts judged unsafe and coherent.
- PPL: perplexity on WikiText-style language modeling.
- mean NLL: mean negative log likelihood; `PPL = exp(mean_nll)`.
- Coherent rate: fraction passing lexical and response-PPL coherence gates.
- Refusal rate: fraction of responses classified as refusal/safe refusal by the experiment's evaluator.
- Crit: selected safety-critical weight subset.
- Wanda survival: fraction of a selected weight set that would be retained by a Wanda pruning mask.
- `s` / `s1`: harm-detection score, `s = r_hat^T a`, where `r_hat` is the dense Arditi harmful-minus-benign
  last-token direction.
- `sK`: k-dimensional harm-detection subspace energy.
- Cliff's delta: rank effect size for complied-vs-refused score separation.
- within-prompt delta: compressed score minus dense score on the same prompt; flip-vs-stay delta compares
  prompts that changed from refused to complied against prompts that stayed refused.

## Phase 0: Compression Harms Safety on Qwen2.5-3B

Source: `results/phase0_qwen25_3b_wanda_fine_joined.csv`.

| condition | sparsity | ASR | refusal rate | PPL |
|---|---:|---:|---:|---:|
| dense | 0.00 | 0.0000 | 1.0000 | 11.1453 |
| wanda_40 | 0.40 | 0.0313 | 0.9688 | 14.0335 |
| wanda_45 | 0.45 | 0.0859 | 0.9141 | 16.3663 |
| wanda_50 | 0.50 | 0.2344 | 0.7656 | 20.5125 |

PPL v2 fixed-window remeasurement on Qwen2.5-3B:

Source: `results/phase1_v2/ppl_v2.csv`.

| condition | context len | sampled windows | tokens | PPL |
|---|---:|---:|---:|---:|
| dense | 1024 | 128 | 65466 | 8.2196 |
| wanda_45 | 1024 | 128 | 65466 | 12.2180 |
| wanda_50 | 1024 | 128 | 65466 | 15.6336 |

Interpretation: the original PPL numbers were inflated by the old short/small evaluation, but Wanda still
causes a real utility drop at 45-50% sparsity. Safety degradation remains visible.

## Phase 1 Crit v2: Cleaner Safety-Critical Sets Exist

Sources:

- `results/phase1_v2_grid_grad/crit_ablation_v2_frontier.csv`
- `results/phase1_v2_grid_grad/crit_ppl_matched_random_v2.csv`
- `results/phase1_v2_grid_grad/crit_wanda_removed_probe_v2.csv`

Representative Pareto candidates:

| candidate | Crit ratio | Crit ASR | random ASR | PPL delta | Wanda50 survival |
|---|---:|---:|---:|---:|---:|
| grad ps=0.003 pu=0.10 | 0.1205% | 0.1250 | 0.0000 | +1.97% | 65.36% |
| grad ps=0.005 pu=0.10 | 0.2135% | 0.2188 | 0.0000 | +2.94% | 65.63% |
| grad ps=0.0075 pu=0.08 | 0.3618% | 0.2656 | 0.0000 | +4.63% | 66.45% |
| grad ps=0.010 pu=0.10 | 0.4716% | 0.3438 | 0.0000 | +6.10% | 65.87% |

PPL-matched random controls retained ASR 0.0 for selected candidates. The Wanda-removed-only probe also
kept ASR 0.0 while removing about one third of grad-Crit, so deleting only the subset of clean Crit that
Wanda would remove is not sufficient to reproduce natural Wanda safety failure.

Interpretation: small clean safety-critical structures exist and are not just "large-weight utility"
artifacts. However, natural Wanda safety loss is not explained by directly deleting only these structures.
This points toward compression perturbing the upstream harm-detection readout rather than simply removing
the executor weights.

## Phase 1.5 vPref Specificity: Harm-Detection Collapse Is Direction-Specific

Sources:

- `results/phase15_vpref_projection/vpref_validation.csv`
- `results/phase15_vpref_projection/vpref_specificity.csv`
- `results/phase15_vpref_projection/vpref_specificity_decision.json`

Decision JSON:

```json
{
  "validation_pass": true,
  "control_specificity_pass": true,
  "random_floor_pass": true,
  "decision_pass": true,
  "interpretation": "harm-detection-specific collapse under compression"
}
```

Causal validation at layer 28:

- benign baseline refusal: 2.34%
- induce at alpha=2: benign refusal rises to 17.19%
- harmful baseline refusal: 100%
- suppress_subtract: harmful refusal falls to 0%
- suppress_project_out harmful refusal: 4.69% for k=1, 10.16% for k=4, 26.56% for k=8

Centered-cosine primary specificity readout:

| config | k_r | harm Cliff's delta | harm flip delta | random null p95 | empirical p |
|---|---:|---:|---:|---:|---:|
| wanda_45 | 1 | -0.6964 | -0.1256 | 0.0119 | 0.0196 |
| wanda_50 | 1 | -0.5418 | -0.1002 | 0.0110 | 0.0196 |

Matched non-safety control directions:

- `ctrl_technical_vs_general`: probe CV accuracy 0.8904
- `ctrl_question_vs_instruction`: probe CV accuracy 0.8283
- both had `control_comparable_to_harm = False` for Wanda45 and Wanda50

Interpretation: the compressed model's harmful compliance is tied to a drop in the harm-detection/safety
trigger score, not to an equally strong collapse of arbitrary matched benign concept directions. This
supports the Method B target: keep `s` above the dense safety threshold under the pruning mask.

## Current Mechanistic Position

Established:

- A causal harm-detection direction/subspace exists.
- Clean safety-critical write-out structures exist.
- Wanda's safety failure is not adequately explained by direct deletion of high-SNIP executor weights.
- Compression causes a direction-specific harm-detection score collapse under Wanda45/Wanda50.

Working hypothesis:

- The executor largely survives pruning; the trigger/readout signal falls below threshold under compression.
- Repair should first test inference-time restoration of `s`; if that works, train-time Method B can aim to
  keep `s` above threshold under pruning.

Next experiment:

- Step 0 restore-s oracle: on compressed Wanda45/Wanda50, add a small gated `beta * r_hat` only at the last
  prompt token and first 6 generated tokens. If harmful refusal returns while benign over-refusal stays low,
  then inference-time steering is a viable non-LoRA repair and the `keep s above tau` target is validated.
