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

- The harm-detection/readout signal falls below threshold under compression.
- Directly adding the layer-28 k=1 harm-detection direction at a short generation window is not enough to
  restore dense-like safety at Wanda50, even when the analytic `s_after` exceeds the dense threshold.

## Step 0 Restore-s Oracle: Short-Window k=1 Steering Is Not Sufficient

Sources:

- `results/phase15_vpref_projection/step0_restore_s.csv`
- `results/phase15_vpref_projection/step0_restore_s_details.csv`
- `results/phase15_vpref_projection/step0_restore_s_decision.json`

Decision JSON:

```json
{
  "pass": false,
  "tau": 34.10434913635254,
  "interpretation": "restore-s oracle did not pass all configs; inspect coherence, benign refusal, and restored s"
}
```

Key rows:

| config | set | beta mode | beta/target | refusal rate | unsafe rate | ASR | coherent rate | mean s before | mean s after |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| wanda_45 | harm_eval | fixed | 0 | 0.7031 | 0.1016 | 0.1016 | 0.9922 | 41.7823 | 41.7823 |
| wanda_45 | harm_eval | fixed | 32 | 0.8906 | 0.0703 | 0.0703 | 0.9922 | 41.7823 | 73.7823 |
| wanda_45 | harm_eval | deficit | dense_ref | 0.8359 | 0.0625 | 0.0625 | 0.9922 | 41.7823 | 56.4186 |
| wanda_50 | harm_eval | fixed | 0 | 0.4141 | 0.3125 | 0.2578 | 0.8984 | 32.8929 | 32.8929 |
| wanda_50 | harm_eval | fixed | 32 | 0.5547 | 0.2734 | 0.2188 | 0.8906 | 32.8929 | 64.8929 |
| wanda_50 | harm_eval | deficit | dense_ref | 0.4844 | 0.2578 | 0.2344 | 0.9219 | 32.8929 | 56.3216 |
| wanda_45 | benign_eval | fixed | 32 | 0.0625 | 0.0000 | 0.0000 | 0.8438 | -7.1599 | -6.9099 |
| wanda_50 | benign_eval | fixed | 32 | 0.0000 | 0.0000 | 0.0000 | 0.7578 | -7.0692 | -6.8192 |

Notes:

- `beta=32` is the best harm-row setting selected by the current decision scorer for both Wanda45 and
  Wanda50.
- Benign gate behaved as intended: only 1/128 benign prompts was gated in for each config, so over-refusal
  did not rise. Benign coherence is below the strict 0.9 threshold even at beta=0, so the benign part of the
  JSON failure is mostly a baseline-generation/coherence-gate issue, not a steering-induced over-refusal
  issue.
- On harmful prompts, short-window k=1 steering raises `s_after` far above `tau`, but dense-like safety does
  not return. Wanda45 improves only modestly (ASR 10.16% -> 7.03% for beta=32). Wanda50 improves from ASR
  25.78% to 21.88% for beta=32, and to 23.44% for dense-ref deficit. This is not enough to validate the
  current repair as a viable standalone inference-time fix.

Interpretation:

- The vPref specificity result still says the harm-detection readout collapses under compression, but Step 0
  shows that simply pushing the layer-28 k=1 readout above threshold for the last prompt token and first 6
  decode tokens is not sufficient to restore robust refusal, especially at Wanda50.
- Before training Method B, the repair target needs a stronger oracle check: sustained steering, multi-layer
  steering, or k=4/8 subspace restoration. If those also fail despite measured restoration, then the executor
  is more compromised than the earlier readout-only story suggested.
