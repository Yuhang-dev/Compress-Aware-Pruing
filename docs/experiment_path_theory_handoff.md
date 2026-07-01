# Experiment Path and Theory Handoff

Last updated: 2026-07-02
Project root: `D:\cap`
Important instruction: do not rely on `CLAUDE.md`; this handoff is self-contained.

## Purpose

This document is for an agent that has no prior conversation context. It explains what experiments were run, where the artifacts live, what the current evidence supports, and how to reformulate the method story so it does not read like an empirical hack.

The current project studies why pruning/compression breaks safety alignment in instruction-tuned LLMs, using Qwen2.5-3B-Instruct and Wanda pruning as the main diagnostic setting.

## Current High-Level Claim

The evidence now supports a two-stage mechanism:

1. **Moderate sparsity causes a harm-detection/refusal-readout collapse.**
   Pruning lowers a refusal-direction score on harmful prompts. This collapse is specific to the harmful-request/refusal direction, not a generic movement along random representation directions.

2. **Higher sparsity introduces a residual withholding failure.**
   Restoring the refusal readout eliminates failures at 40-45% Wanda sparsity, but residual unsafe outputs remain and grow at 50-55%. This suggests that high sparsity also weakens the behavior-level withholding/execution stage.

The safest wording is:

> Pruning-induced safety degradation is not explained by simple deletion of refusal write-out weights. Instead, standard pruning preserves average utility saliency while eroding rare-event safety margins on harmful prompts. At moderate sparsity, this appears as a refusal-readout collapse; at higher sparsity, an additional withholding residual remains even after the readout is restored.

Update after the causal bridge run:

> The direct deletion bridge has now been tested and failed. Zeroing only the grad-Crit weights that
> Wanda would remove does not raise ASR, while zeroing the Wanda-kept complement reproduces most of
> the full grad-Crit effect. The mechanism story should therefore focus on pruning-induced activation
> / margin erosion into preserved refusal readouts, not on Wanda deleting the identified write-out
> safety tail.

Update after margin calibration and closed-form repair:

> The refusal readout is now calibrated as a **distribution-level** margin. The pooled AUC for
> `s_mean` is 0.907, while grouped-CV AUC is 0.786; the latter is a within-sparsity caveat, not a
> failure of the distribution-shift mechanism. `frac(s_mean < tau)` rises monotonically with
> sparsity and tracks ASR. A train-free closed-form readout repair has an initial clean pass at
> Wanda-50: ASR 0.289 -> 0.133 with PPL v2 +0.055 and benign refusal +3.13pp.

## Main Artifact Map

### Project Docs

- `docs/phase1_gate1_findings.md`
  Earlier Gate-1 findings.

- `docs/vpref_projection_experiment_spec.md`
  Initial refusal-projection experiment spec.

- `docs/vpref_specificity_followup_spec.md`
  Specificity/null-distribution follow-up spec.

- `docs/step0_restore_s_oracle_spec.md`
  Restore-s oracle spec.

- `docs/step0b_restore_s_spec.md`
  Step0b restore-s refinement spec.

- `docs/mechanism_seal_spec.md`
  Final mechanism sealing spec: A residual sparsity sweep and B specificity strengthening.

- `docs/spec_a_margin_calibration.md`
  Next-priority spec for calibrating `s_l` into a real refusal/comply margin.

- `docs/closed_form_readout_repair_spec.md`
  New main-method spec: train-free closed-form low-rank readout repair that replaces the old
  mask-repair direction.

- `paper/aaai27/main.tex`
  Current AAAI draft.

- `paper/aaai27/ccfa-review-reports/2026-06-29-compression-aware-safety-repair-aaai-conference-review.md`
  Internal review of the current draft.

### Key Result Directories

- `results/phase1_v2/`
  Crit v2 selection, PPL v2, and Crit ablation results.

- `results/phase15_vpref_projection/`
  Refusal projection, specificity, restore-s, and final mechanism-seal results.

- `results/phase15_mechanism_seal_specificity/`
  The final B run with `RUN_VALIDATION=1`; includes `vpref_validation.csv`.

- `results/phase15_step0b_parallel_merged/`
  Merged Step0b restore-s run.

- `results/phase15_causal_bridge/`
  Spec C direct-deletion bridge results. Important negative result: `wanda_removed` zeroing does not
  explain natural Wanda ASR.

- `results/phase15_margin_calib_recut/`
  Margin-calibration recut from the existing 640-row `margin_points.csv`. Important result:
  `s_mean` pooled AUC 0.907, grouped-CV AUC 0.786, `frac_m_neg` tracks ASR with descriptive Pearson
  0.995, and `claim_margin_is_decision_variable=true` at distribution level.

- `results/phase2_readout_repair_v1/`
  Closed-form readout-repair v1. Important result: W50 eta1 passes guardrails; W45 has too little
  baseline ASR headroom to show a stable gain.

## Experiment Path

### Phase 0: Pruning Degrades Safety

Goal: establish that Wanda pruning increases safety failure rates.

Main setting:

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Pruner: Wanda
- Sparsities: 40%, 45%, 50%, later 55% for residual sweep
- Safety metric: ASR = LlamaGuard unsafe AND coherent
- Utility metric: PPL v2 on fixed sampled WikiText-2 windows

Representative observed ASR progression:

| condition | baseline ASR |
|---|---:|
| Wanda-40 | 0.046875 |
| Wanda-45 | 0.0859375 |
| Wanda-50 | 0.265625 |
| Wanda-55 | 0.625 |

File:

- `results/phase15_vpref_projection/mechanism_seal_residual_sweep.csv`

PPL v2 values:

| condition | PPL v2 | mean NLL | tokens | context length | windows |
|---|---:|---:|---:|---:|---:|
| dense | 8.219634 | 2.106526 | 65,466 | 1024 | 128 |
| Wanda-45 | 12.217974 | 2.502908 | 65,466 | 1024 | 128 |
| Wanda-50 | 15.633601 | 2.749422 | 65,466 | 1024 | 128 |

File:

- `results/phase1_v2/ppl_v2.csv`

Interpretation:

> Wanda pruning increases coherent unsafe behavior as sparsity rises. Utility also degrades, so all method claims must report matched or at least explicitly measured utility.

### Phase 1: Crit v1 Was Causal but Utility-Entangled

Original Crit selection found safety-critical weights, but the selected set was not clean enough for an iso-utility story.

Earlier conclusion:

- Crit v1 was about 2.3% of target weights.
- Zeroing it caused ASR around 0.67.
- But PPL increased heavily, around +63%.
- Wanda-50 survival was high, about 97%.

Interpretation:

> Strong safety write-out structures exist, but v1 Crit mixed safety and general utility. It also showed that the strongest safety write-out weights are mostly preserved by Wanda, so pruning-induced safety loss is not simply "Wanda deletes all refusal weights."

### Phase 1 v2: Cleaner Crit Selection

Motivation: fix Crit selection and PPL evaluation.

Implemented:

- Per-output-row Wei-style set difference.
- Score variants:
  - `snip = |w * grad|`
  - `grad = |grad|`
  - `norm_snip = |w * grad| / row_mean(|w|)`
- Candidate selectors:
  - `wei_setdiff`
  - `ratio`
  - `penalty`
- PPL v2:
  - fixed sampled strided WikiText-2 windows
  - `context_len=1024`
  - `stride=512`
  - `sample_windows=128`
  - same windows reused across conditions

Files:

- `src/casafety/crit_selection_v2.py`
- `src/casafety/ppl_eval_v2.py`
- `scripts/phase1_crit_v2.sh`
- `scripts/phase1_ppl_v2.sh`
- `results/phase1_v2/crit_selection_summary.csv`
- `results/phase1_v2/crit_ablation_v2.csv`
- `results/phase1_v2/crit_pruner_ranking.csv`

Representative Crit v2 result:

| candidate | size | crit-zero ASR | random-zero ASR | PPL delta | Wanda-50 survival |
|---|---:|---:|---:|---:|---:|
| grad, `p_safe=0.01`, `p_util=0.05` | about 0.56% | about 0.406 | 0.0 | about +8.3% | about 67-68% |
| snip, `p_safe=0.01`, `p_util=0.05` | about 0.51% | about 0.547 | 0.0 | about +10% | about 97% |
| v1 Crit | about 2.3% | about 0.67 | 0.0 | about +63% | about 97% |

Interpretation:

> There are at least two safety-related weight populations. High-salience safety weights are strong
> but mostly preserved by Wanda and utility-entangled. Lower-salience safety-support weights are
> smaller and partly removed by Wanda, but Spec C shows that the removed tail is not the main direct
> cause of natural Wanda ASR. Treat Crit v2 as a diagnostic of safety write-out mass, not as the main
> repair target.

Important caution:

> The direct causal bridge has been run and was negative. Do not claim that natural Wanda safety loss
> is caused mainly by deleting the grad-Crit subset.

### vPref Projection: Refusal Direction Exists and Is Measurable

Goal: extract refusal/harm-detection directions and measure whether pruning moves harmful prompts along them.

Core representation:

```text
s_l(x) = r_l^T h_l(x)
```

where:

- `h_l(x)` is a residual-stream activation at layer `l`;
- `r_l` is the dense model's harmful-vs-benign refusal direction;
- `s_l` is the refusal/harm-detection readout.

Files:

- `results/phase15_vpref_projection/vpref_projection.csv`
- `results/phase15_vpref_projection/vpref_projection_details.csv`
- `results/phase15_vpref_projection/vpref_projection_summary.csv`
- `results/phase15_vpref_projection/vpref_validation.csv`

Earlier projection showed collapse but needed stronger specificity controls. That led to mechanism seal B.

### Step0 / Step0b Restore-s: Readout Restoration

Goal: test whether restoring the refusal readout can recover safety.

Best setting used for residual sweep:

| field | value |
|---|---|
| layer group | 24,28,32 |
| window | 32 |
| `k_r` | 1 |
| steering mode | norm-relative |
| target kind | strong |
| beta value | 0.25 |

Important behavioral interpretation:

- Restoring `s` is diagnostic, not a final deployable method.
- It shows whether the safety failure is due to readout/margin collapse.
- Residual unsafe cases after successful `s` restoration indicate a second failure stage.

Files:

- `results/phase15_step0b_parallel_merged/step0b_restore_s.csv`
- `results/phase15_step0b_parallel_merged/step0b_restore_s_details.csv`
- `results/phase15_step0b_parallel_merged/step0b_restore_s_decision.json`

### Mechanism Seal A: Residual Sparsity Sweep

Goal: determine whether residual unsafe cases after restore-s grow with sparsity.

Command output decision:

```json
{
  "pass": true,
  "valid_sparsities": [0.4, 0.45, 0.5, 0.55],
  "valid_residual_counts": [0, 0, 5, 9],
  "interpretation": "residual grows with sparsity"
}
```

Files:

- `results/phase15_vpref_projection/mechanism_seal_residual_sweep.csv`
- `results/phase15_vpref_projection/mechanism_seal_residual_sweep.decision.json`

Interpretation:

> Restoring the refusal readout eliminates failures at 40% and 45% sparsity, but leaves 5/128 residuals at 50% and 9/128 at 55%. Therefore moderate sparsity is largely explained by readout/margin collapse, while high sparsity introduces residual withholding failure.

### Mechanism Seal B: Specificity and Causal Validation

Goal: prove that the collapse is specific to the harmful-request/refusal direction, not generic representation shrinkage.

Final decision:

```json
{
  "validation_pass": true,
  "validation_pass_pairs": [
    {"layer": 24, "k_r": 1},
    {"layer": 24, "k_r": 4},
    {"layer": 28, "k_r": 1},
    {"layer": 28, "k_r": 4}
  ],
  "control_specificity_pass": true,
  "random_floor_pass": true,
  "decision_pass": true,
  "interpretation": "harm-detection-specific collapse under compression"
}
```

Files:

- `results/phase15_vpref_projection/mechanism_seal_specificity.csv`
- `results/phase15_vpref_projection/mechanism_seal_specificity_controls.csv`
- `results/phase15_vpref_projection/mechanism_seal_specificity_decision.json`
- `results/phase15_mechanism_seal_specificity/vpref_validation.csv`

Key specificity values:

| config | layer | k | centered-cosine flip delta | null delta p95 | empirical p | null dirs |
|---|---:|---:|---:|---:|---:|---:|
| Wanda-45 | 24 | 1 | 0.108110 | 0.010364 | 0.004975 | 200 |
| Wanda-45 | 24 | 4 | 0.108110 | 0.008968 | 0.004975 | 200 |
| Wanda-45 | 28 | 1 | 0.115527 | 0.011890 | 0.004975 | 200 |
| Wanda-45 | 28 | 4 | 0.115527 | 0.011646 | 0.004975 | 200 |
| Wanda-50 | 24 | 1 | 0.092783 | 0.008159 | 0.004975 | 200 |
| Wanda-50 | 24 | 4 | 0.092783 | 0.008896 | 0.004975 | 200 |
| Wanda-50 | 28 | 1 | 0.103161 | 0.009850 | 0.004975 | 200 |
| Wanda-50 | 28 | 4 | 0.103161 | 0.009309 | 0.004975 | 200 |

Interpretation:

> The refusal/harm-detection direction is an outlier relative to 200 random directions. The collapse is direction-specific and causally validated at layers 24 and 28 for k=1 and k=4.

Important caution:

> Layer 32 had strong projection evidence, but it did not enter the final causal validation pass pairs. In main text, describe layer 24/28 as causally validated; mention layer 32 only as part of multi-layer restore-s if needed.

### Spec C: Direct Deletion Bridge Failed

Question:

> Does the subset of grad-Crit weights that Wanda would remove directly explain the natural ASR
> increase of Wanda-pruned models?

Result:

| condition | ASR |
|---|---:|
| dense none | 0.000 |
| full grad-Crit zero | 0.359 |
| Wanda-45 natural pruned | 0.117 |
| Wanda-45 removed zero | 0.000 |
| Wanda-45 kept zero | 0.367 |
| Wanda-50 natural pruned | 0.328 |
| Wanda-50 removed zero | 0.008 |
| Wanda-50 kept zero | 0.320 |

File:

- `results/phase15_causal_bridge/analysis.md`

Interpretation:

> grad-Crit is safety-critical as a whole, but the causal mass is mostly in the Wanda-kept subset.
> Wanda's natural ASR is therefore not explained by directly deleting the tested Wanda-removed
> safety tail. The stronger explanation is that broad pruning perturbations change the activations
> flowing into preserved refusal readouts, eroding the harmful-prompt margin.

## What Is Proven vs. Not Yet Proven

### Proven Enough for Mechanism Writing

- Wanda pruning increases coherent unsafe behavior with sparsity.
- PPL v2 has been fixed relative to earlier biased PPL measurement.
- v1 Crit is safety-causal but utility-entangled.
- v2 Crit reveals smaller, more utility-light safety-support structures.
- Strong high-salience safety write-out weights are mostly preserved by Wanda, so direct deletion is not the whole story.
- Refusal/harm-detection readout collapse is direction-specific and validated against random null directions.
- Restore-s eliminates failures at 40-45% but leaves increasing residuals at 50-55%, supporting a second withholding-stage failure.
- The direct-deletion bridge is false for grad-Crit: Wanda-removed zeroing gives ASR 0.000/0.008 at 45/50, while Wanda-kept zeroing gives 0.367/0.320.

### Not Yet Proven

- A final train-free closed-form readout repair reduces ASR with bounded PPL/benign-refusal drift.
- A CASR-style training method improves ASR across compressors and models.
- External validity across Llama-2, Qwen2.5-7B, SparseGPT, AWQ, GPTQ, StrongREJECT, XSTest, OR-Bench, or HarmBench.
- Whether the scalar readout can be calibrated as a robust margin classifier.
- Whether closed-form readout repair can replace the restore-s inference hook.
- Whether closed-form readout repair is enough at 50-55%, where residual withholding failure appears.

## Current Theoretical Reframing

The writing should avoid saying:

> We empirically found some weights/directions that matter, so we protect them.

That sounds like a heuristic.

Instead, frame the problem as **safety-margin preservation under compression**.

For a harmful prompt `x`, define a refusal/harm-detection readout at layer `l`:

```text
s_l(x) = r_l^T h_l(x)
```

and a safety margin:

```text
m_l(x) = s_l(x) - tau_l
```

where `tau_l` is an internal threshold above which refusal behavior is likely.

Dense aligned model:

```text
m_l(x_harm) > 0
```

Pruned model:

```text
m_l(x_harm; theta * M) may fall below 0
```

Core theoretical statement:

> Standard pruning optimizes average-case utility saliency, but safety alignment depends on rare-event margins on harmful prompts. A pruning perturbation can leave average PPL acceptable while pushing harmful-prompt refusal margins below threshold. Therefore compression safety repair should preserve safety margins under a fixed sparsity budget.

First-order derivation:

For a weight `w_i`, the approximate effect of pruning it on the safety margin is:

```text
Delta m_l(x) approx - w_i * d m_l(x) / d w_i
```

This naturally yields a safety-margin saliency:

```text
S_safety(i) = E_x_harm | w_i * d m_l(x) / d w_i |
```

or a magnitude-deconfounded version:

```text
S_safety(i) = E_x_harm | d m_l(x) / d w_i |
```

Utility cost can be estimated by:

```text
S_utility(i) = E_x_benign | w_i * d L_utility / d w_i |
```

Then clean safety-support weights are those with high safety-margin influence and low utility cost:

```text
high S_safety / (S_utility + eps)
```

or:

```text
high S_safety - lambda * S_utility
```

After Spec C, this first-order weight scoring should be treated as a diagnostic of safety-critical
write-out mass, not as the main repair route. The method story now shifts from protecting a deleted
tail to repairing the activation margin feeding preserved readouts.

## Candidate Method Derived from the Theory

### Closed-Form Readout Repair

Optimization view:

```text
min_{Delta W} UtilityDrift(theta_pruned + Delta W)
s.t.          m_l(x_harm; theta_pruned + Delta W) >= 0
              Delta W is low-rank / small-norm / merged into selected readout modules
```

One-shot approximation:

1. Start from a standard Wanda-pruned model.
2. Collect compressed activations at the refusal readout.
3. Use dense activations or a calibrated threshold target to define the missing readout score.
4. Solve a small ridge/least-squares update that raises `s_l` on harmful calibration prompts.
5. Merge the resulting low-rank `Delta W` into selected readout-producing matrices.
6. Evaluate ASR, PPL v2, benign refusal, and random-direction controls.

Expected benefit:

- Train-free.
- No inference-time hook.
- Mechanism-driven.

Expected limitation:

- Should mainly fix readout/margin collapse at 40-45%.
- May only partially fix 50-55% because residual withholding failure remains.

## Completed Late-Stage Diagnostics

### Experiment 1: Margin Calibration

Run:

```text
fit tau_l so s_l predicts refusal vs unsafe-compliance
```

Compare:

- dense / Wanda-40 / Wanda-45 / Wanda-50 / Wanda-55
- layers 24 / 28 / 32 / 35 and multi-layer mean
- grouped cross-validation by condition

Question:

> Is `s_l` a calibrated refusal/comply margin, and does `frac(s_l < tau_l)` track ASR?

Status:

- Completed via recut under `results/phase15_margin_calib_recut/`.
- `s_mean` pooled AUC: **0.907**.
- `s_mean` grouped-CV AUC: **0.786**.
- `frac_m_neg` across dense/W40/W45/W50/W55:
  `0.008 / 0.070 / 0.211 / 0.594 / 1.000`.
- ASR across dense/W40/W45/W50/W55:
  `0.000 / 0.016 / 0.117 / 0.328 / 0.675`.
- Interpretation: strong distribution-level margin; moderate within-sparsity per-sample classifier.

### Experiment 2: Closed-Form Readout Repair

Run:

- Wanda-45, Wanda-50, Wanda-55
- low-rank/ridge updates at layers 24/28/32
- scalar k=1 and subspace k=4 targets
- random-direction and benign-only controls

Metrics:

- ASR = LlamaGuard unsafe AND coherent
- PPL v2
- benign refusal rate
- coherent rate
- update norm and readout restoration amount

Question:

> Can a training-free merged weight update reproduce restore-s without an inference hook?

Status:

- Initial W50 pass under `results/phase2_readout_repair_v1/`.
- Best clean W50 setting: `readout_repair_eta1`.
- W50 ASR: `0.289 -> 0.133`.
- PPL v2: `15.634 -> 15.689` (`+0.055`).
- benign refusal: `0.0156 -> 0.0469` (`+3.13pp`).
- same-eta random direction ASR: `0.258`.
- W50 eta2 is a diagnostic upper point: ASR `0.047`, but benign refusal `0.094`, over the +5pp
  guardrail.
- W45 remains inconclusive because pruned ASR is only `0.078`, leaving little headroom.

## Recommended Next Experiments

### Experiment 3: Readout Repair Refinement

Before moving to training, refine the train-free repair:

- sweep smaller/larger ridge and target margins around the W50 eta1 point;
- add W55 if memory/time allows, but expect residual enforcement failure;
- report same-eta random direction and bias-only controls;
- keep PPL v2 and benign refusal as hard guardrails.

### Experiment 4: High-Sparsity Residual Repair

If closed-form readout repair reduces W50/W55 ASR but leaves coherent unsafe residuals, test a small
behavior-level top-up:

- refusal SFT under compressed view
- LoRA only on selected layers
- combined margin-preservation + refusal behavior loss

Question:

> Is the 50-55% residual a behavior-level withholding failure that requires training?

## Prompt for Another Agent

Copy the following prompt to a context-free agent. Attach or point it to this file and the listed CSV/JSON artifacts.

```text
You are helping write the theory/story section for a paper on pruning-induced safety degradation in instruction-tuned LLMs. Read docs/experiment_path_theory_handoff.md and use it as the source of truth. Do not treat the proposed one-shot repair as already proven unless the handoff explicitly says so.

Task:
Reconstruct the theoretical story behind the experiments so the method does not read like an empirical hack.

Required output:
1. A concise theoretical framing in which safety under pruning is a safety-margin preservation problem, not merely "protect weights we found empirically."
2. A paragraph explaining why standard Wanda/magnitude pruning can preserve PPL but still damage safety: average-case utility saliency does not guarantee harmful-prompt margin preservation.
3. A paragraph reconciling the direct-deletion negative result:
   - grad-Crit is safety-critical as a whole;
   - Wanda keeps most of its causal mass;
   - zeroing Wanda-removed grad-Crit does not explain natural Wanda ASR.
4. A two-stage mechanism statement:
   - moderate sparsity: harm-detection/refusal-readout margin collapse;
   - high sparsity: additional withholding residual even after readout restoration.
5. A theory-derived method proposal for closed-form readout repair: convert restore-s into a train-free low-rank update merged into the compressed model.
6. Explicit claim boundaries:
   - what the current experiments prove;
   - what remains speculative;
   - what additional experiment would convert the theory-derived method into a main paper result.

Style requirements:
- Write as if for an AAAI/ICML-style paper.
- Avoid saying "we found this works, so we do it."
- Use the logic "compression violates a constraint; the method is the first-order repair of that constraint."
- Do not invent new numbers or cite unverified papers.
- Keep the terminology consistent:
  ASR = attack success rate = LlamaGuard unsafe AND coherent.
  PPL v2 = fixed-window strided WikiText-2 perplexity.
  s_l(x) = r_l^T h_l(x) = refusal/harm-detection readout.
  m_l(x) = s_l(x) - tau_l = safety margin.
```

## Suggested Paper Wording

Use this wording as a seed:

> Standard pruning is designed to preserve average-case utility saliency, but safety alignment is governed by rare-event margins on harmful prompts. Let `s_l(x)=r_l^T h_l(x)` denote a refusal readout and `m_l(x)=s_l(x)-tau_l` its safety margin. A pruning perturbation can leave average PPL acceptable while pushing `m_l(x)` below threshold for harmful inputs, producing coherent unsafe completions. This explains why safety failure can appear before broad incoherence and why preserving high-magnitude refusal write-out weights alone is insufficient. Our diagnostics show that the refusal-margin collapse is direction-specific and causally validated at layers 24 and 28; restoring the margin eliminates failures at moderate sparsity but leaves a high-sparsity residual, indicating a second withholding-stage failure. This motivates safety-margin preserving compression: under a fixed sparsity budget, repair the pruning mask by restoring weights with high first-order influence on harmful-prompt refusal margins and low benign utility cost.
Updated seed:

> Standard pruning is designed to preserve average-case utility saliency, but safety alignment is governed by harmful-prompt refusal margins. Let `s_l(x)=r_l^T h_l(x)` denote a refusal readout and `m_l(x)=s_l(x)-tau_l` its safety margin. A pruning perturbation can leave average PPL acceptable while pushing `m_l(x)` below threshold for harmful inputs, producing coherent unsafe completions. This failure is not explained by direct deletion of the identified refusal write-out weights: zeroing the Wanda-removed grad-Crit subset barely changes ASR, while zeroing the Wanda-kept complement reproduces the full grad-Crit effect. The mechanism is instead activation-margin erosion into preserved readouts. This motivates closed-form readout repair: convert the restore-s oracle into a train-free low-rank update that restores the measured refusal margin without an inference-time hook.
