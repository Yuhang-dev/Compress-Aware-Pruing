# Spec A: Refusal-Margin Calibration

Purpose: turn the restore-s mechanism into a measured decision variable. This is the next priority
after Spec C ruled out the direct-deletion bridge.

## Question

Does the scalar refusal readout

```text
s_l(x) = r_l^T h_l(x)
m_l(x) = s_l(x) - tau_l
```

act like a decision margin for harmful prompts, such that pruning-induced ASR is explained by harmful
examples crossing below a threshold `tau_l`?

## Inputs

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Conditions: dense, Wanda-40, Wanda-45, Wanda-50, Wanda-55
- Harmful prompts: same AdvBench-direct eval split used by restore-s and mechanism seal
- Directions: existing dense refusal directions from `artifacts/vpref_projection/`
- Candidate layers: 24, 28, 32, 35 plus a multi-layer mean
- Outcomes: boolean labels from existing LlamaGuard pipeline
  - `safe_refusal = coherent AND NOT unsafe`
  - `unsafe_comply = coherent AND unsafe`
  - drop incoherent rows for the classifier analysis

Text-free rule: output only prompt ids, scalar readouts, boolean outcomes, coherence flags, and
aggregate metrics. Do not write prompt or response text.

## Procedure

1. For every `(condition, prompt_id, layer)` collect `s_l(x)` at the same residual location used by
   vPref projection and restore-s.
2. Join each scalar with the existing boolean outcome under the same generation protocol.
3. Fit a threshold `tau_l` predicting refusal vs unsafe-compliance.
4. Use grouped cross-validation by condition:
   - hold one condition out;
   - fit `tau_l` on the remaining conditions;
   - evaluate AUC and threshold accuracy on the held-out condition.
5. For each condition report:
   - `mean_s`, `median_s`, `p10_s`, `p90_s`;
   - `tau_l`;
   - `frac_m_neg = mean(s_l < tau_l)`;
   - observed ASR;
   - number of coherent examples.
6. Report whether `frac_m_neg` is monotone with sparsity and whether it tracks ASR.

## Outputs

Write under `results/phase15_margin_calib/`:

- `margin_readouts.csv`
  - text-free per prompt/layer scalar table
  - fields: `condition, sparsity, prompt_id, layer, s, outcome, unsafe, coherent, response_ppl`
- `margin_auc.csv`
  - fields: `layer, fold_condition, auc, accuracy, n_train, n_test`
- `margin_thresholds.csv`
  - fields: `layer, fold_condition, tau, criterion`
- `margin_fraction_vs_asr.csv`
  - fields: `condition, sparsity, layer, tau, frac_m_neg, mean_s, median_s, asr, coherent_rate`
- `decision.json`
  - fields:
    - `best_layer`
    - `auc_best_layer`
    - `auc_multilayer`
    - `monotone_frac_m_neg`
    - `frac_m_neg_tracks_asr`
    - `claim_margin_is_decision_variable`
    - `interpretation`

## Decision Rule

Claim passes if:

- best single layer or multi-layer mean has grouped-CV AUC >= 0.80;
- `frac_m_neg` increases with sparsity for Wanda-40/45/50/55;
- `frac_m_neg` qualitatively tracks ASR.

The ASR-vs-`frac_m_neg` correlation has only four sparsity points, so treat it as descriptive. Do
not report a p-value for that part.

## Expected Interpretation

If pass:

> The refusal readout is not just a probe. It behaves as a calibrated safety margin: pruning shifts
> harmful prompts below the threshold, and the fraction below threshold tracks coherent unsafe
> behavior.

If fail:

> Restore-s is still a causal intervention, but the scalar margin is not sufficient as a classifier.
> The mechanism story must use a multi-layer or subspace readout rather than a single threshold.

## Suggested Implementation

Add:

- `src/casafety/margin_calibration.py`
- `scripts/phase15_margin_calibration.sh`

The implementation should reuse:

- vPref projection hidden-state collection;
- LlamaGuard/coherence labels from `phase0_smoke_eval.py`;
- existing Wanda pruning helpers;
- `model_slug` and artifact naming conventions.

Remote smoke:

```bash
cd /root/autodl-tmp/cap
capenv

HARMFUL_LIMIT=16 \
CONDITIONS="dense wanda_45 wanda_50" \
bash scripts/phase15_margin_calibration.sh
```

Full run:

```bash
cd /root/autodl-tmp/cap
capenv

HARMFUL_LIMIT=128 \
CONDITIONS="dense wanda_40 wanda_45 wanda_50 wanda_55" \
bash scripts/phase15_margin_calibration.sh
```
