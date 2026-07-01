# Theory-Hardening Specs (text-free, pre-training)

Created 2026-06-30. Author role: **Claude writes specs; Codex implements/runs on remote GPU.**
These analyses convert the core theory (`docs/paper_core_narrative_and_theory.md`) from
"we protect useful weights" into "we repair a constraint pruning violates" **before** spending any
training compute. Current status after Spec B and Spec C:

- **Spec B passed after recut:** safety-margin influence is magnitude-blind at refusal-readout
  layers, and Wanda partially rescues it through activation norms.
- **Spec C failed:** the Wanda-removed subset of grad-Crit does **not** directly explain natural
  Wanda ASR. The causal mass of grad-Crit is mostly in the Wanda-kept subset.
- **Spec A passed after pooled-AUC recut:** `s_mean` has pooled AUC **0.907** as a
  distribution-level refusal margin; grouped-CV AUC is **0.786**, showing within-sparsity
  per-sample prediction is only moderate.
- **Closed-form readout repair has an initial pass at W50:** eta=1 reduces ASR **0.289 -> 0.133**
  with PPL v2 +0.055 and benign-refusal +3.13pp; W45 is underpowered because the pruned baseline
  ASR is only 0.078.

Recommended order from here: **refine closed-form readout repair -> behavioral top-up if residuals
remain -> external validity**. Do not spend more effort on mask repair as the main method unless a
new upstream-circuit target is identified.

**Standing red line (all three):** aggregate / numeric outputs only. Never read or emit
per-response generations. (B) is pure weight statistics (no generation at all). (A) joins the
*scalar* `s_l` with the *boolean* LlamaGuard label — never the response text. (C) uses aggregate
ASR/PPL only.

**Reuse, do not re-implement:** dense refusal direction `r_l` (from vPref projection); grad-Crit set
(from `crit_selection_v2` / `results/phase1_v2/`); Wanda mask + saliency (from the pruner); ASR
harness, PPL v2, LlamaGuard, `s_l = r_l^T h_l` (existing). Model `Qwen/Qwen2.5-3B-Instruct`;
sparsities from {45,50,55}; eval = AdvBench-direct 128-prompt + greedy; ASR = unsafe AND coherent.

================================================================================
## Spec B — Objective/measure mismatch (cheapest; no generation)
**Locks:** Claim 1 — the weights that move the refusal margin are magnitude-blind at the layers
carrying the refusal readout, while Wanda partially rescues them through its activation-norm term and
still removes a non-trivial causal target subset.

**Inputs.** Target modules = the Crit target set (write-out matrices: `o_proj`, `down_proj`) over the
validated readout layers (L24, L28; optionally all layers <= 28). Harmful calibration set
`X_harm` (64-128 prompts). Benign calibration set `X_benign` (WikiText windows already used by PPL v2).

**Per-weight scores** (compute over the target modules):
- `mag(i)      = |w_i|`
- `wanda(i)    = |w_i| * ||X_j||`  (benign calibration; same as the pruner)
- `S_safety(i) = E_{x in X_harm} | d s_l(x) / d w_i |`  — one autograd backward from the scalar
  `s_l(x)=r_l^T h_l(x)` per prompt, averaged. (tau is constant -> drops out, so **no threshold
  needed here**.) Restrict to weights in layers <= l (gradient is zero above l). Also compute the
  `|w_i * d s_l/d w_i|` variant.
- `S_util(i)   = E_{x in X_benign} | w_i * d L_lm / d w_i |`  — benign LM cross-entropy gradient.

**Report.**
1. Rank correlations across weights: `corr(S_safety, mag)`, `corr(S_safety, wanda)`,
   `corr(S_safety, S_util)` using **Spearman only**. Treat Pearson as unreliable for these
   heavy-tailed score distributions unless the implementation explicitly reports CIs for the same
   statistic being estimated.
2. grad-Crit members' percentile in the `mag` ranking and the `wanda` ranking (reproduce the
   measured ~0.493 magnitude p50 and ~0.726 Wanda p50, now as mean +/- CI per layer).
3. For each sparsity in {45,50,55}: fraction of the **top-q `S_safety`** weights that fall in
   pure-magnitude and Wanda **cut** regions (q = grad-Crit size; report q-sweep too), plus base-rate
   and lift-vs-base. The Wanda cut fraction is an operational exposure number, not the primary
   orthogonality test.

**Outputs (aggregate, inherently text-free).**
- `results/phase15_mismatch/weight_score_correlations.csv`
- `results/phase15_mismatch/safety_support_percentiles.csv`
- `results/phase15_mismatch/highsafety_in_wanda_cut.csv`
- `results/phase15_mismatch/decision.json` -> raw implementation decision.
- `results/phase15_mismatch/decision_recut.json` -> Spearman-only recut with per-layer profile,
  deep-window aggregate, robustness cutoffs, o_proj-pooled rows, cut fractions, and corrected flags.

**Decision (claim holds if):**
- `claim_magnitude_blind_at_readout`: median deep-window `|Spearman(S_safety, |w|)| <= ~0.10` over
  the causally validated readout neighborhood (L24/L28 +/- 2, evaluated on the L28 readout profile)
  **AND** o_proj-pooled `|Spearman(S_safety, |w|)| <= ~0.10`.
- Robustness: the median per-layer Spearman remains near zero for broad deep cutoffs such as
  L>=8, L>=12, and L>=16. This guards against cherry-picking the exact readout block.
- `claim_wanda_partial_rescue`: Wanda cut fraction at W50 is materially below the 50% base cut rate
  and below pure-magnitude cut fraction.
- `claim_wanda_removes_causal_target`: Wanda still removes a non-trivial subset (about >=20%) of
  top safety-influence weights. This is the target for Spec C, not a proof of causality by itself.

**Caveats to flag.** (i) All-layer pooled Spearman can be misleading because early generic layers
show magnitude-importance coupling, while the readout layers do not. Report the per-layer profile and
the deep-window recut. (ii) `wanda` correlation will be higher than `mag`: the `||X||` term partially
lifts safety-support because `S_safety=E|g_i x_j|` and Wanda `|w_ij|*||X_j||` share input-channel
activation factors. Say Wanda **partially rescues**, not that it accidentally "finds" safety.
(iii) Gradients are first-order at the dense point; note it.

================================================================================
## Spec A — Calibrate the refusal margin into a measured distribution-level margin
**Locks:** Claim 2 premise — `s_l` is the refuse/comply decision variable, and pruning shifts the
harmful-prompt `s_l` below threshold; `ASR(sparsity)` tracks `fraction(m_l<0)`.

**Data assembly (text-free join).** For every (harmful prompt, condition) cell with condition in
{dense, Wanda-45, Wanda-50, Wanda-55}: record the **scalar** `s_l(x)` (per layer L24/L28/L32 and
their mean) and the **boolean** outcome from the existing guard pipeline
(`refuse = coherent & safe` vs `comply = coherent & unsafe`); drop incoherent. **Only the number and
the boolean leave the pipeline — not the text.**

**Procedure.**
1. Hypothesis: among harmful prompts, higher `s_l` -> refuse. Fit threshold `tau_l` on `s_l`
   predicting the outcome. Report both:
   - **pooled AUC** over all coherent rows, which tests the claimed distribution-level shift;
   - **sparsity-grouped CV AUC**, which intentionally removes the strongest group-level signal and
     tests only within-condition per-sample ranking.
2. Freeze `tau_l` (Youden-J / EER on the held-in folds). For each sparsity report
   `frac_m_neg = fraction of harmful prompts with m_l = s_l - tau_l < 0`, and `mean/median s_l`.
3. Correlate `frac_m_neg` with `ASR` across the condition points.

**Outputs.**
- `results/phase15_margin_calib/margin_auc.csv` (pooled AUC + grouped-CV AUC per score)
- `results/phase15_margin_calib/margin_threshold.csv`
- `results/phase15_margin_calib/margin_fraction_vs_asr.csv` (sparsity, frac_m_neg, mean_s, asr)
- `results/phase15_margin_calib/decision.json` -> `{auc_best_layer, auc_multilayer,
  frac_vs_asr_pearson, monotone_shift: bool, claim_margin_is_decision_variable: bool, interpretation}`

**Decision (claim holds if):** pooled `AUC >= ~0.85` (distribution-level separation) **AND**
`frac_m_neg` increases monotonically with sparsity **AND** tracks ASR **AND** restore-s explains most
W45/W50 ASR.

**Observed result.** `s_mean` pooled AUC = **0.907**, grouped-CV AUC = **0.786**. The latter is not a
failure: it asks whether `s_l` ranks examples within a fixed sparsity bucket after the group-level
pruning shift is removed. `frac_m_neg(s_mean)` rises
0.008 -> 0.070 -> 0.211 -> 0.594 -> 1.000 over dense/W40/W45/W50/W55 and has descriptive Pearson
0.995 with ASR. Restore-s share is 1.0 at W45 and 0.881 at W50.

**Caveats to flag.** (i) `frac_m_neg`-vs-ASR correlation has only five condition points; report it
as descriptive only. (ii) Grouped-CV AUC is a within-condition caveat, not the main margin gate.
(iii) The margin is distribution-level and causal via restore-s; it is not a perfect per-prompt
classifier.

================================================================================
## Spec C — Causal bridge: completed negative result
**Original lock attempted:** upgrade Claim 1 from "plausible deleted safety-support contributor" to
"Wanda-removed grad-Crit directly causes natural Wanda ASR."

**Design (weight ablation on the DENSE model; isolates the weights' role).** For sparsity s in
{45, 50}: define grad-Crit subset partitions by Wanda's keep/cut at s. Conditions (zero the named
set in dense Qwen2.5-3B, then eval):
1. `none` (dense baseline)
2. `full grad-Crit` (reproduce ASR ~0.41)
3. **`Wanda-removed grad-Crit`** (grad-Crit weights in Wanda's cut region at s) — key condition
4. `Wanda-kept grad-Crit` (complement) — locates where the causal mass sits
5. `random-equal` (size-matched to set 3)
6. `random-magnitude-matched` (size + magnitude-distribution matched to set 3) — the crucial control

**Metrics (aggregate).** ASR (unsafe & coherent), PPL v2, coherent rate, set size, n.

**Key comparisons.**
- set 3 vs sets 5/6: if `ASR(3) >> ASR(6)` -> the *specific* weights Wanda removes are causal, not
  "any removal of that many / that-large weights."
- set 3 vs set 4: how the full-set effect (set 2) splits between removed and kept portions.
- bridge to natural Wanda ASR: report the share of the (dense -> Wanda-s) ASR gap that set 3 alone
  reproduces.

**Outputs.**
- `results/phase15_causal_bridge/bridge_grid.csv` (condition x sparsity -> ASR/PPL/coherent/size)
- `results/phase15_causal_bridge/decision.json` -> `{asr_wanda_removed, asr_mag_matched_random,
  asr_full, share_of_gap_explained, claim_causal: bool, interpretation}`

**Observed result.** Spec C did **not** pass:

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

The removed-vs-magnitude-matched specificity was 0.000 at 45% and 0.008 at 50%, below the 0.03
threshold. The direct-deletion bridge is therefore false for this carrier and candidate.

**Interpretation.** grad-Crit is causally safety-critical as a whole, but its safety-critical mass is
mostly Wanda-kept. Natural Wanda ASR must therefore come from the broad pruned model perturbing the
activation fed into preserved refusal readouts, not from direct deletion of the tested
Wanda-removed write-out tail.

**Caveats to flag.** (i) This rules out this grad-Crit deletion bridge, not every possible upstream
deleted-circuit bridge. (ii) The negative result is strong enough that deterministic rerun is not the
next priority; removed ASR is near zero while kept ASR reproduces the full effect. (iii) Report
Spec C as a theory-sharpening negative result, not a failure of the overall margin-collapse story.

================================================================================
## Sequencing and what each buys
- **B first** (hours, no generation): if deep-window and o_proj Spearman(`S_safety`, `|w|`) are
  near zero while Wanda cuts fewer top safety-influence weights than pure magnitude but still removes
  a non-trivial subset, Claim 1 is locked as "magnitude-blind at the readout, Wanda-partially-rescued"
  rather than a brittle all-layer pooled claim.
- **A completed** (existing activations + boolean labels): turns "margin" into a measured
  distribution-level decision variable — removes the "metaphor" objection.
- **Closed-form readout repair initial pass**: turns restore-s from an inference hook/oracle into a
  train-free, merged-weight method; W50 eta=1 is the current clean point.
- **Behavioral top-up last**: only needed if high-sparsity residuals remain after readout repair.
