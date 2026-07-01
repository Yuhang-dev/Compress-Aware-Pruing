# Core Narrative and Theoretical Reasoning (paper-ready, evidence-consistent)

Created 2026-06-30. Purpose: give the paper a single theory from which the method is *derived*,
not reverse-justified, and which is consistent with every experiment we have run. This sharpens the
"safety-margin preservation" framing in `experiment_path_theory_handoff.md` on three points a
reviewer would otherwise attack (see Section 4).

--------------------------------------------------------------------------------
## 1. Core narrative (the elevator version)

Magnitude/saliency pruning of an aligned LLM degrades **safety far faster than utility**: at
sparsities where perplexity is only moderately worse, the model emits *coherent* unsafe completions
and attack-success rate (ASR) climbs steeply with sparsity. The decisive finding is that this is
**not direct deletion of the refusal write-out machinery**. The clean grad-Crit set is causally
safety-critical as a whole, but Wanda mostly keeps its causal mass; zeroing only the part Wanda would
remove has almost no ASR effect. Instead, pruning violates a constraint that its objective never
represents. Standard pruning preserves benign-distribution reconstruction, but alignment depends on
a low-dimensional **refusal margin** on harmful inputs. Wanda's broad 50% perturbation changes the
activation flowing into otherwise preserved refusal readouts:
`h_l^pruned(x)=h_l(x)+delta h_l(x)`, so `s_l=r_l^T h_l` shifts by
`delta s_l=r_l^T delta h_l`. Harmful prompts sit close to the refusal threshold, and this shift can
push them below it; benign prompts remain far away and usually do not flip. The failure is therefore
**two-staged**: at moderate sparsity it is a direction-specific *readout/margin collapse* that can be
almost entirely reversed by restoring the activation readout; at high sparsity a
*behavioral-enforcement residual* survives even after the readout is restored. The method should now
follow the mechanism: convert the diagnostic activation repair into a **closed-form, train-free
readout repair** that merges a small low-rank update into the compressed model, then use a small
behavioral top-up only for the high-sparsity residual. **The method repairs the violated refusal
margin constraint, not a weight set discovered by ablation.**

--------------------------------------------------------------------------------
## 2. The theory (four claims, each evidence-locked and gap-flagged)

**Setup.** Layer-l residual activation `h_l(x)`; dense refusal direction `r_l` (unit-norm
harmful-minus-benign mean activation); refusal readout `s_l(x) = r_l^T h_l(x)`; refusal margin
`m_l(x) = s_l(x) - tau_l`. Pruning mask `M`, induced perturbation `dW = -W (1-M)` (the removed
weights). To first order, the change in readout on input x is `ds_l(x) = r_l^T dh_l(x)`.

### Claim 1 — Objective/measure mismatch (the central tension)
Wanda keeps weights maximizing **benign-calibration reconstruction saliency** `|W_ij| * ||X_j||`.
Safety depends on `m_l(x)` for x **harmful**. These objectives are misaligned on two axes:
- **distribution:** `||X_j||` is estimated on benign text, under-weighting channels that carry
  harmful-content features;
- **objective:** reconstruction (match `h`) is not margin (keep `r_l^T h` above `tau_l`); even on
  identical inputs, `d(recon)/dw != dm/dw`.

So `argmin_M E_benign|| WX - (W M)X ||` does **not** control `min_{x harmful} m_l(x; W M)`.

*Evidence lock:* the clean safety-support set (grad-Crit) has **magnitude percentile p50 = 0.493
(~random)** yet zeroing it raises ASR 0 -> 0.359 in the causal-bridge harness while equal-size and
magnitude-matched random zeroing gives 0. Recutting
Spec B shows that `S_safety=E|ds_l/dw|` is essentially uncorrelated with `|w|` at the refusal-readout
layers: for the L28 decision readout, per-layer Spearman is near zero over the validated deep window
L22-L28 (median 0.0146, range -0.0073 to 0.0277), and the o_proj-pooled Spearman is 0.0044.
Early generic layers do show magnitude-importance coupling (e.g., L1-L4 are 0.57-0.69), so a uniform
all-layer pooled Spearman of 0.223 is a misleading Simpson-style aggregation artifact, not the
right Claim-1 statistic. Wanda's `||X||` term induces a mild positive correlation (o_proj
Spearman with Wanda = 0.343; all-target pooled Spearman = 0.449), partly mechanical because
`S_safety` and Wanda both share input-channel activation factors. Consequently Wanda removes only
26.4% of the top-q safety-influence weights at 50% sparsity, compared with 53.5% for pure magnitude
pruning. Spec C then shows this Wanda-removed subset is **not** the direct causal source of natural
Wanda ASR: zeroing it in the dense model gives ASR 0.000 at 45% and 0.008 at 50%, while zeroing the
Wanda-kept complement gives ASR 0.367/0.320.

*Updated interpretation:* the objective mismatch is real, but its carrier is **activation-margin
erosion**, not direct deletion of the identified write-out safety weights. Wanda preserves much of
the safety-critical write-out mass; broad upstream perturbations change the representation fed into
that preserved readout.

### Claim 2 — Stage 1: detection-margin collapse (moderate sparsity)
The leading-order effect of `dW` on harmful prompts is a reduction of the scalar margin along the
**fixed** direction `r_l`: `dm_l(x) ~= r_l^T dh_l(x)`.

*Evidence lock:* (i) **Seal B** — the collapse is specific to `r_l` vs 200 random directions
(centered-cosine flip 0.09-0.12 >> null p95 ~0.01, p~0.005), causally validated at L24/28; (ii)
**Seal A** — restoring the lost projection (restore-s) gives residual **0 at 40-45%**; (iii)
**Spec A** — the calibrated multi-layer readout `s_mean` has pooled AUC **0.907** as a
distribution-level refusal margin, `frac(s_mean<tau)` rises monotonically
0.008 -> 0.070 -> 0.211 -> 0.594 -> 1.000 over dense/W40/W45/W50/W55, and this fraction tracks ASR
with descriptive Pearson **0.995**. So at moderate sparsity, safety failure is largely a measured
margin erosion.

*Caveat:* grouped-CV AUC is **0.786** for `s_mean`; within a fixed sparsity bucket, the scalar is only
a moderate per-sample classifier. This is not a contradiction: the main mechanism is a
distribution-level margin shift, not perfect prompt-level ranking.

### Claim 3 — Stage 2: behavioral-enforcement residual (high sparsity)
As sparsity grows, `||dW||` grows and the single-direction first-order picture is no longer
sufficient: restoring `s_l` along `r_l` leaves a residual of coherent unsafe outputs (**5/128 at
50%, 9/128 at 55%**). Interpretation: the map from internal *detection* to refusal *behavior*
(downstream layers / the policy converting "harm detected" into refusal tokens) is itself degraded.
**Two constraints, not one:** detection `m_l(x) >= 0` AND enforcement `g(h_{>l}) -> refuse`.

*Evidence lock:* residual sweep `[0, 0, 5, 9]` monotone in sparsity, measured **after** successful
readout restoration -> a second failure mode that the readout fix cannot reach.

*Gap/flag:* counts are small (5, 9) -> underpowered. Must (a) confirm those residuals are
coherent-unsafe (not garbage), (b) widen n at 50/55 before any *quantitative* Stage-2 claim. State
as "a second, growing failure mode," not a law.

### Claim 4 — Method as first-order constraint repair (derived, not discovered)
**Constrained problem:**
```
min_theta'   UtilityLoss(theta')
s.t.         theta' respects the compression/deployment constraint
             m_l(x; theta') >= 0  for all x in calib_harm
```
The strongest train-free realization is no longer mask swap. Restore-s proves that the intervention
we need is an activation-space map: raise `s_l(x)=r_l^T h_l(x)` for harmful prompts without broadly
changing benign behavior. The deployable version should be a **closed-form low-rank readout repair**:
collect compressed hidden states `h_l^pruned(x)`, dense targets `s_l^dense(x)` or a thresholded target
`tau_l + margin`, solve a small ridge/least-squares update `Delta W` on the readout-producing
matrices, and merge `W' = W_pruned + Delta W`. This is train-free, hook-free at inference, and
derived directly from the measured violated constraint. **Stage-2** (high sparsity) is out of the
single readout's span by Claim 3, so it requires a small **behavioral top-up** only after the closed
form repair saturates. The two method tiers remain the two mechanism stages: readout-margin repair
first, behavioral enforcement second.

--------------------------------------------------------------------------------
## 3. Consistency map (every result -> its slot in the theory)

| Result | Number | Supports |
|---|---|---|
| Phase 0 ASR(sparsity) | 0.047 / 0.086 / 0.266 / 0.625 @ 40/45/50/55 | Phenomenon: safety erodes with sparsity |
| Phase 0 PPL vs ASR | PPL 8.22->12.22->15.63 while ASR 6x | Decoupling -> Claim 1 |
| Crit v1 | 2.3%, ASR->0.67, PPL +63%, **W50 survival 97%** | "Not deletion": strong refusal write-out is *retained* |
| Crit v2 grad | 0.56%, ASR 0.41 (rand 0), PPL +8.3%, **mag p50 0.493**, Wanda p50 0.726, W50 cut 26.4% | Claim 1 readout-layer magnitude-blindness + Wanda partial rescue |
| v2 grid strong | ps0.0075 pu0.08: 0.36%, +26.6pp ASR, +4.63% PPL | A clean, small, low-PPL margin-support set exists |
| Seal B | flip 0.09-0.12 >> null p95 ~0.01, p~0.005, L24/28 | Claim 2: direction-specific, causal |
| Seal A | restore-s residual [0,0,5,9] | Claim 2 (0 at 40-45) + Claim 3 (residual at 50-55) |
| Step0b | multi-layer L24/28/32 steering recovers ASR | Claim 2 mechanism; restore-s is the diagnostic, not the final method |
| Spec C causal bridge | removed-zero ASR 0/0.008; kept-zero ASR 0.367/0.320 @ 45/50 | Direct deletion of Wanda-removed grad-Crit is **not** the natural Wanda ASR cause |
| Spec A margin | `s_mean` pooled AUC 0.907; grouped-CV AUC 0.786; frac_m_neg-ASR corr 0.995 | Claim 2: distribution-level refusal margin |
| Closed-form repair v1 | W50 eta1 ASR 0.289->0.133, PPL +0.055, benign refusal +3.13pp; random same-eta ASR 0.258 | Claim 4: train-free readout repair works at W50 under guardrails |

The table now rules out the old direct-deletion bridge. The surviving theory is that broad pruning
perturbations erode the harmful-prompt readout margin through shifted activations, while the
write-out weights that would execute refusal are mostly still present.

--------------------------------------------------------------------------------
## 4. Where I sharpened Codex's framing (and why)

1. **From "rare event" to "objective/measure mismatch + readout-layer magnitude-blindness."** "Safety is rare in
   calibration" invites the rebuttal "add harmful prompts." The defensible core is that the pruning
   *objective* (benign reconstruction) is the wrong loss, and the data proves the safety weights are
   magnitude-blind where the refusal readout lives (deep-window Spearman ~0, o_proj-pooled Spearman
   0.004, mag p50 0.493). Wanda partially rescues them via `||X||`, but still removes a non-trivial
   subset. Lead with that rather than an absolute orthogonality claim.
2. **Make the two stages fall out of ONE theory.** Codex listed "Stage 1 / Stage 2" as two named
   buckets — itself a mild empirical-hack smell. Here they are one constraint set: detection margin
   `m_l>=0` (1-D, readout-restorable) AND enforcement `g(h_{>l})->refuse` (not in the readout span).
   Seal A's residual is then *predicted*: restoring a 1-D readout cannot fix a failure outside its
   span, and that residual must grow with perturbation size (sparsity). The data `[0,0,5,9]` matches.
3. **Method tiers = mechanism stages, by construction.** Closed-form readout repair is the direct
   deployable analogue of restore-s for the detection-margin constraint; the behavioral top-up is the
   only thing that can touch the enforcement constraint. So "why two method parts" is answered by
   theory, not convenience.

--------------------------------------------------------------------------------
## 5. Three cheap, text-free analyses that harden the theory *before* any new training

(a) **Calibrate the margin into a real classifier. Completed.** On the dense/pruned cells, fit
    `tau_l` as the `s_l` value separating refuse vs comply. The correct headline is distributional:
    pooled `s_mean` AUC is 0.907, while grouped-CV AUC is 0.786 as a within-sparsity caveat. The
    pruned harmful-prompt `s_l` distribution shifts below `tau_l`, and `ASR(sparsity)` tracks the
    fraction with `m_l<0`. This converts "margin" from metaphor to a measured decision boundary.
(b) **Quantify the mismatch.** Report Spearman correlation between margin-influence and `|w|` at the
    refusal-readout layers; show it is ~0 there and robust for L>=8/L>=12/L>=16, while early generic
    layers exhibit the expected coupling. Report Wanda separately as partial rescue, not as the
    orthogonality test.
(c) **Closed-form readout repair. Initial pass.** Replace inference-time restore-s hooks with a
    merged low-rank `Delta W` that raises the measured margin on harmful calibration prompts while
    regularizing benign drift. W50 eta1 reduces ASR 0.289 -> 0.133 with PPL +0.055 and benign
    refusal +3.13pp; eta2 is a useful diagnostic upper point but over-refuses.

Then the readout-repair experiment, not mask repair, is the result that converts the
theory-derived mechanism into a main paper method.

--------------------------------------------------------------------------------
## 6. Paper spine (section logic)

1. **Intro** — safety breaks before utility under compression (hook + Phase-0 curve).
2. **It is not deletion** — refusal write-out mostly survives Wanda (Crit v1 survival 97%) -> a puzzle.
3. **The violated constraint** — objective/measure mismatch; refusal margin; readout-layer magnitude-blindness plus Wanda partial rescue (Claim 1).
4. **Two-stage mechanism** — detection-margin collapse (Claim 2: Seal B + Seal A@40-45) and enforcement residual (Claim 3: Seal A@50-55).
5. **Method** — closed-form readout repair (Claim 4) + behavioral top-up; full derivation from the constrained margin objective.
6. **Experiments** — mechanism validation (done) + readout repair at measured utility/safety tradeoff (initial W50 pass) + behavioral residual analysis.
7. **Related work** — distinguish Wei (harmful-FT brittleness), Circuit-Breakers/RR (attack-robust), ACTOR/ProSafePrune (over-refusal axis), prior pruning-safety; ours = **benign-compression margin preservation**.
8. **Limitations** — Qwen2.5-3B / Wanda / AdvBench-direct scope; external validity (Llama-2, 7B, SparseGPT/AWQ/GPTQ, StrongREJECT/XSTest/OR-Bench) future.

--------------------------------------------------------------------------------
## 7. Abstract-level wording seed

> Pruning an aligned language model can leave perplexity moderately degraded while producing coherent
> unsafe completions, and we show this is not explained by deletion of refusal write-out weights,
> which standard pruners largely retain. We trace the failure to a mismatch of objectives:
> magnitude/saliency pruning preserves benign-distribution reconstruction, but alignment depends on a
> low-dimensional refusal margin on harmful inputs. Broad pruning perturbations shift the activation
> fed into preserved refusal readouts, pushing harmful prompts below the refusal threshold. The
> degradation is two-staged - a direction-specific, causally-validated collapse of the refusal readout
> at moderate sparsity, and a residual behavioral-enforcement failure at high sparsity that persists
> after the readout is restored. From this we derive a training-free repair: a closed-form low-rank
> update that merges the diagnostic readout restoration into the compressed model, with a small
> behavioral top-up only for the high-sparsity residual. The method is the repair of a constraint that
> pruning violates, not an empirically selected set of weights to protect.

--------------------------------------------------------------------------------
## 8. Honest claim boundary (for the abstract/intro to not overclaim)
- **Proven:** phenomenon; non-deletion; direct-deletion bridge is false for grad-Crit; direction-specific
  causal margin collapse; readout restoration clears 40-45% and leaves a growing residual at 50-55%;
  a small utility-light causal safety-support set exists, but its Wanda-removed subset is not the natural ASR cause.
- **Proven/initially shown:** margin as a calibrated **distribution-level** classifier (pooled AUC
  0.907; grouped-CV caveat 0.786); closed-form readout repair reduces W50 ASR without an inference
  hook under PPL/benign-refusal guardrails.
- **Derived but not yet shown:** a behavioral top-up handles the high-sparsity residual.
- **Future:** external validity across models/compressors/benchmarks; whether one-shot repair suffices at 50-55%.
