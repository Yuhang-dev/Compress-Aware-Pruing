# Task for Codex — Mechanism-sealing runs (Qwen2.5-3B, Wanda)

Goal: SEAL the two-mechanism result before any external-validity work. No new mechanism — just tighten.
Same carrier (Qwen2.5-3B-Instruct), reuse Step0b/vpref infra + the already-built directions
(r̂ at layers 24/28/32, c_dense). **Text-free outputs only.** Greedy, seed 0, manifest ids.

Two mechanisms to seal:
- M1 = detection-readout collapse (causal + recoverable). Already strong; Run B tightens specificity.
- M2 = shallow-withholding residual (refuse-then-comply at high sparsity). Confirmed at 50% (5/128, manual);
  Run A turns "appears at 50%" into "scales with sparsity" (or honestly bounds it).

## Run A (PRIMARY) — residual sparsity sweep (seals M2)

Fix the Step0b best setting: layer_group {24,28,32}, steering_mode norm_relative, target strong (β_rel 0.25),
k_r 1, window 32. Sweep Wanda sparsity ∈ {0.40, 0.45, 0.50, 0.55}.

Per sparsity, on harm_eval (n=128, same ids):
- baseline_asr (zero / no steer), steered_asr (best setting), residual_count (# still attack_success under steer),
  coherent_rate, mean s_post_patch / propagated_L35 (confirm detection WAS driven ≫ τ for the residual).
- Record residual prompt_ids per sparsity (text-free) so the refuse-then-comply pattern can be spot-checked.
- benign_eval: over-refusal must stay low (gate control), report benign_refusal_rate.

**Coherence guard:** report coherent_rate; if it falls below ~0.85 at high sparsity, FLAG that sparsity as
coherence-confounded (ASR unreliable there — do not over-read the residual count).

Output: `results/phase15_vpref_projection/mechanism_seal_residual_sweep.csv` with per-sparsity columns above.
Claim test (write to log): is residual_count monotone non-decreasing in sparsity at coherent_rate≥0.85?
Report honestly — monotone growth seals M2 as "scales with sparsity"; a plateau / coherence-confound bounds it.

## Run B (SECONDARY) — detection-collapse specificity tightening (tightens M1)

Extends `docs/vpref_specificity_followup_spec.md`. The current controls only reach probe CV-acc 0.83–0.89 and
null n=50 — too weak to claim "collapse is harm-specific". Two items (drop the induce small-α rescan unless
cheap — restore-s already gives strong causal sufficiency):

1. **Control concepts matched to acc≈1.0**: construct/select control directions (e.g. formal-vs-casual,
   topicA-vs-topicB) whose linear-probe CV-acc ≈ 1.0 — i.e. AS linearly decodable as r̂. Re-run "does the
   control direction's projection collapse under wanda_45/50?" A control as decodable as r̂ should NOT collapse;
   if only r̂ collapses, the collapse is harm-specific (not a generic high-acc-direction artifact).
2. **Null distribution 50 → 200** random directions; report the harm-direction collapse magnitude's percentile
   vs the 200-dir null (for p<0.01). No generation needed — pure s-projection on cached activations.

Output: `results/phase15_vpref_projection/mechanism_seal_specificity.csv`
(per control: cv_acc, dense_s_gap, compressed_s_gap, collapse_magnitude; null: 200-dir distribution + harm percentile).

## Compute

Run A: 4 sparsities × 1 setting × (harm+benign) short generations + judge. Run B: probes + 200-dir projection
(no generation). Reuse built directions → no rebuild. Single 24GB GPU, ~1–2h total.

## After (definition of "sealed")

M2 sealed if Run A shows residual_count grows with sparsity at coherent≥0.85 (or bounds it honestly).
M1 sealed if Run B shows harm-direction collapse ≫ null AND matched-acc controls do NOT collapse.
Then mechanism is closed on Qwen2.5-3B → proceed to parked items (motivating baselines → Method B build →
Llama-2-7B-Chat + quantization external validity).
