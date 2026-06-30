from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import load_config
from .crit_selection_v2 import (
    load_calib_prompts,
    load_harmful_prompts,
    module_kind,
    module_layer,
    read_jsonl,
    selected_linear_modules,
)
from .models import model_slug, resolve_model_id
from .phase0_smoke_eval import collect_wanda_input_norms, format_prompt, load_model_and_tokenizer
from .ppl_eval_v2 import PplWindow, prepare_ppl_inputs
from .pruners import compute_mask


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def active_modules_for_readout(
    modules: list[tuple[str, nn.Linear]],
    readout_layer: int,
    *,
    include_equal: bool = True,
) -> list[tuple[str, nn.Linear]]:
    active = []
    for name, module in modules:
        layer = module_layer(name)
        if layer is None:
            continue
        if layer < readout_layer or (include_equal and layer == readout_layer):
            active.append((name, module))
    return active


def set_trainable_modules(model: nn.Module, modules: Iterable[tuple[str, nn.Linear]]) -> None:
    trainable_ids = {id(module.weight) for _name, module in modules}
    for param in model.parameters():
        param.requires_grad_(id(param) in trainable_ids)


def load_refusal_direction(artifact_dir: Path, model_id: str, layer: int) -> torch.Tensor:
    path = artifact_dir / f"{model_slug(model_id)}_layer{layer}_kr1.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing refusal direction artifact: {path}. "
            "Run phase15_vpref_projection first or set --artifact-dir."
        )
    payload = torch.load(path, map_location="cpu")
    r_hat = payload.get("r_hat")
    if not isinstance(r_hat, torch.Tensor):
        raise ValueError(f"Artifact {path} does not contain tensor key 'r_hat'.")
    return r_hat.detach().float().cpu()


def target_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def accumulate_abs_grads(
    modules: list[tuple[str, nn.Linear]],
    accum: dict[str, torch.Tensor],
    *,
    weighted: bool,
) -> None:
    for name, module in modules:
        grad = module.weight.grad
        if grad is None:
            continue
        value = grad.detach().float()
        if weighted:
            value = value * module.weight.detach().float()
        value = value.abs().cpu()
        if name not in accum:
            accum[name] = value
        else:
            accum[name].add_(value)


def average_accum(accum: dict[str, torch.Tensor], denom: int) -> dict[str, torch.Tensor]:
    return {name: tensor.div(max(1, denom)) for name, tensor in accum.items()}


def safety_scores_for_layer(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    modules: list[tuple[str, nn.Linear]],
    readout_layer: int,
    r_hat_cpu: torch.Tensor,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    grad_accum: dict[str, torch.Tensor] = {}
    wgrad_accum: dict[str, torch.Tensor] = {}
    device = target_device(model)
    r_hat = r_hat_cpu.to(device=device, dtype=torch.float32)

    for pos, prompt in enumerate(prompts, start=1):
        model.zero_grad(set_to_none=True)
        text = format_prompt(tokenizer, prompt)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        final_idx = int(encoded["attention_mask"][0].sum().item()) - 1
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[readout_layer + 1][0, final_idx].float()
        score = hidden.dot(r_hat)
        score.backward()
        accumulate_abs_grads(modules, grad_accum, weighted=False)
        accumulate_abs_grads(modules, wgrad_accum, weighted=True)
        del outputs, encoded, score
        if torch.cuda.is_available() and pos % 16 == 0:
            torch.cuda.empty_cache()

    return average_accum(grad_accum, len(prompts)), average_accum(wgrad_accum, len(prompts))


def utility_scores(
    model: nn.Module,
    input_ids: torch.Tensor,
    windows: list[PplWindow],
    modules: list[tuple[str, nn.Linear]],
) -> dict[str, torch.Tensor]:
    accum: dict[str, torch.Tensor] = {}
    device = target_device(model)
    for pos, window in enumerate(windows, start=1):
        model.zero_grad(set_to_none=True)
        chunk = input_ids[window.begin : window.end].unsqueeze(0).to(device)
        labels = chunk.clone()
        labels[:, : window.target_start] = -100
        outputs = model(input_ids=chunk, labels=labels, use_cache=False)
        outputs.loss.backward()
        accumulate_abs_grads(modules, accum, weighted=True)
        del outputs, chunk, labels
        if torch.cuda.is_available() and pos % 8 == 0:
            torch.cuda.empty_cache()
    return average_accum(accum, len(windows))


def module_wanda_score(module: nn.Linear, input_norms: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    input_norm = input_norms.get(name)
    if input_norm is None:
        input_norm = torch.ones(module.weight.shape[1], dtype=torch.float32)
    return module.weight.detach().float().cpu().abs() * input_norm.float().cpu().view(1, -1)


def module_mag_score(module: nn.Linear) -> torch.Tensor:
    return module.weight.detach().float().cpu().abs()


def sample_module_values(
    modules: list[tuple[str, nn.Linear]],
    value_by_name: dict[str, torch.Tensor],
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sizes = [(name, int(value_by_name.get(name, torch.empty(0)).numel())) for name, _module in modules]
    total = sum(size for _name, size in sizes)
    if total <= 0:
        return np.array([], dtype=np.float32), {}
    rng = np.random.default_rng(seed)
    sample_n = min(max_samples, total)
    sampled_names: list[str] = []
    sampled_offsets: list[np.ndarray] = []
    remaining = sample_n
    remaining_total = total
    for name, size in sizes:
        if size <= 0:
            continue
        count = int(round(sample_n * size / total))
        count = min(size, max(0, count))
        if remaining_total == size:
            count = min(size, remaining)
        remaining -= count
        remaining_total -= size
        if count <= 0:
            continue
        sampled_names.append(name)
        sampled_offsets.append(rng.choice(size, size=count, replace=False))
    arrays: dict[str, list[np.ndarray]] = {}
    for name, offsets in zip(sampled_names, sampled_offsets):
        base = value_by_name[name].flatten()
        arrays.setdefault("primary", []).append(base[offsets].numpy().astype(np.float32, copy=False))
    primary = np.concatenate(arrays.get("primary", [np.array([], dtype=np.float32)]))
    offset_by_name = {name: offsets for name, offsets in zip(sampled_names, sampled_offsets)}
    return primary, offset_by_name


def values_at_offsets(value_by_name: dict[str, torch.Tensor], offsets_by_name: dict[str, np.ndarray]) -> np.ndarray:
    chunks = []
    for name, offsets in offsets_by_name.items():
        tensor = value_by_name.get(name)
        if tensor is None:
            continue
        chunks.append(tensor.flatten()[offsets].numpy().astype(np.float32, copy=False))
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_corr(rx, ry)


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kind: str,
    reps: int,
    sample_size: int,
    seed: int,
) -> tuple[float, float]:
    if x.size < 4 or y.size < 4 or reps <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = x.size
    take = min(sample_size, n)
    values = []
    fn = pearson_corr if kind == "pearson" else spearman_corr
    for _ in range(reps):
        idx = rng.integers(0, n, size=take)
        value = fn(x[idx], y[idx])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def correlation_rows(
    *,
    readout_layer: int,
    safety_variant: str,
    safety_by_name: dict[str, torch.Tensor],
    comparators: dict[str, dict[str, torch.Tensor]],
    modules: list[tuple[str, nn.Linear]],
    scope: str,
    group_value: str = "",
    max_samples: int,
    bootstrap_reps: int,
    bootstrap_sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    safety_sample, offsets = sample_module_values(modules, safety_by_name, max_samples=max_samples, seed=seed)
    rows = []
    for compare_name, compare_by_name in comparators.items():
        compare_sample = values_at_offsets(compare_by_name, offsets)
        n = min(safety_sample.size, compare_sample.size)
        x = safety_sample[:n]
        y = compare_sample[:n]
        pearson = pearson_corr(x, y)
        spearman = spearman_corr(x, y)
        pearson_lo, pearson_hi = bootstrap_corr_ci(
            x,
            y,
            kind="pearson",
            reps=bootstrap_reps,
            sample_size=bootstrap_sample_size,
            seed=seed + 17,
        )
        spearman_lo, spearman_hi = bootstrap_corr_ci(
            x,
            y,
            kind="spearman",
            reps=bootstrap_reps,
            sample_size=bootstrap_sample_size,
            seed=seed + 31,
        )
        rows.append(
            {
                "readout_layer": readout_layer,
                "safety_score": safety_variant,
                "scope": scope,
                "group": group_value,
                "compare_to": compare_name,
                "pearson": pearson,
                "pearson_ci_low": pearson_lo,
                "pearson_ci_high": pearson_hi,
                "spearman": spearman,
                "spearman_ci_low": spearman_lo,
                "spearman_ci_high": spearman_hi,
                "sampled_weights": int(n),
                "active_weights": int(sum(module.weight.numel() for _name, module in modules)),
                "bootstrap_reps": bootstrap_reps,
                "bootstrap_sample_size": min(bootstrap_sample_size, int(n)),
            }
        )
    return rows


def load_crit_indices(path: Path | None) -> tuple[str, dict[str, torch.Tensor]]:
    if path is None:
        raise ValueError("A crit set is required for Spec B. Pass --crit-set-path or --crit-candidate.")
    if not path.exists():
        raise FileNotFoundError(f"Missing expected crit set: {path}")
    payload = torch.load(path, map_location="cpu")
    candidate = str(payload.get("candidate", path.stem))
    indices = payload.get("crit_indices")
    if not isinstance(indices, dict):
        raise ValueError(f"Crit set {path} does not contain dict key 'crit_indices'.")
    clean = {str(name): idx.to(torch.int64).cpu() for name, idx in indices.items() if isinstance(idx, torch.Tensor)}
    return candidate, clean


def percentile_values(reference: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if reference.numel() == 0 or values.numel() == 0:
        return torch.empty(0)
    sorted_ref = torch.sort(reference.flatten().float()).values
    positions = torch.searchsorted(sorted_ref, values.flatten().float(), right=True)
    return positions.float() / max(1, sorted_ref.numel())


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, reps: int = 200, sample_size: int = 100_000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    take = min(sample_size, values.size)
    means = []
    for _ in range(reps):
        idx = rng.integers(0, values.size, size=take)
        means.append(float(values[idx].mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def percentile_summary_rows(
    *,
    readout_layer: int,
    candidate: str,
    crit_indices: dict[str, torch.Tensor],
    modules: list[tuple[str, nn.Linear]],
    input_norms: dict[str, torch.Tensor],
    seed: int,
    max_values_for_ci: int,
) -> list[dict[str, object]]:
    if not crit_indices:
        return [
            {
                "readout_layer": readout_layer,
                "candidate": candidate,
                "scope": "missing_crit_set",
                "selected_count": 0,
            }
        ]
    rows = []
    overall_mag = []
    overall_wanda = []
    for name, module in modules:
        idx = crit_indices.get(name)
        if idx is None or idx.numel() == 0:
            continue
        flat_idx = idx.to(torch.int64)
        mag = module_mag_score(module)
        wanda = module_wanda_score(module, input_norms, name)
        selected_mag = mag.flatten()[flat_idx]
        selected_wanda = wanda.flatten()[flat_idx]
        mag_pct = percentile_values(mag, selected_mag).numpy()
        wanda_pct = percentile_values(wanda, selected_wanda).numpy()
        overall_mag.append(mag_pct)
        overall_wanda.append(wanda_pct)
        layer = module_layer(name)
        row = {
            "readout_layer": readout_layer,
            "candidate": candidate,
            "scope": "module",
            "module": name,
            "module_layer": layer,
            "module_kind": module_kind(name),
            "selected_count": int(flat_idx.numel()),
            "mag_percentile_mean": float(mag_pct.mean()) if mag_pct.size else float("nan"),
            "mag_percentile_p50": float(np.percentile(mag_pct, 50)) if mag_pct.size else float("nan"),
            "wanda_percentile_mean": float(wanda_pct.mean()) if wanda_pct.size else float("nan"),
            "wanda_percentile_p50": float(np.percentile(wanda_pct, 50)) if wanda_pct.size else float("nan"),
        }
        rows.append(row)

    if overall_mag:
        mag_all = np.concatenate(overall_mag)
        wanda_all = np.concatenate(overall_wanda)
        if mag_all.size > max_values_for_ci:
            rng = np.random.default_rng(seed)
            take = rng.choice(mag_all.size, size=max_values_for_ci, replace=False)
            mag_ci_values = mag_all[take]
            wanda_ci_values = wanda_all[take]
        else:
            mag_ci_values = mag_all
            wanda_ci_values = wanda_all
        mag_lo, mag_hi = bootstrap_mean_ci(mag_ci_values, seed=seed + 101)
        wanda_lo, wanda_hi = bootstrap_mean_ci(wanda_ci_values, seed=seed + 211)
        rows.append(
            {
                "readout_layer": readout_layer,
                "candidate": candidate,
                "scope": "overall",
                "selected_count": int(mag_all.size),
                "mag_percentile_mean": float(mag_all.mean()),
                "mag_percentile_ci_low": mag_lo,
                "mag_percentile_ci_high": mag_hi,
                "mag_percentile_p50": float(np.percentile(mag_all, 50)),
                "wanda_percentile_mean": float(wanda_all.mean()),
                "wanda_percentile_ci_low": wanda_lo,
                "wanda_percentile_ci_high": wanda_hi,
                "wanda_percentile_p50": float(np.percentile(wanda_all, 50)),
            }
        )
    return rows


def global_top_threshold(
    score_by_name: dict[str, torch.Tensor],
    modules: list[tuple[str, nn.Linear]],
    k: int,
) -> float:
    local_tops = []
    for name, _module in modules:
        score = score_by_name.get(name)
        if score is None or score.numel() == 0:
            continue
        flat = score.flatten()
        local_k = min(k, flat.numel())
        if local_k <= 0:
            continue
        local_tops.append(torch.topk(flat, k=local_k, largest=True).values.cpu())
    if not local_tops:
        return float("inf")
    candidates = torch.cat(local_tops)
    global_k = min(k, candidates.numel())
    return float(torch.topk(candidates, k=global_k, largest=True).values[-1])


def high_safety_cut_rows(
    *,
    readout_layer: int,
    safety_variant: str,
    score_by_name: dict[str, torch.Tensor],
    modules: list[tuple[str, nn.Linear]],
    input_norms: dict[str, torch.Tensor],
    sparsities: list[float],
    q_specs: list[tuple[str, float]],
) -> list[dict[str, object]]:
    active_weights = int(sum(module.weight.numel() for _name, module in modules))
    cut_masks: dict[tuple[str, str, float], torch.Tensor] = {}
    cut_base_counts: dict[tuple[str, float], int] = {("wanda", sparsity): 0 for sparsity in sparsities}
    cut_base_counts.update({("magnitude", sparsity): 0 for sparsity in sparsities})
    for name, module in modules:
        input_norm = input_norms.get(name)
        if input_norm is None:
            input_norm = torch.ones(module.weight.shape[1], dtype=torch.float32)
        wanda_stats = {"input_norm": input_norm.to(module.weight.device)}
        for sparsity in sparsities:
            with torch.inference_mode():
                wanda_mask = compute_mask(module.weight.detach(), wanda_stats, sparsity, "wanda")
                mag_mask = compute_mask(module.weight.detach(), None, sparsity, "magnitude")
            wanda_cut = ~wanda_mask.detach().flatten().cpu().bool()
            mag_cut = ~mag_mask.detach().flatten().cpu().bool()
            cut_masks[(name, "wanda", sparsity)] = wanda_cut
            cut_masks[(name, "magnitude", sparsity)] = mag_cut
            cut_base_counts[("wanda", sparsity)] += int(wanda_cut.sum().item())
            cut_base_counts[("magnitude", sparsity)] += int(mag_cut.sum().item())
    rows = []
    for q_label, q_fraction in q_specs:
        if q_fraction <= 0:
            continue
        q_count = max(1, int(round(active_weights * q_fraction)))
        threshold = global_top_threshold(score_by_name, modules, q_count)
        selected_total = 0
        selected_wanda_cut = {sparsity: 0 for sparsity in sparsities}
        selected_magnitude_cut = {sparsity: 0 for sparsity in sparsities}
        for name, module in modules:
            score = score_by_name.get(name)
            if score is None:
                continue
            selected = score.flatten() >= threshold
            selected_count = int(selected.sum().item())
            selected_total += selected_count
            if selected_count == 0:
                continue
            for sparsity in sparsities:
                selected_cpu = selected.cpu()
                selected_wanda_cut[sparsity] += int((selected_cpu & cut_masks[(name, "wanda", sparsity)]).sum().item())
                selected_magnitude_cut[sparsity] += int(
                    (selected_cpu & cut_masks[(name, "magnitude", sparsity)]).sum().item()
                )
        for sparsity in sparsities:
            wanda_cut_count = selected_wanda_cut[sparsity]
            mag_cut_count = selected_magnitude_cut[sparsity]
            wanda_frac = wanda_cut_count / selected_total if selected_total else float("nan")
            mag_frac = mag_cut_count / selected_total if selected_total else float("nan")
            wanda_base = cut_base_counts[("wanda", sparsity)] / active_weights if active_weights else float("nan")
            mag_base = cut_base_counts[("magnitude", sparsity)] / active_weights if active_weights else float("nan")
            rows.append(
                {
                    "readout_layer": readout_layer,
                    "safety_score": safety_variant,
                    "q_label": q_label,
                    "q_fraction": q_fraction,
                    "q_count_requested": q_count,
                    "topq_count_observed": selected_total,
                    "sparsity": sparsity,
                    "wanda_cut_count": wanda_cut_count,
                    "magnitude_cut_count": mag_cut_count,
                    "wanda_cut_base_rate": wanda_base,
                    "magnitude_cut_base_rate": mag_base,
                    "frac_topq_in_wanda_cut": wanda_frac,
                    "frac_topq_in_magnitude_cut": mag_frac,
                    "wanda_cut_lift_vs_base": wanda_frac - wanda_base if math.isfinite(wanda_frac) else float("nan"),
                    "magnitude_cut_lift_vs_base": mag_frac - mag_base if math.isfinite(mag_frac) else float("nan"),
                    "threshold": threshold,
                }
            )
    return rows


def default_crit_set_path(output_root: Path, model_id: str, candidate: str) -> Path:
    return output_root / "crit_sets" / f"{model_slug(model_id)}_{candidate}.pt"


def required_direction_path(artifact_dir: Path, model_id: str, layer: int) -> Path:
    return artifact_dir / f"{model_slug(model_id)}_layer{layer}_kr1.pt"


def write_progress_outputs(
    output_dir: Path,
    corr_rows: list[dict[str, object]],
    percentile_rows: list[dict[str, object]],
    highcut_rows: list[dict[str, object]],
    *,
    layer: int,
) -> None:
    pd.DataFrame(corr_rows).to_csv(output_dir / f"weight_score_correlations_through_layer{layer}.csv", index=False)
    pd.DataFrame(percentile_rows).to_csv(output_dir / f"safety_support_percentiles_through_layer{layer}.csv", index=False)
    pd.DataFrame(highcut_rows).to_csv(output_dir / f"highsafety_in_wanda_cut_through_layer{layer}.csv", index=False)


def build_decision(
    correlations: pd.DataFrame,
    highcut: pd.DataFrame,
    *,
    decision_layer: int,
    corr_threshold: float,
    cut_threshold: float,
    cut_base_tolerance: float,
) -> dict[str, object]:
    primary = correlations[
        (correlations["readout_layer"].astype(int) == int(decision_layer))
        & correlations["safety_score"].eq("grad")
        & correlations["scope"].eq("pooled")
        & correlations["compare_to"].isin(["mag", "wanda"])
    ]
    corr_mag = float("nan")
    corr_wanda = float("nan")
    for _, row in primary.iterrows():
        if str(row["compare_to"]) == "mag":
            corr_mag = float(row["spearman"])
        elif str(row["compare_to"]) == "wanda":
            corr_wanda = float(row["spearman"])
    cut_rows = highcut[
        (highcut["readout_layer"].astype(int) == int(decision_layer))
        & highcut["safety_score"].eq("grad")
        & highcut["q_label"].eq("gradcrit")
        & (highcut["sparsity"].astype(float) == 0.50)
    ]
    if cut_rows.empty:
        cut_rows = highcut[
            (highcut["readout_layer"].astype(int) == int(decision_layer))
            & highcut["safety_score"].eq("grad")
            & (highcut["sparsity"].astype(float) == 0.50)
        ].head(1)
    frac_topq = float(cut_rows.iloc[0]["frac_topq_in_wanda_cut"]) if not cut_rows.empty else float("nan")
    wanda_base = float(cut_rows.iloc[0].get("wanda_cut_base_rate", float("nan"))) if not cut_rows.empty else float("nan")
    wanda_lift = float(cut_rows.iloc[0].get("wanda_cut_lift_vs_base", float("nan"))) if not cut_rows.empty else float("nan")
    mag_frac = float(cut_rows.iloc[0].get("frac_topq_in_magnitude_cut", float("nan"))) if not cut_rows.empty else float("nan")
    mag_base = float(cut_rows.iloc[0].get("magnitude_cut_base_rate", float("nan"))) if not cut_rows.empty else float("nan")
    mag_lift = float(cut_rows.iloc[0].get("magnitude_cut_lift_vs_base", float("nan"))) if not cut_rows.empty else float("nan")
    magnitude_cut_base_like = (
        math.isfinite(mag_frac)
        and math.isfinite(mag_base)
        and mag_frac >= mag_base - cut_base_tolerance
    )
    claim_orthogonal = (
        math.isfinite(corr_mag)
        and abs(corr_mag) <= corr_threshold
        and magnitude_cut_base_like
    )
    claim_wanda_exposed = (
        math.isfinite(frac_topq)
        and frac_topq >= cut_threshold
    )
    return {
        "decision_layer": int(decision_layer),
        "corr_safety_magnitude_spearman": corr_mag,
        "corr_safety_wanda_spearman": corr_wanda,
        "frac_topq_in_wanda_cut_at_50": frac_topq,
        "wanda_cut_base_rate_at_50": wanda_base,
        "wanda_cut_lift_vs_base_at_50": wanda_lift,
        "frac_topq_in_magnitude_cut_at_50": mag_frac,
        "magnitude_cut_base_rate_at_50": mag_base,
        "magnitude_cut_lift_vs_base_at_50": mag_lift,
        "corr_threshold": corr_threshold,
        "wanda_operational_exposure_threshold": cut_threshold,
        "cut_base_tolerance": cut_base_tolerance,
        "magnitude_cut_base_like": bool(magnitude_cut_base_like),
        "claim_orthogonal": bool(claim_orthogonal),
        "claim_wanda_operational_exposure": bool(claim_wanda_exposed),
        "interpretation": (
            "safety-margin influence is weakly aligned with magnitude; Wanda exposure is reported separately from base-rate lift"
            if claim_orthogonal
            else "mismatch evidence is incomplete or below the configured decision thresholds"
        ),
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)
    layers = parse_int_list(args.layers)
    missing_artifacts = [str(required_direction_path(args.artifact_dir, model_id, layer)) for layer in layers if not required_direction_path(args.artifact_dir, model_id, layer).exists()]
    if missing_artifacts:
        raise FileNotFoundError(f"Missing refusal direction artifacts: {missing_artifacts}")
    crit_set_path = args.crit_set_path
    if crit_set_path is None:
        crit_set_path = default_crit_set_path(args.crit_output_dir, model_id, args.crit_candidate)
    crit_candidate, crit_indices = load_crit_indices(crit_set_path)
    print(f"[mismatch] loaded crit set {crit_candidate} from {crit_set_path}")

    model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
    model.eval()

    max_layer = max(layers)
    all_modules = selected_linear_modules(model, args.target_suffixes)
    target_modules = [
        (name, module)
        for name, module in all_modules
        if (module_layer(name) is not None and module_layer(name) <= max_layer)
    ]
    if not target_modules:
        raise ValueError(f"No target modules found for suffixes={args.target_suffixes} up to layer {max_layer}.")
    set_trainable_modules(model, target_modules)

    harmful = load_harmful_prompts(args, offset=args.harmful_offset, limit=args.harmful_limit)
    print(f"[mismatch] harmful prompts={len(harmful)}")

    print("[mismatch] preparing WikiText utility windows")
    ppl_input_ids, ppl_windows = prepare_ppl_inputs(
        tokenizer=tokenizer,
        dataset_id=args.ppl_dataset,
        config_name=args.ppl_dataset_config,
        split=args.ppl_split,
        context_len=args.ppl_context_len,
        stride=args.ppl_stride,
        sample_windows=args.ppl_sample_windows,
        seed=args.seed,
        window_index_file=args.window_index_file,
        local_files_only=args.local_files_only,
        force_resample=args.force_resample,
    )

    print("[mismatch] collecting Wanda input norms")
    calib_prompts = load_calib_prompts(args)
    input_norms = collect_wanda_input_norms(model, tokenizer, calib_prompts, args.calib_max_length)

    print("[mismatch] accumulating utility gradients")
    util_scores_all = utility_scores(model, ppl_input_ids, ppl_windows, target_modules)

    corr_rows: list[dict[str, object]] = []
    percentile_rows: list[dict[str, object]] = []
    highcut_rows: list[dict[str, object]] = []
    sparsities = parse_float_list(args.sparsities)
    q_sweep = parse_float_list(args.q_sweep)

    for readout_layer in layers:
        active_modules = active_modules_for_readout(target_modules, readout_layer)
        active_names = {name for name, _module in active_modules}
        print(
            f"[mismatch] readout_layer={readout_layer} active_modules={len(active_modules)} "
            f"active_weights={sum(module.weight.numel() for _name, module in active_modules)}"
        )
        r_hat = load_refusal_direction(args.artifact_dir, model_id, readout_layer)
        safety_grad, safety_wgrad = safety_scores_for_layer(
            model=model,
            tokenizer=tokenizer,
            prompts=harmful,
            modules=active_modules,
            readout_layer=readout_layer,
            r_hat_cpu=r_hat,
            max_length=args.max_length,
        )
        mag_by_name = {name: module_mag_score(module) for name, module in active_modules}
        wanda_by_name = {name: module_wanda_score(module, input_norms, name) for name, module in active_modules}
        util_by_name = {name: util_scores_all[name] for name in active_names if name in util_scores_all}
        comparators = {"mag": mag_by_name, "wanda": wanda_by_name, "S_util": util_by_name}

        for variant, scores in (("grad", safety_grad), ("wgrad", safety_wgrad)):
            corr_rows.extend(
                correlation_rows(
                    readout_layer=readout_layer,
                    safety_variant=variant,
                    safety_by_name=scores,
                    comparators=comparators,
                    modules=active_modules,
                    scope="pooled",
                    max_samples=args.correlation_sample,
                    bootstrap_reps=args.bootstrap_reps,
                    bootstrap_sample_size=args.bootstrap_sample_size,
                    seed=args.seed + 1009 * readout_layer + (0 if variant == "grad" else 17),
                )
            )
            for kind in sorted({module_kind(name) for name, _module in active_modules}):
                subset = [(name, module) for name, module in active_modules if module_kind(name) == kind]
                corr_rows.extend(
                    correlation_rows(
                        readout_layer=readout_layer,
                        safety_variant=variant,
                        safety_by_name=scores,
                        comparators=comparators,
                        modules=subset,
                        scope="module_kind",
                        group_value=kind,
                        max_samples=args.group_correlation_sample,
                        bootstrap_reps=args.group_bootstrap_reps,
                        bootstrap_sample_size=args.group_bootstrap_sample_size,
                        seed=args.seed + 1009 * readout_layer + 97 * len(kind) + (0 if variant == "grad" else 17),
                    )
                )
            for mod_layer in sorted({module_layer(name) for name, _module in active_modules if module_layer(name) is not None}):
                subset = [(name, module) for name, module in active_modules if module_layer(name) == mod_layer]
                corr_rows.extend(
                    correlation_rows(
                        readout_layer=readout_layer,
                        safety_variant=variant,
                        safety_by_name=scores,
                        comparators=comparators,
                        modules=subset,
                        scope="module_layer",
                        group_value=str(mod_layer),
                        max_samples=args.group_correlation_sample,
                        bootstrap_reps=args.group_bootstrap_reps,
                        bootstrap_sample_size=args.group_bootstrap_sample_size,
                        seed=args.seed + 1009 * readout_layer + 131 * int(mod_layer) + (0 if variant == "grad" else 17),
                    )
                )
            active_weight_count = sum(module.weight.numel() for _name, module in active_modules)
            crit_active_count = sum(
                int(crit_indices.get(name, torch.empty(0, dtype=torch.int64)).numel()) for name in active_names
            )
            q_specs = [(f"q{q:g}", q) for q in q_sweep]
            if crit_active_count > 0:
                q_specs.insert(0, ("gradcrit", crit_active_count / active_weight_count))
            highcut_rows.extend(
                high_safety_cut_rows(
                    readout_layer=readout_layer,
                    safety_variant=variant,
                    score_by_name=scores,
                    modules=active_modules,
                    input_norms=input_norms,
                    sparsities=sparsities,
                    q_specs=q_specs,
                )
            )

        percentile_rows.extend(
            percentile_summary_rows(
                readout_layer=readout_layer,
                candidate=crit_candidate or str(crit_set_path),
                crit_indices={name: idx for name, idx in crit_indices.items() if name in active_names},
                modules=active_modules,
                input_norms=input_norms,
                seed=args.seed + readout_layer,
                max_values_for_ci=args.percentile_ci_sample,
            )
        )

        del safety_grad, safety_wgrad, mag_by_name, wanda_by_name, util_by_name, comparators
        write_progress_outputs(args.output_dir, corr_rows, percentile_rows, highcut_rows, layer=readout_layer)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    corr_df = pd.DataFrame(corr_rows)
    percentile_df = pd.DataFrame(percentile_rows)
    highcut_df = pd.DataFrame(highcut_rows)
    corr_path = args.output_dir / "weight_score_correlations.csv"
    percentile_path = args.output_dir / "safety_support_percentiles.csv"
    highcut_path = args.output_dir / "highsafety_in_wanda_cut.csv"
    corr_df.to_csv(corr_path, index=False)
    percentile_df.to_csv(percentile_path, index=False)
    highcut_df.to_csv(highcut_path, index=False)
    print(f"[mismatch] wrote {corr_path}")
    print(f"[mismatch] wrote {percentile_path}")
    print(f"[mismatch] wrote {highcut_path}")

    decision_layer = args.decision_layer if args.decision_layer >= 0 else max(layers)
    decision = build_decision(
        corr_df,
        highcut_df,
        decision_layer=decision_layer,
        corr_threshold=args.decision_corr_threshold,
        cut_threshold=args.decision_cut_threshold,
        cut_base_tolerance=args.decision_cut_base_tolerance,
    )
    decision_path = args.output_dir / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    manifest = {
        "model": model_id,
        "layers": layers,
        "target_suffixes": list(args.target_suffixes),
        "harmful_limit": args.harmful_limit,
        "ppl_sample_windows": args.ppl_sample_windows,
        "sparsities": sparsities,
        "q_sweep": q_sweep,
        "crit_set_path": str(crit_set_path) if crit_set_path else "",
        "decision": decision,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[mismatch] decision {json.dumps(decision, ensure_ascii=False)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spec B: objective/measure mismatch diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase15_mismatch"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/vpref_projection"))
    parser.add_argument("--crit-output-dir", type=Path, default=Path("results/phase1_v2"))
    parser.add_argument("--crit-set-path", type=Path, default=None)
    parser.add_argument("--crit-candidate", default="wei_setdiff__score-grad__ps-0.01__pu-0.05")
    parser.add_argument("--layers", default="24,28")
    parser.add_argument("--target-suffixes", nargs="+", default=["o_proj", "down_proj"])
    parser.add_argument("--harmful-file", type=Path, default=None)
    parser.add_argument("--harmful-dataset", default="walledai/AdvBench")
    parser.add_argument("--harmful-config", default="")
    parser.add_argument("--harmful-split", default="train")
    parser.add_argument("--harmful-column", default="auto")
    parser.add_argument("--harmful-offset", type=int, default=0)
    parser.add_argument("--harmful-limit", type=int, default=128)
    parser.add_argument("--utility-file", type=Path, default=None)
    parser.add_argument("--utility-dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--utility-config", default="")
    parser.add_argument("--utility-split", default="train")
    parser.add_argument("--utility-offset", type=int, default=0)
    parser.add_argument("--utility-limit", type=int, default=128)
    parser.add_argument("--calib-file", type=Path, default=None)
    parser.add_argument("--calib-limit", type=int, default=128)
    parser.add_argument("--calib-max-length", type=int, default=1024)
    parser.add_argument("--ppl-dataset", default="Salesforce/wikitext")
    parser.add_argument("--ppl-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--ppl-context-len", type=int, default=1024)
    parser.add_argument("--ppl-stride", type=int, default=512)
    parser.add_argument("--ppl-sample-windows", type=int, default=128)
    parser.add_argument("--window-index-file", type=Path, default=Path("results/phase1_v2/ppl_windows_wikitext2_seed0.json"))
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--sparsities", default="0.45,0.50,0.55")
    parser.add_argument("--q-sweep", default="0.001,0.002,0.005,0.01")
    parser.add_argument("--correlation-sample", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--bootstrap-sample-size", type=int, default=50_000)
    parser.add_argument("--group-correlation-sample", type=int, default=200_000)
    parser.add_argument("--group-bootstrap-reps", type=int, default=20)
    parser.add_argument("--group-bootstrap-sample-size", type=int, default=20_000)
    parser.add_argument("--percentile-ci-sample", type=int, default=200_000)
    parser.add_argument("--decision-layer", type=int, default=-1)
    parser.add_argument("--decision-corr-threshold", type=float, default=0.15)
    parser.add_argument("--decision-cut-threshold", type=float, default=0.20)
    parser.add_argument("--decision-cut-base-tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
