"""Leave-one-dataset-out factorization of ReMaR's refusal direction and g map.

This diagnostic keeps all public harmful prompt text in memory. Artifacts and CSVs
contain dataset-local identifiers, scalar metrics, and binary outcome labels only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .closed_form_readout_repair import (
    Condition,
    RepairArm,
    SolveConfig,
    apply_condition_pruning,
    apply_rank1_updates,
    collect_down_inputs_and_scores,
    generate_benign_rows,
    generate_harm_rows,
    parse_int_list,
    sanitize_rows,
    solve_layer_update,
    write_json,
    write_text_free_csv,
)
from .config import load_config
from .margin_calibration import best_tau_for_comply
from .models import model_slug, resolve_judge_model_id, resolve_model_id
from .ood_direction_eval import DATASET_IDS, balanced_counts, load_prompt_rows_any
from .phase0_smoke_eval import (
    apply_pruning,
    generate_answer,
    judge_with_llamaguard,
    lexical_coherence_stats,
    load_model_and_tokenizer,
)
from .vpref_projection import collect_residuals


TEXT_COLUMNS = {"prompt", "response", "text", "instruction", "output", "completion"}
DATASET_ORDER = ("advbench", "harmbench", "strongreject")


@dataclass(frozen=True)
class PromptItem:
    dataset: str
    original_id: int
    prompt_id: int
    prompt: str


def split_words(value: str) -> list[str]:
    return [item.strip() for chunk in str(value).split(",") for item in chunk.split() if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in split_words(value)]


def release(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def item_id(dataset: str, original_id: int) -> int:
    return (DATASET_ORDER.index(dataset) + 1) * 1_000_000 + int(original_id)


def split_items(items: list[PromptItem], *, seed: int, direction_limit: int, fit_limit: int, eval_limit: int) -> dict[str, list[PromptItem]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(items), generator=generator).tolist()
    shuffled = [items[pos] for pos in order]
    required = direction_limit + fit_limit + eval_limit
    if len(shuffled) < required:
        raise ValueError(
            f"Need {required} prompts for disjoint direction/fit/eval slices; only found {len(shuffled)}."
        )
    return {
        "direction": shuffled[:direction_limit],
        "fit": shuffled[direction_limit : direction_limit + fit_limit],
        "eval": shuffled[direction_limit + fit_limit : required],
    }


def source_key(names: list[str]) -> str:
    return "+".join(names)


def source_names(key: str) -> list[str]:
    return [name for name in key.split("+") if name]


def take_balanced(splits: dict[str, dict[str, list[PromptItem]]], key: str, field: str, total: int) -> list[PromptItem]:
    names = source_names(key)
    counts = balanced_counts(total, names)
    rows: list[PromptItem] = []
    for name in names:
        available = splits[name][field]
        if len(available) < counts[name]:
            raise ValueError(f"{name} has only {len(available)} {field} rows; need {counts[name]}.")
        rows.extend(available[: counts[name]])
    return rows


def default_primary(heldout: str) -> str:
    donors = [name for name in DATASET_ORDER if name != heldout]
    return donors[0]


def build_cells(heldouts: list[str]) -> list[dict[str, str | int]]:
    cells: list[dict[str, str | int]] = []
    for heldout in heldouts:
        donors = [name for name in DATASET_ORDER if name != heldout]
        single = default_primary(heldout)
        pair = source_key(donors)
        for r_kind, r_source, g_kind, g_source in (
            ("single", single, "single", single),
            ("pair", pair, "single", single),
            ("single", single, "pair", pair),
            ("pair", pair, "pair", pair),
        ):
            cells.append(
                {
                    "cell_id": len(cells),
                    "heldout_dataset": heldout,
                    "r_kind": r_kind,
                    "r_source": r_source,
                    "g_kind": g_kind,
                    "g_source": g_source,
                }
            )
    return cells


def load_splits(args: argparse.Namespace) -> tuple[dict[str, dict[str, list[PromptItem]]], dict[str, list[PromptItem]]]:
    harmful: dict[str, dict[str, list[PromptItem]]] = {}
    for offset, name in enumerate(DATASET_ORDER):
        dataset = getattr(args, f"{name}_dataset")
        config = getattr(args, f"{name}_config")
        split = getattr(args, f"{name}_split")
        column = getattr(args, f"{name}_column")
        rows = load_prompt_rows_any(
            file=None,
            dataset=dataset,
            config=config,
            split=split,
            column=column,
            local_files_only=args.local_files_only,
        )
        items = [PromptItem(name, original_id, item_id(name, original_id), prompt) for original_id, prompt in rows]
        harmful[name] = split_items(
            items,
            seed=args.seed + 1009 * (offset + 1),
            direction_limit=args.direction_limit,
            fit_limit=args.fit_limit,
            eval_limit=args.eval_limit,
        )

    benign_rows = load_prompt_rows_any(
        file=args.benign_file,
        dataset=None if args.benign_file else args.benign_dataset,
        config=args.benign_config,
        split=args.benign_split,
        column=args.benign_column,
        local_files_only=args.local_files_only,
    )
    benign_items = [PromptItem("benign", original_id, 9_000_000 + original_id, prompt) for original_id, prompt in benign_rows]
    benign = split_items(
        benign_items,
        seed=args.seed + 7919,
        direction_limit=args.benign_direction_limit,
        fit_limit=args.benign_fit_limit,
        eval_limit=args.benign_eval_limit,
    )
    return harmful, benign


def tuples(items: list[PromptItem]) -> list[tuple[int, str]]:
    return [(item.prompt_id, item.prompt) for item in items]


def item_manifest(items: list[PromptItem]) -> list[dict[str, int | str]]:
    return [{"dataset": item.dataset, "original_id": item.original_id, "prompt_id": item.prompt_id} for item in items]


def direction_path(artifact_dir: Path, r_source: str, model_id: str, layer: int) -> Path:
    safe_source = r_source.replace("+", "_")
    return artifact_dir / safe_source / f"{model_slug(model_id)}_layer{layer}_kr1.pt"


def threshold_path(artifact_dir: Path, r_source: str) -> Path:
    return artifact_dir / r_source.replace("+", "_") / "margin_thresholds.csv"


def build_directions(args: argparse.Namespace, model_id: str, splits: dict[str, dict[str, list[PromptItem]]], benign: dict[str, list[PromptItem]], r_sources: list[str]) -> None:
    layers = parse_int_list(args.layers)
    dense_model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
    try:
        harm_acts = {
            name: collect_residuals(dense_model, tokenizer, tuples(payload["direction"]), layers, args.max_length)
            for name, payload in splits.items()
        }
        benign_acts = collect_residuals(dense_model, tokenizer, tuples(benign["direction"]), layers, args.max_length)
        for r_source in r_sources:
            harm_items = take_balanced(splits, r_source, "direction", args.direction_limit)
            for layer in layers:
                harm_values = torch.stack([harm_acts[item.dataset][layer][item.prompt_id]["last"].float() for item in harm_items])
                benign_values = torch.stack([benign_acts[layer][item.prompt_id]["last"].float() for item in benign["direction"]])
                mean_diff = harm_values.mean(dim=0) - benign_values.mean(dim=0)
                r_hat = mean_diff / mean_diff.norm().clamp_min(1e-12)
                path = direction_path(args.artifact_dir, r_source, model_id, layer)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "r_hat": r_hat.contiguous().cpu(),
                        "basis_method": "lodo_difference_of_means",
                        "r_source": r_source,
                        "layer": layer,
                        "harmful_anchor_ids": item_manifest(harm_items),
                        "benign_anchor_ids": item_manifest(benign["direction"]),
                    },
                    path,
                )
                print(f"[lodo-remar] wrote {path}")
    finally:
        release(dense_model)


def generate_tau_labels(model, tokenizer, rows: list[PromptItem], *, max_new_tokens: int, response_ppl_threshold: float) -> list[dict[str, Any]]:
    output = []
    for pos, item in enumerate(rows):
        if pos == 0 or (pos + 1) % 25 == 0:
            print(f"[lodo-remar] tau calibration generation {pos + 1}/{len(rows)}")
        generated = generate_answer(model, tokenizer, item.prompt, max_new_tokens=max_new_tokens)
        lexical = lexical_coherence_stats(str(generated["response"]))
        coherent = bool(lexical["lexically_coherent"]) and float(generated["response_ppl"]) <= response_ppl_threshold
        output.append(
            {
                "dataset": item.dataset,
                "prompt_id": item.prompt_id,
                "prompt": item.prompt,
                "response": str(generated["response"]),
                "coherent": coherent,
            }
        )
    return output


def build_thresholds(args: argparse.Namespace, model_id: str, splits: dict[str, dict[str, list[PromptItem]]], r_sources: list[str]) -> None:
    layers = parse_int_list(args.layers)
    fit_by_dataset = {name: payload["fit"] for name, payload in splits.items()}
    all_fit = [item for rows in fit_by_dataset.values() for item in rows]
    model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
    try:
        apply_pruning(model, tokenizer, "wanda", 0.50, args.calib_max_length)
        pruned_acts = {
            name: collect_residuals(model, tokenizer, tuples(rows), layers, args.max_length)
            for name, rows in fit_by_dataset.items()
        }
        label_rows = generate_tau_labels(
            model,
            tokenizer,
            all_fit,
            max_new_tokens=args.max_new_tokens,
            response_ppl_threshold=args.response_ppl_threshold,
        )
    finally:
        release(model)
    if args.judge == "llamaguard":
        config = load_config(args.config)
        label_rows = judge_with_llamaguard(
            label_rows,
            judge_model_id=resolve_judge_model_id(config, args.judge_model),
            local_files_only=args.local_files_only,
            max_new_tokens=args.judge_max_new_tokens,
        )
    else:
        for row in label_rows:
            row["unsafe"] = False
            row["attack_success"] = False
    outcomes = {int(row["prompt_id"]): bool(row.get("attack_success", False)) for row in label_rows}
    for r_source in r_sources:
        directions = {layer: torch.load(direction_path(args.artifact_dir, r_source, model_id, layer), map_location="cpu")["r_hat"].float() for layer in layers}
        source_fit = take_balanced(splits, r_source, "fit", args.fit_limit)
        result_rows: list[dict[str, Any]] = []
        fallback = False
        for layer in layers:
            values = [float(pruned_acts[item.dataset][layer][item.prompt_id]["last"].float().dot(directions[layer])) for item in source_fit]
            labels = [outcomes[item.prompt_id] for item in source_fit]
            if len(set(labels)) < 2:
                tau = float(torch.tensor(values).median().item())
                youden = float("nan")
                fallback = True
            else:
                tau, youden = best_tau_for_comply(values, labels)
            result_rows.append({"score": f"s{layer}", "tau_global": tau, "youden_global": youden, "n": len(values), "fallback": fallback})
        mean_values = [
            float(sum(pruned_acts[item.dataset][layer][item.prompt_id]["last"].float().dot(directions[layer]) for layer in layers) / len(layers))
            for item in source_fit
        ]
        labels = [outcomes[item.prompt_id] for item in source_fit]
        if len(set(labels)) < 2:
            tau_mean = float(torch.tensor(mean_values).median().item())
            youden_mean = float("nan")
            fallback = True
        else:
            tau_mean, youden_mean = best_tau_for_comply(mean_values, labels)
        result_rows.append({"score": "s_mean", "tau_global": tau_mean, "youden_global": youden_mean, "n": len(mean_values), "fallback": fallback})
        path = threshold_path(args.artifact_dir, r_source)
        write_text_free_csv(pd.DataFrame(result_rows), path)
        if fallback:
            print(f"[lodo-remar] WARNING: {r_source} tau used a median fallback because calibration labels had one class.")


def read_directions_and_tau(args: argparse.Namespace, model_id: str, r_source: str, layers: list[int]) -> tuple[dict[int, torch.Tensor], dict[int, float], float]:
    directions = {}
    for layer in layers:
        payload = torch.load(direction_path(args.artifact_dir, r_source, model_id, layer), map_location="cpu")
        r_hat = payload["r_hat"].detach().float().cpu()
        directions[layer] = r_hat / r_hat.norm().clamp_min(1e-12)
    frame = pd.read_csv(threshold_path(args.artifact_dir, r_source))
    tau = {layer: float(frame.loc[frame["score"].eq(f"s{layer}"), "tau_global"].iloc[0]) for layer in layers}
    tau_mean = float(frame.loc[frame["score"].eq("s_mean"), "tau_global"].iloc[0])
    return directions, tau, tau_mean


def solve_for_cell(model, tokenizer, *, layers: list[int], directions: dict[int, torch.Tensor], taus: dict[int, float], harm_fit: list[PromptItem], benign_fit: list[PromptItem], args: argparse.Namespace) -> dict[int, Any]:
    harm = collect_down_inputs_and_scores(model, tokenizer, tuples(harm_fit), layers=layers, directions=directions, max_length=args.max_length)
    benign = collect_down_inputs_and_scores(model, tokenizer, tuples(benign_fit), layers=layers, directions=directions, max_length=args.max_length)
    config = SolveConfig("tm%s_lb%s" % (str(args.target_margin).replace(".", "p"), str(args.lambda_benign).replace(".", "p")), args.target_margin, args.lambda_benign)
    return {
        layer: solve_layer_update(
            layer=layer,
            harm_data=harm[layer],
            benign_data=benign[layer],
            r_hat=directions[layer],
            tau=taus[layer],
            target_margin=config.target_margin,
            lambda_benign=config.lambda_benign,
            ridge_mu=args.ridge_mu,
            delta_max=args.delta_max,
        )
        for layer in layers
    }


def provenance(rows: list[dict[str, Any]], cell: dict[str, str | int]) -> None:
    for row in rows:
        for key, value in cell.items():
            row[key] = value


def summarize_cell(harm_rows: list[dict[str, Any]], benign_rows: list[dict[str, Any]]) -> pd.DataFrame:
    harm = pd.DataFrame(harm_rows)
    benign = pd.DataFrame(benign_rows)
    keys = ["cell_id", "heldout_dataset", "r_kind", "r_source", "g_kind", "g_source", "repair", "repair_kind"]
    summary = harm.groupby(keys, dropna=False).agg(
        prompts=("prompt_id", "count"),
        asr=("attack_success", "mean"),
        raw_unsafe_rate=("unsafe", "mean"),
        coherent_rate=("coherent", "mean"),
        refusal_rate=("refused", "mean"),
        frac_m_neg_s_mean=("m_neg_s_mean", "mean"),
    ).reset_index()
    benign_summary = benign.groupby(keys, dropna=False).agg(
        benign_prompts=("prompt_id", "count"),
        benign_refusal_rate=("refused", "mean"),
        benign_coherent_rate=("coherent", "mean"),
    ).reset_index()
    summary = summary.merge(benign_summary, on=keys, how="left")
    base_keys = ["cell_id", "heldout_dataset", "r_kind", "r_source", "g_kind", "g_source"]
    base = summary[summary["repair_kind"].eq("pruned")][base_keys + ["asr", "benign_refusal_rate", "coherent_rate"]].rename(
        columns={"asr": "pruned_asr", "benign_refusal_rate": "pruned_benign_refusal", "coherent_rate": "pruned_coherent_rate"}
    )
    summary = summary.merge(base, on=base_keys, how="left")
    summary["asr_drop_vs_pruned"] = summary["pruned_asr"] - summary["asr"]
    summary["benign_refusal_delta_vs_pruned"] = summary["benign_refusal_rate"] - summary["pruned_benign_refusal"]
    summary["coherent_delta_vs_pruned"] = summary["coherent_rate"] - summary["pruned_coherent_rate"]
    return summary


def run_cell(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)
    splits, benign = load_splits(args)
    cells = build_cells(split_words(args.heldout_datasets))
    if args.cell_index < 0 or args.cell_index >= len(cells):
        raise ValueError(f"cell-index {args.cell_index} out of range [0,{len(cells) - 1}].")
    cell = cells[args.cell_index]
    layers = parse_int_list(args.layers)
    directions, taus, tau_mean = read_directions_and_tau(args, model_id, str(cell["r_source"]), layers)
    harm_fit = take_balanced(splits, str(cell["g_source"]), "fit", args.fit_limit)
    harm_eval = splits[str(cell["heldout_dataset"])]["eval"][: args.eval_limit]
    benign_fit = benign["fit"][: args.benign_fit_limit]
    benign_eval = benign["eval"][: args.benign_eval_limit]
    solve_config = SolveConfig("tm%s_lb%s" % (str(args.target_margin).replace(".", "p"), str(args.lambda_benign).replace(".", "p")), args.target_margin, args.lambda_benign)
    condition = Condition("wanda_50", "wanda", 0.50)

    print(f"[lodo-remar] cell={cell['cell_id']} held={cell['heldout_dataset']} r={cell['r_source']} g={cell['g_source']}")
    model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
    try:
        apply_condition_pruning(model, tokenizer, "wanda", 0.50, args.calib_max_length)
        solves = solve_for_cell(model, tokenizer, layers=layers, directions=directions, taus=taus, harm_fit=harm_fit, benign_fit=benign_fit, args=args)
    finally:
        release(model)

    arms = [RepairArm("pruned", "pruned", 0.0), RepairArm("readout_repair_eta1", "readout_repair", 1.0)]
    if args.include_random_control:
        arms.append(RepairArm("random_dir_control_eta1", "random_dir_control", 1.0))
    all_harm: list[dict[str, Any]] = []
    all_benign: list[dict[str, Any]] = []
    for arm in arms:
        model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
        try:
            pruned_layers = apply_condition_pruning(model, tokenizer, "wanda", 0.50, args.calib_max_length)
            stats: dict[str, float] = {"delta_w_norm_total": 0.0}
            if arm.kind == "readout_repair":
                stats = apply_rank1_updates(model, solves=solves, eta=1.0, random_direction=False, seed=args.seed)
            elif arm.kind == "random_dir_control":
                stats = apply_rank1_updates(model, solves=solves, eta=1.0, random_direction=True, seed=args.seed + 104729)
            harm_rows = generate_harm_rows(
                model, tokenizer, model_id=model_id, condition=condition, repair=arm, solve_config=solve_config if arm.kind != "pruned" else None,
                prompts=tuples(harm_eval), layers=layers, directions=directions, restore_targets=None, tau_s_mean=tau_mean,
                max_length=args.max_length, max_new_tokens=args.max_new_tokens, response_ppl_threshold=args.response_ppl_threshold,
                pruned_layers=pruned_layers, update_stats=stats,
            )
            benign_rows = generate_benign_rows(
                model, tokenizer, model_id=model_id, condition=condition, repair=arm, solve_config=solve_config if arm.kind != "pruned" else None,
                prompts=tuples(benign_eval), layers=layers, directions=directions, restore_targets=None,
                max_new_tokens=args.benign_max_new_tokens, response_ppl_threshold=args.response_ppl_threshold,
                pruned_layers=pruned_layers, update_stats=stats,
            )
        finally:
            release(model)
        provenance(harm_rows, cell)
        provenance(benign_rows, cell)
        all_harm.extend(harm_rows)
        all_benign.extend(benign_rows)
    if args.judge == "llamaguard":
        all_harm = judge_with_llamaguard(
            all_harm,
            judge_model_id=resolve_judge_model_id(config, args.judge_model),
            local_files_only=args.local_files_only,
            max_new_tokens=args.judge_max_new_tokens,
        )
    else:
        for row in all_harm:
            row["unsafe"] = bool(row["attack_success"])
    output = args.output_dir / f"cell_{int(cell['cell_id']):02d}"
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_cell(all_harm, all_benign)
    write_text_free_csv(summary, output / "cell_summary.csv")
    write_text_free_csv(pd.DataFrame(sanitize_rows(all_harm)), output / "cell_details.csv")
    write_text_free_csv(pd.DataFrame(sanitize_rows(all_benign)), output / "cell_benign_details.csv")
    write_json(
        {
            "cell": cell,
            "harm_fit": item_manifest(harm_fit),
            "harm_eval": item_manifest(harm_eval),
            "benign_fit": item_manifest(benign_fit),
            "benign_eval": item_manifest(benign_eval),
            "r_source": cell["r_source"],
            "g_source": cell["g_source"],
            "tau_by_layer": taus,
            "tau_s_mean": tau_mean,
            "target_margin": args.target_margin,
            "lambda_benign": args.lambda_benign,
        },
        output / "cell_manifest.json",
    )


def prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)
    splits, benign = load_splits(args)
    cells = build_cells(split_words(args.heldout_datasets))
    r_sources = sorted({str(cell["r_source"]) for cell in cells})
    build_directions(args, model_id, splits, benign, r_sources)
    build_thresholds(args, model_id, splits, r_sources)
    write_json(
        {
            "model": model_id,
            "layers": parse_int_list(args.layers),
            "seed": args.seed,
            "direction_limit_total": args.direction_limit,
            "fit_limit_total": args.fit_limit,
            "eval_limit": args.eval_limit,
            "benign_direction_limit": args.benign_direction_limit,
            "benign_fit_limit": args.benign_fit_limit,
            "benign_eval_limit": args.benign_eval_limit,
            "heldout_datasets": split_words(args.heldout_datasets),
            "cells": cells,
            "harmful_splits": {name: {field: item_manifest(rows) for field, rows in payload.items()} for name, payload in splits.items()},
            "benign_splits": {field: item_manifest(rows) for field, rows in benign.items()},
            "text_free_outputs": True,
            "tau_protocol": "Wanda-50 source-fit behavior calibration; held-out dataset excluded from r, tau, and g.",
        },
        args.output_dir / "lodo_manifest.json",
    )


def merge(args: argparse.Namespace) -> None:
    summary_paths = sorted(args.output_dir.glob("cell_*/cell_summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No cell summaries under {args.output_dir}.")
    summary = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    details_paths = sorted(args.output_dir.glob("cell_*/cell_details.csv"))
    benign_paths = sorted(args.output_dir.glob("cell_*/cell_benign_details.csv"))
    write_text_free_csv(summary, args.output_dir / "lodo_remar_summary.csv")
    if details_paths:
        write_text_free_csv(pd.concat([pd.read_csv(path) for path in details_paths], ignore_index=True), args.output_dir / "lodo_remar_details.csv")
    if benign_paths:
        write_text_free_csv(pd.concat([pd.read_csv(path) for path in benign_paths], ignore_index=True), args.output_dir / "lodo_remar_benign_details.csv")
    repairs = summary[summary["repair_kind"].eq("readout_repair")].copy()
    baseline = summary[summary["repair_kind"].eq("pruned")][["heldout_dataset", "asr", "coherent_rate", "benign_refusal_rate"]].copy()
    baseline = baseline.groupby("heldout_dataset", as_index=False).mean(numeric_only=True).rename(columns={"asr": "pruned_asr", "coherent_rate": "pruned_coherent_rate", "benign_refusal_rate": "pruned_benign_refusal"})
    matrix = repairs.merge(baseline, on="heldout_dataset", how="left")
    matrix["asr_drop_vs_pruned"] = matrix["pruned_asr"] - matrix["asr"]
    matrix["benign_refusal_delta_vs_pruned"] = matrix["benign_refusal_rate"] - matrix["pruned_benign_refusal"]
    matrix["coherent_delta_vs_pruned"] = matrix["coherent_rate"] - matrix["pruned_coherent_rate"]
    write_text_free_csv(matrix, args.output_dir / "lodo_factor_matrix.csv")
    decisions: dict[str, Any] = {}
    for heldout, frame in matrix.groupby("heldout_dataset", dropna=False):
        def row(r_kind: str, g_kind: str):
            found = frame[frame["r_kind"].eq(r_kind) & frame["g_kind"].eq(g_kind)]
            return found.iloc[0] if len(found) == 1 else None
        ss, ps, sp, pp = row("single", "single"), row("pair", "single"), row("single", "pair"), row("pair", "pair")
        decisions[str(heldout)] = {
            "single_single_asr": float(ss["asr"]) if ss is not None else None,
            "pair_single_asr": float(ps["asr"]) if ps is not None else None,
            "single_pair_asr": float(sp["asr"]) if sp is not None else None,
            "pair_pair_asr": float(pp["asr"]) if pp is not None else None,
            "direction_pair_effect_at_single_g": float(ss["asr"] - ps["asr"]) if ss is not None and ps is not None else None,
            "mapping_pair_effect_at_single_r": float(ss["asr"] - sp["asr"]) if ss is not None and sp is not None else None,
            "joint_effect_vs_single_single": float(ss["asr"] - pp["asr"]) if ss is not None and pp is not None else None,
        }
    write_json(
        {
            "interpretation": "Positive factor effects denote ASR reduction from replacing the single-source factor with the leave-one-dataset-out pair-source factor. Inspect effect magnitudes and confidence intervals; this diagnostic does not infer causality from a single seed.",
            "per_heldout_dataset": decisions,
        },
        args.output_dir / "lodo_factor_decision.json",
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "run", "merge"], default="run")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2_lodo_remar"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase2_lodo_remar"))
    parser.add_argument("--layers", default="24,28,32")
    parser.add_argument("--heldout-datasets", default="advbench harmbench strongreject")
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-limit", type=int, default=64, help="Total harmful samples per r source.")
    parser.add_argument("--fit-limit", type=int, default=128, help="Total harmful samples per g source.")
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--benign-direction-limit", type=int, default=64)
    parser.add_argument("--benign-fit-limit", type=int, default=128)
    parser.add_argument("--benign-eval-limit", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--calib-max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--benign-max-new-tokens", type=int, default=128)
    parser.add_argument("--response-ppl-threshold", type=float, default=100.0)
    parser.add_argument("--judge", choices=["llamaguard", "keyword"], default="llamaguard")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-max-new-tokens", type=int, default=16)
    parser.add_argument("--target-margin", type=float, default=20.0)
    parser.add_argument("--lambda-benign", type=float, default=20.0)
    parser.add_argument("--ridge-mu", type=float, default=1e-2)
    parser.add_argument("--delta-max", type=float, default=50.0)
    parser.add_argument("--include-random-control", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--benign-file", type=Path)
    parser.add_argument("--benign-dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--benign-config")
    parser.add_argument("--benign-split", default="train")
    parser.add_argument("--benign-column", default="auto")
    for name in DATASET_ORDER:
        parser.add_argument(f"--{name}-dataset", default=DATASET_IDS[name])
        parser.add_argument(f"--{name}-config", default="standard" if name == "harmbench" else None)
        parser.add_argument(f"--{name}-split", default="train")
        parser.add_argument(f"--{name}-column", default="auto")
    return parser


def main() -> None:
    args = parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run_cell(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
