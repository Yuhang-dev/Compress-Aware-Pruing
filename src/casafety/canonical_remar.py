"""Reconstruct and freeze the canonical ReMaR solve with manifest validation.

Historical runs did not persist the g vectors. This module therefore uses the
exact original prompt loader and solver, validates every persisted solve
statistic against the registered main-run manifest, and freezes one artifact
with a content hash for all subsequent workers. Historical bit identity remains
unverifiable because no historical vector hash exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .closed_form_readout_repair import (
    Condition,
    apply_condition_pruning,
    collect_down_inputs_and_scores,
    load_prompt_slice,
    load_refusal_direction,
    load_thresholds,
    parse_int_list,
    solve_layer_update,
    write_text_free_csv,
)
from .config import load_config
from .models import resolve_model_id
from .ood_residual_diag import release, write_decision
from .phase0_smoke_eval import load_model_and_tokenizer


COMPARE_FIELDS = (
    "tau",
    "target",
    "harm_n",
    "benign_n",
    "positive_delta_n",
    "mean_s_pruned",
    "mean_delta",
    "max_delta",
    "ridge_mu_effective",
    "g_norm",
    "delta_w_norm",
    "rank1_added_params",
    "effective_rank",
)


def canonical_manifest_cell(path: Path, *, condition: str, solve_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "shards" in payload:
        payload = payload["shards"]["readout"]
    return payload["conditions"][condition][solve_id]


def tensor_hash(solves: dict[int, Any]) -> str:
    digest = hashlib.sha256()
    for layer in sorted(solves):
        digest.update(str(layer).encode("ascii"))
        for tensor in (solves[layer].r_hat, solves[layer].g):
            value = tensor.detach().float().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def close_enough(actual: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(expected, int):
        return int(actual) == int(expected)
    return math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol)


def run_export(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)
    layers = parse_int_list(args.layers)
    taus, tau_mean = load_thresholds(args.margin_dir / "margin_thresholds.csv", layers)
    directions = {
        layer: load_refusal_direction(args.direction_artifact_dir, model_id, layer)
        for layer in layers
    }
    harmful = load_prompt_slice(
        file=args.harmful_file,
        dataset=None if args.harmful_file else args.harmful_dataset,
        config=args.harmful_config,
        split=args.harmful_split,
        column=args.harmful_column,
        local_files_only=args.local_files_only,
        offset=args.harmful_fit_offset,
        limit=args.fit_limit,
    )
    benign = load_prompt_slice(
        file=args.benign_file,
        dataset=None if args.benign_file else args.benign_dataset,
        config=args.benign_config,
        split=args.benign_split,
        column=args.benign_column,
        local_files_only=args.local_files_only,
        offset=args.benign_fit_offset,
        limit=args.benign_fit_limit,
    )
    model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
    try:
        apply_condition_pruning(
            model, tokenizer, Condition(args.condition, "wanda", args.sparsity), args.calib_max_length
        )
        harm_data = collect_down_inputs_and_scores(
            model,
            tokenizer,
            harmful,
            layers=layers,
            directions=directions,
            max_length=args.max_length,
        )
        benign_data = collect_down_inputs_and_scores(
            model,
            tokenizer,
            benign,
            layers=layers,
            directions=directions,
            max_length=args.max_length,
        )
        solves = {
            layer: solve_layer_update(
                layer=layer,
                harm_data=harm_data[layer],
                benign_data=benign_data[layer],
                r_hat=directions[layer],
                tau=taus[layer],
                target_margin=args.target_margin,
                lambda_benign=args.lambda_benign,
                ridge_mu=args.ridge_mu,
                delta_max=args.delta_max,
            )
            for layer in layers
        }
    finally:
        release(model)

    solve_id = f"tm{args.target_margin:g}_lb{args.lambda_benign:g}"
    expected = canonical_manifest_cell(
        args.canonical_manifest, condition=args.condition, solve_id=solve_id
    )
    rows = []
    all_match = True
    for layer in layers:
        solve = solves[layer]
        expected_layer = expected["layers"][str(layer)]
        for field in COMPARE_FIELDS:
            actual_value = getattr(solve, field)
            expected_value = expected_layer[field]
            matches = close_enough(
                actual_value, expected_value, atol=args.match_atol, rtol=args.match_rtol
            )
            all_match = all_match and matches
            rows.append(
                {
                    "layer": layer,
                    "field": field,
                    "actual": float(actual_value),
                    "expected": float(expected_value),
                    "absolute_error": abs(float(actual_value) - float(expected_value)),
                    "matches": matches,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_text_free_csv(
        pd.DataFrame(rows), args.output_dir / "canonical_remar_manifest_validation.csv"
    )
    vector_sha256 = tensor_hash(solves)
    decision = {
        "model": model_id,
        "condition": args.condition,
        "solve_id": solve_id,
        "canonical_manifest": str(args.canonical_manifest),
        "historical_vector_available": False,
        "historical_bit_identical": None,
        "historical_bit_identity_reason": "The historical run persisted scalar solve statistics but no g vector or vector hash.",
        "manifest_exact_reconstruction": bool(all_match),
        "match_atol": float(args.match_atol),
        "match_rtol": float(args.match_rtol),
        "current_vector_sha256": vector_sha256,
        "current_run_internal_bit_identical": bool(all_match),
    }
    write_decision(decision, args.output_dir / "canonical_remar_decision.json")
    if not all_match:
        raise RuntimeError(
            "Canonical ReMaR reconstruction does not match the registered main-run manifest. "
            "Refusing to save or evaluate this g."
        )
    artifact = {
        "model": model_id,
        "layers": layers,
        "taus": taus,
        "tau_mean": float(tau_mean),
        "solve_id": solve_id,
        "target_margin": float(args.target_margin),
        "lambda_benign": float(args.lambda_benign),
        "eta": 1.0,
        "canonical_manifest": str(args.canonical_manifest),
        "manifest_exact_reconstruction": True,
        "historical_bit_identical": None,
        "vector_sha256": vector_sha256,
        "solves": {
            layer: {
                "g": solves[layer].g.float().cpu(),
                "r_hat": solves[layer].r_hat.float().cpu(),
                "g_norm": float(solves[layer].g_norm),
                "delta_w_norm": float(solves[layer].delta_w_norm),
            }
            for layer in layers
        },
    }
    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output_artifact)
    print(f"[canonical-remar] wrote {args.output_artifact} sha256={vector_sha256}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--condition", default="wanda_50")
    parser.add_argument("--sparsity", type=float, default=0.50)
    parser.add_argument("--layers", default="24,28,32")
    parser.add_argument("--direction-artifact-dir", type=Path, default=Path("artifacts/vpref_projection"))
    parser.add_argument("--margin-dir", type=Path, default=Path("results/phase15_margin_calib"))
    parser.add_argument("--canonical-manifest", type=Path, default=Path("results/phase2_readout_repair_v2_w50_parallel/g_solve_manifest.json"))
    parser.add_argument("--output-artifact", type=Path, default=Path("artifacts/phase2_oracle_hs_coverage_v2/canonical_remar.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2_oracle_hs_coverage_v2"))
    parser.add_argument("--target-margin", type=float, default=20.0)
    parser.add_argument("--lambda-benign", type=float, default=20.0)
    parser.add_argument("--ridge-mu", type=float, default=0.01)
    parser.add_argument("--delta-max", type=float, default=50.0)
    parser.add_argument("--fit-limit", type=int, default=128)
    parser.add_argument("--harmful-fit-offset", type=int, default=0)
    parser.add_argument("--benign-fit-offset", type=int, default=0)
    parser.add_argument("--benign-fit-limit", type=int, default=128)
    parser.add_argument("--harmful-file", type=Path)
    parser.add_argument("--harmful-dataset", default="walledai/AdvBench")
    parser.add_argument("--harmful-config")
    parser.add_argument("--harmful-split", default="train")
    parser.add_argument("--harmful-column", default="auto")
    parser.add_argument("--benign-file", type=Path)
    parser.add_argument("--benign-dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--benign-config")
    parser.add_argument("--benign-split", default="train")
    parser.add_argument("--benign-column", default="auto")
    parser.add_argument("--calib-max-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--match-atol", type=float, default=1e-5)
    parser.add_argument("--match-rtol", type=float, default=1e-6)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    run_export(parser().parse_args())


if __name__ == "__main__":
    main()
