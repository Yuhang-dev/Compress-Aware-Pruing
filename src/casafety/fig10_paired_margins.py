"""Collect real, text-free paired margins for Appendix Figure 10.

The run evaluates the same held-out AdvBench prompts under Dense, Wanda-50,
and decode-aware ReMaR. Only prompt hashes and scalar readouts are persisted;
the script does not generate model completions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from .margin_calibration import collect_prompt_readouts
from .remar_coverage_diag import prompt_identity
from .remar_coverage_v2 import payload_hash
from .vpref_projection import load_prompt_rows
from .xstest_orbench_eval import (
    apply_remar,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    git_head,
    load_model_and_tokenizer,
    read_json,
    sha256_text,
    verify_checkpoint,
    verify_repair,
)


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CONDITIONS = ("dense", "pruned", "remar")
IDENTITY_FIELDS = (
    "dataset_id",
    "dataset_config",
    "dataset_split",
    "dataset_column",
    "prompt_count",
    "prompt_numeric_id_sha256",
    "prompt_id_sha256",
    "prompt_content_sha256",
    "prompt_split_sha256",
)
BASE_SHARD_COLUMNS = (
    "condition",
    "prompt_id",
    "eval_order",
    "prompt_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError(f"Expected a non-empty unique layer list, got {value!r}")
    return layers


def normalized_prompt_sha256(prompt: str) -> str:
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
    return sha256_text(normalized)


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_identity_matches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in IDENTITY_FIELDS
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(
            "Held-out AdvBench identity does not match the shipped evaluation split: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _solve_for_layer(solves: dict[Any, Any], layer: int) -> dict[str, Any]:
    solve = solves.get(layer, solves.get(str(layer)))
    if not isinstance(solve, dict) or not {"r_hat", "g"}.issubset(solve):
        raise ValueError(f"Repair artifact has no complete solve for layer {layer}")
    return solve


def load_repair_components(
    path: Path,
    *,
    expected_model: str,
    expected_layers: Sequence[int],
) -> tuple[dict[str, Any], dict[int, torch.Tensor], dict[int, float], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Repair artifact must contain a dictionary")
    if payload.get("model") != expected_model:
        raise ValueError("Repair artifact model mismatch")
    if payload.get("variant") != "adv_decode":
        raise ValueError(f"Expected adv_decode artifact, got {payload.get('variant')!r}")
    layers = [int(value) for value in payload.get("layers", [])]
    if layers != list(expected_layers):
        raise ValueError(
            f"Repair layer mismatch: expected {list(expected_layers)}, got {layers}"
        )

    actual_vector_sha256 = payload_hash(payload)
    if actual_vector_sha256 != payload.get("vector_sha256"):
        raise ValueError("Repair artifact vector hash mismatch")

    solves = payload.get("solves") or {}
    raw_taus = payload.get("taus") or {}
    directions: dict[int, torch.Tensor] = {}
    taus: dict[int, float] = {}
    layer_metadata: dict[str, Any] = {}
    for layer in layers:
        solve = _solve_for_layer(solves, layer)
        direction = solve["r_hat"].detach().float().cpu().reshape(-1).contiguous()
        g = solve["g"].detach().float().cpu().reshape(-1).contiguous()
        if direction.numel() == 0 or g.numel() == 0:
            raise ValueError(f"Empty repair vector at layer {layer}")
        if not torch.isfinite(direction).all() or not torch.isfinite(g).all():
            raise ValueError(f"Non-finite repair vector at layer {layer}")
        tau_value = raw_taus.get(layer, raw_taus.get(str(layer)))
        if tau_value is None:
            raise ValueError(f"Repair artifact has no tau for layer {layer}")
        tau = float(tau_value)
        if not torch.isfinite(torch.tensor(tau)):
            raise ValueError(f"Non-finite tau at layer {layer}")
        directions[layer] = direction
        taus[layer] = tau
        layer_metadata[str(layer)] = {
            "r_hat_norm": float(direction.norm()),
            "g_norm": float(g.norm()),
            "hidden_size": int(direction.numel()),
            "input_size": int(g.numel()),
            "tau": tau,
        }

    metadata = {
        "variant": payload["variant"],
        "vector_sha256": actual_vector_sha256,
        "layers": layers,
        "layer_metadata": layer_metadata,
    }
    return payload, directions, taus, metadata


def shard_columns(layers: Sequence[int]) -> list[str]:
    return [*BASE_SHARD_COLUMNS, *(f"s{layer}" for layer in layers), "s_mean"]


def validate_shard(
    frame: pd.DataFrame,
    *,
    condition: str,
    prompts: Sequence[tuple[int, str]],
    layers: Sequence[int],
) -> pd.DataFrame:
    expected_columns = shard_columns(layers)
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{condition} shard is missing columns: {missing}")
    frame = frame[expected_columns].copy()
    if frame.empty:
        return frame
    if set(frame["condition"].astype(str)) != {condition}:
        raise ValueError(f"Shard contains a condition other than {condition}")
    if frame["prompt_id"].duplicated().any():
        raise ValueError(f"Duplicate prompt_id in {condition} shard")

    expected = {
        int(prompt_id): (order, normalized_prompt_sha256(prompt))
        for order, (prompt_id, prompt) in enumerate(prompts)
    }
    for row in frame.itertuples(index=False):
        prompt_id = int(row.prompt_id)
        if prompt_id not in expected:
            raise ValueError(f"Unexpected prompt_id {prompt_id} in {condition} shard")
        expected_order, expected_hash = expected[prompt_id]
        if int(row.eval_order) != expected_order or str(row.prompt_sha256) != expected_hash:
            raise ValueError(f"Prompt identity mismatch in {condition} shard: {prompt_id}")
    return frame.sort_values("eval_order").reset_index(drop=True)


def build_paired_frame(
    long_frame: pd.DataFrame,
    *,
    layers: Sequence[int],
    taus: dict[int, float],
) -> pd.DataFrame:
    expected_columns = set(shard_columns(layers))
    missing = sorted(expected_columns.difference(long_frame.columns))
    if missing:
        raise ValueError(f"Long margin table is missing columns: {missing}")
    if set(long_frame["condition"].astype(str)) != set(CONDITIONS):
        raise ValueError(f"Expected exactly the conditions {CONDITIONS}")
    if long_frame.duplicated(["condition", "prompt_id"]).any():
        raise ValueError("Duplicate condition/prompt_id pair in long margin table")

    indexed: dict[str, pd.DataFrame] = {}
    for condition in CONDITIONS:
        arm = (
            long_frame.loc[long_frame["condition"] == condition]
            .copy()
            .sort_values("eval_order")
            .set_index("prompt_id", drop=False)
        )
        indexed[condition] = arm

    reference = indexed["dense"]
    reference_ids = list(reference.index.astype(int))
    for condition in CONDITIONS[1:]:
        arm = indexed[condition]
        if list(arm.index.astype(int)) != reference_ids:
            raise ValueError(f"Prompt order differs between dense and {condition}")
        if list(arm["prompt_sha256"].astype(str)) != list(
            reference["prompt_sha256"].astype(str)
        ):
            raise ValueError(f"Prompt hashes differ between dense and {condition}")

    paired = reference[["prompt_id", "eval_order", "prompt_sha256"]].reset_index(drop=True)
    for condition in CONDITIONS:
        arm = indexed[condition].reset_index(drop=True)
        for layer in layers:
            paired[f"{condition}_s{layer}"] = arm[f"s{layer}"].astype(float)
        paired[f"{condition}_margin"] = arm["s_mean"].astype(float)

    for layer in layers:
        paired[f"tau_s{layer}"] = float(taus[layer])
    tau_mean = float(sum(taus[layer] for layer in layers) / len(layers))
    paired["tau"] = tau_mean
    for condition in CONDITIONS:
        paired[f"{condition}_below_tau"] = (
            paired[f"{condition}_margin"] < tau_mean
        ).astype(int)

    paired["dense_above_pruned_below"] = (
        (paired["dense_margin"] >= tau_mean)
        & (paired["pruned_margin"] < tau_mean)
    ).astype(int)
    paired["dense_above_remar_below"] = (
        (paired["dense_margin"] >= tau_mean)
        & (paired["remar_margin"] < tau_mean)
    ).astype(int)
    paired["pruned_below_remar_above"] = (
        (paired["pruned_margin"] < tau_mean)
        & (paired["remar_margin"] >= tau_mean)
    ).astype(int)
    paired["pruned_delta_vs_dense"] = paired["pruned_margin"] - paired["dense_margin"]
    paired["remar_delta_vs_pruned"] = paired["remar_margin"] - paired["pruned_margin"]
    return paired


def build_quadrant_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison in ("pruned", "remar"):
        for dense_state, dense_below in (("above", False), ("below", True)):
            for compared_state, compared_below in (("above", False), ("below", True)):
                mask = (
                    paired["dense_below_tau"].astype(bool).eq(dense_below)
                    & paired[f"{comparison}_below_tau"].astype(bool).eq(compared_below)
                )
                rows.append(
                    {
                        "pair": f"dense_vs_{comparison}",
                        "dense_state": dense_state,
                        "comparison_state": compared_state,
                        "count": int(mask.sum()),
                        "fraction": float(mask.mean()),
                    }
                )
    return pd.DataFrame(rows)


def source_hashes(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "src/casafety/fig10_paired_margins.py",
        "src/casafety/margin_calibration.py",
        "src/casafety/remar_coverage_diag.py",
        "src/casafety/remar_coverage_v2.py",
        "src/casafety/vpref_projection.py",
        "src/casafety/xstest_orbench_eval.py",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = file_sha256(path)
    return result


def initialize_run_config(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    path = output_dir / "run_config.json"
    fingerprint_payload = dict(payload)
    fingerprint = canonical_json_sha256(fingerprint_payload)
    config = {
        **fingerprint_payload,
        "fingerprint_sha256": fingerprint,
        "created_at_utc": utc_now(),
    }
    if path.exists():
        existing = read_json(path)
        if not resume:
            raise FileExistsError(f"Existing run config requires --resume: {path}")
        if existing.get("fingerprint_sha256") != fingerprint:
            raise ValueError("Existing output directory belongs to a different run")
        return existing
    atomic_write_json(config, path)
    return config


def collect_condition(
    args: argparse.Namespace,
    *,
    condition: str,
    prompts: Sequence[tuple[int, str]],
    layers: Sequence[int],
    directions: dict[int, torch.Tensor],
) -> pd.DataFrame:
    shard_path = args.output_dir / "shards" / f"{condition}.csv"
    if shard_path.exists() and args.resume:
        frame = validate_shard(
            pd.read_csv(shard_path),
            condition=condition,
            prompts=prompts,
            layers=layers,
        )
    else:
        frame = pd.DataFrame(columns=shard_columns(layers))

    completed = set(frame["prompt_id"].astype(int)) if not frame.empty else set()
    if len(completed) == len(prompts):
        print(f"[fig10] reuse complete {condition} shard ({len(completed)} rows)")
        return frame

    model_path = args.dense_model if condition == "dense" else str(args.pruned_model_dir)
    model = tokenizer = None
    try:
        print(f"[fig10] load {condition}: {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path, args)
        if condition == "remar":
            applied = apply_remar(
                model,
                args.repair_artifact,
                expected_model=args.dense_model,
                expected_layers=layers,
                eta=args.eta,
            )
            print(f"[fig10] applied ReMaR: {applied}")

        records = frame.to_dict("records")
        for eval_order, (prompt_id, prompt) in enumerate(prompts):
            if int(prompt_id) in completed:
                continue
            readouts = collect_prompt_readouts(
                model,
                tokenizer,
                prompt,
                list(layers),
                directions,
                args.max_length,
            )
            record: dict[str, Any] = {
                "condition": condition,
                "prompt_id": int(prompt_id),
                "eval_order": int(eval_order),
                "prompt_sha256": normalized_prompt_sha256(prompt),
            }
            record.update({key: float(value) for key, value in readouts.items()})
            records.append(record)
            frame = pd.DataFrame(records, columns=shard_columns(layers))
            frame = frame.sort_values("eval_order").reset_index(drop=True)
            atomic_write_csv(frame, shard_path)
            if len(frame) % 16 == 0 or len(frame) == len(prompts):
                print(f"[fig10] {condition}: {len(frame)}/{len(prompts)}")
        return validate_shard(
            frame,
            condition=condition,
            prompts=prompts,
            layers=layers,
        )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-model", default=MODEL_ID)
    parser.add_argument(
        "--pruned-model-dir",
        type=Path,
        default=Path("artifacts/phase2_oracle_policy_distill/pruned_model"),
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=Path("results/phase2_oracle_policy_distill/prepare_manifest.json"),
    )
    parser.add_argument(
        "--repair-artifact",
        type=Path,
        default=Path("artifacts/phase2_oracle_policy_distill/adv_decode.pt"),
    )
    parser.add_argument(
        "--repair-manifest",
        type=Path,
        default=Path("results/phase2_oracle_policy_distill/prepare_manifest.json"),
    )
    parser.add_argument("--dataset", default="walledai/AdvBench")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-column", default="auto")
    parser.add_argument("--offset", type=int, default=128)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--layers", default="24,28,32")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fig10_paired_margins_real"),
    )
    parser.add_argument(
        "--verify-checkpoint",
        choices=("none", "config", "all"),
        default="config",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.offset < 0 or args.limit <= 0:
        parser.error("--offset must be nonnegative and --limit must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.eta <= 0:
        parser.error("--eta must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    args.output_dir = args.output_dir.resolve()
    args.pruned_model_dir = args.pruned_model_dir.resolve()
    args.checkpoint_manifest = args.checkpoint_manifest.resolve()
    args.repair_artifact = args.repair_artifact.resolve()
    args.repair_manifest = args.repair_manifest.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    checkpoint_info = verify_checkpoint(
        args.pruned_model_dir,
        args.checkpoint_manifest,
        args.dense_model,
        args.verify_checkpoint,
    )
    repair_info = verify_repair(
        args.repair_artifact,
        args.repair_manifest,
        args.dense_model,
    )
    payload, directions, taus, artifact_metadata = load_repair_components(
        args.repair_artifact,
        expected_model=args.dense_model,
        expected_layers=layers,
    )

    all_prompts = load_prompt_rows(
        file=None,
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.dataset_split,
        column=args.dataset_column,
        local_files_only=args.local_files_only,
    )
    prompts = all_prompts[args.offset : args.offset + args.limit]
    if len(prompts) != args.limit:
        raise ValueError(
            f"Requested {args.limit} prompts at offset {args.offset}, got {len(prompts)}"
        )
    actual_identity = prompt_identity(
        list(prompts),
        dataset="advbench_evaluation",
        dataset_id=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_column=args.dataset_column,
    )
    prepare_manifest = read_json(args.checkpoint_manifest)
    try:
        expected_identity = prepare_manifest["calibration"]["evaluation_identities"]["advbench"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Checkpoint manifest has no AdvBench evaluation identity") from exc
    assert_identity_matches(actual_identity, expected_identity)
    print(f"[fig10] held-out split verified: {actual_identity['prompt_split_sha256']}")

    run_payload = {
        "schema_version": 1,
        "model": args.dense_model,
        "conditions": list(CONDITIONS),
        "pruner": "wanda",
        "sparsity": 0.5,
        "layers": layers,
        "eta": float(args.eta),
        "max_length": int(args.max_length),
        "dtype": args.dtype,
        "dataset_identity": actual_identity,
        "checkpoint": checkpoint_info,
        "repair": repair_info,
        "repair_vectors": artifact_metadata,
        "taus": {str(layer): float(taus[layer]) for layer in layers},
        "tau_mean": float(sum(taus.values()) / len(taus)),
        "source_sha256": source_hashes(repo_root),
        "git_head": git_head(),
        "privacy": {
            "generation_performed": False,
            "prompt_text_persisted": False,
            "response_text_persisted": False,
        },
    }
    run_config = initialize_run_config(args.output_dir, run_payload, resume=args.resume)

    frames = [
        collect_condition(
            args,
            condition=condition,
            prompts=prompts,
            layers=layers,
            directions=directions,
        )
        for condition in CONDITIONS
    ]
    long_frame = pd.concat(frames, ignore_index=True)
    paired = build_paired_frame(long_frame, layers=layers, taus=taus)
    quadrants = build_quadrant_summary(paired)
    long_path = args.output_dir / "fig10_margin_readouts_long.csv"
    paired_path = args.output_dir / "fig10_paired_margins.csv"
    quadrant_path = args.output_dir / "fig10_quadrant_counts.csv"
    atomic_write_csv(long_frame, long_path)
    atomic_write_csv(paired, paired_path)
    atomic_write_csv(quadrants, quadrant_path)

    failure_counts = {
        "dense_above_pruned_below": int(paired["dense_above_pruned_below"].sum()),
        "dense_above_remar_below": int(paired["dense_above_remar_below"].sum()),
        "pruned_below_remar_above": int(paired["pruned_below_remar_above"].sum()),
    }
    final_manifest = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "run_fingerprint_sha256": run_config["fingerprint_sha256"],
        "prompt_count": len(paired),
        "failure_counts": failure_counts,
        "claim_supported": failure_counts["dense_above_remar_below"] == 0,
        "outputs": {
            path.name: {"sha256": file_sha256(path), "rows": rows}
            for path, rows in (
                (long_path, len(long_frame)),
                (paired_path, len(paired)),
                (quadrant_path, len(quadrants)),
            )
        },
        "privacy": run_payload["privacy"],
    }
    atomic_write_json(final_manifest, args.output_dir / "manifest.json")
    print(f"[fig10] complete: {paired_path}")
    print(f"[fig10] failure counts: {failure_counts}")
    del payload
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
