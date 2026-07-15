"""Prune a model once and persist an auditable reusable checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import phase0_smoke_eval as phase0_module
from . import pruners as pruners_module
from . import sparsegpt as sparsegpt_module
from .phase0_smoke_eval import (
    CALIB_PROMPTS,
    apply_pruning,
    load_model_and_tokenizer,
    target_linear_modules,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hashes(path: Path) -> dict[str, str]:
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    if not files:
        raise ValueError(f"Checkpoint is empty: {path}")
    return {str(item.relative_to(path)): file_sha256(item) for item in files}


def target_sparsity_summary(model) -> dict[str, Any]:
    modules = target_linear_modules(model)
    total = 0
    zeros = 0
    for _name, module in modules:
        weight = module.weight.detach()
        total += int(weight.numel())
        zeros += int(weight.eq(0).sum().item())
    return {
        "target_linear_modules": len(modules),
        "target_weights": total,
        "zero_weights": zeros,
        "realized_zero_fraction": float(zeros / total) if total else 0.0,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    result.add_argument("--pruner", choices=("sparsegpt", "wanda", "magnitude"), required=True)
    result.add_argument("--sparsity", type=float, required=True)
    result.add_argument("--condition", required=True)
    result.add_argument("--checkpoint-dir", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--calib-max-length", type=int, default=256)
    result.add_argument("--calibration-sequences", type=int, default=128)
    result.add_argument("--calibration-dataset", default="Salesforce/wikitext")
    result.add_argument("--calibration-dataset-config", default="wikitext-2-raw-v1")
    result.add_argument("--calibration-split", default="train")
    result.add_argument("--calibration-seed", type=int, default=0)
    result.add_argument(
        "--calibration-index-file",
        type=Path,
        default=Path(
            "results/phase2_qwen3b_sparsegpt_calibration/"
            "wikitext2_train_seed0_128x1024.json"
        ),
    )
    result.add_argument("--local-files-only", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint pruning, but no GPU is available.")
    if args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty checkpoint: {args.checkpoint_dir}"
        )
    if not 0.0 < args.sparsity < 1.0:
        raise ValueError(f"sparsity must be in (0, 1), got {args.sparsity}")

    started_at = utc_now()
    total_start = time.perf_counter()

    stage = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args.model, args.local_files_only)
    load_seconds = time.perf_counter() - stage

    calibration_input_ids = None
    calibration_module = None
    calibration_load_seconds = 0.0
    calibration_metadata: dict[str, Any]
    if args.pruner == "sparsegpt":
        from . import calibration as calibration_module
        from .calibration import load_raw_text_calibration

        stage = time.perf_counter()
        calibration = load_raw_text_calibration(
            tokenizer,
            dataset_id=args.calibration_dataset,
            dataset_config=args.calibration_dataset_config,
            split=args.calibration_split,
            num_sequences=args.calibration_sequences,
            sequence_length=args.calib_max_length,
            seed=args.calibration_seed,
            index_file=args.calibration_index_file,
            local_files_only=args.local_files_only,
        )
        calibration_input_ids = calibration.input_ids
        calibration_metadata = calibration.metadata
        calibration_load_seconds = time.perf_counter() - stage
        print(
            f"[checkpoint] calibration sequences={calibration_input_ids.shape[0]} "
            f"tokens={calibration_input_ids.shape[1]} rows={calibration_input_ids.numel()} "
            f"sha256={calibration_metadata['token_ids_sha256']}",
            flush=True,
        )
    else:
        calibration_metadata = {
            "dataset": "inline_benign_chat_prompts",
            "num_sequences": len(CALIB_PROMPTS),
            "sequence_length": int(args.calib_max_length),
            "chat_template_applied": True,
            "raw_text": False,
            "prompt_content_sha256": hashlib.sha256(
                json.dumps(
                    CALIB_PROMPTS,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    stage = time.perf_counter()
    pruning_kwargs: dict[str, Any] = {}
    if args.pruner == "sparsegpt":
        # Older repository revisions do not expose this keyword.  Wanda and
        # magnitude do not need it, so keep their persisted-checkpoint path
        # compatible with those revisions.
        pruning_kwargs["sparsegpt_calibration_input_ids"] = calibration_input_ids
    pruned_layers = apply_pruning(
        model,
        tokenizer,
        args.pruner,
        float(args.sparsity),
        args.calib_max_length,
        **pruning_kwargs,
    )
    prune_seconds = time.perf_counter() - stage
    del calibration_input_ids

    stage = time.perf_counter()
    sparsity = target_sparsity_summary(model)
    sparsity_scan_seconds = time.perf_counter() - stage

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage = time.perf_counter()
    model.save_pretrained(
        args.checkpoint_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(args.checkpoint_dir)
    save_seconds = time.perf_counter() - stage

    del model
    gc.collect()
    torch.cuda.empty_cache()

    stage = time.perf_counter()
    hashes = checkpoint_hashes(args.checkpoint_dir)
    hash_seconds = time.perf_counter() - stage
    total_seconds = time.perf_counter() - total_start

    manifest = {
        "model": args.model,
        "condition": args.condition,
        "pruner": args.pruner,
        "requested_sparsity": float(args.sparsity),
        "calib_max_length": int(args.calib_max_length),
        "calibration": calibration_metadata,
        "pruned_layers_reported": int(pruned_layers),
        "sparsity": sparsity,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_files_sha256": hashes,
        "sparsegpt": {
            "blocksize": int(os.environ.get("SPARSEGPT_BLOCKSIZE", "128")),
            "damp": float(os.environ.get("SPARSEGPT_DAMP", "0.01")),
            "hessian_block": int(os.environ.get("SPARSEGPT_HESSIAN_BLOCK", "2048")),
            "max_exact_in_features": int(
                os.environ.get("SPARSEGPT_MAX_EXACT_IN_FEATURES", "4096")
            ),
            "max_samples": int(os.environ.get("SPARSEGPT_MAX_SAMPLES", "0")),
            "seed": int(os.environ.get("SPARSEGPT_SEED", "0")),
            "calibration_batch_size": int(
                os.environ.get("SPARSEGPT_CALIB_BATCH_SIZE", "1")
            ),
            "statistics_mode": (
                "streaming_layerwise_hessian" if args.pruner == "sparsegpt" else None
            ),
        },
        "runtime": {
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "load_model_seconds": load_seconds,
            "load_calibration_seconds": calibration_load_seconds,
            "prune_seconds": prune_seconds,
            "sparsity_scan_seconds": sparsity_scan_seconds,
            "save_checkpoint_seconds": save_seconds,
            "hash_checkpoint_seconds": hash_seconds,
            "total_seconds": total_seconds,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "source_sha256": {
            str(Path(__file__)): file_sha256(Path(__file__)),
            str(Path(phase0_module.__file__)): file_sha256(Path(phase0_module.__file__)),
            str(Path(pruners_module.__file__)): file_sha256(Path(pruners_module.__file__)),
            str(Path(sparsegpt_module.__file__)): file_sha256(Path(sparsegpt_module.__file__)),
            **(
                {
                    str(Path(calibration_module.__file__)): file_sha256(
                        Path(calibration_module.__file__)
                    )
                }
                if calibration_module is not None
                else {}
            ),
        },
        "privacy": {
            "prompt_text_persisted": False,
            "response_text_persisted": False,
            "calibration_text_persisted": False,
            "calibration_token_ids_persisted": False,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.manifest.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        temporary_manifest = Path(handle.name)
    os.replace(temporary_manifest, args.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
