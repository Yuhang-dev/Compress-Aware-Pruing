"""Measure the inference overhead of an unmerged LoRA sidecar."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .common import (
    MODEL_ID,
    atomic_write_json,
    directory_hashes,
    file_sha256,
    read_json,
    source_identity,
    utc_now,
    verify_wanda_checkpoint,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-id", default=MODEL_ID)
    result.add_argument("--wanda-checkpoint-dir", type=Path, required=True)
    result.add_argument("--wanda-manifest", type=Path, required=True)
    result.add_argument("--artifact-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--sequence-length", type=int, default=256)
    result.add_argument("--warmup-iterations", type=int, default=5)
    result.add_argument("--iterations", type=int, default=40)
    result.add_argument("--seed", type=int, default=0)
    return result


def _timed_forward(model, inputs: dict[str, Any], *, adapter_enabled: bool) -> float:
    import torch

    context = nullcontext() if adapter_enabled else model.disable_adapter()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with context, torch.inference_mode():
        model(**inputs, use_cache=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.model_id != MODEL_ID:
        raise ValueError(f"This benchmark is registered only for {MODEL_ID}")
    if args.sequence_length <= 0 or args.iterations < 5 or args.warmup_iterations < 1:
        raise ValueError("Invalid latency benchmark size")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency benchmark")
    wanda = verify_wanda_checkpoint(
        args.wanda_checkpoint_dir,
        args.wanda_manifest,
        expected_model=args.model_id,
        verify_all_files=True,
    )
    artifact = read_json(args.artifact_manifest)
    if artifact.get("status") != "completed" or artifact.get("method") != "sft":
        raise ValueError("Incomplete or non-SFT artifact manifest")
    if artifact.get("model") != args.model_id:
        raise ValueError("SFT artifact model mismatch")
    base_path = Path(str((artifact.get("base_checkpoint") or {}).get("path", "")))
    if base_path.resolve() != args.wanda_checkpoint_dir.resolve():
        raise ValueError("SFT artifact was trained on a different Wanda checkpoint")
    artifact_base = artifact.get("base_checkpoint") or {}
    if artifact_base.get("manifest_sha256") != wanda["manifest_sha256"]:
        raise ValueError("SFT artifact references a different Wanda manifest")
    if artifact_base.get("checkpoint_files_sha256") != wanda["checkpoint_files_sha256"]:
        raise ValueError("SFT artifact references different Wanda checkpoint files")
    adapter_dir = Path(str(artifact.get("adapter_dir", "")))
    if directory_hashes(adapter_dir) != artifact.get("adapter_files_sha256"):
        raise ValueError("Adapter hashes differ from the training manifest")

    started_at = utc_now()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.wanda_checkpoint_dir.resolve()),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_dir.resolve()),
        is_trainable=False,
        local_files_only=True,
    )
    model.to(torch.device("cuda"))
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (1, args.sequence_length),
        generator=generator,
        device=device,
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    for _ in range(args.warmup_iterations):
        _timed_forward(model, inputs, adapter_enabled=False)
        _timed_forward(model, inputs, adapter_enabled=True)

    base_times: list[float] = []
    adapter_times: list[float] = []
    for index in range(args.iterations):
        # Alternate order to reduce thermal/clock-order bias.
        if index % 2 == 0:
            base_times.append(_timed_forward(model, inputs, adapter_enabled=False))
            adapter_times.append(_timed_forward(model, inputs, adapter_enabled=True))
        else:
            adapter_times.append(_timed_forward(model, inputs, adapter_enabled=True))
            base_times.append(_timed_forward(model, inputs, adapter_enabled=False))

    base_median = statistics.median(base_times)
    adapter_median = statistics.median(adapter_times)
    overhead = 100.0 * (adapter_median / base_median - 1.0)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "model": args.model_id,
        "method": "sft",
        "artifact_manifest_sha256": file_sha256(args.artifact_manifest),
        "wanda_manifest_sha256": wanda["manifest_sha256"],
        "adapter_files_sha256": artifact["adapter_files_sha256"],
        "measurement": "unmerged_lora_sidecar_forward_latency",
        "base_median_ms": 1000.0 * base_median,
        "adapter_median_ms": 1000.0 * adapter_median,
        "overhead_percent": overhead,
        "base_mean_ms": 1000.0 * statistics.fmean(base_times),
        "adapter_mean_ms": 1000.0 * statistics.fmean(adapter_times),
        "sequence_length": args.sequence_length,
        "batch_size": 1,
        "warmup_iterations": args.warmup_iterations,
        "timed_iterations_per_arm": args.iterations,
        "dtype": "bfloat16",
        "gpu_model": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "runtime": {
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "total_seconds": time.perf_counter() - started,
        },
        "source": source_identity(extra_paths=(Path(__file__),)),
        "privacy": {"input_text_persisted": False, "synthetic_token_ids_persisted": False},
    }
    atomic_write_json(args.output, payload)
    print(
        f"[latency] base={payload['base_median_ms']:.3f}ms "
        f"adapter={payload['adapter_median_ms']:.3f}ms overhead={overhead:+.2f}%",
        flush=True,
    )
    del model, input_ids, inputs
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
