from __future__ import annotations

import gc
import os
from collections.abc import Callable, Iterable

import torch
from torch import nn


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _target_suffixes(default_suffixes: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get("SPARSEGPT_TARGET_SUFFIXES", "").strip()
    if not raw:
        return default_suffixes
    return tuple(item.strip() for item in raw.replace(",", " ").split() if item.strip())


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _collect_inputs_for_module(
    model: nn.Module,
    tokenizer,
    module: nn.Linear,
    prompts: list[str],
    max_length: int,
    format_prompt: Callable[[object, str], str],
    max_samples: int,
    seed: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []

    def hook(_module, inputs):
        x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu()
        if x.numel():
            chunks.append(x)

    handle = module.register_forward_pre_hook(hook)
    try:
        for prompt in prompts:
            text = format_prompt(tokenizer, prompt)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            device = _module_device(model)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if not chunks:
        raise RuntimeError("SparseGPT calibration captured no inputs for a target module.")
    x = torch.cat(chunks, dim=0)
    if max_samples > 0 and x.shape[0] > max_samples:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=generator)[:max_samples]
        x = x[idx]
    return x.contiguous()


def _stable_cholesky_inverse_factor(hessian: torch.Tensor, damp: float) -> torch.Tensor | None:
    hessian = hessian.float()
    diag = torch.diag(hessian)
    dead = diag <= 0
    if dead.any():
        hessian = hessian.clone()
        hessian[dead, dead] = 1.0
    mean_diag = torch.diag(hessian).mean()
    if not torch.isfinite(mean_diag) or float(mean_diag) <= 0:
        return None
    eye = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    hessian = hessian + eye * (damp * mean_diag)
    try:
        chol = torch.linalg.cholesky(hessian)
        inv = torch.cholesky_inverse(chol)
        return torch.linalg.cholesky(inv, upper=True)
    except RuntimeError:
        return None


def _prune_block_sparsegpt(
    weight_block: torch.Tensor,
    hessian_block: torch.Tensor,
    sparsity: float,
    *,
    damp: float,
    blocksize: int,
) -> torch.Tensor:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if weight_block.numel() == 0:
        return weight_block

    device = weight_block.device
    hessian_block = hessian_block.to(device=device, dtype=torch.float32)
    h_inv = _stable_cholesky_inverse_factor(hessian_block, damp)
    if h_inv is None:
        # Diagonal fallback keeps the run alive for ill-conditioned tiny smoke
        # calibrations. The score is still OBS-like: w^2 / H^{-1}_{jj}^2.
        diag = torch.diag(hessian_block).clamp_min(1e-8)
        scores = weight_block.float().square() * diag.view(1, -1)
        keep = max(1, int(scores.numel() * (1.0 - sparsity)))
        idx = scores.flatten().topk(keep, largest=True).indices
        mask = torch.zeros(scores.numel(), device=device, dtype=weight_block.dtype)
        mask.scatter_(0, idx, 1.0)
        return weight_block * mask.reshape_as(weight_block)

    w = weight_block.float().clone()
    out = torch.zeros_like(w)
    columns = w.shape[1]
    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        w1 = w[:, start:end].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        h1 = h_inv[start:end, start:end]
        diag = torch.diag(h1).clamp_min(1e-8)
        scores = w1.square() / diag.view(1, -1).square()
        num_prune = int(scores.numel() * sparsity)
        if num_prune <= 0:
            mask1 = torch.zeros_like(scores, dtype=torch.bool)
        else:
            threshold = torch.sort(scores.flatten())[0][min(num_prune, scores.numel() - 1)]
            mask1 = scores <= threshold
        for i in range(end - start):
            current = w1[:, i]
            d = h1[i, i].clamp_min(1e-8)
            q = current.clone()
            q[mask1[:, i]] = 0.0
            q1[:, i] = q
            err = (current - q) / d
            if i + 1 < end - start:
                w1[:, i + 1 :] -= err.unsqueeze(1).matmul(h1[i, i + 1 :].unsqueeze(0))
            err1[:, i] = err
        out[:, start:end] = q1
        if end < columns:
            w[:, end:] -= err1.matmul(h_inv[start:end, end:])
    return out.to(dtype=weight_block.dtype)


def _prune_linear_from_inputs(
    module: nn.Linear,
    inputs: torch.Tensor,
    sparsity: float,
    *,
    damp: float,
    blocksize: int,
    hessian_block: int,
    max_exact_in_features: int,
) -> None:
    device = module.weight.device
    original = module.weight.detach()
    in_features = original.shape[1]
    pruned_chunks: list[torch.Tensor] = []
    if in_features <= max_exact_in_features:
        x = inputs.to(device=device, dtype=torch.float32)
        hessian = x.t().matmul(x) / max(1, x.shape[0])
        pruned = _prune_block_sparsegpt(original.float(), hessian, sparsity, damp=damp, blocksize=blocksize)
        with torch.no_grad():
            module.weight.copy_(pruned.to(device=device, dtype=module.weight.dtype))
        del x, hessian, pruned
        return

    # Full SparseGPT needs a dense in_features x in_features inverse. For Qwen
    # MLP down-projections this is very expensive, so use an explicit
    # block-diagonal Hessian approximation over input columns.
    for start in range(0, in_features, hessian_block):
        end = min(start + hessian_block, in_features)
        x = inputs[:, start:end].to(device=device, dtype=torch.float32)
        hessian = x.t().matmul(x) / max(1, x.shape[0])
        block_weight = original[:, start:end].float()
        pruned = _prune_block_sparsegpt(block_weight, hessian, sparsity, damp=damp, blocksize=blocksize)
        pruned_chunks.append(pruned.detach().cpu())
        del x, hessian, block_weight, pruned
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    merged = torch.cat(pruned_chunks, dim=1).to(device=device, dtype=module.weight.dtype)
    with torch.no_grad():
        module.weight.copy_(merged)
    del merged, pruned_chunks


def apply_sparsegpt_pruning(
    model: nn.Module,
    tokenizer,
    modules: Iterable[tuple[str, nn.Linear]],
    prompts: list[str],
    sparsity: float | str,
    max_length: int,
    format_prompt: Callable[[object, str], str],
    default_suffixes: tuple[str, ...],
) -> int:
    if not isinstance(sparsity, float):
        raise TypeError(f"SparseGPT only supports unstructured float sparsity, got {sparsity!r}")
    suffixes = _target_suffixes(default_suffixes)
    damp = _env_float("SPARSEGPT_DAMP", 0.01)
    blocksize = _env_int("SPARSEGPT_BLOCKSIZE", 128)
    hessian_block = _env_int("SPARSEGPT_HESSIAN_BLOCK", 2048)
    max_exact = _env_int("SPARSEGPT_MAX_EXACT_IN_FEATURES", 4096)
    max_samples = _env_int("SPARSEGPT_MAX_SAMPLES", 512)
    seed = _env_int("SPARSEGPT_SEED", 0)

    selected = [(name, module) for name, module in modules if name.endswith(suffixes)]
    if not selected:
        raise ValueError(f"SparseGPT selected no modules for suffixes={suffixes}")

    pruned = 0
    total = len(selected)
    for index, (name, module) in enumerate(selected, start=1):
        print(
            f"[sparsegpt] {index}/{total} collecting inputs for {name} "
            f"shape={tuple(module.weight.shape)} sparsity={sparsity:g}",
            flush=True,
        )
        inputs = _collect_inputs_for_module(
            model,
            tokenizer,
            module,
            prompts,
            max_length,
            format_prompt,
            max_samples,
            seed + index,
        )
        print(f"[sparsegpt] {index}/{total} pruning {name} samples={inputs.shape[0]}", flush=True)
        _prune_linear_from_inputs(
            module,
            inputs,
            sparsity,
            damp=damp,
            blocksize=blocksize,
            hessian_block=hessian_block,
            max_exact_in_features=max_exact,
        )
        pruned += 1
        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pruned
