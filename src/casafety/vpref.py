from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch import nn

from .config import load_config
from .models import model_slug, resolve_model_id
from .phase0_smoke_eval import (
    CALIB_PROMPTS,
    HARMFUL_SMOKE_PROMPTS,
    format_prompt,
    generate_answer,
    infer_prompt_column,
    is_refusal,
    load_model_and_tokenizer,
)


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj: object = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return list(obj)
    raise ValueError("Could not locate decoder layers on model.")


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    if spec == "auto":
        if num_layers <= 8:
            return list(range(num_layers))
        start = max(0, (2 * num_layers) // 3)
        selected = set(range(start, num_layers, 4))
        selected.add(max(0, num_layers - 2))
        selected.add(num_layers - 1)
        return sorted(layer for layer in selected if layer < num_layers)

    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            selected.update(range(int(lo), int(hi) + 1))
        else:
            selected.add(int(part))
    layers = sorted(selected)
    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {invalid}")
    return layers


def load_hf_column(
    dataset_id: str,
    config_name: str | None,
    split: str,
    column: str,
    local_files_only: bool,
) -> list[str]:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - dependency checked on remote
        raise ImportError("Install datasets to load Hugging Face datasets.") from exc

    download_config = DownloadConfig(local_files_only=local_files_only)
    if config_name:
        dataset = load_dataset(dataset_id, config_name, split=split, download_config=download_config)
    else:
        dataset = load_dataset(dataset_id, split=split, download_config=download_config)

    selected_column = infer_prompt_column(dataset.column_names) if column == "auto" else column
    values = []
    for row in dataset:
        value = row.get(selected_column)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    if not values:
        raise ValueError(f"No text found in {dataset_id}:{split} column={selected_column!r}")
    print(f"[vpref] loaded {len(values)} rows from {dataset_id}:{split} column={selected_column}")
    return values


def slice_items(items: list[str], limit: int, offset: int = 0) -> list[str]:
    sliced = items[offset:]
    return sliced[:limit] if limit else sliced


def load_prompt_file(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_harmful_prompts(args: argparse.Namespace) -> list[str]:
    prompts = load_prompt_file(args.harmful_file)
    if prompts is None and args.harmful_dataset:
        prompts = load_hf_column(
            args.harmful_dataset,
            args.harmful_config,
            args.harmful_split,
            args.harmful_column,
            args.local_files_only,
        )
    if prompts is None:
        prompts = HARMFUL_SMOKE_PROMPTS
    return slice_items(prompts, args.harmful_limit, args.harmful_offset)


def load_benign_prompts(args: argparse.Namespace) -> list[str]:
    prompts = load_prompt_file(args.benign_file)
    if prompts is None and args.benign_dataset:
        prompts = load_hf_column(
            args.benign_dataset,
            args.benign_config,
            args.benign_split,
            args.benign_column,
            args.local_files_only,
        )
    if prompts is None:
        prompts = CALIB_PROMPTS
    return slice_items(prompts, args.benign_limit, args.benign_offset)


def collect_final_token_residuals(
    model: nn.Module,
    tokenizer,
    prompts: Iterable[str],
    layer_indices: list[int],
    max_length: int,
) -> dict[int, torch.Tensor]:
    layers = decoder_layers(model)
    collected: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}
    current_final_idx = {"value": 0}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            final_idx = current_final_idx["value"]
            collected[layer_idx].append(hidden[0, final_idx].detach().float().cpu())

        return hook

    for layer_idx in layer_indices:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    try:
        for prompt in prompts:
            text = format_prompt(tokenizer, prompt)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            current_final_idx["value"] = int(inputs["attention_mask"][0].sum().item()) - 1
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return {idx: torch.stack(values) for idx, values in collected.items() if values}


def orient_basis(diff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    oriented = basis.clone()
    for col in range(oriented.shape[1]):
        projection_mean = float(diff.matmul(oriented[:, col]).mean())
        if projection_mean < 0:
            oriented[:, col].mul_(-1)
    return oriented


def build_arditi_plus_svd_basis(
    harm_acts: torch.Tensor,
    benign_mean: torch.Tensor,
    max_kr: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
    """Return a basis whose first column is exactly mean(harm)-mean(benign).

    For k_r > 1, remaining columns are principal directions of the harmful
    residuals after removing the Arditi mean-difference component.
    """

    diff = (harm_acts - benign_mean).float()
    mean_diff = harm_acts.float().mean(dim=0) - benign_mean.float()
    mean_diff_norm = float(mean_diff.norm().clamp_min(1e-12))
    if mean_diff_norm <= 1e-8:
        _u, singular_values, vh = torch.linalg.svd(diff, full_matrices=False)
        basis = orient_basis(diff, vh[:max_kr].transpose(0, 1).contiguous())
        scales = singular_values[:max_kr] / max(1.0, diff.shape[0] ** 0.5)
        return basis, scales.contiguous(), singular_values[:max_kr].contiguous(), mean_diff_norm, "svd_fallback"

    arditi = (mean_diff / mean_diff_norm).contiguous()
    columns = [arditi]
    component_scales = [torch.tensor(mean_diff_norm, dtype=torch.float32)]
    residual_singular_values = []

    if max_kr > 1:
        residual = diff - diff.matmul(arditi).unsqueeze(1) * arditi.unsqueeze(0)
        _u, singular_values, vh = torch.linalg.svd(residual, full_matrices=False)
        extra = orient_basis(diff, vh[: max_kr - 1].transpose(0, 1).contiguous())
        for col in range(extra.shape[1]):
            columns.append(extra[:, col].contiguous())
        residual_singular_values = [value.detach().float() for value in singular_values[: max_kr - 1]]
        component_scales.extend(
            (singular_values[: max_kr - 1].float() / max(1.0, diff.shape[0] ** 0.5)).detach().cpu()
        )

    basis = torch.stack(columns, dim=1)[:, :max_kr].contiguous()
    scales = torch.stack([scale.float().cpu() for scale in component_scales])[:max_kr].contiguous()
    singular_summary = torch.cat(
        [
            torch.tensor([mean_diff_norm], dtype=torch.float32),
            torch.stack(residual_singular_values) if residual_singular_values else torch.empty(0),
        ]
    )[:max_kr].contiguous()
    return basis, scales, singular_summary, mean_diff_norm, "arditi_plus_residual_svd"


def extract_refusal_bases(
    model_id: str,
    harm: dict[int, torch.Tensor],
    benign: dict[int, torch.Tensor],
    kr_values: list[int],
    artifact_root: Path,
) -> dict[tuple[int, int], Path]:
    slug = model_slug(model_id)
    output_dir = artifact_root / "vpref"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[tuple[int, int], Path] = {}
    max_kr = max(kr_values)

    for layer_idx, harm_acts in harm.items():
        benign_acts = benign[layer_idx]
        benign_mean = benign_acts.mean(dim=0)
        diff = harm_acts - benign_mean
        if diff.shape[0] < max_kr:
            raise ValueError(
                f"Need at least {max_kr} harmful examples for k_r={max_kr}, got {diff.shape[0]}"
            )
        basis, component_scales, singular_values, mean_diff_norm, basis_method = build_arditi_plus_svd_basis(
            harm_acts=harm_acts,
            benign_mean=benign_mean,
            max_kr=max_kr,
        )

        for kr in kr_values:
            path = output_dir / f"{slug}_layer{layer_idx}_kr{kr}.pt"
            torch.save(
                {
                    "model": model_id,
                    "layer": layer_idx,
                    "k_r": kr,
                    "basis": basis[:, :kr].contiguous(),
                    "basis_method": basis_method,
                    "component_scales": component_scales[:kr].contiguous(),
                    "singular_values": singular_values[:kr].contiguous(),
                    "mean_diff_norm": mean_diff_norm,
                    "benign_mean": benign_mean.contiguous(),
                    "num_harmful": int(harm_acts.shape[0]),
                    "num_benign": int(benign_acts.shape[0]),
                },
                path,
            )
            artifacts[(layer_idx, kr)] = path
            print(f"[vpref] wrote {path}")
    return artifacts


def generate_refusal_rate(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    layer_idx: int | None = None,
    vector: torch.Tensor | None = None,
) -> float:
    handle = None
    if layer_idx is not None and vector is not None:
        layers = decoder_layers(model)
        patch_vector = vector.detach()

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim == 3 and hidden.shape[1] >= 1:
                patched = hidden.clone()
                delta = patch_vector.to(device=hidden.device, dtype=hidden.dtype)
                patched[:, -1, :] = patched[:, -1, :] + delta
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return output

        handle = layers[layer_idx].register_forward_hook(hook)

    try:
        refusals = 0
        for prompt in prompts:
            answer = str(generate_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)["response"])
            refusals += int(is_refusal(answer))
        return refusals / max(1, len(prompts))
    finally:
        if handle is not None:
            handle.remove()


def validate_refusal_bases(
    model: nn.Module,
    tokenizer,
    model_id: str,
    artifacts: dict[tuple[int, int], Path],
    harmful_prompts: list[str],
    benign_prompts: list[str],
    kr_values: list[int],
    validation_components: int,
    intervention_scales: list[float],
    max_new_tokens: int,
) -> list[dict[str, object]]:
    validation_harm = harmful_prompts
    validation_benign = benign_prompts
    base_benign_refusal = generate_refusal_rate(
        model, tokenizer, validation_benign, max_new_tokens=max_new_tokens
    )
    base_harmful_refusal = generate_refusal_rate(
        model, tokenizer, validation_harm, max_new_tokens=max_new_tokens
    )
    rows: list[dict[str, object]] = []

    for (layer_idx, kr), path in sorted(artifacts.items()):
        if kr not in kr_values:
            continue
        payload = torch.load(path, map_location="cpu")
        basis = payload["basis"].float()
        component_scales = payload.get("component_scales", torch.ones(basis.shape[1])).float()
        components = min(validation_components, basis.shape[1])
        for component in range(components):
            direction = basis[:, component]
            natural_scale = float(component_scales[component])
            for intervention_scale in intervention_scales:
                scaled = direction * natural_scale * intervention_scale
                induced_benign_refusal = generate_refusal_rate(
                    model,
                    tokenizer,
                    validation_benign,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    vector=scaled,
                )
                suppressed_harmful_refusal = generate_refusal_rate(
                    model,
                    tokenizer,
                    validation_harm,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    vector=-scaled,
                )
                rows.append(
                    {
                        "model": model_id,
                        "layer": layer_idx,
                        "k_r": kr,
                        "component": component,
                        "artifact": str(path),
                        "num_harmful_extract": payload["num_harmful"],
                        "num_benign_extract": payload["num_benign"],
                        "validation_harmful_prompts": len(validation_harm),
                        "validation_benign_prompts": len(validation_benign),
                        "intervention_scale": intervention_scale,
                        "natural_component_scale": natural_scale,
                        "effective_intervention_norm": float(scaled.norm()),
                        "basis_method": payload.get("basis_method", "unknown"),
                        "mean_diff_norm": payload.get("mean_diff_norm", float("nan")),
                        "base_benign_refusal_rate": base_benign_refusal,
                        "induced_benign_refusal_rate": induced_benign_refusal,
                        "induce_delta": induced_benign_refusal - base_benign_refusal,
                        "base_harmful_refusal_rate": base_harmful_refusal,
                        "suppressed_harmful_refusal_rate": suppressed_harmful_refusal,
                        "suppress_delta": base_harmful_refusal - suppressed_harmful_refusal,
                        "status": "ok",
                    }
                )
                print(
                    "[vpref] "
                    f"layer={layer_idx} kr={kr} component={component} "
                    f"scale={intervention_scale:.3f} "
                    f"effective_norm={float(scaled.norm()):.3f} "
                    f"induce_delta={induced_benign_refusal - base_benign_refusal:.3f} "
                    f"suppress_delta={base_harmful_refusal - suppressed_harmful_refusal:.3f}"
                )
    return rows


def run_vpref_validation(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)
    model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
    model.eval()

    layers = decoder_layers(model)
    layer_indices = parse_layers(args.layers, len(layers))
    harmful_prompts = load_harmful_prompts(args)
    benign_prompts = load_benign_prompts(args)
    if len(harmful_prompts) < max(args.kr):
        raise ValueError(f"harmful_limit must be >= max k_r ({max(args.kr)}).")
    if not benign_prompts:
        raise ValueError("No benign prompts loaded.")

    print(
        "[vpref] "
        f"model={model_id} layers={layer_indices} "
        f"harmful={len(harmful_prompts)} benign={len(benign_prompts)} kr={args.kr}"
    )
    harm = collect_final_token_residuals(
        model, tokenizer, harmful_prompts, layer_indices, max_length=args.max_length
    )
    benign = collect_final_token_residuals(
        model, tokenizer, benign_prompts, layer_indices, max_length=args.max_length
    )
    artifacts = extract_refusal_bases(
        model_id=model_id,
        harm=harm,
        benign=benign,
        kr_values=args.kr,
        artifact_root=args.artifact_dir,
    )

    validation_harm = harmful_prompts[: args.validation_limit]
    validation_benign = benign_prompts[: args.validation_limit]
    rows = validate_refusal_bases(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        artifacts=artifacts,
        harmful_prompts=validation_harm,
        benign_prompts=validation_benign,
        kr_values=args.kr,
        validation_components=args.validation_components,
        intervention_scales=args.intervention_scales
        if args.intervention_scale is None
        else [args.intervention_scale],
        max_new_tokens=args.validation_max_new_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"[vpref] wrote {args.output}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", help="Model id; defaults to config model.default.")
    parser.add_argument("--output", type=Path, default=Path("results/vpref_validation.csv"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--layers", default="auto", help="'auto', 'all', or comma/range list like 20,24-28")
    parser.add_argument("--kr", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--harmful-file", type=Path)
    parser.add_argument("--harmful-dataset", default="walledai/AdvBench")
    parser.add_argument("--harmful-config")
    parser.add_argument("--harmful-split", default="train")
    parser.add_argument("--harmful-column", default="auto")
    parser.add_argument("--harmful-limit", type=int, default=64)
    parser.add_argument("--harmful-offset", type=int, default=0)
    parser.add_argument("--benign-file", type=Path)
    parser.add_argument("--benign-dataset", default="Salesforce/wikitext")
    parser.add_argument("--benign-config", default="wikitext-2-raw-v1")
    parser.add_argument("--benign-split", default="train")
    parser.add_argument("--benign-column", default="text")
    parser.add_argument("--benign-limit", type=int, default=64)
    parser.add_argument("--benign-offset", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=32)
    parser.add_argument("--validation-components", type=int, default=1)
    parser.add_argument("--validation-max-new-tokens", type=int, default=64)
    parser.add_argument("--intervention-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--intervention-scale", type=float)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    run_vpref_validation(args)


if __name__ == "__main__":
    main()
