from __future__ import annotations

import argparse
import gc
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch import nn

from .config import load_config
from .models import model_slug, resolve_model_id
from .phase0_smoke_eval import CALIB_PROMPTS, collect_wanda_input_norms, infer_prompt_column
from .phase0_smoke_eval import load_model_and_tokenizer, target_linear_modules
from .pruners import compute_mask


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def module_layer(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", f".{name}.")
    return int(match.group(1)) if match else None


def module_kind(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def selected_linear_modules(model: nn.Module, suffixes: Iterable[str]) -> list[tuple[str, nn.Linear]]:
    suffix_tuple = tuple(suffixes)
    return [(name, module) for name, module in target_linear_modules(model) if name.endswith(suffix_tuple)]


def load_hf_column(
    dataset_id: str,
    config_name: str | None,
    split: str,
    column: str,
    local_files_only: bool,
) -> list[str]:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - remote dependency
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
    print(
        f"[phase1-followup] loaded {len(values)} rows from "
        f"{dataset_id}:{split} column={selected_column}"
    )
    return values


def read_prompt_file(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_benign_prompts(args: argparse.Namespace) -> list[str]:
    prompts = read_prompt_file(args.benign_file)
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
    prompts = prompts[args.benign_offset :]
    return prompts[: args.benign_limit] if args.benign_limit else prompts


def default_crit_set_path(model_id: str, p: float) -> Path:
    return Path("results/crit_sets") / f"{model_slug(model_id)}_p{p:g}.pt"


def tensor_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().detach().cpu())


def run_crit_wanda_survival(
    model_id: str,
    modules: list[tuple[str, nn.Linear]],
    input_norms: dict[str, torch.Tensor],
    crit_payload: dict,
    sparsities: list[float],
) -> pd.DataFrame:
    crit_indices: dict[str, torch.Tensor] = crit_payload["crit_indices"]
    random_indices: dict[str, torch.Tensor] = crit_payload["random_indices"]
    rows: list[dict[str, object]] = []

    for sparsity in sparsities:
        for name, module in modules:
            if name not in crit_indices and name not in random_indices:
                continue
            input_norm = input_norms.get(name)
            if input_norm is None:
                print(f"[phase1-followup] missing input_norm for {name}; skipping survival")
                continue
            stats = {"input_norm": input_norm.to(module.weight.device)}
            with torch.inference_mode():
                mask = compute_mask(module.weight.detach(), stats, sparsity, "wanda")
                flat_mask = mask.detach().flatten().to(device="cpu", dtype=torch.float32)
            for group, index_map in (("crit", crit_indices), ("random", random_indices)):
                idx = index_map.get(name, torch.empty(0, dtype=torch.int64)).to(torch.int64)
                selected = flat_mask[idx] if idx.numel() else torch.empty(0)
                survived = int(selected.sum().item()) if selected.numel() else 0
                rows.append(
                    {
                        "model": model_id,
                        "module": name,
                        "layer": module_layer(name),
                        "module_kind": module_kind(name),
                        "group": group,
                        "sparsity": sparsity,
                        "count": int(idx.numel()),
                        "survived": survived,
                        "survival_rate": tensor_mean(selected),
                    }
                )
            del mask, flat_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    overall_rows = []
    for (group, sparsity), sub in df.groupby(["group", "sparsity"], dropna=False):
        count = int(sub["count"].sum())
        survived = int(sub["survived"].sum())
        overall_rows.append(
            {
                "model": model_id,
                "module": "__overall__",
                "layer": None,
                "module_kind": "overall",
                "group": group,
                "sparsity": sparsity,
                "count": count,
                "survived": survived,
                "survival_rate": survived / count if count else float("nan"),
                "module_count": int(len(sub)),
            }
        )
    return pd.concat([df, pd.DataFrame(overall_rows)], ignore_index=True, sort=False)


def input_norm_percentiles(input_norm: torch.Tensor) -> torch.Tensor:
    input_norm = input_norm.float().cpu()
    order = torch.argsort(input_norm)
    ranks = torch.empty_like(order, dtype=torch.float32)
    if input_norm.numel() <= 1:
        ranks.fill_(1.0)
        return ranks
    ranks[order] = torch.arange(input_norm.numel(), dtype=torch.float32) / float(input_norm.numel() - 1)
    return ranks


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    weights = weights.float().cpu()
    values = values.float().cpu()
    denom = weights.sum().clamp_min(1e-12)
    return float((values * weights).sum() / denom)


def top_energy_percentile_summary(
    percentiles: torch.Tensor,
    energy: torch.Tensor,
    top_fractions: list[float],
) -> dict[str, float]:
    if energy.numel() == 0:
        return {}
    energy = energy.float().cpu()
    percentiles = percentiles.float().cpu()
    result: dict[str, float] = {}
    total_energy = energy.sum().clamp_min(1e-12)
    for threshold in (0.25, 0.50, 0.75):
        result[f"energy_mass_input_norm_bottom_{int(threshold * 100)}"] = float(
            energy[percentiles <= threshold].sum() / total_energy
        )
    for fraction in top_fractions:
        k = max(1, int(round(energy.numel() * fraction)))
        top_idx = torch.topk(energy, k=k, largest=True).indices
        result[f"top_{fraction:g}_energy_input_norm_percentile_mean"] = float(percentiles[top_idx].mean())
        result[f"top_{fraction:g}_energy_input_norm_percentile_median"] = float(
            torch.quantile(percentiles[top_idx], 0.50)
        )
    return result


def parse_vpref_layer(path: Path) -> int | None:
    match = re.search(r"_layer(\d+)_kr(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def parse_vpref_kr(path: Path) -> int | None:
    match = re.search(r"_layer(\d+)_kr(\d+)\.pt$", path.name)
    return int(match.group(2)) if match else None


def load_vpref_artifacts(vpref_dir: Path, model_id: str, layers: set[int] | None, kr_values: set[int] | None) -> list[Path]:
    slug = model_slug(model_id)
    paths = sorted(vpref_dir.glob(f"{slug}_layer*_kr*.pt"))
    selected = []
    for path in paths:
        layer = parse_vpref_layer(path)
        kr = parse_vpref_kr(path)
        if layers is not None and layer not in layers:
            continue
        if kr_values is not None and kr not in kr_values:
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(
            f"No vpref artifacts found in {vpref_dir} for model={model_id}, layers={layers}, kr={kr_values}"
        )
    return selected


def run_a2_channel_diagnostic(
    model_id: str,
    modules: list[tuple[str, nn.Linear]],
    input_norms: dict[str, torch.Tensor],
    vpref_paths: list[Path],
    top_energy_fractions: list[float],
) -> pd.DataFrame:
    modules_by_layer: dict[int, list[tuple[str, nn.Linear]]] = {}
    for name, module in modules:
        layer = module_layer(name)
        if layer is not None:
            modules_by_layer.setdefault(layer, []).append((name, module))

    rows: list[dict[str, object]] = []
    for path in vpref_paths:
        payload = torch.load(path, map_location="cpu")
        layer = int(payload["layer"])
        basis = payload["basis"].float()
        kr = int(payload["k_r"])
        if layer not in modules_by_layer:
            continue
        for name, module in modules_by_layer[layer]:
            input_norm = input_norms.get(name)
            if input_norm is None:
                print(f"[phase1-followup] missing input_norm for {name}; skipping A2 diagnostic")
                continue
            if module.weight.shape[0] != basis.shape[0]:
                print(
                    "[phase1-followup] skipping incompatible module "
                    f"{name}: weight_out={module.weight.shape[0]} basis_dim={basis.shape[0]}"
                )
                continue

            with torch.inference_mode():
                device = module.weight.device
                r = basis.to(device=device, dtype=torch.float32)
                weight = module.weight.detach().to(dtype=torch.float32)
                a = weight.transpose(0, 1).matmul(r)
                a_energy = a.square().sum(dim=1).detach().cpu()
            input_norm_cpu = input_norm.float().cpu()
            if input_norm_cpu.numel() != a_energy.numel():
                raise ValueError(
                    f"input_norm length mismatch for {name}: "
                    f"{input_norm_cpu.numel()} vs A channels {a_energy.numel()}"
                )
            input_pct = input_norm_percentiles(input_norm_cpu)
            weighted_energy = a_energy * input_norm_cpu.square()

            row: dict[str, object] = {
                "model": model_id,
                "vpref_artifact": str(path),
                "layer": layer,
                "k_r": kr,
                "basis_method": payload.get("basis_method", "unknown"),
                "module": name,
                "module_kind": module_kind(name),
                "input_channels": int(a_energy.numel()),
                "a_energy_sum": float(a_energy.sum()),
                "a_energy_mean": float(a_energy.mean()),
                "weighted_a_energy_sum": float(weighted_energy.sum()),
                "input_norm_percentile_energy_weighted_mean": weighted_mean(input_pct, a_energy),
                "input_norm_percentile_weighted_a_energy_mean": weighted_mean(input_pct, weighted_energy),
            }
            row.update(top_energy_percentile_summary(input_pct, a_energy, top_energy_fractions))
            rows.append(row)
            del a, a_energy, weighted_energy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def parse_optional_int_set(values: list[int] | None) -> set[int] | None:
    return set(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", help="Model id; defaults to config model.default.")
    parser.add_argument("--crit-set", type=Path)
    parser.add_argument("--crit-p", type=float, default=0.03)
    parser.add_argument("--vpref-dir", type=Path, default=Path("artifacts/vpref"))
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--kr", nargs="+", type=int)
    parser.add_argument("--target-suffixes", nargs="+", default=["o_proj", "down_proj"])
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.45, 0.50])
    parser.add_argument("--survival-output", type=Path, default=Path("results/phase1_crit_wanda_survival.csv"))
    parser.add_argument("--a2-output", type=Path, default=Path("results/phase1_a2_channel_diagnostic.csv"))
    parser.add_argument("--benign-file", type=Path)
    parser.add_argument("--benign-dataset", default="Salesforce/wikitext")
    parser.add_argument("--benign-config", default="wikitext-2-raw-v1")
    parser.add_argument("--benign-split", default="train")
    parser.add_argument("--benign-column", default="text")
    parser.add_argument("--benign-limit", type=int, default=64)
    parser.add_argument("--benign-offset", type=int, default=0)
    parser.add_argument("--calib-max-length", type=int, default=1024)
    parser.add_argument("--top-energy-fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)
    crit_set = args.crit_set or default_crit_set_path(model_id, args.crit_p)
    if not crit_set.exists():
        raise FileNotFoundError(f"Missing Crit set: {crit_set}. Run mechanism_diagnosis first.")

    print(f"[phase1-followup] loading model={model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, args.local_files_only)
    model.eval()
    modules = selected_linear_modules(model, args.target_suffixes)
    if not modules:
        raise ValueError(f"No modules found for suffixes={args.target_suffixes}")
    print(f"[phase1-followup] selected {len(modules)} modules: {args.target_suffixes}")

    benign_prompts = load_benign_prompts(args)
    print(f"[phase1-followup] collecting Wanda input norms on {len(benign_prompts)} benign prompts")
    input_norms = collect_wanda_input_norms(model, tokenizer, benign_prompts, args.calib_max_length)

    crit_payload = torch.load(crit_set, map_location="cpu")
    survival = run_crit_wanda_survival(
        model_id=model_id,
        modules=modules,
        input_norms=input_norms,
        crit_payload=crit_payload,
        sparsities=args.sparsities,
    )
    args.survival_output.parent.mkdir(parents=True, exist_ok=True)
    survival.to_csv(args.survival_output, index=False)
    print(f"[phase1-followup] wrote {args.survival_output}")

    vpref_paths = load_vpref_artifacts(
        args.vpref_dir,
        model_id=model_id,
        layers=parse_optional_int_set(args.layers),
        kr_values=parse_optional_int_set(args.kr),
    )
    print(f"[phase1-followup] loaded {len(vpref_paths)} vpref artifacts")
    a2 = run_a2_channel_diagnostic(
        model_id=model_id,
        modules=modules,
        input_norms=input_norms,
        vpref_paths=vpref_paths,
        top_energy_fractions=args.top_energy_fractions,
    )
    args.a2_output.parent.mkdir(parents=True, exist_ok=True)
    a2.to_csv(args.a2_output, index=False)
    print(f"[phase1-followup] wrote {args.a2_output}")

    print("\n== crit wanda survival overall ==")
    print(survival[survival["module"].astype(str).eq("__overall__")].to_string(index=False))
    print("\n== A2 channel diagnostic ==")
    cols = [
        "layer",
        "k_r",
        "module_kind",
        "input_norm_percentile_energy_weighted_mean",
        "input_norm_percentile_weighted_a_energy_mean",
        "energy_mass_input_norm_bottom_50",
        "top_0.01_energy_input_norm_percentile_median",
    ]
    available_cols = [col for col in cols if col in a2.columns]
    print(a2[available_cols].sort_values(["layer", "k_r", "module_kind"]).to_string(index=False))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
