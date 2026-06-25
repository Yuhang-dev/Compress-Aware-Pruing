from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

from .config import load_config
from .models import resolve_model_id
from .phase0_smoke_eval import EvalCondition, apply_pruning, load_model_and_tokenizer, resolve_cache_dir
from .phase0_smoke_eval import format_sparsity_name, parse_sparsity


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class PplWindow:
    index: int
    begin: int
    end: int
    target_start: int
    target_count: int


@dataclass(frozen=True)
class PplV2Result:
    model: str
    condition: str
    pruner: str
    sparsity: float | str
    context_len: int
    stride: int
    sample_windows: int
    windows_evaluated: int
    tokens: int
    mean_nll: float
    ppl: float
    seed: int
    window_index_file: str
    pruned_layers: int


def load_wikitext_token_ids(
    tokenizer,
    dataset_id: str,
    config_name: str,
    split: str,
    local_files_only: bool,
) -> torch.Tensor:
    dataset = load_dataset(
        dataset_id,
        config_name,
        split=split,
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    texts = [row["text"] for row in dataset if isinstance(row.get("text"), str) and row["text"].strip()]
    tokenized = tokenizer("\n\n".join(texts), return_tensors="pt")
    return tokenized["input_ids"][0].cpu()


def build_strided_windows(num_tokens: int, context_len: int, stride: int) -> list[PplWindow]:
    if context_len < 2:
        raise ValueError(f"context_len must be >= 2, got {context_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > context_len:
        raise ValueError(f"stride must be <= context_len, got stride={stride}, context_len={context_len}")
    windows: list[PplWindow] = []
    index = 0
    for target_begin in range(0, num_tokens - 1, stride):
        target_end = min(target_begin + stride, num_tokens)
        begin = max(target_end - context_len, 0)
        end = target_end
        target_count = target_end - target_begin
        if end - begin < 2 or target_count <= 0:
            continue
        target_start = (end - begin) - target_count
        windows.append(
            PplWindow(
                index=index,
                begin=begin,
                end=end,
                target_start=target_start,
                target_count=target_count,
            )
        )
        index += 1
    if not windows:
        raise ValueError("No PPL windows were produced; check token count/context/stride.")
    return windows


def load_or_create_window_index(
    path: Path,
    all_windows: list[PplWindow],
    dataset_id: str,
    config_name: str,
    split: str,
    context_len: int,
    stride: int,
    sample_windows: int,
    seed: int,
    force_resample: bool = False,
) -> list[PplWindow]:
    if path.exists() and not force_resample:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "dataset": dataset_id,
            "dataset_config": config_name,
            "split": split,
            "context_len": context_len,
            "stride": stride,
            "sample_windows": sample_windows,
            "seed": seed,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Window index metadata mismatch for {path}: {mismatches}. "
                "Use --force-resample or a different --window-index-file."
            )
        windows = [PplWindow(**item) for item in payload["windows"]]
        print(f"[ppl-v2] loaded {len(windows)} sampled windows from {path}")
        return windows

    rng = random.Random(seed)
    if sample_windows <= 0 or sample_windows >= len(all_windows):
        selected = list(all_windows)
    else:
        selected_indices = sorted(rng.sample(range(len(all_windows)), sample_windows))
        selected = [all_windows[idx] for idx in selected_indices]

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset": dataset_id,
        "dataset_config": config_name,
        "split": split,
        "context_len": context_len,
        "stride": stride,
        "sample_windows": sample_windows,
        "seed": seed,
        "total_windows": len(all_windows),
        "windows": [window.__dict__ for window in selected],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[ppl-v2] wrote {len(selected)} sampled windows to {path}")
    return selected


def eval_ppl_on_windows(model, input_ids: torch.Tensor, windows: list[PplWindow]) -> tuple[float, float, int]:
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for window in windows:
            chunk = input_ids[window.begin : window.end].unsqueeze(0).to(device)
            labels = chunk.clone()
            labels[:, : window.target_start] = -100
            outputs = model(input_ids=chunk, labels=labels, use_cache=False)
            token_count = max(0, int((labels[:, 1:] != -100).sum().detach().cpu()))
            if token_count == 0:
                continue
            total_nll += float(outputs.loss.detach().cpu()) * token_count
            total_tokens += token_count
    mean_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(mean_nll) if mean_nll < 100 else float("inf")
    return mean_nll, ppl, total_tokens


def evaluate_condition(
    model_id: str,
    condition: EvalCondition,
    input_ids: torch.Tensor,
    windows: list[PplWindow],
    context_len: int,
    stride: int,
    sample_windows: int,
    seed: int,
    window_index_file: Path,
    local_files_only: bool,
    calib_max_length: int,
) -> PplV2Result:
    model, tokenizer = load_model_and_tokenizer(model_id, local_files_only)
    pruned_layers = 0
    if condition.pruner is not None and condition.sparsity is not None:
        pruned_layers = apply_pruning(model, tokenizer, condition.pruner, condition.sparsity, calib_max_length)
    mean_nll, ppl, tokens = eval_ppl_on_windows(model, input_ids, windows)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return PplV2Result(
        model=model_id,
        condition=condition.name,
        pruner=condition.pruner or "none",
        sparsity=condition.sparsity if condition.sparsity is not None else 0.0,
        context_len=context_len,
        stride=stride,
        sample_windows=sample_windows,
        windows_evaluated=len(windows),
        tokens=tokens,
        mean_nll=mean_nll,
        ppl=ppl,
        seed=seed,
        window_index_file=str(window_index_file),
        pruned_layers=pruned_layers,
    )


def build_conditions(pruners: list[str], sparsities: list[str]) -> list[EvalCondition]:
    conditions = [EvalCondition("dense", None, None)]
    for pruner in pruners:
        for sparsity_text in sparsities:
            sparsity = parse_sparsity(sparsity_text)
            conditions.append(EvalCondition(f"{pruner}_{format_sparsity_name(sparsity)}", pruner, sparsity))
    return conditions


def prepare_ppl_inputs(
    tokenizer,
    dataset_id: str,
    config_name: str,
    split: str,
    context_len: int,
    stride: int,
    sample_windows: int,
    seed: int,
    window_index_file: Path,
    local_files_only: bool,
    force_resample: bool = False,
) -> tuple[torch.Tensor, list[PplWindow]]:
    input_ids = load_wikitext_token_ids(tokenizer, dataset_id, config_name, split, local_files_only)
    all_windows = build_strided_windows(input_ids.numel(), context_len, stride)
    windows = load_or_create_window_index(
        path=window_index_file,
        all_windows=all_windows,
        dataset_id=dataset_id,
        config_name=config_name,
        split=split,
        context_len=context_len,
        stride=stride,
        sample_windows=sample_windows,
        seed=seed,
        force_resample=force_resample,
    )
    return input_ids, windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", help="Model id to evaluate; defaults to config model.default.")
    parser.add_argument("--output", type=Path, default=Path("results/phase1_v2/ppl_v2.csv"))
    parser.add_argument("--window-index-file", type=Path, default=Path("results/phase1_v2/ppl_windows_wikitext2_seed0.json"))
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--sample-windows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--calib-max-length", type=int, default=256)
    parser.add_argument("--sparsities", nargs="+", default=["0.45", "0.50"])
    parser.add_argument("--pruners", nargs="+", default=["wanda"])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=args.local_files_only,
        cache_dir=resolve_cache_dir(model_id),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids, windows = prepare_ppl_inputs(
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        config_name=args.dataset_config,
        split=args.split,
        context_len=args.context_len,
        stride=args.stride,
        sample_windows=args.sample_windows,
        seed=args.seed,
        window_index_file=args.window_index_file,
        local_files_only=args.local_files_only,
        force_resample=args.force_resample,
    )

    rows = []
    for condition in build_conditions(args.pruners, args.sparsities):
        print(f"[ppl-v2] running {condition.name}")
        rows.append(
            evaluate_condition(
                model_id=model_id,
                condition=condition,
                input_ids=input_ids,
                windows=windows,
                context_len=args.context_len,
                stride=args.stride,
                sample_windows=args.sample_windows,
                seed=args.seed,
                window_index_file=args.window_index_file,
                local_files_only=args.local_files_only,
                calib_max_length=args.calib_max_length,
            ).__dict__
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"[ppl-v2] wrote {args.output}")


if __name__ == "__main__":
    main()
