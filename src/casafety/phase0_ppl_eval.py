from __future__ import annotations

import argparse
import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path

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
class PplResult:
    condition: str
    pruner: str
    sparsity: float | str
    sequences: int
    tokens: int
    mean_nll: float
    ppl: float
    pruned_layers: int


def load_wikitext_sequences(
    tokenizer,
    dataset_id: str,
    config_name: str,
    split: str,
    seq_len: int,
    limit: int,
    local_files_only: bool,
) -> list[torch.Tensor]:
    dataset = load_dataset(
        dataset_id,
        config_name,
        split=split,
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    texts = [row["text"] for row in dataset if isinstance(row.get("text"), str) and row["text"].strip()]
    tokenized = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    sequences = []
    for start in range(0, input_ids.numel() - seq_len, seq_len):
        sequences.append(input_ids[start : start + seq_len])
        if len(sequences) >= limit:
            break
    if not sequences:
        raise ValueError("No WikiText sequences were produced; check dataset/cache settings.")
    return sequences


def eval_ppl(model, sequences: list[torch.Tensor]) -> tuple[float, float, int]:
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for seq in sequences:
            input_ids = seq.unsqueeze(0).to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            token_count = input_ids.shape[1] - 1
            total_nll += float(outputs.loss.detach().cpu()) * token_count
            total_tokens += token_count
    mean_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(mean_nll) if mean_nll < 100 else float("inf")
    return mean_nll, ppl, total_tokens


def evaluate_condition(
    model_id: str,
    condition: EvalCondition,
    sequences: list[torch.Tensor],
    local_files_only: bool,
    calib_max_length: int,
) -> PplResult:
    model, tokenizer = load_model_and_tokenizer(model_id, local_files_only)
    pruned_layers = 0
    if condition.pruner is not None and condition.sparsity is not None:
        pruned_layers = apply_pruning(model, tokenizer, condition.pruner, condition.sparsity, calib_max_length)
    mean_nll, ppl, tokens = eval_ppl(model, sequences)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return PplResult(
        condition=condition.name,
        pruner=condition.pruner or "none",
        sparsity=condition.sparsity if condition.sparsity is not None else 0.0,
        sequences=len(sequences),
        tokens=tokens,
        mean_nll=mean_nll,
        ppl=ppl,
        pruned_layers=pruned_layers,
    )


def build_conditions(pruners: list[str], sparsities: list[str]) -> list[EvalCondition]:
    conditions = [EvalCondition("dense", None, None)]
    for pruner in pruners:
        for sparsity_text in sparsities:
            sparsity = parse_sparsity(sparsity_text)
            conditions.append(EvalCondition(f"{pruner}_{format_sparsity_name(sparsity)}", pruner, sparsity))
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model", help="Model id to evaluate; defaults to config model.default.")
    parser.add_argument("--output", type=Path, default=Path("results/phase0_wikitext_ppl_grid.csv"))
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--calib-max-length", type=int, default=256)
    parser.add_argument("--sparsities", nargs="+", default=["0.5", "0.6", "0.7"])
    parser.add_argument("--pruners", nargs="+", default=["magnitude", "wanda"])
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

    sequences = load_wikitext_sequences(
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        config_name=args.dataset_config,
        split=args.split,
        seq_len=args.seq_len,
        limit=args.limit,
        local_files_only=args.local_files_only,
    )

    rows = []
    for condition in build_conditions(args.pruners, args.sparsities):
        print(f"[phase0-ppl] running {condition.name}")
        rows.append(
            evaluate_condition(
                model_id=model_id,
                condition=condition,
                sequences=sequences,
                local_files_only=args.local_files_only,
                calib_max_length=args.calib_max_length,
            ).__dict__
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
