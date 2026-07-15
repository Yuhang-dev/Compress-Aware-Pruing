"""Shared CLI arguments for baseline data and evaluation splits."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_data_download_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-data-download",
        action="store_true",
        help="Allow dataset downloads. Model and judge checkpoints remain local-only.",
    )


def add_registered_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adv-eval-offset", type=int, default=128)
    parser.add_argument("--ood-eval-offset", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--benign-eval-offset", type=int, default=128)
    parser.add_argument("--benign-eval-limit", type=int, default=128)
    parser.add_argument("--advbench-dataset", default="walledai/AdvBench")
    parser.add_argument("--advbench-config")
    parser.add_argument("--advbench-split", default="train")
    parser.add_argument("--advbench-column", default="auto")
    parser.add_argument("--harmbench-dataset", default="walledai/HarmBench")
    parser.add_argument("--harmbench-config", default="standard")
    parser.add_argument("--harmbench-split", default="train")
    parser.add_argument("--harmbench-column", default="auto")
    parser.add_argument("--strongreject-dataset", default="walledai/StrongREJECT")
    parser.add_argument("--strongreject-config")
    parser.add_argument("--strongreject-split", default="train")
    parser.add_argument("--strongreject-column", default="auto")
    parser.add_argument(
        "--benign-file", type=Path, default=Path("data/alpaca_cleaned_train.jsonl")
    )
    parser.add_argument("--benign-dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--benign-config")
    parser.add_argument("--benign-split", default="train")
    parser.add_argument("--benign-column", default="auto")


def add_ppl_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ppl-dataset", default="Salesforce/wikitext")
    parser.add_argument("--ppl-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--sample-windows", type=int, default=128)
    parser.add_argument(
        "--window-index-file",
        type=Path,
        default=Path("results/phase1_v2/ppl_windows_wikitext2_seed0.json"),
    )


def registered_split_cli(args) -> list[str]:
    result = [
        "--adv-eval-offset",
        str(args.adv_eval_offset),
        "--ood-eval-offset",
        str(args.ood_eval_offset),
        "--eval-limit",
        str(args.eval_limit),
        "--benign-eval-offset",
        str(args.benign_eval_offset),
        "--benign-eval-limit",
        str(args.benign_eval_limit),
        "--advbench-dataset",
        args.advbench_dataset,
        "--advbench-split",
        args.advbench_split,
        "--advbench-column",
        args.advbench_column,
        "--harmbench-dataset",
        args.harmbench_dataset,
        "--harmbench-split",
        args.harmbench_split,
        "--harmbench-column",
        args.harmbench_column,
        "--strongreject-dataset",
        args.strongreject_dataset,
        "--strongreject-split",
        args.strongreject_split,
        "--strongreject-column",
        args.strongreject_column,
        "--benign-file",
        str(args.benign_file),
        "--benign-dataset",
        args.benign_dataset,
        "--benign-split",
        args.benign_split,
        "--benign-column",
        args.benign_column,
    ]
    optional = (
        ("--advbench-config", args.advbench_config),
        ("--harmbench-config", args.harmbench_config),
        ("--strongreject-config", args.strongreject_config),
        ("--benign-config", args.benign_config),
    )
    for flag, value in optional:
        if value is not None:
            result.extend((flag, str(value)))
    return result


def ppl_cli(args) -> list[str]:
    return [
        "--ppl-dataset",
        args.ppl_dataset,
        "--ppl-dataset-config",
        args.ppl_dataset_config,
        "--ppl-split",
        args.ppl_split,
        "--context-len",
        str(args.context_len),
        "--stride",
        str(args.stride),
        "--sample-windows",
        str(args.sample_windows),
        "--window-index-file",
        str(args.window_index_file),
    ]
