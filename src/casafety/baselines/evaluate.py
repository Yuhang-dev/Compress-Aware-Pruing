"""Aggregate-only safety, benign-refusal, coherence, and PPL evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .arguments import (
    add_data_download_argument,
    add_ppl_arguments,
    add_registered_split_arguments,
)
from .common import (
    METHODS,
    MODEL_ID,
    TRAINING_METHODS,
    atomic_write_json,
    directory_hashes,
    file_sha256,
    read_json,
    source_identity,
    utc_now,
    verify_hashes,
    verify_wanda_checkpoint,
)
from .data import (
    load_registered_evaluation_splits,
    prompt_hashes,
    split_identity,
)


HARMFUL_DATASETS = ("advbench", "harmbench", "strongreject")


def load_artifact_model(args: argparse.Namespace, artifact: dict[str, Any]):
    from casafety.phase0_smoke_eval import load_model_and_tokenizer

    if artifact.get("status") != "completed" or artifact.get("method") != args.method:
        raise ValueError("Baseline artifact manifest is incomplete or for another method")
    if artifact.get("model") != args.model_id:
        raise ValueError("Baseline artifact model mismatch")
    if args.method in TRAINING_METHODS:
        base = artifact.get("base_checkpoint") or {}
        if Path(base.get("path", "")).resolve() != args.wanda_checkpoint_dir.resolve():
            raise ValueError("LoRA artifact was trained on a different Wanda checkpoint")
        verify_wanda_checkpoint(
            args.wanda_checkpoint_dir,
            args.wanda_manifest,
            expected_model=args.model_id,
            verify_all_files=True,
        )
        adapter_dir = Path(artifact["adapter_dir"])
        actual = directory_hashes(adapter_dir)
        if actual != artifact.get("adapter_files_sha256"):
            raise ValueError("LoRA adapter hashes do not match its manifest")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to evaluate SFT/DPO adapters") from exc
        model, tokenizer = load_model_and_tokenizer(
            str(args.wanda_checkpoint_dir.resolve()), True
        )
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir.resolve()),
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()
        return model, tokenizer

    checkpoint_dir = Path(artifact["checkpoint_dir"])
    verify_hashes(checkpoint_dir, artifact.get("checkpoint_files_sha256") or {})
    model, tokenizer = load_model_and_tokenizer(str(checkpoint_dir.resolve()), True)
    return model, tokenizer


def _generate(
    args: argparse.Namespace,
    model,
    tokenizer,
    splits: dict[str, list[tuple[int, str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from casafety.phase0_smoke_eval import (
        generate_answer,
        is_refusal,
        lexical_coherence_stats,
    )

    harmful_rows: list[dict[str, Any]] = []
    for dataset in HARMFUL_DATASETS:
        prompts = splits[dataset]
        for order, (prompt_id, prompt) in enumerate(prompts):
            if order == 0 or (order + 1) % 20 == 0:
                print(f"[baseline-eval] {args.method}/{dataset} {order + 1}/{len(prompts)}")
            generated = generate_answer(
                model, tokenizer, prompt, max_new_tokens=args.max_new_tokens
            )
            response = str(generated["response"])
            lexical = lexical_coherence_stats(response)
            coherent = bool(lexical["lexically_coherent"]) and (
                float(generated["response_ppl"]) <= args.response_ppl_threshold
            )
            harmful_rows.append(
                {
                    "model": args.model_id,
                    "condition": args.method,
                    "dataset": dataset,
                    "prompt_id": int(prompt_id),
                    "eval_order": int(order),
                    "prompt": prompt,
                    "response": response,
                    "response_tokens": int(generated["response_tokens"]),
                    "response_ppl": float(generated["response_ppl"]),
                    "coherent": coherent,
                    "ppl_coherent": float(generated["response_ppl"])
                    <= args.response_ppl_threshold,
                }
            )

    benign = {
        "n": 0,
        "refusal_n": 0,
        "coherent_n": 0,
        "response_ppl_sum": 0.0,
        "response_tokens_sum": 0,
    }
    for order, (_prompt_id, prompt) in enumerate(splits["benign"]):
        if order == 0 or (order + 1) % 20 == 0:
            print(
                f"[baseline-eval] {args.method}/benign "
                f"{order + 1}/{len(splits['benign'])}"
            )
        generated = generate_answer(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens
        )
        response = str(generated["response"])
        lexical = lexical_coherence_stats(response)
        coherent = bool(lexical["lexically_coherent"]) and (
            float(generated["response_ppl"]) <= args.response_ppl_threshold
        )
        benign["n"] += 1
        benign["refusal_n"] += int(is_refusal(response) and coherent)
        benign["coherent_n"] += int(coherent)
        benign["response_ppl_sum"] += float(generated["response_ppl"])
        benign["response_tokens_sum"] += int(generated["response_tokens"])
    return harmful_rows, benign


def _ppl(args: argparse.Namespace, model, tokenizer) -> dict[str, Any]:
    from casafety.ppl_eval_v2 import eval_ppl_on_windows, prepare_ppl_inputs

    if not args.window_index_file.is_file():
        raise FileNotFoundError(
            f"Registered PPL window index is missing: {args.window_index_file}"
        )
    input_ids, windows = prepare_ppl_inputs(
        tokenizer=tokenizer,
        dataset_id=args.ppl_dataset,
        config_name=args.ppl_dataset_config,
        split=args.ppl_split,
        context_len=args.context_len,
        stride=args.stride,
        sample_windows=args.sample_windows,
        seed=args.seed,
        window_index_file=args.window_index_file,
        local_files_only=not args.allow_data_download,
        force_resample=False,
    )
    mean_nll, ppl, tokens = eval_ppl_on_windows(model, input_ids, windows)
    return {
        "model": args.model_id,
        "condition": args.method,
        "dataset": "wikitext2",
        "context_len": args.context_len,
        "stride": args.stride,
        "sample_windows": args.sample_windows,
        "windows_evaluated": len(windows),
        "tokens": int(tokens),
        "mean_nll": float(mean_nll),
        "ppl": float(ppl),
        "seed": args.seed,
        "window_index_sha256": file_sha256(args.window_index_file),
    }


def _harmful_summaries(
    judged: list[dict[str, Any]], identities: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    frame = pd.DataFrame(judged)
    rows: list[dict[str, Any]] = []
    for dataset in HARMFUL_DATASETS:
        group = frame[frame["dataset"].eq(dataset)]
        if len(group) != identities[dataset]["count"]:
            raise ValueError(f"Unexpected judged row count for {dataset}: {len(group)}")
        unsafe = group["unsafe"].astype(bool)
        coherent = group["coherent"].astype(bool)
        rows.append(
            {
                "model": str(group["model"].iloc[0]),
                "condition": str(group["condition"].iloc[0]),
                "dataset": dataset,
                "n": len(group),
                "attack_success_n": int(group["attack_success"].sum()),
                "asr": float(group["attack_success"].mean()),
                "raw_unsafe_n": int(unsafe.sum()),
                "raw_unsafe_rate": float(unsafe.mean()),
                "refusal_n": int(group["refused"].sum()),
                "refusal_rate": float(group["refused"].mean()),
                "coherent_n": int(coherent.sum()),
                "coherent_rate": float(coherent.mean()),
                "unsafe_coherent_n": int((unsafe & coherent).sum()),
                "unsafe_incoherent_n": int((unsafe & ~coherent).sum()),
                "safe_coherent_n": int((~unsafe & coherent).sum()),
                "safe_incoherent_n": int((~unsafe & ~coherent).sum()),
                "mean_response_ppl": float(group["response_ppl"].mean()),
                "mean_response_tokens": float(group["response_tokens"].mean()),
                "prompt_content_sha256": identities[dataset]["content_set_sha256"],
            }
        )
    return pd.DataFrame(rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", choices=METHODS, required=True)
    result.add_argument("--model-id", default=MODEL_ID)
    result.add_argument("--wanda-checkpoint-dir", type=Path, required=True)
    result.add_argument("--wanda-manifest", type=Path, required=True)
    result.add_argument("--artifact-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    result.add_argument("--max-new-tokens", type=int, default=128)
    result.add_argument("--response-ppl-threshold", type=float, default=100.0)
    result.add_argument("--judge-model")
    result.add_argument("--judge-max-new-tokens", type=int, default=16)
    result.add_argument("--seed", type=int, default=0)
    add_data_download_argument(result)
    add_registered_split_arguments(result)
    add_ppl_arguments(result)
    return result


def run(args: argparse.Namespace) -> None:
    from casafety.closed_form_readout_repair import write_text_free_csv
    from casafety.config import load_config
    from casafety.ood_residual_diag import judge_rows, release

    if args.model_id != MODEL_ID:
        raise ValueError(f"This baseline protocol is registered for {MODEL_ID}")
    expected_outputs = (
        args.output_dir / "safety_summary.csv",
        args.output_dir / "benign_summary.csv",
        args.output_dir / "ppl_summary.csv",
        args.output_dir / "main_summary.csv",
        args.output_dir / "evaluation_manifest.json",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("Refusing to overwrite baseline evaluation outputs")
    artifact = read_json(args.artifact_manifest)
    total_start = time.perf_counter()
    started_at = utc_now()
    splits = load_registered_evaluation_splits(args)
    identities = {
        name: split_identity(rows, label=name) for name, rows in splits.items()
    }
    harmful_hashes = {
        identities[name]["content_set_sha256"] for name in HARMFUL_DATASETS
    }
    if len(harmful_hashes) != len(HARMFUL_DATASETS):
        raise ValueError("Harmful evaluation datasets have identical content hashes")
    evaluation_union = set().union(*(prompt_hashes(rows) for rows in splits.values()))
    if args.method in TRAINING_METHODS:
        registered = (
            artifact.get("training_data", {})
            .get("evaluation_exclusion", {})
            .get("union_content_set_sha256")
        )
        from .common import digest_strings

        if registered != digest_strings(evaluation_union):
            raise ValueError("Evaluation split changed after SFT/DPO de-duplication")

    model, tokenizer = load_artifact_model(args, artifact)
    generation_start = time.perf_counter()
    generation_seconds = 0.0
    ppl_seconds = 0.0
    try:
        harmful_rows, benign_counts = _generate(args, model, tokenizer, splits)
        generation_seconds = time.perf_counter() - generation_start
        ppl_start = time.perf_counter()
        ppl_row = _ppl(args, model, tokenizer)
        ppl_seconds = time.perf_counter() - ppl_start
    finally:
        del model, tokenizer
        gc.collect()
        release()

    # Dataset downloads may be enabled, but judge/model downloads never are.
    args.local_files_only = True
    judge_start = time.perf_counter()
    judged = judge_rows(args, harmful_rows, load_config(args.config))
    judge_seconds = time.perf_counter() - judge_start
    safety = _harmful_summaries(judged, identities)
    del harmful_rows, judged

    benign_n = int(benign_counts["n"])
    benign = pd.DataFrame(
        [
            {
                "model": args.model_id,
                "condition": args.method,
                "dataset": "benign",
                "n": benign_n,
                "refusal_n": int(benign_counts["refusal_n"]),
                "refusal_rate": float(benign_counts["refusal_n"] / benign_n),
                "coherent_n": int(benign_counts["coherent_n"]),
                "coherent_rate": float(benign_counts["coherent_n"] / benign_n),
                "mean_response_ppl": float(
                    benign_counts["response_ppl_sum"] / benign_n
                ),
                "mean_response_tokens": float(
                    benign_counts["response_tokens_sum"] / benign_n
                ),
                "prompt_content_sha256": identities["benign"]["content_set_sha256"],
            }
        ]
    )
    ppl = pd.DataFrame([ppl_row])
    indexed = safety.set_index("dataset")
    main = pd.DataFrame(
        [
            {
                "model": args.model_id,
                "condition": args.method,
                "advbench_asr": float(indexed.loc["advbench", "asr"]),
                "harmbench_asr": float(indexed.loc["harmbench", "asr"]),
                "strongreject_asr": float(indexed.loc["strongreject", "asr"]),
                "ood_asr_mean": float(
                    (
                        indexed.loc["harmbench", "asr"]
                        + indexed.loc["strongreject", "asr"]
                    )
                    / 2
                ),
                "minimum_harmful_coherence": float(safety["coherent_rate"].min()),
                "benign_refusal_rate": float(benign.loc[0, "refusal_rate"]),
                "benign_coherent_rate": float(benign.loc[0, "coherent_rate"]),
                "ppl": float(ppl.loc[0, "ppl"]),
            }
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_text_free_csv(safety, args.output_dir / "safety_summary.csv")
    write_text_free_csv(benign, args.output_dir / "benign_summary.csv")
    write_text_free_csv(ppl, args.output_dir / "ppl_summary.csv")
    write_text_free_csv(main, args.output_dir / "main_summary.csv")
    total_seconds = time.perf_counter() - total_start
    csv_paths = expected_outputs[:-1]
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "method": args.method,
        "model": args.model_id,
        "artifact_manifest": str(args.artifact_manifest.resolve()),
        "artifact_manifest_sha256": file_sha256(args.artifact_manifest),
        "evaluation_splits": identities,
        "dataset_content_hashes_distinct": True,
        "protocol": {
            "decoding": {
                "do_sample": False,
                "max_new_tokens": int(args.max_new_tokens),
                "response_ppl_threshold": float(args.response_ppl_threshold),
                "seed": int(args.seed),
            },
            "judge": {
                "model": args.judge_model,
                "max_new_tokens": int(args.judge_max_new_tokens),
                "local_files_only": True,
                "config_sha256": file_sha256(args.config),
            },
            "ppl": {
                "dataset": args.ppl_dataset,
                "dataset_config": args.ppl_dataset_config,
                "split": args.ppl_split,
                "context_len": int(args.context_len),
                "stride": int(args.stride),
                "sample_windows": int(args.sample_windows),
                "seed": int(args.seed),
                "window_index_sha256": ppl_row["window_index_sha256"],
            },
            "registered_slices": {
                "adv_eval_offset": int(args.adv_eval_offset),
                "ood_eval_offset": int(args.ood_eval_offset),
                "eval_limit": int(args.eval_limit),
                "benign_eval_offset": int(args.benign_eval_offset),
                "benign_eval_limit": int(args.benign_eval_limit),
            },
        },
        "metrics": json.loads(main.to_json(orient="records"))[0],
        "artifacts_sha256": {str(path): file_sha256(path) for path in csv_paths},
        "runtime": {
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "generation_seconds": generation_seconds,
            "ppl_seconds": ppl_seconds,
            "judge_seconds": judge_seconds,
            "total_seconds": total_seconds,
            "gpu_count": 1,
            "gpu_hours": total_seconds / 3600.0,
        },
        "source": source_identity(),
        "privacy": {
            "prompt_text_persisted": False,
            "response_text_persisted": False,
            "per_example_rows_persisted": False,
            "aggregate_only": True,
        },
    }
    atomic_write_json(args.output_dir / "evaluation_manifest.json", manifest)
    print(main.to_string(index=False))


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
