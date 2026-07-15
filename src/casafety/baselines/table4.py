"""Build the audited Table 4 record for the post-pruning Safety-SFT row."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json, file_sha256, read_json, utc_now


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact-manifest", type=Path, required=True)
    result.add_argument("--evaluation-manifest", type=Path, required=True)
    result.add_argument("--evaluation-summary", type=Path, required=True)
    result.add_argument("--latency-manifest", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    result.add_argument(
        "--window-index-file",
        type=Path,
        default=Path("results/phase1_v2/ppl_windows_wikitext2_seed0.json"),
    )
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def _one_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    outputs = (
        args.output_dir / "table4_safety_sft_record.json",
        args.output_dir / "table4_safety_sft_record.csv",
        args.output_dir / "table4_safety_sft_row.tex",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite an existing Table 4 record")
    artifact = read_json(args.artifact_manifest)
    evaluation_manifest = read_json(args.evaluation_manifest)
    latency = read_json(args.latency_manifest)
    evaluation = _one_csv_row(args.evaluation_summary)
    if artifact.get("status") != "completed" or artifact.get("method") != "sft":
        raise ValueError("Training artifact is not a completed SFT run")
    if latency.get("status") != "completed" or latency.get("method") != "sft":
        raise ValueError("Latency manifest is not a completed SFT measurement")
    artifact_sha256 = file_sha256(args.artifact_manifest)
    if evaluation_manifest.get("artifact_manifest_sha256") != artifact_sha256:
        raise ValueError("Evaluation manifest is bound to a different SFT artifact")
    if latency.get("artifact_manifest_sha256") != artifact_sha256:
        raise ValueError("Latency manifest is bound to a different SFT artifact")
    protocol = evaluation_manifest.get("protocol") or {}
    decoding = protocol.get("decoding") or {}
    judge = protocol.get("judge") or {}
    ppl_protocol = protocol.get("ppl") or {}
    slices = protocol.get("registered_slices") or {}
    expected_protocol = {
        "decoding.do_sample": (decoding.get("do_sample"), False),
        "decoding.max_new_tokens": (decoding.get("max_new_tokens"), 128),
        "decoding.response_ppl_threshold": (
            decoding.get("response_ppl_threshold"),
            100.0,
        ),
        "decoding.seed": (decoding.get("seed"), 0),
        "judge.model": (judge.get("model"), "meta-llama/Llama-Guard-3-8B"),
        "judge.max_new_tokens": (judge.get("max_new_tokens"), 16),
        "judge.local_files_only": (judge.get("local_files_only"), True),
        "judge.config_sha256": (judge.get("config_sha256"), file_sha256(args.config)),
        "ppl.dataset": (ppl_protocol.get("dataset"), "Salesforce/wikitext"),
        "ppl.dataset_config": (
            ppl_protocol.get("dataset_config"),
            "wikitext-2-raw-v1",
        ),
        "ppl.split": (ppl_protocol.get("split"), "test"),
        "ppl.context_len": (ppl_protocol.get("context_len"), 1024),
        "ppl.stride": (ppl_protocol.get("stride"), 512),
        "ppl.sample_windows": (ppl_protocol.get("sample_windows"), 128),
        "ppl.seed": (ppl_protocol.get("seed"), 0),
        "ppl.window_index_sha256": (
            ppl_protocol.get("window_index_sha256"),
            file_sha256(args.window_index_file),
        ),
        "slices.adv_eval_offset": (slices.get("adv_eval_offset"), 128),
        "slices.ood_eval_offset": (slices.get("ood_eval_offset"), 64),
        "slices.eval_limit": (slices.get("eval_limit"), 128),
        "slices.benign_eval_offset": (slices.get("benign_eval_offset"), 128),
        "slices.benign_eval_limit": (slices.get("benign_eval_limit"), 128),
    }
    mismatches = {
        name: actual
        for name, (actual, expected) in expected_protocol.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"Evaluation does not use the registered Table 4 protocol: {mismatches}")
    latency_expected = {
        "measurement": "unmerged_lora_sidecar_forward_latency",
        "sequence_length": 256,
        "batch_size": 1,
        "timed_iterations_per_arm": 40,
        "warmup_iterations": 5,
        "dtype": "bfloat16",
        "seed": 0,
    }
    latency_mismatches = {
        key: latency.get(key)
        for key, expected in latency_expected.items()
        if latency.get(key) != expected
    }
    if latency_mismatches:
        raise ValueError(f"Latency result uses another protocol: {latency_mismatches}")
    manifest_metrics = evaluation_manifest.get("metrics") or {}
    for key in ("advbench_asr", "benign_refusal_rate", "ppl"):
        if abs(float(manifest_metrics[key]) - float(evaluation[key])) > 1e-8:
            raise ValueError(f"Evaluation CSV and manifest disagree on {key}")

    runtime = artifact.get("runtime") or {}
    parameters = artifact.get("parameters") or {}
    training_data = artifact.get("training_data") or {}
    train_stats = training_data.get("train_statistics") or {}
    validation_stats = training_data.get("validation_statistics") or {}
    fit_seconds = float(runtime.get("accumulated_fit_seconds", runtime["total_seconds"]))
    wall_hours = fit_seconds / 3600.0
    gpu_hours = float(runtime.get("accumulated_gpu_hours", runtime["gpu_hours"]))
    overhead = float(latency["overhead_percent"])
    trainable = int(parameters["trainable"])
    record: dict[str, Any] = {
        "table_id": "tab:cost",
        "row_id": 3,
        "model": "Qwen2.5-3B-Instruct",
        "condition": "wanda_50",
        "method": "Prune-to-Safety-SFT",
        "asr_pct": 100.0 * float(evaluation["advbench_asr"]),
        "benign_refusal_pct": 100.0 * float(evaluation["benign_refusal_rate"]),
        "ppl": float(evaluation["ppl"]),
        "wall_hours": wall_hours,
        "gpu_hours": gpu_hours,
        "backpropagation": "yes",
        "added_trainable_parameters": trainable,
        "added_trainable_parameters_millions": trainable / 1_000_000.0,
        "inference_overhead_percent": overhead,
        "inference_overhead_measurement": latency["measurement"],
        "training_examples": int(train_stats["output_rows"]),
        "training_safety_examples": int(train_stats["safety_rows"]),
        "validation_examples": int(validation_stats["output_rows"]),
        "epochs": float((artifact.get("hyperparameters") or {})["epochs"]),
        "training_file_sha256": training_data["training_file_sha256"],
        "artifact_manifest_sha256": artifact_sha256,
        "evaluation_manifest_sha256": file_sha256(args.evaluation_manifest),
        "evaluation_summary_sha256": file_sha256(args.evaluation_summary),
        "latency_manifest_sha256": file_sha256(args.latency_manifest),
        "config_sha256": file_sha256(args.config),
        "ppl_window_index_sha256": file_sha256(args.window_index_file),
        "generated_at_utc": utc_now(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(outputs[0], record)
    _write_csv(outputs[1], record)
    sign = "+" if overhead >= 0 else ""
    latex = (
        "Prune$\\to$SFT "
        f"& {record['asr_pct']:.1f} & {record['benign_refusal_pct']:.1f} "
        f"& {record['ppl']:.2f} "
        f"& {wall_hours:.2f}\\,h / {gpu_hours:.2f} "
        f"& yes / {trainable / 1_000_000.0:.1f}M / {sign}{overhead:.1f}\\%\\\\\n"
    )
    outputs[2].write_text(latex, encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
