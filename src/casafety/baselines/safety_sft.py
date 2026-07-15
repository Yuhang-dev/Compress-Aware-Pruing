"""Post-Wanda LoRA safety recovery with the ICLR'24 Saferpaca-500 recipe."""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .arguments import add_data_download_argument, add_registered_split_arguments
from .common import (
    MODEL_ID,
    atomic_write_json,
    content_sha256,
    digest_strings,
    directory_hashes,
    file_sha256,
    repo_root,
    source_identity,
    utc_now,
    verify_wanda_checkpoint,
)
from .data import load_evaluation_exclusions


SAFETY_TUNED_COMMIT = "36a4b8d5c2177ed165bf61f59b590161394f7f12"
MIX_SHA256 = "7d94c1e9fdc6123d59bfd978aa062ad4e5e83c086ac9f4d410c9d1adb030436d"
SAFETY_REFERENCE_SHA256 = "7dde08acf87f10cf74fd1b76f56538eae8e413ae8e6b09e3f7bcd97769058dbe"
TARGET_MODULES = ("q_proj", "v_proj")
MINIMUM_VERSIONS = {
    "torch": (2, 1),
    "transformers": (4, 40),
    "peft": (0, 10),
    "accelerate": (0, 27),
}


@dataclass(frozen=True)
class SafetyTuningRecord:
    prompt: str
    response: str
    prompt_sha256: str
    example_sha256: str
    is_safety: bool
    source_index: int


class TokenizedDataset:
    def __init__(self, rows: Sequence[dict[str, list[int]]]) -> None:
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]):
        import torch

        maximum = max(len(item["input_ids"]) for item in features)
        width = int(math.ceil(maximum / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        values = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = width - len(item["input_ids"])
            values["input_ids"].append(
                [*item["input_ids"], *([self.pad_token_id] * padding)]
            )
            values["attention_mask"].append(
                [*item["attention_mask"], *([0] * padding)]
            )
            values["labels"].append([*item["labels"], *([-100] * padding)])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in values.items()}


class FitCostLedger:
    """Persist active attempt time so checkpoint resumes do not under-report cost."""

    def __init__(self, path: Path, *, run_fingerprint: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise ValueError(f"Invalid fit-cost ledger: {path}")
            if value.get("run_fingerprint") != run_fingerprint:
                raise ValueError(
                    "Fit-cost ledger belongs to another data/checkpoint/protocol run"
                )
            self.payload = value
        else:
            self.payload = {
                "schema_version": 1,
                "method": "sft",
                "run_fingerprint": run_fingerprint,
                "attempts": [],
            }
        now = time.time()
        for attempt in self.payload["attempts"]:
            if attempt.get("status") == "running":
                last = float(attempt.get("last_heartbeat_epoch", attempt["started_epoch"]))
                attempt["active_seconds"] = max(
                    float(attempt.get("active_seconds", 0.0)),
                    last - float(attempt["started_epoch"]),
                )
                attempt["status"] = "interrupted_lower_bound"
                attempt["reconciled_at_epoch"] = now
        self.attempt_id = uuid.uuid4().hex
        self.started_epoch = now
        self.finished = False
        self.payload["attempts"].append(
            {
                "attempt_id": self.attempt_id,
                "status": "running",
                "started_epoch": now,
                "last_heartbeat_epoch": now,
                "active_seconds": 0.0,
            }
        )
        self._write()
        self._exit_callback = lambda: self.finish("process_exit")
        atexit.register(self._exit_callback)

    def _current(self) -> dict[str, Any]:
        for attempt in self.payload["attempts"]:
            if attempt.get("attempt_id") == self.attempt_id:
                return attempt
        raise AssertionError("Current cost-ledger attempt is missing")

    def _write(self) -> None:
        atomic_write_json(self.path, self.payload)

    def heartbeat(self) -> None:
        if self.finished:
            return
        now = time.time()
        attempt = self._current()
        attempt["last_heartbeat_epoch"] = now
        attempt["active_seconds"] = max(0.0, now - self.started_epoch)
        self._write()

    def finish(self, status: str) -> None:
        if self.finished:
            return
        self.heartbeat()
        attempt = self._current()
        attempt["status"] = status
        attempt["completed_epoch"] = time.time()
        self.finished = True
        self._write()

    def accumulated_seconds(self) -> float:
        return sum(float(item.get("active_seconds", 0.0)) for item in self.payload["attempts"])


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = [int(item) for item in re.findall(r"\d+", value)[:3]]
    return tuple(numbers or [0])


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, minimum in MINIMUM_VERSIONS.items():
        try:
            value = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing training dependency: {package}") from exc
        if _version_tuple(value) < minimum:
            required = ".".join(str(item) for item in minimum)
            raise RuntimeError(f"{package}>={required} is required; found {value}")
        versions[package] = value
    return versions


def prompt_from_instruction(row: Mapping[str, Any]) -> str | None:
    """Use the same Alpaca prompt normalization as the registered benign split."""

    instruction = str(row.get("instruction") or "").strip()
    if not instruction:
        return None
    extra = str(row.get("input") or "").strip()
    return f"{instruction}\n\n{extra}" if extra and extra != instruction else instruction


def example_sha256(row: Mapping[str, Any]) -> str:
    prompt = prompt_from_instruction(row) or ""
    response = str(row.get("output") or "").strip()
    return hashlib.sha256(f"{prompt}\0{response}".encode("utf-8")).hexdigest()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON array of objects: {path}")
    return value


def prepare_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    safety_example_hashes: set[str],
    excluded_prompt_hashes: set[str],
) -> tuple[list[SafetyTuningRecord], dict[str, Any]]:
    counts = {
        "source_rows": 0,
        "invalid_rows": 0,
        "evaluation_overlap_rows": 0,
        "duplicate_prompt_rows": 0,
        "general_rows": 0,
        "safety_rows": 0,
    }
    records: list[SafetyTuningRecord] = []
    seen_prompts: set[str] = set()
    for source_index, row in enumerate(rows):
        counts["source_rows"] += 1
        prompt = prompt_from_instruction(row)
        response = str(row.get("output") or "").strip()
        if not prompt or not response:
            counts["invalid_rows"] += 1
            continue
        prompt_hash = content_sha256(prompt)
        if prompt_hash in excluded_prompt_hashes:
            counts["evaluation_overlap_rows"] += 1
            continue
        if prompt_hash in seen_prompts:
            counts["duplicate_prompt_rows"] += 1
            continue
        seen_prompts.add(prompt_hash)
        digest = example_sha256(row)
        is_safety = digest in safety_example_hashes
        counts["safety_rows" if is_safety else "general_rows"] += 1
        records.append(
            SafetyTuningRecord(
                prompt=prompt,
                response=response,
                prompt_sha256=prompt_hash,
                example_sha256=digest,
                is_safety=is_safety,
                source_index=int(source_index),
            )
        )
    if set(record.prompt_sha256 for record in records).intersection(excluded_prompt_hashes):
        raise AssertionError("Safety-SFT data overlaps a registered evaluation prompt")
    return records, {
        **counts,
        "eligible_unique_rows": len(records),
        "excluded_prompt_count": len(excluded_prompt_hashes),
        "excluded_prompt_set_sha256": digest_strings(excluded_prompt_hashes),
        "selected_eval_intersection_count": 0,
    }


def stratified_train_validation_split(
    records: Sequence[SafetyTuningRecord],
    *,
    validation_size: int,
    seed: int,
) -> tuple[list[SafetyTuningRecord], list[SafetyTuningRecord]]:
    if validation_size < 0 or validation_size >= len(records):
        raise ValueError("validation_size must be non-negative and smaller than the data")
    safety = [record for record in records if record.is_safety]
    general = [record for record in records if not record.is_safety]
    generator = random.Random(seed)
    generator.shuffle(safety)
    generator.shuffle(general)
    safety_validation = round(validation_size * len(safety) / len(records))
    safety_validation = min(safety_validation, len(safety))
    general_validation = validation_size - safety_validation
    if general_validation > len(general):
        raise ValueError("Not enough general rows for the validation split")
    validation = [*safety[:safety_validation], *general[:general_validation]]
    training = [*safety[safety_validation:], *general[general_validation:]]
    generator.shuffle(training)
    generator.shuffle(validation)
    return training, validation


def tokenize_completion_only(
    tokenizer,
    records: Sequence[SafetyTuningRecord],
    *,
    max_length: int,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    tokenized: list[dict[str, list[int]]] = []
    selected_hashes: set[str] = set()
    counts = {
        "input_rows": len(records),
        "output_rows": 0,
        "safety_rows": 0,
        "zero_supervised_token_rows": 0,
        "truncated_rows": 0,
        "prefix_mismatch_rows": 0,
        "total_tokens": 0,
        "supervised_tokens": 0,
    }
    for record in records:
        user_messages = [{"role": "user", "content": record.prompt}]
        prompt_ids = list(
            tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        full_ids = list(
            tokenizer.apply_chat_template(
                [*user_messages, {"role": "assistant", "content": record.response}],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
        )
        prefix = min(len(prompt_ids), len(full_ids))
        if full_ids[:prefix] != prompt_ids[:prefix]:
            counts["prefix_mismatch_rows"] += 1
            prefix = 0
            for left, right in zip(prompt_ids, full_ids):
                if left != right:
                    break
                prefix += 1
        if len(full_ids) > max_length:
            counts["truncated_rows"] += 1
        input_ids = full_ids[:max_length]
        labels = list(input_ids)
        labels[: min(prefix, len(labels))] = [-100] * min(prefix, len(labels))
        supervised = sum(token != -100 for token in labels)
        if supervised == 0:
            counts["zero_supervised_token_rows"] += 1
            continue
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
        selected_hashes.add(record.prompt_sha256)
        counts["output_rows"] += 1
        counts["safety_rows"] += int(record.is_safety)
        counts["total_tokens"] += len(input_ids)
        counts["supervised_tokens"] += supervised
    counts["selected_prompt_set_sha256"] = digest_strings(selected_hashes)
    counts["truncation_rate"] = counts["truncated_rows"] / max(1, len(records))
    counts["assistant_only_loss"] = True
    return tokenized, counts


def _training_arguments(training_arguments, values: Mapping[str, Any]):
    parameters = inspect.signature(training_arguments).parameters
    accepted = {key: value for key, value in values.items() if key in parameters}
    if "eval_strategy" in parameters:
        accepted["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        accepted["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError("Installed Transformers lacks an evaluation strategy argument")
    return training_arguments(**accepted)


def _numeric_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def recover_pending_finalization(args: argparse.Namespace) -> bool:
    """Complete an adapter/manifest commit interrupted between atomic renames."""

    adapter_dir = args.output_dir / "adapter"
    adapter_staging = args.output_dir / "adapter.pending"
    manifest_pending = args.artifact_manifest.with_name(
        f"{args.artifact_manifest.name}.pending"
    )
    if args.artifact_manifest.exists():
        raise FileExistsError("Refusing to overwrite a completed SFT artifact")
    if manifest_pending.is_file():
        payload = json.loads(manifest_pending.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("method") != "sft":
            raise ValueError("Pending artifact manifest is invalid")
        candidate = adapter_dir if adapter_dir.is_dir() else adapter_staging
        if not candidate.is_dir():
            raise FileNotFoundError("Pending artifact manifest has no adapter directory")
        if directory_hashes(candidate) != payload.get("adapter_files_sha256"):
            raise ValueError("Pending adapter hashes do not match its manifest")
        if not adapter_dir.exists():
            os.replace(adapter_staging, adapter_dir)
        os.replace(manifest_pending, args.artifact_manifest)
        print(f"[safety-sft] recovered pending finalization: {args.artifact_manifest}")
        return True
    if adapter_dir.exists() or adapter_staging.exists():
        raise FileExistsError(
            "An adapter exists without a recoverable pending manifest; move it aside "
            "before resuming."
        )
    return False


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-id", default=MODEL_ID)
    result.add_argument("--wanda-checkpoint-dir", type=Path, required=True)
    result.add_argument("--wanda-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--artifact-manifest", type=Path, required=True)
    result.add_argument(
        "--training-file",
        type=Path,
        default=Path("data/saferpaca_Instructions_500.json"),
    )
    result.add_argument(
        "--safety-reference-file",
        type=Path,
        default=Path("data/safety_only_data_Instructions.json"),
    )
    result.add_argument("--validation-size", type=int, default=500)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", type=float, default=4.0)
    result.add_argument("--max-length", type=int, default=512)
    result.add_argument("--micro-batch-size", type=int, default=4)
    result.add_argument("--gradient-accumulation-steps", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--warmup-steps", type=int, default=10)
    result.add_argument("--logging-steps", type=int, default=10)
    result.add_argument("--eval-steps", type=int, default=50)
    result.add_argument("--save-steps", type=int, default=50)
    result.add_argument("--lora-r", type=int, default=4)
    result.add_argument("--lora-alpha", type=int, default=16)
    result.add_argument("--lora-dropout", type=float, default=0.05)
    result.add_argument("--dataloader-workers", type=int, default=2)
    result.add_argument("--no-resume", action="store_true")
    # This flag applies only to registered evaluation-exclusion datasets.  The
    # two Safety-Tuned training files remain explicit and hash-pinned.
    add_data_download_argument(result)
    add_registered_split_arguments(result)
    return result


def run(args: argparse.Namespace) -> None:
    if args.model_id != MODEL_ID:
        raise ValueError(f"This baseline is registered only for {MODEL_ID}")
    if args.epochs != 4.0 or args.max_length != 512 or args.validation_size != 500:
        raise ValueError("Registered Saferpaca protocol is 4 epochs, length 512, val 500")
    effective_batch = args.micro_batch_size * args.gradient_accumulation_steps
    if effective_batch != 128:
        raise ValueError(f"Registered effective batch is 128, got {effective_batch}")
    registered = {
        "learning_rate": (args.learning_rate, 1e-4),
        "warmup_steps": (args.warmup_steps, 10),
        "eval_steps": (args.eval_steps, 50),
        "save_steps": (args.save_steps, 50),
        "lora_r": (args.lora_r, 4),
        "lora_alpha": (args.lora_alpha, 16),
        "lora_dropout": (args.lora_dropout, 0.05),
    }
    changed = {
        name: actual
        for name, (actual, expected) in registered.items()
        if actual != expected
    }
    if changed:
        raise ValueError(f"Non-registered Saferpaca hyperparameters: {changed}")
    if recover_pending_finalization(args):
        return
    for path, expected in (
        (args.training_file, MIX_SHA256),
        (args.safety_reference_file, SAFETY_REFERENCE_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned Safety-Tuned data file: {path}")
        if file_sha256(path).lower() != expected:
            raise ValueError(f"Pinned Safety-Tuned data hash mismatch: {path}")

    versions = dependency_versions()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    total_start = time.perf_counter()
    started_at = utc_now()
    wanda = verify_wanda_checkpoint(
        args.wanda_checkpoint_dir,
        args.wanda_manifest,
        expected_model=args.model_id,
        verify_all_files=True,
    )
    excluded_hashes, evaluation_identity = load_evaluation_exclusions(args)

    data_start = time.perf_counter()
    raw_rows = load_json_records(args.training_file)
    safety_reference = load_json_records(args.safety_reference_file)
    if len(raw_rows) != 20_500:
        raise ValueError(f"Expected 20,500 Saferpaca-500 rows, found {len(raw_rows)}")
    safety_hashes = {example_sha256(row) for row in safety_reference}
    raw_safety_rows = sum(example_sha256(row) in safety_hashes for row in raw_rows)
    if raw_safety_rows != 500:
        raise ValueError(
            f"Saferpaca-500 must contain 500 safety rows, found {raw_safety_rows}"
        )
    records, selection_statistics = prepare_records(
        raw_rows,
        safety_example_hashes=safety_hashes,
        excluded_prompt_hashes=excluded_hashes,
    )
    training_records, validation_records = stratified_train_validation_split(
        records,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.wanda_checkpoint_dir.resolve()),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    training_rows, training_statistics = tokenize_completion_only(
        tokenizer, training_records, max_length=args.max_length
    )
    validation_rows, validation_statistics = tokenize_completion_only(
        tokenizer, validation_records, max_length=args.max_length
    )
    if training_statistics["safety_rows"] < 400:
        raise ValueError("Fewer than 400 safety demonstrations remain in the train split")
    training_dataset = TokenizedDataset(training_rows)
    validation_dataset = TokenizedDataset(validation_rows)
    del raw_rows, safety_reference, records, training_records, validation_records
    del training_rows, validation_rows
    data_seconds = time.perf_counter() - data_start

    run_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "protocol": "safety_tuned_llamas_saferpaca500_post_wanda_lora",
                "model": args.model_id,
                "wanda_manifest_sha256": wanda["manifest_sha256"],
                "training_file_sha256": MIX_SHA256,
                "safety_reference_file_sha256": SAFETY_REFERENCE_SHA256,
                "training_source_sha256": file_sha256(Path(__file__)),
                "evaluation_union_sha256": evaluation_identity[
                    "union_content_set_sha256"
                ],
                "hyperparameters": {
                    "epochs": args.epochs,
                    "max_length": args.max_length,
                    "validation_size": args.validation_size,
                    "effective_batch": effective_batch,
                    "learning_rate": args.learning_rate,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "seed": args.seed,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cost_ledger = FitCostLedger(
        args.output_dir / "fit_cost_ledger.json",
        run_fingerprint=run_fingerprint,
    )
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.wanda_checkpoint_dir.resolve()),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(TARGET_MODULES),
        ),
    )
    trainable_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(int(parameter.numel()) for parameter in model.parameters())

    trainer_dir = args.output_dir / "trainer"
    values = {
        "output_dir": str(trainer_dir),
        "num_train_epochs": float(args.epochs),
        "per_device_train_batch_size": int(args.micro_batch_size),
        "per_device_eval_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "lr_scheduler_type": "linear",
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "logging_steps": int(args.logging_steps),
        "logging_strategy": "steps",
        "eval_steps": int(args.eval_steps),
        "save_strategy": "steps",
        "save_steps": int(args.save_steps),
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "seed": int(args.seed),
        "data_seed": int(args.seed),
        "dataloader_num_workers": int(args.dataloader_workers),
    }
    training_args = _training_arguments(TrainingArguments, values)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        callbacks=[
            type(
                "FitCostHeartbeatCallback",
                (TrainerCallback,),
                {
                    "on_log": lambda self, *unused_args, **unused_kwargs: cost_ledger.heartbeat(),
                    "on_save": lambda self, *unused_args, **unused_kwargs: cost_ledger.heartbeat(),
                },
            )()
        ],
    )
    resume_checkpoint = None
    if not args.no_resume and trainer_dir.is_dir():
        resume_checkpoint = get_last_checkpoint(str(trainer_dir))
    train_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    training_seconds = time.perf_counter() - train_start

    adapter_dir = args.output_dir / "adapter"
    adapter_staging = args.output_dir / "adapter.pending"
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_staging, safe_serialization=True)
    adapter_hashes = directory_hashes(adapter_staging)
    adapter_bytes = sum(
        path.stat().st_size for path in adapter_staging.rglob("*") if path.is_file()
    )
    peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated())
    trainer_metrics = _numeric_metrics(getattr(train_result, "metrics", {}))
    optimizer_steps = int(getattr(trainer.state, "global_step", 0))

    del trainer, model, tokenizer, training_dataset, validation_dataset
    gc.collect()
    torch.cuda.empty_cache()
    wanda_after = verify_wanda_checkpoint(
        args.wanda_checkpoint_dir,
        args.wanda_manifest,
        expected_model=args.model_id,
        verify_all_files=True,
    )
    if wanda_after["checkpoint_files_sha256"] != wanda["checkpoint_files_sha256"]:
        raise RuntimeError("Source Wanda checkpoint changed during Safety-SFT")

    total_seconds = time.perf_counter() - total_start
    cost_ledger.finish("completed")
    accumulated_fit_seconds = cost_ledger.accumulated_seconds()
    payload = {
        "schema_version": 2,
        "status": "completed",
        "method": "sft",
        "protocol": "safety_tuned_llamas_saferpaca500_post_wanda_lora",
        "model": args.model_id,
        "artifact_type": "peft_lora_adapter",
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_files_sha256": adapter_hashes,
        "adapter_size_bytes": adapter_bytes,
        "base_checkpoint": wanda,
        "sparsity_semantics": {
            "base_weights": "unchanged_wanda_50",
            "adapter": "separate_dense_low_rank_sidecar",
            "merged_into_base": False,
        },
        "training_data": {
            "name": "Saferpaca-500",
            "source_repository": "vinid/safety-tuned-llamas",
            "source_commit": SAFETY_TUNED_COMMIT,
            "license": "CC BY-NC 4.0",
            "training_file": str(args.training_file.resolve()),
            "training_file_sha256": MIX_SHA256,
            "safety_reference_file_sha256": SAFETY_REFERENCE_SHA256,
            "published_composition": {"general": 20_000, "safety": 500},
            "selection_statistics": selection_statistics,
            "train_statistics": training_statistics,
            "validation_statistics": validation_statistics,
            "evaluation_exclusion": evaluation_identity,
            "selected_eval_intersection_count": 0,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "max_length": args.max_length,
            "validation_size": args.validation_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "optimizer": "adamw_torch",
            "lr_scheduler": "linear",
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing": True,
            "assistant_only_loss": True,
            "qwen_chat_template": True,
            "adaptation_note": (
                "Paper protocol adapted from base LLaMA to Qwen2.5-Instruct; "
                "Qwen chat formatting and assistant-only loss are used."
            ),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": list(TARGET_MODULES),
            "seed": args.seed,
        },
        "parameters": {
            "trainable": trainable_parameters,
            "total_with_adapter": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
        },
        "runtime": {
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "data_preparation_seconds": data_seconds,
            "training_seconds": training_seconds,
            "total_seconds": total_seconds,
            "gpu_count": 1,
            "gpu_hours": total_seconds / 3600.0,
            "accumulated_fit_seconds": accumulated_fit_seconds,
            "accumulated_gpu_hours": accumulated_fit_seconds / 3600.0,
            "fit_cost_ledger": str(cost_ledger.path.resolve()),
            "fit_cost_accounting": (
                "Sum of active process attempts. A hard-killed attempt is a lower "
                "bound through its last Trainer log/save heartbeat."
            ),
            "gpu_model": torch.cuda.get_device_name(0),
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "optimizer_steps": optimizer_steps,
            "resumed_from_checkpoint": resume_checkpoint,
            "trainer_metrics": trainer_metrics,
        },
        "dependencies": versions,
        "source": source_identity(
            extra_paths=(
                Path(__file__),
                repo_root() / "scripts/phase4_safety_sft_qwen3b.sh",
            )
        ),
        "privacy": {
            "prompt_text_persisted": False,
            "response_text_persisted": False,
            "selected_dataset_persisted": False,
            "trainer_reporting_disabled": True,
        },
    }
    manifest_pending = args.artifact_manifest.with_name(
        f"{args.artifact_manifest.name}.pending"
    )
    atomic_write_json(manifest_pending, payload)
    os.replace(adapter_staging, adapter_dir)
    os.replace(manifest_pending, args.artifact_manifest)
    print(
        f"[safety-sft] complete train={training_statistics['output_rows']} "
        f"safety={training_statistics['safety_rows']} trainable={trainable_parameters} "
        f"wall_seconds={total_seconds:.1f}",
        flush=True,
    )


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
