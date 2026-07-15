"""Shared provenance, checkpoint, timing, and resume helpers for baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
METHODS = ("sft", "dpo", "dsnot", "wandaplusplus", "optima")
TRAINING_METHODS = ("sft", "dpo")
VENDOR_METHODS = ("dsnot", "wandaplusplus", "optima")
TARGET_LINEAR_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_content(value: str) -> str:
    """Canonicalize prompt content before split-overlap hashing."""

    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = " ".join(normalized.split())
    return normalized.strip().casefold()


def content_sha256(value: str) -> str:
    return hashlib.sha256(normalize_content(value).encode("utf-8")).hexdigest()


def digest_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    if not files:
        raise ValueError(f"Directory has no files: {path}")
    return {str(item.relative_to(path)): file_sha256(item) for item in files}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def require_local_model(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"{label} must be a local model directory: {resolved}")
    return resolved


def verify_hashes(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    if not expected:
        raise ValueError("Checkpoint manifest has no checkpoint_files_sha256")
    actual: dict[str, str] = {}
    for relative, expected_hash in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        actual[str(relative)] = actual_hash
    return actual


def verify_wanda_checkpoint(
    checkpoint_dir: Path,
    manifest_path: Path,
    *,
    expected_model: str = MODEL_ID,
    verify_all_files: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = require_local_model(checkpoint_dir, label="Wanda checkpoint")
    manifest = read_json(manifest_path)
    if str(manifest.get("model")) != expected_model:
        raise ValueError(
            f"Wanda model mismatch: {manifest.get('model')!r} != {expected_model!r}"
        )
    if str(manifest.get("pruner", "")).lower() != "wanda":
        raise ValueError(f"Expected Wanda checkpoint, got {manifest.get('pruner')!r}")
    sparsity = float(manifest.get("requested_sparsity", -1))
    if abs(sparsity - 0.5) > 1e-9:
        raise ValueError(f"Expected Wanda-50, got requested_sparsity={sparsity}")
    expected_hashes = manifest.get("checkpoint_files_sha256") or {}
    config_expected = expected_hashes.get("config.json")
    config_actual = file_sha256(checkpoint_dir / "config.json")
    if not config_expected or config_expected != config_actual:
        raise ValueError("Wanda config hash does not match its manifest")
    verified = verify_hashes(checkpoint_dir, expected_hashes) if verify_all_files else {
        "config.json": config_actual
    }
    realized = (manifest.get("sparsity") or {}).get("realized_zero_fraction")
    if realized is not None and abs(float(realized) - 0.5) > 0.01:
        raise ValueError(f"Wanda checkpoint realized sparsity is not near 50%: {realized}")
    return {
        "path": str(checkpoint_dir),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_files_sha256": verified,
        "requested_sparsity": sparsity,
        "realized_zero_fraction": None if realized is None else float(realized),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_head(path: Path | None = None) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path or repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    value = result.stdout.strip().lower()
    return value if result.returncode == 0 and HEX40.fullmatch(value) else None


def source_identity(extra_paths: Sequence[Path] = ()) -> dict[str, Any]:
    root = repo_root()
    paths = [
        *sorted((root / "src/casafety/baselines").glob("*.py")),
        *sorted((root / "scripts/baselines").glob("*")),
        *extra_paths,
    ]
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        try:
            label = str(resolved.relative_to(root))
        except ValueError:
            label = str(resolved)
        hashes[label] = file_sha256(resolved)
    return {"git_head": git_head(root), "files_sha256": hashes}


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    expected_outputs: tuple[Path, ...]


class RunState:
    """Atomic stage ledger used for conservative resume behavior."""

    def __init__(self, path: Path, *, method: str) -> None:
        self.path = path
        self.method = method
        if path.is_file():
            self.payload = read_json(path)
            if self.payload.get("method") != method:
                raise ValueError(f"Run-state method mismatch in {path}")
        else:
            self.payload = {
                "schema_version": 1,
                "method": method,
                "status": "pending",
                "created_at_utc": utc_now(),
                "stages": {},
            }

    def stage_complete(self, stage: Stage, *, infer_from_outputs: bool = False) -> bool:
        record = (self.payload.get("stages") or {}).get(stage.name, {})
        outputs_exist = bool(stage.expected_outputs) and all(
            path.is_file() for path in stage.expected_outputs
        )
        if record.get("status") == "completed":
            if not outputs_exist:
                raise FileNotFoundError(
                    f"State says {stage.name} completed but outputs are missing"
                )
            return True
        return bool(infer_from_outputs and outputs_exist)

    def mark_running(self, stage: Stage) -> None:
        self.payload["status"] = "running"
        self.payload.setdefault("stages", {})[stage.name] = {
            "status": "running",
            "started_at_utc": utc_now(),
            "command_sha256": hashlib.sha256(
                json.dumps(stage.command).encode("utf-8")
            ).hexdigest(),
        }
        atomic_write_json(self.path, self.payload)

    def mark_completed(
        self,
        stage: Stage,
        *,
        seconds: float,
        gpu_count: int,
        inferred: bool = False,
    ) -> None:
        missing = [str(path) for path in stage.expected_outputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Stage {stage.name} produced no outputs: {missing}")
        record = self.payload.setdefault("stages", {}).setdefault(stage.name, {})
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "wall_clock_seconds": float(seconds),
                "gpu_count": int(gpu_count),
                "gpu_hours": float(seconds * gpu_count / 3600.0),
                "inferred_from_outputs": bool(inferred),
                "outputs_sha256": {
                    str(path): file_sha256(path) for path in stage.expected_outputs
                },
            }
        )
        if all(
            item.get("status") == "completed"
            for item in self.payload.get("stages", {}).values()
        ):
            self.payload["status"] = "completed"
            self.payload["completed_at_utc"] = utc_now()
        atomic_write_json(self.path, self.payload)

    def mark_failed(self, stage: Stage, *, returncode: int, seconds: float) -> None:
        record = self.payload.setdefault("stages", {}).setdefault(stage.name, {})
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "returncode": int(returncode),
                "wall_clock_seconds": float(seconds),
            }
        )
        self.payload["status"] = "failed"
        atomic_write_json(self.path, self.payload)


def run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] command={json.dumps(list(command))}\n")
        log.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode), time.perf_counter() - started


def parse_gpu_map(
    methods: Sequence[str], default_gpus: str, entries: Sequence[str]
) -> dict[str, str]:
    mapping = {method: default_gpus for method in methods}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"GPU mapping must be method=devices: {entry!r}")
        method, devices = (part.strip() for part in entry.split("=", 1))
        if method not in methods:
            raise ValueError(f"GPU mapping references unselected method: {method}")
        if not devices or any(not item.strip().isdigit() for item in devices.split(",")):
            raise ValueError(f"Invalid GPU list for {method}: {devices!r}")
        mapping[method] = devices
    return mapping


def gpu_count(devices: str) -> int:
    return len({item.strip() for item in devices.split(",") if item.strip()})


def offline_environment(
    base: Mapping[str, str], *, devices: str, allow_data_download: bool
) -> dict[str, str]:
    result = dict(base)
    result["CUDA_VISIBLE_DEVICES"] = devices
    result["TOKENIZERS_PARALLELISM"] = "false"
    result["WANDB_DISABLED"] = "true"
    if not allow_data_download:
        result["HF_HUB_OFFLINE"] = "1"
        result["HF_DATASETS_OFFLINE"] = "1"
        result["TRANSFORMERS_OFFLINE"] = "1"
    return result
