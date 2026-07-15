"""PKU-SafeRLHF pair selection with content-hash evaluation exclusion."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import content_sha256, digest_strings


@dataclass(frozen=True)
class SafePair:
    prompt: str
    safe_response: str
    unsafe_response: str
    prompt_sha256: str
    source_index: int


@dataclass(frozen=True)
class PairSelection:
    pairs: tuple[SafePair, ...]
    statistics: dict[str, Any]


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return None


def extract_safe_pair(row: Mapping[str, Any], source_index: int) -> SafePair | None:
    prompt = row.get("prompt")
    response_0 = row.get("response_0")
    response_1 = row.get("response_1")
    safe_0 = _strict_bool(row.get("is_response_0_safe"))
    safe_1 = _strict_bool(row.get("is_response_1_safe"))
    if not all(isinstance(value, str) and value.strip() for value in (prompt, response_0, response_1)):
        return None
    if safe_0 is None or safe_1 is None or safe_0 == safe_1:
        return None
    prompt_text = str(prompt).strip()
    return SafePair(
        prompt=prompt_text,
        safe_response=str(response_0 if safe_0 else response_1).strip(),
        unsafe_response=str(response_1 if safe_0 else response_0).strip(),
        prompt_sha256=content_sha256(prompt_text),
        source_index=int(source_index),
    )


def select_safe_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    excluded_prompt_hashes: set[str],
    limit: int = 5000,
    seed: int = 0,
) -> PairSelection:
    if limit <= 0:
        raise ValueError("limit must be positive")
    counts = {
        "source_rows": 0,
        "exactly_one_safe_rows": 0,
        "invalid_or_same_safety_rows": 0,
        "evaluation_overlap_rows": 0,
        "duplicate_prompt_rows": 0,
    }
    candidates: list[SafePair] = []
    seen: set[str] = set()
    for source_index, row in enumerate(rows):
        counts["source_rows"] += 1
        pair = extract_safe_pair(row, source_index)
        if pair is None:
            counts["invalid_or_same_safety_rows"] += 1
            continue
        counts["exactly_one_safe_rows"] += 1
        if pair.prompt_sha256 in excluded_prompt_hashes:
            counts["evaluation_overlap_rows"] += 1
            continue
        if pair.prompt_sha256 in seen:
            counts["duplicate_prompt_rows"] += 1
            continue
        seen.add(pair.prompt_sha256)
        candidates.append(pair)
    generator = random.Random(seed)
    generator.shuffle(candidates)
    if len(candidates) < limit:
        raise ValueError(
            f"Only {len(candidates)} disjoint one-safe/one-unsafe pairs available; need {limit}"
        )
    selected = tuple(candidates[:limit])
    selected_hashes = {pair.prompt_sha256 for pair in selected}
    if selected_hashes.intersection(excluded_prompt_hashes):
        raise AssertionError("Selected PKU prompts overlap evaluation prompts")
    statistics = {
        **counts,
        "eligible_unique_rows": len(candidates),
        "selected_rows": len(selected),
        "seed": int(seed),
        "selected_prompt_set_sha256": digest_strings(selected_hashes),
        "excluded_prompt_count": len(excluded_prompt_hashes),
        "excluded_prompt_set_sha256": digest_strings(excluded_prompt_hashes),
        "selected_eval_intersection_count": 0,
    }
    return PairSelection(pairs=selected, statistics=statistics)


def prompt_hashes(rows: Sequence[tuple[int, str]]) -> set[str]:
    return {content_sha256(prompt) for _prompt_id, prompt in rows}


def split_identity(rows: Sequence[tuple[int, str]], *, label: str) -> dict[str, Any]:
    hashes = prompt_hashes(rows)
    return {
        "label": label,
        "count": len(rows),
        "unique_content_count": len(hashes),
        "content_set_sha256": digest_strings(hashes),
    }


def load_pku_rows(
    *,
    dataset_id: str,
    config: str | None,
    split: str,
    allow_data_download: bool,
):
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets before loading PKU-SafeRLHF") from exc
    download_config = DownloadConfig(local_files_only=not allow_data_download)
    kwargs = {
        "split": split,
        "download_config": download_config,
    }
    if config:
        return load_dataset(dataset_id, config, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def load_registered_evaluation_splits(args) -> dict[str, list[tuple[int, str]]]:
    """Load the exact evaluation slices shared by training exclusion and evaluation."""

    from casafety.ood_residual_diag import load_benign_slice, load_slice

    data_local_only = not bool(args.allow_data_download)
    args.local_files_only = data_local_only
    return {
        "advbench": load_slice(
            args, "advbench", offset=args.adv_eval_offset, limit=args.eval_limit
        ),
        "harmbench": load_slice(
            args, "harmbench", offset=args.ood_eval_offset, limit=args.eval_limit
        ),
        "strongreject": load_slice(
            args, "strongreject", offset=args.ood_eval_offset, limit=args.eval_limit
        ),
        "benign": load_benign_slice(
            args, offset=args.benign_eval_offset, limit=args.benign_eval_limit
        ),
    }


def load_evaluation_exclusions(args) -> tuple[set[str], dict[str, Any]]:
    """Load only registered evaluation slices and return aggregate identities."""

    splits = load_registered_evaluation_splits(args)
    identities = {
        name: split_identity(rows, label=name) for name, rows in splits.items()
    }
    union = set().union(*(prompt_hashes(rows) for rows in splits.values()))
    return union, {
        "splits": identities,
        "union_unique_content_count": len(union),
        "union_content_set_sha256": digest_strings(union),
    }


def sft_records(pairs: Sequence[SafePair]) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": pair.prompt},
                {"role": "assistant", "content": pair.safe_response},
            ]
        }
        for pair in pairs
    ]


def dpo_records(pairs: Sequence[SafePair]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [{"role": "user", "content": pair.prompt}],
            "chosen": [{"role": "assistant", "content": pair.safe_response}],
            "rejected": [{"role": "assistant", "content": pair.unsafe_response}],
        }
        for pair in pairs
    ]
