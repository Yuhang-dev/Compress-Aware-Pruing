"""Verify and summarize a same-split dense/pruned/ReMaR OOD comparison.

The direct OOD ReMaR runs historically stored pruned and repaired outcomes but
not a dense arm. This utility joins a newly generated dense-only arm only after
checking that harmful and benign prompt identifiers exactly match every arm.
It never writes prompt or response text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .closed_form_readout_repair import write_text_free_csv


TEXT_COLUMNS = {"prompt", "response", "text", "instruction", "output", "completion"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    leaked = sorted(TEXT_COLUMNS.intersection({str(column).lower() for column in frame.columns}))
    if leaked:
        raise ValueError(f"Text-bearing columns are not allowed in {path}: {leaked}.")
    return frame


def select_arm(frame: pd.DataFrame, *, condition: str, repair_kind: str) -> pd.DataFrame:
    selected = frame[frame["condition"].eq(condition) & frame["repair_kind"].eq(repair_kind)].copy()
    if selected.empty:
        raise ValueError(f"Missing condition={condition!r}, repair_kind={repair_kind!r}.")
    if selected["prompt_id"].duplicated().any():
        raise ValueError(f"Duplicate prompt IDs for condition={condition!r}, repair_kind={repair_kind!r}.")
    return selected.sort_values("prompt_id").reset_index(drop=True)


def id_digest(frame: pd.DataFrame) -> str:
    payload = ",".join(str(int(value)) for value in frame["prompt_id"].tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_harmful(label: str, frame: pd.DataFrame, source_dir: Path) -> dict[str, Any]:
    return {
        "arm": label,
        "source_dir": str(source_dir),
        "condition": str(frame["condition"].iloc[0]),
        "repair": str(frame["repair"].iloc[0]),
        "repair_kind": str(frame["repair_kind"].iloc[0]),
        "prompts": int(len(frame)),
        "asr": float(frame["attack_success"].mean()),
        "raw_unsafe_rate": float(frame["unsafe"].mean()),
        "coherent_rate": float(frame["coherent"].mean()),
        "refusal_rate": float(frame["refused"].mean()),
        "negative_margin_rate": float(frame["m_neg_s_mean"].mean()),
        "harmful_prompt_id_sha256": id_digest(frame),
    }


def attach_benign(summary: dict[str, Any], frame: pd.DataFrame) -> None:
    summary.update(
        {
            "benign_prompts": int(len(frame)),
            "benign_refusal_rate": float(frame["refused"].mean()),
            "benign_coherent_rate": float(frame["coherent"].mean()),
            "benign_prompt_id_sha256": id_digest(frame),
        }
    )


def assert_same_ids(reference: pd.DataFrame, candidate: pd.DataFrame, *, label: str) -> None:
    expected = reference["prompt_id"].tolist()
    actual = candidate["prompt_id"].tolist()
    if expected != actual:
        raise ValueError(f"{label} prompt IDs do not exactly match the direct pruned baseline.")


def run(args: argparse.Namespace) -> None:
    direct_details = read_csv(args.direct_dir / "repair_details.csv")
    direct_benign = read_csv(args.direct_dir / "repair_benign_details.csv")
    dense_details = read_csv(args.dense_dir / "repair_details.csv")
    dense_benign = read_csv(args.dense_dir / "repair_benign_details.csv")

    pruned_harm = select_arm(direct_details, condition="wanda_50", repair_kind="pruned")
    repair_harm = select_arm(direct_details, condition="wanda_50", repair_kind="readout_repair")
    dense_harm = select_arm(dense_details, condition="dense", repair_kind="pruned")
    pruned_benign = select_arm(direct_benign, condition="wanda_50", repair_kind="pruned")
    repair_benign = select_arm(direct_benign, condition="wanda_50", repair_kind="readout_repair")
    dense_benign = select_arm(dense_benign, condition="dense", repair_kind="pruned")

    for label, candidate in (
        ("dense harmful", dense_harm),
        ("ReMaR harmful", repair_harm),
    ):
        assert_same_ids(pruned_harm, candidate, label=label)
    for label, candidate in (
        ("dense benign", dense_benign),
        ("ReMaR benign", repair_benign),
    ):
        assert_same_ids(pruned_benign, candidate, label=label)

    rows = []
    for label, harm, benign, source_dir in (
        ("dense", dense_harm, dense_benign, args.dense_dir),
        ("pruned", pruned_harm, pruned_benign, args.direct_dir),
        ("remar", repair_harm, repair_benign, args.direct_dir),
    ):
        row = {"eval_dataset": args.eval_dataset, **summarize_harmful(label, harm, source_dir)}
        attach_benign(row, benign)
        rows.append(row)

    summary = pd.DataFrame(rows)
    alignment = pd.DataFrame(
        [
            {
                "eval_dataset": args.eval_dataset,
                "arm": row["arm"],
                "harmful_prompt_count": row["prompts"],
                "harmful_prompt_id_sha256": row["harmful_prompt_id_sha256"],
                "benign_prompt_count": row["benign_prompts"],
                "benign_prompt_id_sha256": row["benign_prompt_id_sha256"],
                "same_split_verified": True,
            }
            for row in rows
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_text_free_csv(summary, args.output_dir / "strict_ood_remar_summary.csv")
    write_text_free_csv(alignment, args.output_dir / "strict_ood_remar_prompt_alignment.csv")

    by_arm = summary.set_index("arm")
    decision = {
        "eval_dataset": args.eval_dataset,
        "same_split_verified": True,
        "harmful_prompt_count": int(by_arm.loc["dense", "prompts"]),
        "benign_prompt_count": int(by_arm.loc["dense", "benign_prompts"]),
        "dense_asr": float(by_arm.loc["dense", "asr"]),
        "pruned_asr": float(by_arm.loc["pruned", "asr"]),
        "remar_asr": float(by_arm.loc["remar", "asr"]),
        "pruning_asr_increase": float(by_arm.loc["pruned", "asr"] - by_arm.loc["dense", "asr"]),
        "remar_asr_reduction": float(by_arm.loc["pruned", "asr"] - by_arm.loc["remar", "asr"]),
        "dense_benign_refusal": float(by_arm.loc["dense", "benign_refusal_rate"]),
        "pruned_benign_refusal": float(by_arm.loc["pruned", "benign_refusal_rate"]),
        "remar_benign_refusal": float(by_arm.loc["remar", "benign_refusal_rate"]),
        "interpretation": "All three arms were evaluated on exactly matching harmful and benign prompt IDs.",
    }
    (args.output_dir / "strict_ood_remar_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"[same-split-ood] wrote {args.output_dir / 'strict_ood_remar_summary.csv'}")
    print(f"[same-split-ood] wrote {args.output_dir / 'strict_ood_remar_prompt_alignment.csv'}")
    print(f"[same-split-ood] wrote {args.output_dir / 'strict_ood_remar_decision.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dataset", required=True, choices=["harmbench", "strongreject"])
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
