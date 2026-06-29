from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = {"prompt", "response", "text", "instruction", "output", "completion"}


def text_free_assert(df: pd.DataFrame, path: Path) -> None:
    banned = set(df.columns).intersection(TEXT_COLUMNS)
    if banned:
        raise ValueError(f"Refusing to write text columns {sorted(banned)} to {path}")


def write_csv_text_free(df: pd.DataFrame, path: Path) -> None:
    text_free_assert(df, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[mechanism-seal] wrote {path}")


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def infer_sparsity(config: str) -> float:
    if config.startswith("wanda_"):
        suffix = config.rsplit("_", 1)[-1]
        try:
            return float(suffix) / 100.0
        except ValueError:
            pass
    return float("nan")


def mean_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def first_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if frame.empty:
        raise ValueError(f"Missing row for {description}")
    return frame.iloc[0]


def residual_sweep(args: argparse.Namespace) -> None:
    summary_frames = []
    detail_frames = []
    for input_dir in args.input_dirs:
        input_dir = Path(input_dir)
        summary_path = input_dir / "step0b_restore_s.csv"
        details_path = input_dir / "step0b_restore_s_details.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        if not details_path.exists():
            raise FileNotFoundError(details_path)
        summary_frames.append(pd.read_csv(summary_path))
        detail_frames.append(pd.read_csv(details_path))
    summary = pd.concat(summary_frames, ignore_index=True)
    details = pd.concat(detail_frames, ignore_index=True)

    setting = (
        summary["set"].eq("harm_eval")
        & summary["layer_group"].astype(str).eq(args.layer_group)
        & summary["window"].astype(str).eq(str(args.window))
        & summary["k_r"].astype(int).eq(args.k_r)
        & summary["steering_mode"].eq(args.steering_mode)
    )
    configs = sorted(summary[setting]["config"].unique().tolist(), key=infer_sparsity)
    rows = []
    for config in configs:
        sparsity = infer_sparsity(str(config))
        common_base = (
            summary["config"].eq(config)
            & summary["layer_group"].astype(str).eq(args.layer_group)
            & summary["window"].astype(str).eq(str(args.window))
            & summary["k_r"].astype(int).eq(args.k_r)
            & summary["steering_mode"].eq(args.steering_mode)
        )
        common_zero = (
            summary["config"].eq(config)
            & summary["layer_group"].astype(str).eq(args.layer_group)
            & summary["window"].astype(str).eq(str(args.window))
            & summary["k_r"].astype(int).eq(args.k_r)
        )
        common_steer = common_base
        if math.isfinite(args.beta_value):
            common_steer = common_steer & pd.to_numeric(summary["beta_value"], errors="coerce").sub(args.beta_value).abs().le(1e-9)
        harm_zero = first_row(
            summary[common_zero & summary["set"].eq("harm_eval") & summary["target_kind"].eq("zero")],
            f"{config} harm zero",
        )
        harm_steer = first_row(
            summary[common_steer & summary["set"].eq("harm_eval") & summary["target_kind"].eq(args.target_kind)],
            f"{config} harm {args.target_kind}",
        )
        benign_steer = first_row(
            summary[common_steer & summary["set"].eq("benign_eval") & summary["target_kind"].eq(args.target_kind)],
            f"{config} benign {args.target_kind}",
        )
        detail_common = (
            details["config"].eq(config)
            & details["set"].eq("harm_eval")
            & details["layer_group"].astype(str).eq(args.layer_group)
            & details["window"].astype(str).eq(str(args.window))
            & details["k_r"].astype(int).eq(args.k_r)
            & details["steering_mode"].eq(args.steering_mode)
            & details["target_kind"].eq(args.target_kind)
        )
        if math.isfinite(args.beta_value):
            detail_common = detail_common & pd.to_numeric(details["beta_value"], errors="coerce").sub(args.beta_value).abs().le(1e-9)
        steered_details = details[detail_common].copy()
        residual = steered_details[steered_details["attack_success"].astype(bool)]
        residual_ids = sorted(int(pid) for pid in residual["prompt_id"].tolist())
        row = {
            "model": args.model,
            "config": config,
            "sparsity": sparsity,
            "layer_group": args.layer_group,
            "window": str(args.window),
            "k_r": args.k_r,
            "steering_mode": args.steering_mode,
            "target_kind": args.target_kind,
            "beta_value": float(harm_steer["beta_value"]),
            "baseline_asr": float(harm_zero["asr"]),
            "steered_asr": float(harm_steer["asr"]),
            "asr_drop": float(harm_steer["asr_drop"]),
            "residual_count": int(len(residual)),
            "n_harm": int(harm_steer["n"]),
            "steered_coherent_rate": float(harm_steer["coherent_rate"]),
            "baseline_coherent_rate": float(harm_zero["coherent_rate"]),
            "coherence_confounded": bool(float(harm_steer["coherent_rate"]) < args.coherent_min),
            "benign_refusal_rate": float(benign_steer.get("refusal_keyword_coherent_rate", float("nan"))),
            "benign_refusal_keyword_rate": float(benign_steer["refusal_keyword_rate"]),
            "benign_coherent_rate": float(benign_steer["coherent_rate"]),
            "s_post_patch_mean": float(harm_steer["s_post_patch_decode_mean"]),
            "s_propagated_l35_mean": float(harm_steer["s_propagated_decode_mean_L35"]),
            "propagated_gain_l35": float(harm_steer["propagated_gain_L35"]),
            "tau_l35": float(harm_steer["propagated_tau_L35"]),
            "residual_s_post_patch_mean": mean_or_nan(residual.get("s_post_patch_decode_mean", pd.Series(dtype=float))),
            "residual_s_propagated_l35_mean": mean_or_nan(
                residual.get("s_propagated_decode_mean_L35", pd.Series(dtype=float))
            ),
            "residual_prompt_ids": ";".join(str(pid) for pid in residual_ids),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("sparsity")
    valid = out[~out["coherence_confounded"].astype(bool)]
    counts = valid["residual_count"].tolist()
    monotone = bool(all(left <= right for left, right in zip(counts, counts[1:]))) if len(counts) >= 2 else False
    out["monotone_non_decreasing_valid_sparsities"] = monotone
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv_text_free(out, args.output)
    decision = {
        "pass": monotone,
        "valid_sparsities": [float(value) for value in valid["sparsity"].tolist()],
        "valid_residual_counts": [int(value) for value in counts],
        "interpretation": "residual grows with sparsity" if monotone else "residual does not monotonically grow or is bounded/confounded",
        "output": str(args.output),
    }
    decision_path = args.output.with_suffix(".decision.json")
    decision_path.write_text(json.dumps(decision, indent=2, default=json_default), encoding="utf-8")
    print(f"[mechanism-seal] wrote {decision_path}")
    print(f"[mechanism-seal] decision {json.dumps(decision, ensure_ascii=False, default=json_default)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    residual = sub.add_parser("residual-sweep")
    residual.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    residual.add_argument("--output", type=Path, default=Path("results/phase15_vpref_projection/mechanism_seal_residual_sweep.csv"))
    residual.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    residual.add_argument("--layer-group", default="24,28,32")
    residual.add_argument("--window", default="32")
    residual.add_argument("--k-r", type=int, default=1)
    residual.add_argument("--steering-mode", default="norm_relative")
    residual.add_argument("--target-kind", default="strong")
    residual.add_argument("--beta-value", type=float, default=0.25)
    residual.add_argument("--coherent-min", type=float, default=0.85)
    residual.set_defaults(func=residual_sweep)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
