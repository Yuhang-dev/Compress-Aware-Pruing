from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "results"
    / "phase2_readout_repair_v1_shards"
    / "wanda_50"
    / "repair_details.csv"
)
DEFAULT_TAU_MANIFEST = DEFAULT_INPUT.with_name("g_solve_manifest.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "aaai2_full_review" / "figure_data"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def parse_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("--layers must contain unique comma-separated integers")
    return layers


def relative_source(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def read_filtered_rows(
    path: Path,
    *,
    condition: str,
    repair_kind: str,
    split: str,
    label_column: str,
    layers: Iterable[int],
) -> list[dict[str, str]]:
    required = {
        "model",
        "condition",
        "repair_kind",
        "split",
        "prompt_id",
        "coherent",
        label_column,
        *(f"s{layer}" for layer in layers),
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        rows = [
            row
            for row in reader
            if row["condition"].strip() == condition
            and row["repair_kind"].strip() == repair_kind
            and row["split"].strip() == split
            and parse_bool(row["coherent"])
        ]
    if not rows:
        raise ValueError(
            f"No coherent rows matched condition={condition!r}, repair_kind={repair_kind!r}, split={split!r}"
        )
    prompt_ids = [row["prompt_id"].strip() for row in rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Filtered rows contain duplicate prompt_id values; an arm may have been counted twice")
    models = {row["model"].strip() for row in rows}
    if len(models) != 1:
        raise ValueError(f"Expected one model after filtering, found {sorted(models)}")
    return rows


def load_tau_by_layer(path: Path, layers: Iterable[int]) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("tau_by_layer")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a tau_by_layer mapping")
    result: dict[int, float] = {}
    for layer in layers:
        key = str(layer)
        if key not in raw:
            raise ValueError(f"{path} has no fixed threshold for layer {layer}")
        tau = float(raw[key])
        if not math.isfinite(tau):
            raise ValueError(f"Non-finite tau for layer {layer}: {tau}")
        result[layer] = tau
    return result


def empirical_roc(labels: np.ndarray, risk_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    labels = np.asarray(labels, dtype=bool)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if labels.ndim != 1 or risk_scores.ndim != 1 or labels.size != risk_scores.size:
        raise ValueError("labels and risk_scores must be aligned one-dimensional arrays")
    if not np.isfinite(risk_scores).all():
        raise ValueError("risk_scores contain non-finite values")
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    if not n_positive or not n_negative:
        raise ValueError("ROC requires at least one positive and one negative row")

    order = np.argsort(-risk_scores, kind="stable")
    sorted_scores = risk_scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=float)
    cumulative_fp = np.cumsum(~sorted_labels, dtype=float)
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), sorted_scores.size - 1]

    tpr = np.r_[0.0, cumulative_tp[distinct_ends] / n_positive]
    fpr = np.r_[0.0, cumulative_fp[distinct_ends] / n_negative]
    thresholds = np.r_[np.inf, sorted_scores[distinct_ends]]
    auc = float(np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]) * 0.5))
    return fpr, tpr, thresholds, auc


def auc_pairwise(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    positive_scores = np.asarray(positive_scores, dtype=float)
    negative_scores = np.sort(np.asarray(negative_scores, dtype=float))
    if not positive_scores.size or not negative_scores.size:
        raise ValueError("AUC requires non-empty positive and negative samples")
    lower = np.searchsorted(negative_scores, positive_scores, side="left")
    upper = np.searchsorted(negative_scores, positive_scores, side="right")
    wins = lower + 0.5 * (upper - lower)
    return float(wins.sum() / (positive_scores.size * negative_scores.size))


def stratified_bootstrap_auc(
    labels: np.ndarray,
    risk_scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    positive = np.asarray(risk_scores, dtype=float)[np.asarray(labels, dtype=bool)]
    negative = np.asarray(risk_scores, dtype=float)[~np.asarray(labels, dtype=bool)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_positive = positive[rng.integers(0, positive.size, size=positive.size)]
        sampled_negative = negative[rng.integers(0, negative.size, size=negative.size)]
        estimates[index] = auc_pairwise(sampled_positive, sampled_negative)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    layers = args.layers
    rows = read_filtered_rows(
        args.input,
        condition=args.condition,
        repair_kind=args.repair_kind,
        split=args.split,
        label_column=args.label_column,
        layers=layers,
    )
    if args.expected_n is not None and len(rows) != args.expected_n:
        raise ValueError(f"Expected n={args.expected_n}, found n={len(rows)}")

    labels = np.asarray([parse_bool(row[args.label_column]) for row in rows], dtype=bool)
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    model = rows[0]["model"].strip()
    tau_by_layer = load_tau_by_layer(args.tau_manifest, layers)
    source = relative_source(args.input)
    tau_source = relative_source(args.tau_manifest) + ":tau_by_layer"

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for layer_offset, layer in enumerate(layers):
        score_column = f"s{layer}"
        margins = np.asarray([float(row[score_column]) for row in rows], dtype=float)
        risk_scores = -margins
        fpr, tpr, thresholds, auc = empirical_roc(labels, risk_scores)
        ci_low, ci_high = stratified_bootstrap_auc(
            labels,
            risk_scores,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + layer_offset,
        )
        tau = tau_by_layer[layer]
        tau_predictions = margins < tau
        tau_tpr = float(tau_predictions[labels].mean())
        tau_fpr = float(tau_predictions[~labels].mean())

        summary_rows.append(
            {
                "figure_id": "fig:speca",
                "model": model,
                "condition": args.condition,
                "repair_kind": args.repair_kind,
                "split": args.split,
                "layer": layer,
                "score_column": score_column,
                "risk_score": f"-{score_column}",
                "positive_label": f"coherent_{args.label_column}",
                "auc": f"{auc:.10f}",
                "auc_ci_low": f"{ci_low:.10f}",
                "auc_ci_high": f"{ci_high:.10f}",
                "ci_method": f"stratified_prompt_bootstrap_{args.bootstrap_replicates}",
                "bootstrap_seed": args.bootstrap_seed + layer_offset,
                "n": len(rows),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "tau": f"{tau:.10f}",
                "tau_rule": f"{score_column}<tau",
                "tau_tpr": f"{tau_tpr:.10f}",
                "tau_fpr": f"{tau_fpr:.10f}",
                "tau_source": tau_source,
                "status": "measured",
                "source": source,
            }
        )

        for point_order, (point_fpr, point_tpr, risk_threshold) in enumerate(zip(fpr, tpr, thresholds)):
            finite_threshold = math.isfinite(float(risk_threshold))
            curve_rows.append(
                {
                    "figure_id": "fig:speca",
                    "model": model,
                    "condition": args.condition,
                    "layer": layer,
                    "point_order": point_order,
                    "fpr": f"{float(point_fpr):.10f}",
                    "tpr": f"{float(point_tpr):.10f}",
                    "risk_threshold": f"{float(risk_threshold):.10f}" if finite_threshold else "inf",
                    "margin_threshold": f"{-float(risk_threshold):.10f}" if finite_threshold else "-inf",
                    "status": "measured",
                    "source": source,
                }
            )

    curve_path = args.output_dir / "fig_speca_within_w50_roc.csv"
    summary_path = args.output_dir / "fig_speca_within_w50_roc_summary.csv"
    manifest_path = args.output_dir / "fig_speca_within_w50_roc_manifest.json"
    write_csv(
        curve_path,
        curve_rows,
        [
            "figure_id",
            "model",
            "condition",
            "layer",
            "point_order",
            "fpr",
            "tpr",
            "risk_threshold",
            "margin_threshold",
            "status",
            "source",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "figure_id",
            "model",
            "condition",
            "repair_kind",
            "split",
            "layer",
            "score_column",
            "risk_score",
            "positive_label",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "ci_method",
            "bootstrap_seed",
            "n",
            "n_positive",
            "n_negative",
            "tau",
            "tau_rule",
            "tau_tpr",
            "tau_fpr",
            "tau_source",
            "status",
            "source",
        ],
    )
    manifest = {
        "figure_id": "fig:speca",
        "model": model,
        "condition": args.condition,
        "repair_kind": args.repair_kind,
        "split": args.split,
        "label_column": args.label_column,
        "positive_label": f"coherent_{args.label_column}",
        "risk_score_definition": "rho_l(x) = -s_l(x)",
        "prompt_boundary": "last prompt token before generation",
        "layers": layers,
        "n": len(rows),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "tau_rule": "s_l < tau_l",
        "tau_source": tau_source,
        "input": source,
        "curve_csv": curve_path.name,
        "summary_csv": summary_path.name,
        "status": "measured",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for row in summary_rows:
        print(
            f"L{row['layer']}: AUC={float(row['auc']):.3f} "
            f"[{float(row['auc_ci_low']):.3f}, {float(row['auc_ci_high']):.3f}], "
            f"tau point=({float(row['tau_fpr']):.3f}, {float(row['tau_tpr']):.3f})"
        )
    print(f"n={len(rows)}; positive={n_positive}; negative={n_negative}")
    print(f"wrote {curve_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    return curve_path, summary_path, manifest_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build auditable within-condition ROC source data for Fig. speca from cached "
            "Qwen2.5-3B Wanda-50 prompt-boundary readouts."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tau-manifest", type=Path, default=DEFAULT_TAU_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition", default="wanda_50")
    parser.add_argument("--repair-kind", default="pruned")
    parser.add_argument("--split", default="harm_eval")
    parser.add_argument("--label-column", default="unsafe")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("24,28,32"))
    parser.add_argument("--expected-n", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    build(args)


if __name__ == "__main__":
    main()
