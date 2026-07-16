from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_figure_style import COLORS, configure_matplotlib, save_figure


ROOT = Path(__file__).resolve().parent
ROC_DATA = ROOT / "figure_data" / "fig_speca_within_w50_roc.csv"
SUMMARY_DATA = ROOT / "figure_data" / "fig_speca_within_w50_roc_summary.csv"

LAYER_ORDER = (24, 28, 32)
LAYER_STYLE = {
    24: {"color": "#6EA6B6", "linewidth": 1.05, "zorder": 3},
    28: {"color": "#2F758B", "linewidth": 1.75, "zorder": 5},
    32: {"color": "#AAC8D0", "linewidth": 0.95, "zorder": 2},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(
    roc_path: Path,
    summary_path: Path,
) -> tuple[dict[int, list[dict[str, str]]], dict[int, dict[str, str]]]:
    curve_rows = read_rows(roc_path)
    summary_rows = read_rows(summary_path)
    if any(row.get("status", "").lower() != "measured" for row in curve_rows + summary_rows):
        raise ValueError("Fig. speca accepts measured rows only")

    curves: dict[int, list[dict[str, str]]] = {}
    for row in curve_rows:
        layer = int(row["layer"])
        curves.setdefault(layer, []).append(row)
    summaries = {int(row["layer"]): row for row in summary_rows}
    if set(curves) != set(LAYER_ORDER) or set(summaries) != set(LAYER_ORDER):
        raise ValueError("Fig. speca requires exactly L24, L28, and L32")

    shared = {
        (row["model"], row["condition"], row["n"], row["n_positive"], row["n_negative"])
        for row in summary_rows
    }
    if len(shared) != 1:
        raise ValueError("All layers must share model, condition, and sample counts")
    for layer, rows in curves.items():
        rows.sort(key=lambda row: int(row["point_order"]))
        fpr = [float(row["fpr"]) for row in rows]
        tpr = [float(row["tpr"]) for row in rows]
        if abs(fpr[0]) > 1e-9 or abs(tpr[0]) > 1e-9 or abs(fpr[-1] - 1.0) > 1e-9 or abs(tpr[-1] - 1.0) > 1e-9:
            raise ValueError(f"L{layer} ROC must run from (0,0) to (1,1)")
        if any(right < left for left, right in zip(fpr, fpr[1:])) or any(
            right < left for left, right in zip(tpr, tpr[1:])
        ):
            raise ValueError(f"L{layer} ROC coordinates are not monotone")
    return curves, summaries


def draw_figure_speca(
    *,
    roc_path: Path = ROC_DATA,
    summary_path: Path = SUMMARY_DATA,
    output_dir: Path = ROOT,
    stem: str = "figure_speca",
) -> None:
    curves, summaries = validate(roc_path, summary_path)
    configure_matplotlib()

    fig = plt.figure(figsize=(3.35, 2.65), facecolor="white")
    ax = fig.add_axes([0.18, 0.17, 0.627, 0.792])
    ax.set_aspect("equal", adjustable="box")

    ax.plot(
        [0, 1],
        [0, 1],
        color="#AEB4BA",
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    ax.text(
        0.64,
        0.60,
        "chance",
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#8A9096",
    )

    legend_handles: list[Line2D] = []
    for layer in LAYER_ORDER:
        style = LAYER_STYLE[layer]
        rows = curves[layer]
        fpr = [float(row["fpr"]) for row in rows]
        tpr = [float(row["tpr"]) for row in rows]
        summary = summaries[layer]
        auc = float(summary["auc"])
        ci_low = float(summary["auc_ci_low"])
        ci_high = float(summary["auc_ci_high"])
        ax.step(
            fpr,
            tpr,
            where="post",
            color=style["color"],
            linewidth=style["linewidth"],
            solid_capstyle="round",
            zorder=style["zorder"],
        )
        ax.scatter(
            [float(summary["tau_fpr"])],
            [float(summary["tau_tpr"])],
            s=23 if layer == 28 else 18,
            marker="o",
            facecolor=style["color"],
            edgecolor=style["color"],
            linewidth=0.5,
            zorder=style["zorder"] + 2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=style["linewidth"],
                label=f"L{layer}  {auc:.2f} [{ci_low:.2f}, {ci_high:.2f}]",
            )
        )

    first = summaries[LAYER_ORDER[0]]
    model = first["model"].replace("Qwen/", "").replace("-Instruct", "")
    condition = first["condition"].replace("wanda_", "Wanda-") + "%"
    n = int(first["n"])
    n_positive = int(first["n_positive"])
    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.015),
        borderaxespad=0.0,
        handlelength=2.2,
        handletextpad=0.65,
        labelspacing=0.45,
        fontsize=6.3,
        title=f"Within condition\n{model} · {condition}\nn={n} ({n_positive} unsafe)",
        title_fontsize=6.3,
        frameon=True,
        fancybox=False,
        borderpad=0.42,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.90)
    ax.text(
        0.025,
        0.965,
        "Filled dots: fixed $\\tau_\\ell$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["neutral"],
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", colors=COLORS["neutral"], labelsize=6.8, width=0.65, length=3, pad=2.5)
    ax.grid(True, color="#ECEEEF", linewidth=0.42, zorder=-2)
    ax.set_xlabel("False-positive rate", fontsize=7.6, labelpad=4)
    ax.set_ylabel("True-positive rate", fontsize=7.6, labelpad=4)
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw the single-panel within-W50 ROC for Fig. speca")
    parser.add_argument("--roc-data", type=Path, default=ROC_DATA)
    parser.add_argument("--summary-data", type=Path, default=SUMMARY_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--stem", default="figure_speca")
    return parser


if __name__ == "__main__":
    cli = make_parser().parse_args()
    draw_figure_speca(
        roc_path=cli.roc_data,
        summary_path=cli.summary_data,
        output_dir=cli.output_dir,
        stem=cli.stem,
    )
