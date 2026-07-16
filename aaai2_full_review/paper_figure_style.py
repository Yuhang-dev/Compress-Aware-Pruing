from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


COLORS = {
    "ink": "#30343B",
    "neutral": "#7A7F87",
    "blue": "#3E8FA8",
    "blue_light": "#D9E8EC",
    "red": "#C65D52",
    "amber": "#E1A23A",
    "grid": "#E5E5E5",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, root: Path, stem: str, dpi: int = 400) -> None:
    pdf_path = root / f"{stem}.pdf"
    svg_path = root / f"{stem}.svg"
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    fig.savefig(svg_path, format="svg", facecolor="white")
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    fig.savefig(root / f"{stem}.png", format="png", dpi=dpi, facecolor="white")
