#!/usr/bin/env python3
"""Fill table_adaptive_attack_tbd.tex from GCG summary JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ARM_LABELS = {"dense": "Dense", "wanda50": "Wanda-50", "remar": r"+\method{}"}


def cell(summary: dict) -> str:
    return f"{float(summary['asr_pct']):.1f} / {float(summary['coherence_pct']):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/gcg_qwen3b"))
    parser.add_argument("--table", type=Path, default=Path("aaai2_full_review/draft_tables/table_adaptive_attack_tbd.tex"))
    parser.add_argument("--csv", type=Path, default=Path("aaai2_full_review/table_data/table_adaptive_attack_tbd.csv"))
    args = parser.parse_args()

    summaries = {}
    for arm in ARM_LABELS:
        for attack in ("standard", "adaptive"):
            path = args.root / attack / arm / "summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            summaries[(arm, attack)] = json.loads(path.read_text(encoding="utf-8"))

    text = args.table.read_text(encoding="utf-8")
    for arm, label in ARM_LABELS.items():
        standard = cell(summaries[(arm, "standard")])
        adaptive = cell(summaries[(arm, "adaptive")])
        replacement = f"{label} & {standard} & {adaptive} \\\\"  # noqa: W605
        pattern = rf"^{re.escape(label)}\s*&.*$"
        text, count = re.subn(pattern, lambda _match, value=replacement: value, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"Could not update row {label}")
    reference = summaries[("dense", "standard")]
    budget_text = (
        f"L={reference['suffix_tokens']} tokens, $T={reference['steps']}$ steps, "
        f"$R={reference['restarts']}$ restart{'' if int(reference['restarts']) == 1 else 's'}, "
        f"$n={reference['n_prompts']}$ prompts"
    )
    text = text.replace(
        r"L=\textit{TBD}$ tokens, $T=\textit{TBD}$ steps, $R=\textit{TBD}$ restarts, $n=\textit{TBD}$ prompts",
        budget_text,
    )
    args.table.write_text(text, encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["row_order,arm,model,pruner,sparsity_pct,attack,attack_set,standard_asr_pct,standard_coherence_pct,adaptive_asr_pct,adaptive_coherence_pct,suffix_tokens,steps,restarts,n_prompts,status,include"]
    csv_arm = {"dense": "dense", "wanda50": "pruned", "remar": "remar"}
    for index, arm in enumerate(("dense", "wanda50", "remar"), start=1):
        std, ada = summaries[(arm, "standard")], summaries[(arm, "adaptive")]
        lines.append(",".join(map(str, [index, csv_arm[arm], "Qwen2.5-3B", "Wanda", 50, "GCG", "AdvBench", f"{std['asr_pct']:.6f}", f"{std['coherence_pct']:.6f}", f"{ada['asr_pct']:.6f}", f"{ada['coherence_pct']:.6f}", std["suffix_tokens"], std["steps"], std["restarts"], std["n_prompts"], "measured", 1])))
    args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated {args.table}")
    print(f"updated {args.csv}")


if __name__ == "__main__":
    main()
