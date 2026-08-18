"""Plot the IEMOCAP context-length ablation (CMC revision).

Reads the results CSV produced by scripts/ablation/run_context_len_iemocap.sh
(columns: context_len, weighted_f1, macro_f1, ...) and draws weighted-F1 and
macro-F1 as a function of the number of preceding utterances fed to the text
encoder ({0, 2, 4, 8, full}).

Usage:
    python scripts/analysis/plot_context_len.py \
        --results scripts/analysis/context_len_results.csv \
        --out scripts/analysis/context_len_curve.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def to_x(cl, full_value):
    """Map context_len label to a numeric x; 'full' -> full_value."""
    s = str(cl).strip().lower()
    return float(full_value) if s in ("full", "-1", "") else float(s)


def main(args):
    df = pd.read_csv(args.results)
    df = df[df["context_len"].notna()].copy()
    # order by numeric context, with 'full' plotted last
    df["is_full"] = df["context_len"].astype(str).str.lower().eq("full")
    df["x"] = [to_x(c, args.full_x) for c in df["context_len"]]
    df = df.sort_values(["is_full", "x"], kind="mergesort").reset_index(drop=True)

    xs = df["x"].to_numpy()
    xticklabels = [str(c) for c in df["context_len"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, label, marker in [("weighted_f1", "weighted-F1", "o"),
                               ("macro_f1", "macro-F1", "s")]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isfinite(y).any():
                ax.plot(xs, y, marker=marker, label=label)

    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("# preceding utterances in text context")
    ax.set_ylabel("F1 (%)")
    ax.set_title("IEMOCAP: effect of text-context length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="scripts/analysis/context_len_results.csv")
    p.add_argument("--out", default="scripts/analysis/context_len_curve.png")
    p.add_argument("--full_x", type=float, default=12.0,
                   help="numeric x position to plot the 'full' setting at")
    args = p.parse_args()
    main(args)
