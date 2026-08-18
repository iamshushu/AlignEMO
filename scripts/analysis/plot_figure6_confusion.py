"""Regenerate Figure 6: row-normalized confusion matrices, Stage 1 vs Stage 1 + Stage 2.

A faithful copy of draw_combined() from EACL2TelME/EACL/TelME/draw_confusion.py -- same
figsize, aspect, type sizes, grid, and margins -- so the output drops into the manuscript
at the identical aspect ratio as figure/all_confusion_modified.pdf. Five of the six panels
reproduce that figure value-for-value.

Only K-MIND's Stage 1 + Stage 2 panel is new: it comes from paper_768_v1/stage12/hmf/
seed_100 (validation-selected, test w-F1 75.29), replacing the earlier panel whose Sad
recall had fallen from 0.27 to 0.26.

    python scripts/analysis/plot_figure6_confusion.py --out figures/all_confusion_v2.pdf

NOTE: K-MIND's two panels come from different runs -- Stage 1 from the original 74.61
pipeline (1280-d, legacy attention) and Stage 1 + Stage 2 from the 768-d corrected-
attention run. The difference between them is therefore not attributable to Stage 2 alone.
Passing --kmind-stage1 matched swaps in that run's own Stage 1 (75.09) so both panels
describe one model.
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

IEMOCAP = ["Happy", "Sad", "Neutral", "Anger", "Excited", "Frustrated"]
MELD = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
KMIND = ["Joy", "Sad", "Anger", "Surprise", "Fear", "Disgust", "Neutral"]

CM = {
    ("IEMOCAP", "s1"): [
        [0.53, 0.06, 0.15, 0.00, 0.26, 0.01],
        [0.00, 0.69, 0.12, 0.00, 0.00, 0.19],
        [0.03, 0.01, 0.76, 0.01, 0.02, 0.17],
        [0.00, 0.01, 0.05, 0.69, 0.00, 0.25],
        [0.25, 0.03, 0.10, 0.01, 0.60, 0.02],
        [0.00, 0.01, 0.13, 0.14, 0.00, 0.72]],
    ("IEMOCAP", "s2"): [
        [0.65, 0.03, 0.12, 0.00, 0.17, 0.03],
        [0.00, 0.70, 0.10, 0.00, 0.00, 0.20],
        [0.03, 0.01, 0.74, 0.01, 0.03, 0.18],
        [0.00, 0.00, 0.03, 0.70, 0.00, 0.27],
        [0.23, 0.02, 0.09, 0.00, 0.64, 0.01],
        [0.00, 0.01, 0.12, 0.14, 0.00, 0.73]],
    ("MELD", "s1"): [
        [0.53, 0.02, 0.01, 0.08, 0.21, 0.04, 0.11],
        [0.26, 0.18, 0.00, 0.03, 0.38, 0.04, 0.10],
        [0.02, 0.04, 0.22, 0.08, 0.34, 0.12, 0.18],
        [0.05, 0.01, 0.00, 0.65, 0.21, 0.01, 0.06],
        [0.03, 0.00, 0.00, 0.05, 0.85, 0.03, 0.04],
        [0.12, 0.00, 0.01, 0.06, 0.38, 0.36, 0.06],
        [0.09, 0.00, 0.01, 0.11, 0.14, 0.01, 0.64]],
    ("MELD", "s2"): [
        [0.51, 0.06, 0.01, 0.06, 0.18, 0.05, 0.12],
        [0.09, 0.43, 0.00, 0.03, 0.34, 0.06, 0.06],
        [0.02, 0.04, 0.22, 0.08, 0.32, 0.14, 0.18],
        [0.04, 0.03, 0.00, 0.65, 0.20, 0.01, 0.07],
        [0.03, 0.02, 0.01, 0.05, 0.81, 0.04, 0.04],
        [0.09, 0.05, 0.03, 0.05, 0.31, 0.40, 0.07],
        [0.07, 0.03, 0.01, 0.11, 0.11, 0.02, 0.65]],
    # published Stage 1 panel (74.61 pipeline)
    ("K-MIND", "s1"): [
        [0.40, 0.02, 0.01, 0.05, 0.01, 0.00, 0.52],
        [0.08, 0.27, 0.01, 0.02, 0.03, 0.01, 0.58],
        [0.06, 0.09, 0.19, 0.04, 0.02, 0.04, 0.57],
        [0.03, 0.00, 0.00, 0.65, 0.00, 0.00, 0.31],
        [0.03, 0.10, 0.01, 0.07, 0.23, 0.01, 0.55],
        [0.06, 0.04, 0.02, 0.02, 0.01, 0.17, 0.68],
        [0.04, 0.02, 0.00, 0.03, 0.01, 0.00, 0.90]],
    # NEW: stage12/hmf/seed_100, test w-F1 75.29
    ("K-MIND", "s2"): [
        [0.41, 0.02, 0.01, 0.06, 0.01, 0.00, 0.49],
        [0.06, 0.35, 0.02, 0.02, 0.02, 0.01, 0.50],
        [0.05, 0.11, 0.25, 0.06, 0.01, 0.03, 0.50],
        [0.03, 0.00, 0.00, 0.74, 0.00, 0.00, 0.23],
        [0.02, 0.10, 0.01, 0.07, 0.27, 0.00, 0.52],
        [0.06, 0.06, 0.02, 0.02, 0.02, 0.22, 0.60],
        [0.04, 0.02, 0.01, 0.04, 0.01, 0.00, 0.88]],
    # same run's own Stage 1 (75.09), for --kmind-stage1 matched
    ("K-MIND", "s1_matched"): [
        [0.38, 0.02, 0.00, 0.06, 0.00, 0.00, 0.54],
        [0.06, 0.30, 0.01, 0.02, 0.02, 0.01, 0.57],
        [0.04, 0.10, 0.20, 0.06, 0.01, 0.02, 0.57],
        [0.02, 0.00, 0.00, 0.70, 0.00, 0.00, 0.26],
        [0.02, 0.08, 0.01, 0.07, 0.26, 0.00, 0.56],
        [0.06, 0.05, 0.02, 0.02, 0.01, 0.18, 0.66],
        [0.04, 0.02, 0.00, 0.03, 0.01, 0.00, 0.90]],
}

ROWS = [("IEMOCAP", IEMOCAP), ("MELD", MELD), ("K-MIND", KMIND)]
COL_TITLES = ["Stage 1", "Stage 1 + Stage 2"]


def register_times():
    """Pull a Times-metric serif out of fontconfig, bypassing matplotlib's cache.

    Same trick as the original script. When fontconfig has nothing (headless nodes),
    the rcParam list below falls through to STIXGeneral, which ships with matplotlib
    and is metric-compatible with Times New Roman.
    """
    try:
        out = subprocess.check_output(["fc-list", ":", "file", "family"], text=True)
    except Exception:
        return
    keys = ["times new roman", "nimbus roman", "liberation serif",
            "tex gyre termes", "freeserif"]
    for line in out.splitlines():
        if any(k in line.lower() for k in keys):
            path = line.split(":")[0].strip()
            if path.endswith((".ttf", ".otf")):
                try:
                    fm.fontManager.addfont(path)
                except Exception:
                    pass


def draw_combined(rows, out, kmind_stage1):
    fig, axes = plt.subplots(len(rows), 2, figsize=(11.0, 4.9 * len(rows)))
    for r, (name, labels) in enumerate(rows):
        s1 = "s1_matched" if (name == "K-MIND" and kmind_stage1 == "matched") else "s1"
        for c, stage in enumerate((s1, "s2")):
            M = np.asarray(CM[(name, stage)])
            ax, n = axes[r][c], len(labels)
            ax.imshow(M, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
            ax.set_yticklabels(labels, fontsize=13)
            ax.set_xticks(np.arange(-.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="both", length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            for i in range(n):
                for j in range(n):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v >= 0.5 else "#222222", fontsize=15)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=19, pad=12)
            ax.set_xlabel("Predicted", fontsize=14)
            ax.set_ylabel("True", fontsize=14)

    fig.tight_layout(rect=[0.06, 0.0, 1, 1])
    for r, (name, _) in enumerate(rows):
        pos = axes[r][0].get_position()
        fig.text(0.018, (pos.y0 + pos.y1) / 2.0, f"({chr(97 + r)}) {name}",
                 ha="center", va="center", rotation=90, fontsize=21, fontweight="bold")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/all_confusion_v2.pdf")
    p.add_argument("--kmind-stage1", choices=["published", "matched"], default="published",
                   help="'published' keeps the 74.61-pipeline panel; 'matched' uses the "
                        "75.09 Stage 1 from the same run as the new Stage 1 + Stage 2")
    args = p.parse_args()

    register_times()
    mpl.rcParams["pdf.fonttype"] = 42          # embed TrueType, not Type 3
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                                  "TeX Gyre Termes", "FreeSerif", "STIXGeneral",
                                  "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.linewidth"] = 0.8

    out = Path(args.out)
    draw_combined(ROWS, out, args.kmind_stage1)
    print(f"[written] {out}")
    print(f"[written] {out.with_suffix('.png')}")
    print(f"  K-MIND Stage 1 panel: {args.kmind_stage1}")
    print("  K-MIND Stage 1 + Stage 2 panel: stage12/hmf/seed_100, test w-F1 75.29")


if __name__ == "__main__":
    main()
