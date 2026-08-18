#!/usr/bin/env python3
"""Build Markdown and LaTeX tables from immutable corrected-result summaries."""
import argparse
import copy
import json
from pathlib import Path


METHODS = (
    ("concat", "Concatenation"),
    ("crossattn", "Cross-attention"),
    ("hmf_nogate", "HMF w/o gate"),
    ("hmf", "HMF"),
    ("shiftaware", "Shift-aware HMF"),
)

# Numbers stated by the released manuscript/README.  These are retained as a
# reference table and must not be presented as results of the corrected rerun.
REPORTED_HEADLINE = (
    ("IEMOCAP", 70.83),
    ("MELD", 67.49),
    ("K-MIND", 74.61),
)


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def score(entry):
    return f'{entry["test_weighted_f1_mean"]:.3f} $\\pm$ {entry["test_weighted_f1_sample_std"]:.3f}'


def md_score(entry):
    return f'{entry["test_weighted_f1_mean"]:.3f} ± {entry["test_weighted_f1_sample_std"]:.3f}'


def main(args):
    iem = load(args.iemocap)
    meld = load(args.meld)
    kmind = load(args.kmind)
    if args.kmind_beta is not None:
        beta = load(args.kmind_beta)
        kmind = copy.deepcopy(kmind)
        kmind["variants"]["hmf"].update({
            "test_weighted_f1": beta["hmf_selected_beta"]["values"],
            "test_weighted_f1_mean": beta["hmf_selected_beta"]["mean"],
            "test_weighted_f1_sample_std": beta["hmf_selected_beta"]["sample_std"],
        })
        kmind["selected_hmf_beta"] = beta["selected_beta"]
    datasets = (("IEMOCAP", iem), ("MELD", meld), ("K-MIND", kmind))
    best = {
        name: max(data["variants"], key=lambda key: data["variants"][key]["test_weighted_f1_mean"])
        for name, data in datasets
    }

    md = [
        "# Paper-ready corrected result tables",
        "",
        "## Previously reported end-to-end headline (reference only)",
        "",
        "| Dataset | Reported test weighted F1 | Corrected-rerun status |",
        "|---|---:|---|",
        *[
            f"| {dataset} | {value:.2f} | Not a corrected-rerun result |"
            for dataset, value in REPORTED_HEADLINE
        ],
        "",
        "These values are the manuscript/README headline numbers. They are not interchangeable with the cached fusion ablation below and remain historical reference values until the corrected end-to-end reruns finish.",
        "",
        "## Cached fusion-method ablation",
        "",
        "| Method | IEMOCAP test weighted F1 | MELD test weighted F1 | K-MIND test weighted F1 |",
        "|---|---:|---:|---:|",
    ]
    tex_rows = []
    for key, label in METHODS:
        values = []
        tex_values = []
        for name, data in datasets:
            entry = data["variants"][key]
            rendered = md_score(entry)
            rendered_tex = score(entry)
            if key == best[name]:
                rendered = f"**{rendered}**"
                rendered_tex = f"\\textbf{{{rendered_tex}}}"
            values.append(rendered)
            tex_values.append(rendered_tex)
        md.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
        tex_rows.append(f"{label} & {tex_values[0]} & {tex_values[1]} & {tex_values[2]} \\\\")

    seeds = ", ".join(str(seed) for seed in iem["seeds"])
    md.extend([
        "",
        f"Values are mean ± sample standard deviation over fusion-head seeds {seeds}.",
        "Epoch selection uses validation weighted F1; test is evaluated once after checkpoint selection.",
        "All runs reuse one fixed, frozen encoder cache per dataset, so this is a fusion-module ablation and not full-pipeline seed variance.",
        "K-MIND uses the reconstructed paper-aligned 768-d projection cache; its HMF row uses validation-selected beta=0.3.",
        "Bold denotes the highest test mean in each dataset column.",
        "",
        "## LaTeX: cached fusion-method ablation",
        "",
        "```latex",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cached fusion-module ablation. Values are test weighted F1 (mean $\\pm$ sample standard deviation) over five fusion-head seeds with fixed frozen encoder features. Checkpoints are selected only on validation weighted F1. K-MIND uses the reconstructed 768-d projection cache.}",
        "\\label{tab:cached_fusion_ablation}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & IEMOCAP & MELD & K-MIND \\\\",
        "\\midrule",
        *tex_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "```",
        "",
        "## LaTeX: previously reported headline",
        "",
        "```latex",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Previously reported end-to-end headline test weighted F1. These historical manuscript values are shown for reference and are not corrected-rerun results.}",
        "\\label{tab:reported_headline_reference}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Dataset & Reported weighted F1 \\\\",
        "\\midrule",
        *[f"{dataset} & {value:.2f} \\\\" for dataset, value in REPORTED_HEADLINE],
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "```",
        "",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(md), encoding="utf-8")

    latex_blocks = []
    cursor = 0
    while True:
        try:
            latex_start = md.index("```latex", cursor) + 1
        except ValueError:
            break
        latex_end = md.index("```", latex_start)
        latex_blocks.append("\n".join(md[latex_start:latex_end]))
        cursor = latex_end + 1
    args.latex.write_text("\n\n".join(latex_blocks) + "\n", encoding="utf-8")
    print(args.markdown)
    print(args.latex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap", required=True, type=Path)
    parser.add_argument("--meld", required=True, type=Path)
    parser.add_argument("--kmind", required=True, type=Path)
    parser.add_argument("--kmind-beta", type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--latex", required=True, type=Path)
    main(parser.parse_args())
