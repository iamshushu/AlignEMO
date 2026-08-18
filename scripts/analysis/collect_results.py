"""Collect metrics from finished job logs into mean±std tables (CMC revision).

Scans log files for a metric pattern and aggregates (mean, std, n, values) per
group, so multi-seed / gate results are ready when you come back — no manual
grepping. Define groups as  NAME=GLOB::REGEX  (REGEX must have one capturing
group with the number; the LAST match per file is used).

Example:
    python scripts/analysis/collect_results.py \
      --group "kmind_multiseed=kmind_multiseed_*_*.out::best test F1:\\s*([0-9.]+)" \
      --group "iemocap_gate=scripts/analysis/gate_seed*.log::Best F-Score based on test:\\s*([0-9.]+)" \
      --out_csv scripts/analysis/collected_results.csv
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract(files, regex):
    pat = re.compile(regex)
    out = []
    for f in sorted(files):
        try:
            txt = Path(f).read_text(errors="ignore")
        except Exception:
            continue
        m = pat.findall(txt)
        if m:
            out.append((Path(f).name, float(m[-1])))
    return out


def main(args):
    rows = []
    for g in args.group:
        name, rest = g.split("=", 1)
        pattern_glob, regex = rest.split("::", 1)
        found = extract(glob.glob(pattern_glob), regex)
        vals = np.array([v for _, v in found], dtype=float)
        if len(vals) == 0:
            print(f"[{name}] no matches for glob '{pattern_glob}'")
            rows.append(dict(group=name, n=0, mean="", std="", values=""))
            continue
        print(f"\n=== {name}  (n={len(vals)}) ===")
        for fn, v in found:
            print(f"  {v:7.2f}   {fn}")
        print(f"  mean±std = {vals.mean():.2f} ± {vals.std(ddof=0):.2f}   "
              f"(min {vals.min():.2f}, max {vals.max():.2f})")
        rows.append(dict(group=name, n=len(vals), mean=round(vals.mean(), 2),
                         std=round(vals.std(ddof=0), 2),
                         values=";".join(f"{v:.2f}" for v in vals)))
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[saved] {args.out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", action="append", required=True,
                   help='NAME=GLOB::REGEX (regex has one () capture with the number)')
    p.add_argument("--out_csv", default="scripts/analysis/collected_results.csv")
    main(p.parse_args())
