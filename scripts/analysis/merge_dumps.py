"""Merge per-variant prediction dumps into one wide CSV for significance testing.

Each input dump (from train_fusion_cached.py --dump_csv) has columns
  id, dataset, gold, variant, pred_stage1
This aligns them on `id` and emits  id, dataset, gold, pred_<variantA>, pred_<variantB>, ...
so significance.py can compare two models with --model_a_col / --model_b_col.
"""
import argparse
from pathlib import Path

import pandas as pd


def main(args):
    base = None
    for path in args.dumps:
        df = pd.read_csv(path)
        variant = str(df["variant"].iloc[0])
        col = f"pred_{variant}"
        part = df[["id", "dataset", "gold"]].copy()
        part[col] = df["pred_stage1"].values
        base = part if base is None else base.merge(
            df[["id"]].assign(**{col: df["pred_stage1"].values}), on="id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out, index=False)
    print(f"[merged] {args.out}  rows={len(base)}  cols={list(base.columns)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", nargs="+", required=True)
    p.add_argument("--out", default="scripts/analysis/iemocap_dumps_merged.csv")
    main(p.parse_args())
