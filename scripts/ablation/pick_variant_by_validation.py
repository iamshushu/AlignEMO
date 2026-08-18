"""Print the fusion variant with the highest validation w-F1 in a sweep directory.

Used by run_elacl_cached_5seed.sbatch so the follow-up job can be queued with
--dependency before the sweep that decides the winner has even started. The choice
reads `best_validation_f1` only; the test column in those CSVs is never consulted,
which is what keeps the selection honest when the job runs unattended.

    python scripts/ablation/pick_variant_by_validation.py <sweep_dir> [--seed 42]

With --seed, only that seed's rows count (the sweep's seed-42 rows are the ones the
method choice was meant to rest on). Without it, every row is considered.
"""
import argparse
import csv
import glob
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    best, pick = None, None
    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*.csv")))
    if not files:
        sys.exit(f"no CSVs under {args.sweep_dir}")

    for path in files:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip(): v for k, v in row.items() if k}
                if "best_validation_f1" not in row or "variant" not in row:
                    continue
                if args.seed is not None and int(row.get("seed", -1)) != args.seed:
                    continue
                try:
                    score = float(row["best_validation_f1"])
                except (TypeError, ValueError):
                    continue
                if best is None or score > best:
                    best, pick = score, row["variant"]

    if pick is None:
        sys.exit(f"no usable best_validation_f1 rows under {args.sweep_dir}")
    print(pick)


if __name__ == "__main__":
    main()
