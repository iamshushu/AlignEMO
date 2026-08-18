"""Significance testing for two ERC models' predictions (AlignEMO, CMC revision).

Compares model B vs model A (by default Stage1+2 ELACL vs Stage1 HMF) on
Δ weighted-F1 and Δ macro-F1, using two paired tests:

  1. Paired bootstrap resampling (Berg-Kirkpatrick et al., 2012):
     resample utterances with replacement (same indices for both models),
     recompute Δmetric  ->  95% percentile CI and a two-sided bootstrap p-value.
  2. Approximate randomization test (Yeh 2000; Dror et al., 2018):
     under H0 the two systems are exchangeable, so per-utterance we swap the two
     models' predictions with prob 0.5 and recompute |Δmetric|;
     p = (#{|Δ_rand| >= |Δ_obs|} + 1) / (R + 1).

Input
-----
One or more per-utterance prediction CSVs (``--pred_csv``), each with columns
(configurable): gold, pred_stage1, pred_stage2  (+ optional ``dataset``).
Predictions of the two models must be paired row-by-row (same test utterance).
Multiple CSVs are treated as **seeds**: per-model mean±std across seeds is
reported, and the paired tests are run per seed and aggregated.

Refs: Berg-Kirkpatrick, Burkett & Klein (2012); Dror, Baumer, Shlomov & Reichart
(2018) "The Hitchhiker's Guide to Testing Statistical Significance in NLP".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DEFAULT_COLS = dict(gold="gold", modelA="pred_stage1", modelB="pred_stage2", dataset="dataset")


def w_f1(gold, pred):
    return f1_score(gold, pred, average="weighted", zero_division=0) * 100.0


def m_f1(gold, pred):
    return f1_score(gold, pred, average="macro", zero_division=0) * 100.0


METRICS = {"weighted_f1": w_f1, "macro_f1": m_f1}


def paired_bootstrap(gold, predA, predB, metric_fn, n_boot, rng):
    N = len(gold)
    obs = metric_fn(gold, predB) - metric_fn(gold, predA)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)          # paired resample w/ replacement
        deltas[b] = metric_fn(gold[idx], predB[idx]) - metric_fn(gold[idx], predA[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p-value: mass of the distribution on the null side of 0
    p = 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return dict(delta=obs, ci_low=lo, ci_high=hi, p_value=min(p, 1.0))


def approx_randomization(gold, predA, predB, metric_fn, n_iter, rng):
    N = len(gold)
    obs = abs(metric_fn(gold, predB) - metric_fn(gold, predA))
    count = 0
    for _ in range(n_iter):
        swap = rng.random(N) < 0.5
        pa = np.where(swap, predB, predA)
        pb = np.where(swap, predA, predB)
        if abs(metric_fn(gold, pb) - metric_fn(gold, pa)) >= obs:
            count += 1
    return dict(delta_abs=obs, p_value=(count + 1) / (n_iter + 1))


def run_one_seed(df, cols, n_boot, n_rand, seed):
    gold = df[cols["gold"]].to_numpy()
    a = df[cols["modelA"]].to_numpy()
    b = df[cols["modelB"]].to_numpy()
    rng = np.random.default_rng(seed)
    res = {}
    for mname, mfn in METRICS.items():
        res[mname] = dict(
            A=round(mfn(gold, a), 3),
            B=round(mfn(gold, b), 3),
            bootstrap=paired_bootstrap(gold, a, b, mfn, n_boot, rng),
            randomization=approx_randomization(gold, a, b, mfn, n_rand, rng),
        )
    return res


def aggregate_seeds(per_seed):
    """per_seed: list of run_one_seed() dicts. Return mean/std summary."""
    agg = {}
    for mname in METRICS:
        A = np.array([s[mname]["A"] for s in per_seed])
        B = np.array([s[mname]["B"] for s in per_seed])
        dboot = np.array([s[mname]["bootstrap"]["delta"] for s in per_seed])
        pboot = np.array([s[mname]["bootstrap"]["p_value"] for s in per_seed])
        prand = np.array([s[mname]["randomization"]["p_value"] for s in per_seed])
        agg[mname] = dict(
            A_mean=round(A.mean(), 3), A_std=round(A.std(ddof=0), 3),
            B_mean=round(B.mean(), 3), B_std=round(B.std(ddof=0), 3),
            delta_mean=round(dboot.mean(), 3), delta_std=round(dboot.std(ddof=0), 3),
            boot_p_mean=round(float(pboot.mean()), 5),
            rand_p_mean=round(float(prand.mean()), 5),
            n_seeds=len(per_seed),
            n_seeds_sig_boot=int((pboot < 0.05).sum()),
            n_seeds_sig_rand=int((prand < 0.05).sum()),
        )
    return agg


def print_report(name, per_seed):
    n = len(per_seed)
    print(f"\n================ {name}  ({n} seed{'s' if n != 1 else ''}) ================")
    if n == 1:
        r = per_seed[0]
        for mname in METRICS:
            m = r[mname]
            bt, rd = m["bootstrap"], m["randomization"]
            print(f"\n[{mname}]  A={m['A']}  B={m['B']}  Δ(B-A)={bt['delta']:+.3f}")
            print(f"  bootstrap 95% CI = [{bt['ci_low']:+.3f}, {bt['ci_high']:+.3f}]  "
                  f"p={bt['p_value']:.4f}  {'(sig)' if bt['p_value'] < 0.05 else '(n.s.)'}")
            print(f"  approx. randomization  p={rd['p_value']:.4f}  "
                  f"{'(sig)' if rd['p_value'] < 0.05 else '(n.s.)'}")
    else:
        agg = aggregate_seeds(per_seed)
        for mname in METRICS:
            a = agg[mname]
            print(f"\n[{mname}]")
            print(f"  A = {a['A_mean']} ± {a['A_std']}   B = {a['B_mean']} ± {a['B_std']}")
            print(f"  Δ(B-A) = {a['delta_mean']:+.3f} ± {a['delta_std']}")
            print(f"  bootstrap  mean p = {a['boot_p_mean']:.4f}   "
                  f"sig in {a['n_seeds_sig_boot']}/{a['n_seeds']} seeds")
            print(f"  randomiz.  mean p = {a['rand_p_mean']:.4f}   "
                  f"sig in {a['n_seeds_sig_rand']}/{a['n_seeds']} seeds")


def load_by_dataset(pred_csvs, cols):
    """Return {dataset_name: [df_seed0, df_seed1, ...]} (one df per input CSV)."""
    out = {}
    for path in pred_csvs:
        df = pd.read_csv(path)
        for req in ("gold", "modelA", "modelB"):
            if cols[req] not in df.columns:
                raise ValueError(f"{path}: missing column '{cols[req]}'. Have {list(df.columns)}")
        if cols["dataset"] in df.columns:
            for name, sub in df.groupby(cols["dataset"], sort=False):
                out.setdefault(str(name), []).append(sub.reset_index(drop=True))
        else:
            out.setdefault(Path(path).stem, []).append(df.reset_index(drop=True))
    return out


def main(args):
    cols = dict(DEFAULT_COLS, gold=args.gold_col, modelA=args.model_a_col,
                modelB=args.model_b_col, dataset=args.dataset_col)
    by_ds = load_by_dataset(args.pred_csv, cols)

    summary_rows = []
    for name, dfs in by_ds.items():
        per_seed = [run_one_seed(df, cols, args.n_boot, args.n_rand, args.seed + i)
                    for i, df in enumerate(dfs)]
        print_report(name, per_seed)
        if len(per_seed) == 1:
            r = per_seed[0]
            for mname in METRICS:
                m = r[mname]
                summary_rows.append(dict(
                    dataset=name, metric=mname, n_seeds=1,
                    A=m["A"], B=m["B"], delta=round(m["bootstrap"]["delta"], 3),
                    ci_low=round(m["bootstrap"]["ci_low"], 3),
                    ci_high=round(m["bootstrap"]["ci_high"], 3),
                    boot_p=round(m["bootstrap"]["p_value"], 5),
                    rand_p=round(m["randomization"]["p_value"], 5)))
        else:
            agg = aggregate_seeds(per_seed)
            for mname, a in agg.items():
                summary_rows.append(dict(
                    dataset=name, metric=mname, n_seeds=a["n_seeds"],
                    A=f"{a['A_mean']}±{a['A_std']}", B=f"{a['B_mean']}±{a['B_std']}",
                    delta=f"{a['delta_mean']}±{a['delta_std']}",
                    ci_low="", ci_high="",
                    boot_p=a["boot_p_mean"], rand_p=a["rand_p_mean"]))

    out = pd.DataFrame(summary_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[saved] {args.out_csv}  ({len(out)} rows)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_csv", nargs="+", required=True,
                   help="per-utterance pred CSV(s); multiple = seeds")
    p.add_argument("--out_csv", default="scripts/analysis/significance_results.csv")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--n_rand", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gold_col", default=DEFAULT_COLS["gold"])
    p.add_argument("--model_a_col", default=DEFAULT_COLS["modelA"], help="baseline model column")
    p.add_argument("--model_b_col", default=DEFAULT_COLS["modelB"], help="proposed model column")
    p.add_argument("--dataset_col", default=DEFAULT_COLS["dataset"])
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
