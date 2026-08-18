"""Similar-emotion pair confusion analysis (AlignEMO, CMC revision).

For a set of emotion pairs (A, B), measures the *cross-confusion rate*:

    rate = ( #(gold=A, pred=B) + #(gold=B, pred=A) ) / ( #(gold=A) + #(gold=B) )

computed for Stage 1 (HMF only) and Stage 1+2 (ELACL), so you can see whether
ELACL reduces confusion between similar emotions.

Pair selection
--------------
- IEMOCAP: fixed similar pairs (Happy<->Excited), (Anger<->Frustrated) by default
  (label values 'hap'/'exc', 'ang'/'fru'); override with --iemocap_pairs.
- MELD / K-MIND (any non-IEMOCAP dataset): the top-``--topk`` most-confused pairs
  are auto-selected from the **Stage 1** confusion matrix (largest off-diagonal
  mass  cm[A,B] + cm[B,A]).

Input
-----
Per-utterance prediction CSV(s) (``--pred_csv``) with columns (configurable):
    gold, pred_stage1, pred_stage2   (+ optional ``dataset`` column)
Dataset name comes from the ``dataset`` column, else the file stem. A dataset is
treated as IEMOCAP if its name contains "iemocap" (case-insensitive).

Output: console table per dataset + tidy CSV (--out_csv).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


IEMOCAP_DEFAULT_PAIRS = [("hap", "exc"), ("ang", "fru")]
# IEMOCAP dataset emoList order (label index == position); used to map the
# name-based pairs above onto integer gold labels when needed.
IEMOCAP_EMOLIST = ["ang", "exc", "fru", "hap", "neu", "sad"]

DEFAULT_COLS = dict(gold="gold", stage1="pred_stage1", stage2="pred_stage2", dataset="dataset")


def all_labels(df, *cols):
    labs = set()
    for c in cols:
        labs |= set(df[c].dropna().unique().tolist())
    return sorted(labs, key=str)


def top_confused_pairs(gold, pred, labels, k):
    cm = confusion_matrix(gold, pred, labels=labels)
    scored = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            scored.append(((labels[i], labels[j]), int(cm[i, j] + cm[j, i])))
    scored.sort(key=lambda x: -x[1])
    return [pair for pair, mass in scored[:k] if mass > 0]


def cross_confusion(gold, pred, A, B):
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    nA = int((gold == A).sum())
    nB = int((gold == B).sum())
    ab = int(((gold == A) & (pred == B)).sum())   # A mistaken as B
    ba = int(((gold == B) & (pred == A)).sum())   # B mistaken as A
    denom = nA + nB
    rate = (100.0 * (ab + ba) / denom) if denom else float("nan")
    return dict(rate=rate, ab=ab, ba=ba, nA=nA, nB=nB)


def analyze_dataset(df, cols, is_iemocap, iemocap_pairs, topk):
    gold = df[cols["gold"]].to_numpy()
    p1 = df[cols["stage1"]].to_numpy()
    has_s2 = cols["stage2"] in df.columns
    p2 = df[cols["stage2"]].to_numpy() if has_s2 else None
    lab_cols = [cols["gold"], cols["stage1"]] + ([cols["stage2"]] if has_s2 else [])
    labels = all_labels(df, *lab_cols)

    if is_iemocap:
        pairs = [tuple(p) for p in iemocap_pairs]
        # gold may be integer indices (from the cache) rather than emotion names;
        # map name-based pairs onto indices via the IEMOCAP emoList order.
        if np.issubdtype(np.asarray(gold).dtype, np.number):
            idx = {e: i for i, e in enumerate(IEMOCAP_EMOLIST)}
            pairs = [(idx.get(A, A), idx.get(B, B)) for A, B in pairs]
        pair_src = "fixed (IEMOCAP)"
    else:
        pairs = top_confused_pairs(gold, p1, labels, topk)
        pair_src = f"auto top-{topk} from Stage1 CM"

    rows = []
    for (A, B) in pairs:
        c1 = cross_confusion(gold, p1, A, B)
        row = dict(
            pair=f"{A}↔{B}",
            n=c1["nA"] + c1["nB"],
            stage1_confusion_pct=round(c1["rate"], 2),
            s1_AtoB=c1["ab"], s1_BtoA=c1["ba"],
        )
        if has_s2:
            c2 = cross_confusion(gold, p2, A, B)
            row.update(
                stage2_confusion_pct=round(c2["rate"], 2),
                delta_pct=round(c2["rate"] - c1["rate"], 2),  # negative = ELACL reduced confusion
                s2_AtoB=c2["ab"], s2_BtoA=c2["ba"],
            )
        rows.append(row)
    return rows, pair_src


def print_table(name, rows, pair_src):
    print(f"\n=== {name}  (pairs: {pair_src}) ===")
    has_s2 = bool(rows) and "stage2_confusion_pct" in rows[0]
    if has_s2:
        print(f"{'pair':<12}{'n':>6}{'S1 conf%':>10}{'S2 conf%':>10}{'Δ (S2-S1)':>12}")
        print("-" * 50)
        for r in rows:
            arrow = "  ↓ reduced" if r["delta_pct"] < 0 else ("  ↑ worse" if r["delta_pct"] > 0 else "")
            print(f"{r['pair']:<12}{r['n']:>6}{r['stage1_confusion_pct']:>10}"
                  f"{r['stage2_confusion_pct']:>10}{r['delta_pct']:>12}{arrow}")
    else:
        print(f"{'pair':<12}{'n':>6}{'S1 conf%':>10}   (pred_stage2 absent — Stage1 only)")
        print("-" * 40)
        for r in rows:
            print(f"{r['pair']:<12}{r['n']:>6}{r['stage1_confusion_pct']:>10}")


def load_frames(pred_csvs, cols):
    for path in pred_csvs:
        df = pd.read_csv(path)
        for req in ("gold", "stage1"):
            if cols[req] not in df.columns:
                raise ValueError(f"{path}: missing column '{cols[req]}'. Have {list(df.columns)}")
        if cols["dataset"] in df.columns:
            for name, sub in df.groupby(cols["dataset"], sort=False):
                yield str(name), sub.reset_index(drop=True)
        else:
            yield Path(path).stem, df.reset_index(drop=True)


def parse_pairs(s):
    # "hap:exc,ang:fru" -> [("hap","exc"),("ang","fru")]
    out = []
    for chunk in s.split(","):
        a, b = chunk.split(":")
        out.append((a.strip(), b.strip()))
    return out


def main(args):
    cols = dict(DEFAULT_COLS, gold=args.gold_col, stage1=args.stage1_col,
                stage2=args.stage2_col, dataset=args.dataset_col)
    iemo_pairs = parse_pairs(args.iemocap_pairs) if args.iemocap_pairs else IEMOCAP_DEFAULT_PAIRS

    all_rows = []
    for name, df in load_frames(args.pred_csv, cols):
        is_iemocap = "iemocap" in name.lower()
        rows, pair_src = analyze_dataset(df, cols, is_iemocap, iemo_pairs, args.topk)
        print_table(name, rows, pair_src)
        for r in rows:
            all_rows.append(dict(dataset=name, pair_source=pair_src, **r))

    out = pd.DataFrame(all_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[saved] {args.out_csv}  ({len(out)} rows)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_csv", nargs="+", required=True)
    p.add_argument("--out_csv", default="scripts/analysis/pair_confusion_results.csv")
    p.add_argument("--iemocap_pairs", default=None,
                   help='override IEMOCAP pairs, e.g. "hap:exc,ang:fru"')
    p.add_argument("--topk", type=int, default=2,
                   help="number of auto-selected pairs for non-IEMOCAP datasets")
    p.add_argument("--gold_col", default=DEFAULT_COLS["gold"])
    p.add_argument("--stage1_col", default=DEFAULT_COLS["stage1"])
    p.add_argument("--stage2_col", default=DEFAULT_COLS["stage2"])
    p.add_argument("--dataset_col", default=DEFAULT_COLS["dataset"])
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
