"""Emotion-shift error analysis for AlignEMO (CMC revision).

Splits the test set into *shift* / *no-shift* utterances and reports
weighted-F1 and macro-F1 for each subset, comparing Stage 1 (HMF only) vs
Stage 1+2 (ELACL).

Shift definition
----------------
An utterance is a **shift** iff its GOLD label differs from the gold label of
the *previous utterance by the same speaker within the same dialogue*.
An utterance with no previous same-speaker utterance (that speaker's first turn
in the dialogue — which includes the dialogue's opening turn) is **no-shift**.
(Shift is a property of the ground-truth label sequence, not of predictions.)

Input
-----
One or more per-utterance prediction CSVs (``--pred_csv``), each with columns
(names configurable via flags):

    dialogue_id, speaker, turn_idx, gold, pred_stage1, pred_stage2

- ``turn_idx`` is optional; if absent, the row order within the file is used
  (the CSV is assumed to list each dialogue's utterances in conversation order).
- An optional ``dataset`` column groups rows by dataset; otherwise the file stem
  is used as the dataset name.
- Labels may be strings or ints (compared as-is).

NOTE: the training/eval scripts in this repo do not yet dump per-utterance
predictions. Add that dump (e.g. in IEMOCAP/eval_stage12.py and
K-MIND/stage1b_per_class_eval.py) to produce these CSVs. Run with ``--selftest``
to verify the shift logic without any data.

Output
------
- A console table per dataset (shift/no-shift x Stage1/Stage2: n, w-F1, macro-F1).
- A tidy CSV (``--out_csv``) with columns:
  dataset, subset, model, n, weighted_f1, macro_f1, shift_ratio.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DEFAULT_COLS = dict(
    dialogue="dialogue_id",
    speaker="speaker",
    turn="turn_idx",
    gold="gold",
    stage1="pred_stage1",
    stage2="pred_stage2",
    dataset="dataset",
)


# ---------------------------------------------------------------------------
# Core shift logic
# ---------------------------------------------------------------------------
def compute_shift_flags(df, dialogue_col, speaker_col, turn_col, gold_col):
    """Boolean np.ndarray aligned to df.index: True = shift.

    A row is a shift iff its gold label differs from the previous row of the
    same (dialogue, speaker) in turn order. A speaker's first turn = no-shift.
    Flags are returned in the ORIGINAL row order of ``df``.
    """
    if turn_col in df.columns:
        order = df.sort_values([dialogue_col, turn_col], kind="mergesort").index
    else:
        # No explicit turn index: assume file order == conversation order.
        order = df.index

    prev_gold = {}
    flag = {}
    for idx in order:
        key = (df.at[idx, dialogue_col], df.at[idx, speaker_col])
        g = df.at[idx, gold_col]
        if key not in prev_gold:
            flag[idx] = False            # first same-speaker turn -> no-shift
        else:
            flag[idx] = bool(g != prev_gold[key])
        prev_gold[key] = g
    return np.array([flag[i] for i in df.index], dtype=bool)


def subset_metrics(gold, pred):
    if len(gold) == 0:
        return dict(n=0, weighted_f1=float("nan"), macro_f1=float("nan"))
    return dict(
        n=int(len(gold)),
        weighted_f1=round(f1_score(gold, pred, average="weighted", zero_division=0) * 100, 2),
        macro_f1=round(f1_score(gold, pred, average="macro", zero_division=0) * 100, 2),
    )


def analyze_dataset(df, cols):
    """Return (list_of_result_rows, shift_ratio) for one dataset's df."""
    is_shift = compute_shift_flags(
        df, cols["dialogue"], cols["speaker"], cols["turn"], cols["gold"]
    )
    shift_ratio = round(100 * is_shift.mean(), 2) if len(is_shift) else float("nan")

    stages = [("Stage1 (HMF)", cols["stage1"]), ("Stage1+2 (ELACL)", cols["stage2"])]
    rows = []
    for subset_name, mask in [("shift", is_shift), ("no-shift", ~is_shift)]:
        sub = df[mask]
        for stage_name, pred_col in stages:
            if pred_col not in df.columns:
                continue
            m = subset_metrics(sub[cols["gold"]].to_numpy(), sub[pred_col].to_numpy())
            rows.append(dict(subset=subset_name, model=stage_name, **m))
    return rows, shift_ratio


# ---------------------------------------------------------------------------
# I/O + reporting
# ---------------------------------------------------------------------------
def load_frames(pred_csvs, cols):
    """Yield (dataset_name, df) for each dataset across the given CSVs."""
    for path in pred_csvs:
        df = pd.read_csv(path)
        missing = [cols[k] for k in ("dialogue", "speaker", "gold") if cols[k] not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}. "
                             f"Have: {list(df.columns)}")
        if cols["dataset"] in df.columns:
            for name, sub in df.groupby(cols["dataset"], sort=False):
                yield str(name), sub.reset_index(drop=True)
        else:
            yield Path(path).stem, df.reset_index(drop=True)


def print_table(dataset_name, rows, shift_ratio):
    print(f"\n=== {dataset_name}  (shift ratio: {shift_ratio}%  of test utterances) ===")
    print(f"{'subset':<9}{'model':<20}{'n':>7}{'w-F1':>9}{'macro-F1':>10}")
    print("-" * 55)
    for r in rows:
        print(f"{r['subset']:<9}{r['model']:<20}{r['n']:>7}"
              f"{r['weighted_f1']:>9}{r['macro_f1']:>10}")


def main(args):
    cols = dict(DEFAULT_COLS)
    cols.update(dialogue=args.dialogue_col, speaker=args.speaker_col,
                turn=args.turn_col, gold=args.gold_col,
                stage1=args.stage1_col, stage2=args.stage2_col,
                dataset=args.dataset_col)

    all_rows = []
    for name, df in load_frames(args.pred_csv, cols):
        rows, shift_ratio = analyze_dataset(df, cols)
        print_table(name, rows, shift_ratio)
        for r in rows:
            all_rows.append(dict(dataset=name, shift_ratio=shift_ratio, **r))

    out_df = pd.DataFrame(all_rows, columns=[
        "dataset", "subset", "model", "n", "weighted_f1", "macro_f1", "shift_ratio"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\n[saved] {args.out_csv}  ({len(out_df)} rows)")


# ---------------------------------------------------------------------------
# Unit test (toy example)
# ---------------------------------------------------------------------------
def selftest():
    # (dialogue, speaker, turn, gold, expected_shift)
    toy = pd.DataFrame(
        [
            ("d1", "A", 0, "hap", False),  # A's 1st turn -> no-shift
            ("d1", "B", 1, "sad", False),  # B's 1st turn -> no-shift
            ("d1", "A", 2, "hap", False),  # A: hap -> hap = no-shift
            ("d1", "A", 3, "ang", True),   # A: hap -> ang = shift
            ("d1", "B", 4, "sad", False),  # B: sad -> sad = no-shift
            ("d1", "B", 5, "hap", True),   # B: sad -> hap = shift
            ("d2", "A", 0, "neu", False),  # new dialogue, A's 1st turn -> no-shift
        ],
        columns=["dialogue_id", "speaker", "turn_idx", "gold", "expected"],
    )
    flags = compute_shift_flags(toy, "dialogue_id", "speaker", "turn_idx", "gold")
    exp = toy["expected"].to_numpy()
    print("in-order   flags:", list(flags.astype(int)))
    print("in-order   expect:", list(exp.astype(int)))
    assert (flags == exp).all(), "shift flags mismatch (in-order)"

    # order-independence: shuffle rows, flags must still match each row's expectation
    shuf = toy.sample(frac=1.0, random_state=7).reset_index(drop=True)
    flags_s = compute_shift_flags(shuf, "dialogue_id", "speaker", "turn_idx", "gold")
    assert (flags_s == shuf["expected"].to_numpy()).all(), "shift flags mismatch (shuffled)"
    print("shuffled   flags:", list(flags_s.astype(int)))
    print("shuffled   expect:", list(shuf["expected"].to_numpy().astype(int)))

    # metrics sanity: perfect predictions -> 100 F1, on a subset
    g = np.array(["a", "b", "a", "c"])
    assert subset_metrics(g, g)["weighted_f1"] == 100.0
    assert subset_metrics(np.array([]), np.array([]))["n"] == 0

    print("\nSELFTEST PASSED ✅  (shift logic order-independent; metrics OK)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_csv", nargs="+",
                   help="one or more per-utterance prediction CSVs")
    p.add_argument("--out_csv", default="scripts/analysis/emotion_shift_results.csv")
    p.add_argument("--dialogue_col", default=DEFAULT_COLS["dialogue"])
    p.add_argument("--speaker_col", default=DEFAULT_COLS["speaker"])
    p.add_argument("--turn_col", default=DEFAULT_COLS["turn"])
    p.add_argument("--gold_col", default=DEFAULT_COLS["gold"])
    p.add_argument("--stage1_col", default=DEFAULT_COLS["stage1"])
    p.add_argument("--stage2_col", default=DEFAULT_COLS["stage2"])
    p.add_argument("--dataset_col", default=DEFAULT_COLS["dataset"])
    p.add_argument("--selftest", action="store_true",
                   help="run the toy unit test for the shift logic and exit")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        sys.exit(0)
    if not args.pred_csv:
        build_parser().error("--pred_csv is required (or use --selftest)")
    main(args)
