"""Dump per-utterance test predictions for K-MIND (CMC revision — plumbing check).

Reads the K-MIND rowwise cache (same format as ELACL_fusion.py /
stage1b_per_class_eval.py) in deterministic sorted-file order, runs a fusion
checkpoint pair to get predictions, and joins them with each row's cached
``meta`` (dialogue / speaker / turn / gold) into a per-utterance CSV consumable
by emotion_shift.py / pair_confusion.py / significance.py.

This validates the prediction-dump plumbing on the dataset where per-utterance
metadata is already available (in the cache ``meta``). Run it twice with the two
model checkpoints and merge to obtain both ``pred_stage1`` and ``pred_stage2``:

  # Stage 1 (HMF only) fusion checkpoints:
  python scripts/analysis/dump_predictions_kmind.py \
      --cache_root ./K-MIND/cache_total_multimodal_rowwise_kfdb --split test \
      --fusion_visual_ckpt <fv.bin> --fusion_ckpt <fusion.bin> \
      --pred_col pred_stage1 --out_csv kmind_stage1.csv

  # Stage 1+ELACL fusion checkpoints (the ELACL-fine-tuned fusion pair):
  python scripts/analysis/dump_predictions_kmind.py ... \
      --fusion_visual_ckpt <fv_elacl.bin> --fusion_ckpt <fusion_elacl.bin> \
      --pred_col pred_stage2 --out_csv kmind_stage2.csv

  # merge on the utterance key:
  python scripts/analysis/dump_predictions_kmind.py --merge \
      kmind_stage1.csv kmind_stage2.csv --out_csv kmind_preds.csv
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "K-MIND"))


# label order used when the cache was built (see stage1b_per_class_eval.py)
CODE_ORDER = ["joy", "sad", "ang", "sur", "fear", "dis", "neu"]


class CacheWithMeta(Dataset):
    """Returns (text, audio, v2, v1, y) — mirrors stage1b_per_class_eval — and
    keeps the sorted file list so meta can be re-read in the same order."""
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not self.files:
            raise RuntimeError(f"No .pt files found in: {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        o = torch.load(self.files[idx], map_location="cpu")
        return o["text"], o["audio"], o["v2"], o["v1"], o["y"]


def stack_collate(batch):
    text, audio, v2, v1, y = zip(*batch)
    return (torch.stack(text, 0), torch.stack(audio, 0), torch.stack(v2, 0),
            torch.stack(v1, 0), torch.stack([yy.view(-1)[0] for yy in y], 0).long())


@torch.no_grad()
def predict(fusion_visual, fusion, loader):
    fusion_visual.eval(); fusion.eval()
    preds, golds = [], []
    for data in loader:
        text_h, audio_h, v2_h, v1_h, y = data
        text_h = text_h.cuda(non_blocking=True).float()
        audio_h = audio_h.cuda(non_blocking=True).float()
        v2_h = v2_h.cuda(non_blocking=True).float()
        v1_h = v1_h.cuda(non_blocking=True).float()
        pred_hidden, _ = fusion_visual(audio_h, v2_h, v1_h)          # K-MIND arg convention
        _, logits = fusion(audio_h, pred_hidden, text_h)
        preds.append(logits.argmax(1).cpu().numpy())
        golds.append(y.numpy())
    return np.concatenate(preds), np.concatenate(golds)


def parse_meta(path):
    """Extract (id, dialogue_id, speaker, turn_idx, gold_name) from a cache .pt."""
    o = torch.load(path, map_location="cpu")
    meta = o.get("meta", {}) if isinstance(o, dict) else {}
    row_key = o.get("id") or meta.get("row_key") or Path(path).stem
    wav = str(meta.get("wav_path", ""))
    # wav path e.g. .../KMIND/wavs/SET156_M003_S002/00225/audio.wav
    parts = [p for p in wav.split("/") if p]
    dialogue_id, turn_idx = "", ""
    for i, p in enumerate(parts):
        if p.upper().startswith("SET"):
            dialogue_id = p
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                turn_idx = int(parts[i + 1])
            break
    if turn_idx == "":
        turn_idx = int(o.get("row_index", meta.get("row_index", -1)))
    return dict(
        id=str(row_key),
        dialogue_id=dialogue_id or str(row_key),
        speaker=str(meta.get("speaker", "")),
        turn_idx=turn_idx,
        gold_name=str(meta.get("emotion", "")),
    )


def do_merge(paths, out_csv):
    base = pd.read_csv(paths[0])
    key = "id"
    for p in paths[1:]:
        df = pd.read_csv(p)
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        base = base.merge(df[[key] + pred_cols], on=key, how="inner")
    base.to_csv(out_csv, index=False)
    print(f"[merged] {out_csv}  rows={len(base)}  cols={list(base.columns)}")


def main(args):
    if args.merge:
        do_merge(args.merge, args.out_csv)
        return

    from teacher_audio_model import ASF, ASF_visual

    fv = ASF_visual(args.clsNum, args.fv_hidden_size, beta_shift=args.fv_beta_shift,
                    dropout_prob=args.fv_dropout_prob, num_head=args.fv_num_head)
    fv.load_state_dict(torch.load(args.fusion_visual_ckpt, map_location="cpu"))
    fv = fv.cuda().eval()

    fu = ASF(args.clsNum, args.hidden_size, args.beta_shift, args.dropout_prob, args.num_head)
    fu.load_state_dict(torch.load(args.fusion_ckpt, map_location="cpu"))
    fu = fu.cuda().eval()

    cache_dir = os.path.join(args.cache_root, args.split)
    ds = CacheWithMeta(cache_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=stack_collate)
    print(f"[info] {args.split}: {len(ds)} utterances")

    preds, golds = predict(fv, fu, loader)
    assert len(preds) == len(ds.files), "pred/file count mismatch (order must be preserved)"

    rows = []
    for i, path in enumerate(ds.files):
        m = parse_meta(path)
        rows.append(dict(dataset="K-MIND", **m, gold=int(golds[i]), **{args.pred_col: int(preds[i])}))
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}  rows={len(out)}  cols={list(out.columns)}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merge", nargs="+", help="merge these per-column CSVs on 'id' and exit")
    p.add_argument("--cache_root", default="./K-MIND/cache_total_multimodal_rowwise_kfdb")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--fusion_visual_ckpt")
    p.add_argument("--fusion_ckpt")
    p.add_argument("--pred_col", default="pred_stage1")
    p.add_argument("--out_csv", default="scripts/analysis/kmind_preds.csv")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--clsNum", type=int, default=7)
    p.add_argument("--fv_hidden_size", type=int, default=1280)
    p.add_argument("--fv_beta_shift", type=float, default=0.2)
    p.add_argument("--fv_dropout_prob", type=float, default=0.2)
    p.add_argument("--fv_num_head", type=int, default=8)
    p.add_argument("--hidden_size", type=int, default=1280)
    p.add_argument("--beta_shift", type=float, default=0.3)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    p.add_argument("--num_head", type=int, default=8)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
