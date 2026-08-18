"""
Evaluate a trained Stage 1B (HMF) fusion checkpoint on the test split,
producing per-class F1 + weighted F1 + macro F1 + confusion matrix.

Usage (K-MIND, kfdb):
    python K-MIND/stage1b_per_class_eval.py \\
        --cache_root         ./K-MIND/cache_total_multimodal_rowwise_kfdb \\
        --fusion_visual_ckpt ./K-MIND/save_model/0325_fusion_visual_grid/fv_lr1e-5_bs4_dp0.2_h8_b0.2_hs1280_seed42.bin \\
        --fusion_ckpt        ./K-MIND/save_model/0421_fusion_kfdb/fusion_lr1e-5_bs4_dp0.1_h8_b0.3_hs1280_seed42.bin \\
        --clsNum 7 \\
        --fv_hidden_size 1280 --fv_beta_shift 0.2 --fv_dropout_prob 0.2 --fv_num_head 8 \\
        --hidden_size 1280    --beta_shift 0.3    --dropout_prob 0.1    --num_head 8 \\
        --split test \\
        --batch_size 8

Outputs printed:
  - per-class F1 (%) for each of the 7 K-MIND emotions (paper label order)
  - weighted F1 (%)
  - macro F1 (%)
  - confusion matrix (rows=true, cols=pred)
"""
import argparse
import glob
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)

import torch
from torch.utils.data import Dataset, DataLoader

from teacher_audio_model import ASF, ASF_visual


# Code label order used when the cache was built
#   idx 0..6 = [기쁨, 슬픔, 화남, 놀람, 무서움, 혐오, 중립]
#            = [joy,  sad, ang,  sur,  fear,  dis, neu]
CODE_ORDER = ["joy", "sad", "ang", "sur", "fear", "dis", "neu"]

# Paper label order used in Table 6 and discussion
PAPER_ORDER = ["joy", "sur", "fear", "sad", "neu", "dis", "ang"]


class FinalMultimodalCacheDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not self.files:
            raise RuntimeError(f"No .pt files found in: {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj = torch.load(self.files[idx], map_location="cpu")
        required = ["text", "audio", "v2", "v1", "y"]
        for k in required:
            if k not in obj:
                raise KeyError(f"Missing key '{k}' in cache file: {self.files[idx]}")
        return obj["text"], obj["audio"], obj["v2"], obj["v1"], obj["y"]


def stack_collate(batch):
    text, audio, visual2, visual, y = zip(*batch)
    text = torch.stack(text, 0)
    audio = torch.stack(audio, 0)
    visual2 = torch.stack(visual2, 0)
    visual = torch.stack(visual, 0)
    y = torch.stack([yy.view(-1)[0] for yy in y], 0).long()
    return text, audio, visual2, visual, y


@torch.no_grad()
def evaluate(fusion_visual, fusion, dataloader):
    # NOTE: teacher_audio_model.ASF/ASF_visual has hard-coded .cuda()
    # inside forward(); GPU is required.
    fusion_visual.eval()
    fusion.eval()

    preds = []
    labels = []

    for data in dataloader:
        if data is None:
            continue
        text_h, audio_h, v2_h, v1_h, y = data
        text_h = text_h.cuda(non_blocking=True).float()
        audio_h = audio_h.cuda(non_blocking=True).float()
        v2_h = v2_h.cuda(non_blocking=True).float()
        v1_h = v1_h.cuda(non_blocking=True).float()

        pred_hidden, _ = fusion_visual(audio_h, v2_h, v1_h)
        _, pred_logits_all = fusion(audio_h, pred_hidden, text_h)
        pred = pred_logits_all.argmax(1).detach().cpu().numpy()

        preds.append(pred)
        labels.append(y.numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--fusion_visual_ckpt", required=True)
    ap.add_argument("--fusion_ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--num_workers", default=2, type=int)
    ap.add_argument("--clsNum", default=7, type=int)

    ap.add_argument("--fv_hidden_size", default=1280, type=int)
    ap.add_argument("--fv_beta_shift", default=0.2, type=float)
    ap.add_argument("--fv_dropout_prob", default=0.2, type=float)
    ap.add_argument("--fv_num_head", default=8, type=int)

    ap.add_argument("--hidden_size", default=1280, type=int)
    ap.add_argument("--beta_shift", default=0.3, type=float)
    ap.add_argument("--dropout_prob", default=0.1, type=float)
    ap.add_argument("--num_head", default=8, type=int)
    args = ap.parse_args()

    # Build models
    fv = ASF_visual(
        args.clsNum, args.fv_hidden_size,
        beta_shift=args.fv_beta_shift,
        dropout_prob=args.fv_dropout_prob,
        num_head=args.fv_num_head,
    )
    fv.load_state_dict(torch.load(args.fusion_visual_ckpt, map_location="cpu"))
    fv = fv.cuda()
    fv.eval()

    fu = ASF(
        args.clsNum, args.hidden_size,
        args.beta_shift, args.dropout_prob, args.num_head,
    )
    fu.load_state_dict(torch.load(args.fusion_ckpt, map_location="cpu"))
    fu = fu.cuda()
    fu.eval()

    # Dataloader
    cache_dir = os.path.join(args.cache_root, args.split)
    dataset = FinalMultimodalCacheDataset(cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stack_collate,
    )
    print(f"[info] {args.split} samples: {len(dataset)}")

    preds, labels = evaluate(fv, fu, loader)

    # Overall metrics
    w_p, w_r, w_f, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    macro_f = f1_score(labels, preds, average="macro", zero_division=0)

    # Per-class F1 (by code index)
    per_class_f = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(args.clsNum)))
    per_class_f_pct = (per_class_f * 100).round(2)

    # Confusion matrix in code order
    cm = confusion_matrix(labels, preds, labels=list(range(args.clsNum)))

    # Sample counts per class
    counts = np.bincount(labels, minlength=args.clsNum)

    print("\n===== Per-class F1 (code order: joy, sad, ang, sur, fear, dis, neu) =====")
    for i, name in enumerate(CODE_ORDER):
        print(f"  [{i}] {name:5s}  n={int(counts[i])}  F1={per_class_f_pct[i]:.2f}")

    print("\n===== Per-class F1 (paper order) =====")
    code2idx = {n: i for i, n in enumerate(CODE_ORDER)}
    paper_f1 = [per_class_f_pct[code2idx[n]] for n in PAPER_ORDER]
    header = "  " + "  ".join(f"{n:>5s}" for n in PAPER_ORDER)
    values = "  " + "  ".join(f"{v:>5.2f}" for v in paper_f1)
    print(header)
    print(values)

    print("\n===== Summary =====")
    print(f"  weighted F1 = {w_f * 100:.2f}")
    print(f"  macro F1    = {macro_f * 100:.2f}")
    print(f"  weighted P  = {w_p * 100:.2f}")
    print(f"  weighted R  = {w_r * 100:.2f}")

    print("\n===== Confusion matrix (rows=true, cols=pred, code order) =====")
    print(cm)


if __name__ == "__main__":
    main()
