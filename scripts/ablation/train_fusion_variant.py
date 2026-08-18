"""Train one fusion variant on frozen AlignEMO encoders (CMC revision, M4).

Loads the frozen Stage-1A encoders + the (frozen) visual fusion, then trains ONLY
the final fusion module (variant selectable) and reports test weighted-F1 and
macro-F1. Encoders are shared across variants for a fair comparison.

    python scripts/ablation/train_fusion_variant.py --variant concat --seed 42 \
        --teacher_ckpt ... --audio_ckpt ... --visual_ckpt ... \
        --visual_masked_ckpt ... --fusion_visual_ckpt ...
"""
import argparse
import copy
import os
import sys
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "IEMOCAP"))
sys.path.insert(0, str(REPO / "scripts" / "ablation"))

from preprocessing import preprocessing_pickle          # noqa: E402
from dataset import iemocap_dataset                      # noqa: E402
from utils import make_batchs_pickle                     # noqa: E402
from model import Teacher_model, Student_Audio, Student_Video, ASF_visual  # noqa: E402
from fusion_variants import build_fusion                 # noqa: E402


def seed_everything(seed):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    mask_time_length: int = 3


@torch.no_grad()
def encode(batch, model_t, audio_s, video_s, video_s_masked, fusion_visual, device):
    bi, am, ai, vi, vi2, bl = [x.to(device) for x in batch]
    th, _ = model_t(bi, am)
    ah, _ = audio_s(ai)
    vh, _ = video_s(vi)
    vh2, _ = video_s_masked(vi2)
    pvh, _ = fusion_visual(text_embedding=th, visual=vh2, visual_2=vh)
    return th, pvh, ah, bl


@torch.no_grad()
def evaluate(loader, encoders, fusion, device):
    fusion.eval()
    preds, golds = [], []
    for batch in loader:
        if batch is None:
            continue
        th, pvh, ah, bl = encode(batch, *encoders, device)
        _, logits = fusion(text_embedding=th, visual=pvh, acoustic=ah)
        preds.append(logits.argmax(1).cpu().numpy()); golds.append(bl.cpu().numpy())
    preds = np.concatenate(preds); golds = np.concatenate(golds)
    wf1 = f1_score(golds, preds, average="weighted", zero_division=0) * 100
    mf1 = f1_score(golds, preds, average="macro", zero_division=0) * 100
    return wf1, mf1, preds, golds


def reconstruct_meta(sessions, emo_list):
    """Per-sample (dialogue_id, turn_idx, speaker, gold, gold_name) from the
    growing-context session list — aligned 1:1 with the shuffle=False loader."""
    rows, dlg = [], -1
    for sess in sessions:
        if len(sess) == 1:
            dlg += 1
        target = sess[-1]
        rows.append(dict(
            dialogue_id=f"dlg{dlg:04d}", turn_idx=len(sess) - 1,
            speaker=int(target[0]), gold_name=str(target[6]),
            gold=emo_list.index(target[6]) if target[6] in emo_list else -1))
    return rows


def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_config = Config()

    dp = args.data_path
    train_ds = iemocap_dataset(preprocessing_pickle(dp + args.train_csv, dp + args.pickle))
    dev_ds = iemocap_dataset(preprocessing_pickle(dp + args.dev_csv, dp + args.pickle))
    test_sessions = preprocessing_pickle(dp + args.test_csv, dp + args.pickle)
    test_ds = iemocap_dataset(test_sessions)
    clsNum = len(train_ds.emoList)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=make_batchs_pickle)
    dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=make_batchs_pickle)
    tl_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=make_batchs_pickle)

    def frozen(m, ck, strict=True):
        m.load_state_dict(torch.load(ck, map_location="cpu"), strict=strict)
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    model_t = frozen(Teacher_model(args.text_model, clsNum), args.teacher_ckpt, strict=False)
    audio_s = frozen(Student_Audio(args.audio_model, clsNum, init_config), args.audio_ckpt)
    video_s = frozen(Student_Video(args.video_model, clsNum), args.visual_ckpt)
    video_s_masked = frozen(Student_Video(args.video_model, clsNum), args.visual_masked_ckpt)
    fusion_visual = frozen(ASF_visual(clsNum, 768, 2e-1, 0.2, 4), args.fusion_visual_ckpt)
    encoders = (model_t, audio_s, video_s, video_s_masked, fusion_visual)

    fusion = build_fusion(args.variant, clsNum, hidden_size=768,
                          beta_shift=2e-1, dropout_prob=0.2, num_head=4).to(device)
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss()

    best_wf1, best_mf1 = 0.0, 0.0
    best_state = copy.deepcopy(fusion.state_dict())
    for epoch in range(args.epochs):
        fusion.train()
        for batch in tl:
            if batch is None:
                continue
            th, pvh, ah, bl = encode(batch, *encoders, device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                _, logits = fusion(text_embedding=th, visual=pvh, acoustic=ah)
                loss = ce(logits, bl)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        wf1, mf1, _, _ = evaluate(dl, encoders, fusion, device)
        print(f"[{args.variant}][seed {args.seed}] epoch {epoch+1}: validation w-F1={wf1:.2f} macro-F1={mf1:.2f}")
        if wf1 > best_wf1:
            best_wf1, best_mf1 = wf1, mf1
            best_state = copy.deepcopy(fusion.state_dict())

    # Select exclusively on validation, then evaluate test once.
    fusion.load_state_dict(best_state)
    test_wf1, test_mf1, test_preds, test_golds = evaluate(tl_test, encoders, fusion, device)
    print(f"[{args.variant}][seed {args.seed}] validation-best test w-F1={test_wf1:.2f} macro-F1={test_mf1:.2f}")

    # per-utterance dump of the validation-best model, for shift analysis
    if args.dump_csv:
        meta = reconstruct_meta(test_sessions, test_ds.emoList)
        n = min(len(meta), len(test_preds))
        if len(meta) != len(test_preds):
            print(f"[warn] meta={len(meta)} preds={len(test_preds)}; truncating to {n}")
        drows = [dict(dataset="IEMOCAP", id=f"utt{i:05d}", variant=args.variant,
                      seed=args.seed, **meta[i], pred_stage1=int(test_preds[i]))
                 for i in range(n)]
        Path(args.dump_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(drows).to_csv(args.dump_csv, index=False)
        print(f"[dumped] {args.dump_csv}  rows={n}")

    row = dict(dataset="IEMOCAP", variant=args.variant, seed=args.seed,
               weighted_f1=round(test_wf1, 2), macro_f1=round(test_mf1, 2),
               selection_split="validation", selection_metric="weighted_f1")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    hdr = not os.path.exists(args.out_csv)
    pd.DataFrame([row]).to_csv(args.out_csv, mode="a", header=hdr, index=False)
    print(f"[appended] {args.out_csv}: {row}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=["concat", "crossattn", "hmf_nogate", "hmf", "shiftaware"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--data_path", default="./dataset/IEMOCAP_full_release/")
    p.add_argument("--train_csv", default="IEMOCAP_train_2.csv")
    p.add_argument("--dev_csv", default="IEMOCAP_dev_2.csv")
    p.add_argument("--test_csv", default="IEMOCAP_test_2.csv")
    p.add_argument("--pickle", default="faces_data_627.pkl")
    p.add_argument("--text_model", default="roberta-large")
    p.add_argument("--audio_model", default="facebook/data2vec-audio-base-960h")
    p.add_argument("--video_model", default="facebook/timesformer-base-finetuned-k400")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--audio_ckpt", required=True)
    p.add_argument("--visual_ckpt", required=True)
    p.add_argument("--visual_masked_ckpt", required=True)
    p.add_argument("--fusion_visual_ckpt", required=True)
    p.add_argument("--out_csv", default="scripts/analysis/fusion_ablation_results.csv")
    p.add_argument("--dump_csv", default=None,
                   help="also dump per-utterance best-model predictions here (for shift analysis)")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
