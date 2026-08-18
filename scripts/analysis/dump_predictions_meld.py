"""Dump per-utterance Stage-1 (HMF) predictions for MELD (CMC revision).

Runs the trained MELD Stage-1 pipeline (text teacher + audio/visual students +
HMF fusion) on the MELD test set and writes a per-utterance CSV with
    dataset, id, dialogue_id, speaker, turn_idx, gold, gold_name, pred_stage1
so emotion_shift.py / pair_confusion.py can bucket MELD like IEMOCAP/K-MIND.
gold comes from the loader label (same space as pred); dialogue/speaker/turn are
reconstructed from the growing-context session list (aligned to the shuffle=False
loader). Point --*_ckpt at the MELD checkpoints (e.g. meld_gate_seed42/...).
"""
import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "MELD"))

from preprocessing import preprocessing_pickle          # noqa: E402
from dataset import meld_dataset                         # noqa: E402
from utils import make_batchs_pickle                     # noqa: E402
from model import Teacher_model, Student_Audio, Student_Video, ASF, ASF_visual  # noqa: E402


@dataclass
class Config:
    mask_time_length: int = 3


def reconstruct_meta(sessions, emo_list):
    """(dialogue_id, turn_idx, speaker, gold, gold_name) per sample; a new dialogue
    starts whenever the growing context resets to length 1. MELD utterance tuple =
    [speaker, utt, video_path, emotion, bbox]."""
    rows, dlg = [], -1
    for sess in sessions:
        if len(sess) == 1:
            dlg += 1
        t = sess[-1]
        rows.append(dict(dialogue_id=f"dlg{dlg:04d}", turn_idx=len(sess) - 1,
                         speaker=int(t[0]), gold_name=str(t[3]),
                         gold=emo_list.index(t[3]) if t[3] in emo_list else -1))
    return rows


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    sessions = preprocessing_pickle(args.test_csv, args.pickle_path)
    ds = meld_dataset(sessions)
    clsNum = len(ds.emoList)
    emo_list = ds.emoList
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=make_batchs_pickle)
    meta = reconstruct_meta(sessions, emo_list)

    def frozen(m, ck, strict=True):
        m.load_state_dict(torch.load(ck, map_location="cpu"), strict=strict)
        return m.to(device).eval()

    model_t = frozen(Teacher_model(args.text_model, clsNum), args.teacher_ckpt, strict=False)
    audio_s = frozen(Student_Audio(args.audio_model, clsNum, cfg), args.audio_ckpt)
    video_s = frozen(Student_Video(args.video_model, clsNum), args.visual_ckpt)
    video_sm = frozen(Student_Video(args.video_model, clsNum), args.visual_masked_ckpt)
    fv = frozen(ASF_visual(clsNum, 768, 2e-1, 0.2, 4), args.fusion_visual_ckpt)
    fusion = frozen(ASF(clsNum, 768, 2e-1, 0.2, 4), args.fusion_ckpt)

    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            bi, am, ai, vi, vi2, bl = [x.to(device) for x in batch]
            th, _ = model_t(bi, am); ah, _ = audio_s(ai)
            vh, _ = video_s(vi); vh2, _ = video_sm(vi2)
            pvh, _ = fv(th, vh2, vh)
            _, logits = fusion(text_embedding=th, visual=pvh, acoustic=ah)
            preds.append(logits.argmax(1).cpu().numpy()); golds.append(bl.cpu().numpy())
    preds = np.concatenate(preds); golds = np.concatenate(golds)

    n = min(len(meta), len(preds))
    mism = sum(1 for i in range(n) if meta[i]["gold"] != int(golds[i]))
    print(f"[dump] meta-gold vs loader-y mismatch: {mism}/{n} "
          f"({'using loader-y' if mism else 'aligned'})")
    rows = []
    for i in range(n):
        m = dict(meta[i]); m["gold"] = int(golds[i])
        m["gold_name"] = emo_list[int(golds[i])] if 0 <= int(golds[i]) < len(emo_list) else str(int(golds[i]))
        rows.append(dict(dataset="MELD", id=f"utt{i:05d}", **m, pred_stage1=int(preds[i])))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[dumped] {args.out_csv}  rows={n}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_csv", default="./dataset/MELD.Raw/test_meld_emo.csv")
    p.add_argument("--pickle_path", default="./dataset/MELD.Raw/test_faces_data.pkl")
    p.add_argument("--out_csv", default="scripts/analysis/meld_preds.csv")
    p.add_argument("--text_model", default="roberta-large")
    p.add_argument("--audio_model", default="facebook/data2vec-audio-base-960h")
    p.add_argument("--video_model", default="facebook/timesformer-base-finetuned-k400")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--audio_ckpt", required=True)
    p.add_argument("--visual_ckpt", required=True)
    p.add_argument("--visual_masked_ckpt", required=True)
    p.add_argument("--fusion_visual_ckpt", required=True)
    p.add_argument("--fusion_ckpt", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
