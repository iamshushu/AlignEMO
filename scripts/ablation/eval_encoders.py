"""Standalone per-modality encoder F1 on IEMOCAP test (CMC revision, gap diagnosis).

Loads a set of unimodal encoder checkpoints (text teacher, audio/video students)
and reports each one's own test weighted-F1 (argmax of its classifier head). Use
it to compare the REPORTED encoders (which fuse to ~70) against the from-scratch
gate encoders (teacher 63 / audio 47 / video 20) and localize the 60-vs-70 gap.

    python scripts/ablation/eval_encoders.py --tag reported \
        --teacher_ckpt ... --audio_ckpt ... --visual_ckpt ... --visual_masked_ckpt ...
"""
import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "IEMOCAP"))

from preprocessing import preprocessing_pickle          # noqa: E402
from dataset import iemocap_dataset                      # noqa: E402
from utils import make_batchs_pickle                     # noqa: E402
from model import Teacher_model, Student_Audio, Student_Video  # noqa: E402


@dataclass
class Config:
    mask_time_length: int = 3


@torch.no_grad()
def eval_modality(kind, model, loader, device):
    preds, golds = [], []
    for batch in loader:
        if batch is None:
            continue
        bi, am, ai, vi, vi2, bl = [x.to(device) for x in batch]
        if kind == "text":
            _, logits = model(bi, am)
        elif kind == "audio":
            _, logits = model(ai)
        elif kind == "video":
            _, logits = model(vi)
        elif kind == "video_masked":
            _, logits = model(vi2)
        preds.append(logits.argmax(1).cpu().numpy()); golds.append(bl.cpu().numpy())
    preds = np.concatenate(preds); golds = np.concatenate(golds)
    return f1_score(golds, preds, average="weighted", zero_division=0) * 100


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    ds = iemocap_dataset(preprocessing_pickle(args.test_csv, args.pickle))
    clsNum = len(ds.emoList)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=make_batchs_pickle)

    def load(m, ck):
        m.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
        return m.to(device).eval()

    results = {}
    if args.teacher_ckpt:
        results["text"] = eval_modality("text", load(Teacher_model(args.text_model, clsNum), args.teacher_ckpt), loader, device)
    if args.audio_ckpt:
        results["audio"] = eval_modality("audio", load(Student_Audio(args.audio_model, clsNum, cfg), args.audio_ckpt), loader, device)
    if args.visual_ckpt:
        results["video"] = eval_modality("video", load(Student_Video(args.video_model, clsNum), args.visual_ckpt), loader, device)
    if args.visual_masked_ckpt:
        results["video_masked"] = eval_modality("video_masked", load(Student_Video(args.video_model, clsNum), args.visual_masked_ckpt), loader, device)

    print(f"\n=== [{args.tag}] per-modality test w-F1 (IEMOCAP) ===")
    for k, v in results.items():
        print(f"  {k:<14} {v:.2f}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="reported")
    p.add_argument("--test_csv", default="./dataset/IEMOCAP_full_release/IEMOCAP_test_2.csv")
    p.add_argument("--pickle", default="./dataset/IEMOCAP_full_release/faces_data_627.pkl")
    p.add_argument("--text_model", default="roberta-large")
    p.add_argument("--audio_model", default="facebook/data2vec-audio-base-960h")
    p.add_argument("--video_model", default="facebook/timesformer-base-finetuned-k400")
    p.add_argument("--teacher_ckpt", default=None)
    p.add_argument("--audio_ckpt", default=None)
    p.add_argument("--visual_ckpt", default=None)
    p.add_argument("--visual_masked_ckpt", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
