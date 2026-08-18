"""Dump per-utterance test predictions for IEMOCAP (CMC revision).

Runs the trained pipeline on the IEMOCAP test set and writes a per-utterance CSV
with BOTH model columns aligned by utterance key:

    dataset, id, dialogue_id, speaker, turn_idx, gold, gold_name,
    pred_stage1 (HMF only), pred_stage2 (Stage 1 + ELACL)

Both stages are produced in a single deterministic (shuffle=False) pass, so the
two columns are already paired — no separate merge needed. Per-utterance metadata
(dialogue_id, turn_idx, speaker, gold) is reconstructed from the growing-context
session structure (a new dialogue starts whenever the context length resets to 1;
turn_idx = position of the target utterance within its dialogue).

Point the --*_ckpt flags at the checkpoints that produced the reported number
(e.g. teacher_csv2_lr1e_6.bin, student_audio_csv/…, plus the Stage-1/Stage-2
ELACL .pkl files). Mirrors IEMOCAP/eval_stage12.py.
"""
import argparse
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
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "IEMOCAP"))

from preprocessing import preprocessing_pickle          # noqa: E402
from dataset import iemocap_dataset                      # noqa: E402
from utils import make_batchs_pickle                     # noqa: E402
from model import (Teacher_model, Student_Audio, Student_Video,   # noqa: E402
                   ASF, ASF_visual, CLModel, Classifier)
from ELACL_loss import loss_function                     # noqa: E402
from ELACL_trainer import train_or_eval_model, retrain   # noqa: E402


def seed_everything(seed):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def reconstruct_meta(sessions, emo_list):
    """Per-sample (dialogue_id, turn_idx, speaker, gold_idx, gold_name) from the
    growing-context session list — aligned 1:1 with the dataset / loader order."""
    rows = []
    dlg = -1
    for sess in sessions:
        if len(sess) == 1:            # a dialogue's first utterance (context reset)
            dlg += 1
        target = sess[-1]             # [speaker, utt, wav, video, start, end, emotion, (bbox)]
        speaker = target[0]
        emotion = target[6]
        rows.append(dict(
            dialogue_id=f"dlg{dlg:04d}",
            turn_idx=len(sess) - 1,
            speaker=int(speaker),
            gold_name=str(emotion),
            gold=emo_list.index(emotion) if emotion in emo_list else -1,
        ))
    return rows


def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        args.fp16 = False

    @dataclass
    class Config:
        mask_time_length: int = 3

    text_model = args.text_model
    audio_model = args.audio_model
    video_model = args.video_model
    init_config = Config()

    sessions = preprocessing_pickle(args.test_csv, args.pickle_path)
    test_dataset = iemocap_dataset(sessions)
    clsNum = len(test_dataset.emoList)
    emo_list = test_dataset.emoList
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=make_batchs_pickle)

    meta = reconstruct_meta(sessions, emo_list)     # aligned to loader order

    # ---- models ----
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu"), strict=False)
    model_t = model_t.to(device).eval()
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load(args.audio_ckpt, map_location="cpu"))
    audio_s = audio_s.to(device).eval()
    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load(args.visual_ckpt, map_location="cpu"))
    video_s = video_s.to(device).eval()
    video_s_masked = Student_Video(video_model, clsNum)
    video_s_masked.load_state_dict(torch.load(args.visual_masked_ckpt, map_location="cpu"))
    video_s_masked = video_s_masked.to(device).eval()

    hs, bs, dp, nh = 768, 2e-1, 0.2, 4
    fusion_visual = ASF_visual(clsNum, hs, bs, dp, nh)
    fusion_visual.load_state_dict(torch.load(args.fusion_visual_ckpt, map_location="cpu"))
    fusion_visual = fusion_visual.to(device).eval()
    fusion = ASF(clsNum, hs, bs, dp, nh)
    fusion.load_state_dict(torch.load(args.fusion_ckpt, map_location="cpu"))
    fusion = fusion.to(device).eval()

    stage1_model = CLModel(args, clsNum, fusion).to(device)
    stage1_model.load_state_dict(torch.load(args.stage1_ckpt, map_location="cpu"))
    stage1_model.eval()

    # ---- Stage 1 (HMF only) predictions ----
    _, _, s1_label, s1_pred, s1_fscore, _, _ = train_or_eval_model(
        model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion,
        stage1_model, loss_function, test_loader, 0, device, args)
    print(f"[Stage 1] w-F1 = {s1_fscore:.4f}  (n={len(s1_pred)})")

    # ---- build Stage-2 embeddings, then Stage 1+2 predictions ----
    emb, lab = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            bi, am, ai, vi, vi2, bl = [x.to(device) for x in batch]
            th, _ = model_t(bi, am); vh, _ = video_s(vi); vh2, _ = video_s_masked(vi2)
            ah, _ = audio_s(ai)
            pvh, _ = fusion_visual(text_embedding=th, visual=vh2, visual_2=vh)
            ph, _ = fusion(text_embedding=th, visual=pvh, acoustic=ah)
            if args.fp16:
                with torch.autocast(device_type=device):
                    _, mmo, _, _ = stage1_model(ph, return_mask_output=True)
            else:
                _, mmo, _, _ = stage1_model(ph, return_mask_output=True)
            emb.append(mmo.detach().cpu()); lab.append(bl.cpu())
    emb = torch.cat(emb); lab = torch.cat(lab)

    with torch.no_grad():
        anchors = stage1_model.map_function_anchor(stage1_model.emo_anchor)
    clf = Classifier(args, anchors).to(device)
    clf.load_state_dict(torch.load(args.stage2_ckpt, map_location="cpu"))
    clf.eval()
    loader2 = DataLoader(TensorDataset(emb, lab), batch_size=args.stage2_batch_size, shuffle=False)
    opt = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr)
    _, _, _, s2_label, s2_pred, s2_fscore, _ = retrain(
        clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), loader2, 0, device, args, opt, train=False)
    print(f"[Stage 1+2] w-F1 = {s2_fscore:.4f}  (n={len(s2_pred)})")

    # ---- align + dump ----
    n = min(len(meta), len(s1_pred), len(s2_pred))
    if not (len(meta) == len(s1_pred) == len(s2_pred)):
        print(f"[warn] length mismatch meta={len(meta)} s1={len(s1_pred)} s2={len(s2_pred)}; "
              f"truncating to {n} (check that no test batch was dropped).")
    rows = []
    for i in range(n):
        rows.append(dict(dataset="IEMOCAP", id=f"utt{i:05d}", **meta[i],
                         gold_check=int(s1_label[i]),
                         pred_stage1=int(s1_pred[i]), pred_stage2=int(s2_pred[i])))
    out = pd.DataFrame(rows)
    # sanity: reconstructed gold should equal the loader's label
    mism = int((out["gold"] != out["gold_check"]).sum())
    if mism:
        print(f"[warn] {mism}/{n} gold mismatch between reconstruction and loader label.")
    out = out.drop(columns=["gold_check"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}  rows={len(out)}  cols={list(out.columns)}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_csv", default="./dataset/IEMOCAP_full_release/IEMOCAP_test_2.csv")
    p.add_argument("--pickle_path", default="./dataset/IEMOCAP_full_release/faces_data_627.pkl")
    p.add_argument("--out_csv", default="scripts/analysis/iemocap_preds.csv")
    p.add_argument("--text_model", default="roberta-large")
    p.add_argument("--audio_model", default="facebook/data2vec-audio-base-960h")
    p.add_argument("--video_model", default="facebook/timesformer-base-finetuned-k400")
    # checkpoints (point at the ones that produced the reported number)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--audio_ckpt", required=True)
    p.add_argument("--visual_ckpt", required=True)
    p.add_argument("--visual_masked_ckpt", required=True)
    p.add_argument("--fusion_visual_ckpt", required=True)
    p.add_argument("--fusion_ckpt", required=True)
    p.add_argument("--stage1_ckpt", required=True, help="Stage-1 ELACL CLModel .pkl")
    p.add_argument("--stage2_ckpt", required=True, help="Stage-2 Classifier .pkl")
    p.add_argument("--anchor_path", default="./IEMOCAP/emo_anchors/roberta-large")
    # knobs mirrored from eval_stage12 (needed by CLModel/Classifier/trainer)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--cuda", type=bool, default=True)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--ce_loss_weight", type=float, default=0.9)
    p.add_argument("--angle_loss_weight", type=float, default=1.0)
    p.add_argument("--use_nearest_neighbour", type=bool, default=True)
    p.add_argument("--disable_emo_anchor", action="store_true")
    p.add_argument("--stage_two_lr", type=float, default=1e-4)
    p.add_argument("--stage2_batch_size", type=int, default=64)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
