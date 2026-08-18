"""
IEMOCAP evaluation: Stage 1 vs Stage 1+Stage 2 on test set.
Reports per-class F1 + confusion matrix for both stages.

Usage:
  cd /path/to/AlignEMO   # repo root
  python IEMOCAP/eval_stage12.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import argparse
import random
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing import *
from utils import *
from dataset import *
from model import *
from ELACL_loss import loss_function
from ELACL_trainer import train_or_eval_model, retrain


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', default=8, type=int)
    p.add_argument('--learning_rate', default=1e-5, type=float)
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--max_grad_norm', type=float, default=5.0)
    p.add_argument('--dataset_name', default='IEMOCAP', type=str)
    p.add_argument("--ce_loss_weight", type=float, default=0.9)
    p.add_argument("--angle_loss_weight", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument('--ptmlr', type=float, default=1e-5)
    p.add_argument("--fp16", type=bool, default=True)
    p.add_argument("--cuda", type=bool, default=True)
    p.add_argument('--accumulation_step', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument("--use_nearest_neighbour", type=bool, default=True)
    p.add_argument("--anchor_path", type=str, default='./IEMOCAP/emo_anchors/roberta-large')
    p.add_argument("--disable_emo_anchor", action='store_true')
    p.add_argument("--disable_two_stage_training", type=bool, default=True)
    p.add_argument("--stage_two_lr", default=1e-4, type=float)
    p.add_argument("--save_stage_two_cache", type=bool, default=True)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--stage2_batch_size', type=int, default=64)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_cm_and_per_class(label, pred, class_names, tag):
    cm = confusion_matrix(label, pred, labels=list(range(len(class_names))))
    print(f"\n=== {tag} | Confusion Matrix ===")
    print("        " + " ".join(f"{n[:6]:>7s}" for n in class_names))
    for i, n in enumerate(class_names):
        print(f"{n[:6]:>6s}  " + " ".join(f"{cm[i, j]:>7d}" for j in range(len(class_names))))
    print(f"\n=== {tag} | Per-class F1 ===")
    print(classification_report(label, pred, target_names=class_names, digits=4))


def main(args):
    seed_everything(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: CUDA unavailable — inference will be slow.")
        args.cuda = False
        args.fp16 = False

    @dataclass
    class Config:
        mask_time_length: int = 3

    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"
    data_path = './dataset/IEMOCAP_full_release/'
    test_path = data_path + 'IEMOCAP_test.csv'
    pickle_path = data_path + 'faces_data_627.pkl'

    test_dataset = iemocap_dataset(preprocessing_pickle(test_path, pickle_path))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=16, collate_fn=make_batchs_pickle)

    save_path = './IEMOCAP/save_model'
    clsNum = len(test_dataset.emoList)
    class_names = test_dataset.emoList
    init_config = Config()
    print(f"clsNum={clsNum}, classes={class_names}")

    model_t = Teacher_model(text_model, clsNum)
    try:
        safe_load_state_dict(model_t, f'{save_path}/teacher.bin',
                             ignore_keys=['text_model.embeddings.position_ids'], strict=False)
    except NameError:
        sd = torch.load(f'{save_path}/teacher.bin', map_location='cpu')
        model_t.load_state_dict(sd, strict=False)
    for p in model_t.parameters(): p.requires_grad = False
    model_t = model_t.to(DEVICE).eval()

    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load(f'{save_path}/student_audio/total_student.bin', map_location='cpu'))
    for p in audio_s.parameters(): p.requires_grad = False
    audio_s = audio_s.to(DEVICE).eval()

    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load(f'{save_path}/student_video/total_student.bin', map_location='cpu'))
    for p in video_s.parameters(): p.requires_grad = False
    video_s = video_s.to(DEVICE).eval()

    video_s_masked = Student_Video(video_model, clsNum)
    video_s_masked.load_state_dict(torch.load(f'{save_path}/student_video_masked2/total_student.bin', map_location='cpu'))
    for p in video_s_masked.parameters(): p.requires_grad = False
    video_s_masked = video_s_masked.to(DEVICE).eval()

    hidden_size, beta_shift, dropout_prob, num_head = 768, 2e-1, 0.2, 4
    fusion_visual = ASF_visual(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion_visual.load_state_dict(torch.load(f'{save_path}/total_fusion_visual3.bin', map_location='cpu'))
    for p in fusion_visual.parameters(): p.requires_grad = False
    fusion_visual = fusion_visual.to(DEVICE).eval()

    fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion.load_state_dict(torch.load(f'{save_path}/total_fusion3.bin', map_location='cpu'))
    for p in fusion.parameters(): p.requires_grad = False
    fusion = fusion.to(DEVICE).eval()

    stage1_model = CLModel(args, clsNum, fusion).to(DEVICE)
    stage1_ckpt = f'{save_path}/stage1_model_roberta-large3.pkl'
    print(f"\nLoading Stage 1 ckpt: {stage1_ckpt}")
    stage1_model.load_state_dict(torch.load(stage1_ckpt, map_location='cpu'))
    stage1_model.eval()

    # ===== Stage 1 evaluation =====
    print("\n>>> Stage 1 (HMF only) evaluation on test set...")
    test_loss, test_acc, test_label, test_pred, test_fscore, test_detail_f1, _ = \
        train_or_eval_model(model_t, audio_s, video_s, video_s_masked, fusion_visual,
                            fusion, stage1_model, loss_function, test_loader, 0, DEVICE, args)
    print(f"[Stage 1] test_acc={test_acc:.4f}  test_fscore={test_fscore:.4f}")
    print(f"[Stage 1] per-class F1: {test_detail_f1}")
    print_cm_and_per_class(test_label, test_pred, class_names, "Stage 1")

    # ===== Build test embeddings for Stage 2 =====
    print("\n>>> Building test embeddings for Stage 2...")
    emb_test, label_test = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            bi, am, ai, vi, vi2, bl = batch
            bi, am, ai, vi, vi2, bl = bi.to(DEVICE), am.to(DEVICE), ai.to(DEVICE), vi.to(DEVICE), vi2.to(DEVICE), bl.to(DEVICE)
            text_hidden, _ = model_t(bi, am)
            video_hidden, _ = video_s(vi)
            video_hidden_2, _ = video_s_masked(vi2)
            audio_hidden, _ = audio_s(ai)
            pred_video_hidden, _ = fusion_visual(text_embedding=text_hidden, visual=video_hidden_2, visual_2=video_hidden)
            pred_hidden, _ = fusion(text_embedding=text_hidden, visual=pred_video_hidden, acoustic=audio_hidden)
            if args.fp16:
                with torch.autocast(device_type=DEVICE):
                    _, masked_mapped_output, _, _ = stage1_model(pred_hidden, return_mask_output=True)
            else:
                _, masked_mapped_output, _, _ = stage1_model(pred_hidden, return_mask_output=True)
            emb_test.append(masked_mapped_output.detach().cpu())
            label_test.append(bl.cpu())
    emb_test = torch.cat(emb_test, dim=0)
    label_test = torch.cat(label_test, dim=0)
    print(f"Test embeddings: {emb_test.shape}")

    # ===== Stage 2 evaluation =====
    print("\n>>> Stage 1+Stage 2 evaluation on test set...")
    with torch.no_grad():
        anchors = stage1_model.map_function_anchor(stage1_model.emo_anchor)
    clf = Classifier(args, anchors).to(DEVICE)
    stage2_ckpt = f'{save_path}/stage2_model_roberta-large3.pkl'
    print(f"Loading Stage 2 ckpt: {stage2_ckpt}")
    clf.load_state_dict(torch.load(stage2_ckpt, map_location='cpu'))
    clf.eval()

    testset = TensorDataset(emb_test, label_test)
    test_loader2 = DataLoader(testset, batch_size=args.stage2_batch_size, shuffle=False, num_workers=8)
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr)
    test_loss, test_ce_loss, test_acc, test_label2, test_pred2, test_fscore2, test_detail_f12 = \
        retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(DEVICE),
                test_loader2, 0, DEVICE, args, optimizer, train=False)
    print(f"[Stage 1+2] test_acc={test_acc:.4f}  test_fscore={test_fscore2:.4f}")
    print(f"[Stage 1+2] per-class F1: {test_detail_f12}")
    print_cm_and_per_class(test_label2, test_pred2, class_names, "Stage 1 + Stage 2")

    print("\n============ SUMMARY ============")
    print(f"Stage 1 (HMF only)       :  w-F1 = {test_fscore:.4f}")
    print(f"Stage 1 + Stage 2 (ELACL):  w-F1 = {test_fscore2:.4f}")
    print(f"Delta                    :  {test_fscore2 - test_fscore:+.4f}")


if __name__ == "__main__":
    main(parse_args())
