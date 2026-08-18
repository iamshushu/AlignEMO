"""
IEMOCAP Stage 1B (HMF) per-class 평가.
fusion.py 의 evaluation() 을 그대로 재현(예측 = pred_logits_all.argmax)하되,
per-class F1 + confusion matrix + weighted/macro F1 를 출력한다.
체크포인트/텍스트인코더는 CLI 로 덮어쓸 수 있다(정식 ckpt 가 다른 경우 대비).

  cd /path/to/AlignEMO   # repo root
  python IEMOCAP/eval_stage1b.py \
      --text_model microsoft/deberta-v3-large \
      --fusion_ckpt ./IEMOCAP/save_model/total_fusion2.bin

weighted-F1 이 논문 Table 7 의 68.63 와 일치하면 정식 ckpt 가 맞다.
"""
import os, sys, argparse, random, warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from preprocessing import *
from utils import *
from dataset import *
from model import *


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--text_model', default="microsoft/deberta-v3-large", type=str)
    p.add_argument('--audio_model', default="microsoft/wavlm-large", type=str)
    p.add_argument('--video_model', default="facebook/timesformer-base-finetuned-k400", type=str)
    p.add_argument('--teacher_ckpt',       default='./IEMOCAP/save_model/teacher.bin')
    p.add_argument('--audio_ckpt',         default='./IEMOCAP/save_model/student_audio/total_student.bin')
    p.add_argument('--video_ckpt',         default='./IEMOCAP/save_model/student_video2/total_student.bin')
    p.add_argument('--video_masked_ckpt',  default='./IEMOCAP/save_model/student_video_masked2/total_student.bin')
    p.add_argument('--fusion_visual_ckpt', default='./IEMOCAP/save_model/total_fusion_visual.bin')
    p.add_argument('--fusion_ckpt',        default='./IEMOCAP/save_model/total_fusion2.bin')
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluation(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, dataloader):
    fusion.eval()
    label_list, pred_list = [], []
    for data in dataloader:
        bi, am, ai, vi, vi2, bl = [x.cuda() for x in data]
        text_hidden, _ = model_t(bi, am)
        video_hidden, _ = video_s(vi)
        audio_hidden, _ = audio_s(ai)
        video_hidden_2, _ = video_s_masked(vi2)
        pred_hidden, _ = fusion_visual(text_embedding=text_hidden, visual=video_hidden_2, visual_2=video_hidden)
        _, pred_logits_all = fusion(text_embedding=text_hidden, visual=pred_hidden, acoustic=audio_hidden)
        pred_list.extend(pred_logits_all.argmax(1).cpu().numpy())
        label_list.extend(bl.cpu().numpy())
    return np.array(label_list), np.array(pred_list)


def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config: mask_time_length: int = 3
    init_config = Config()

    data_path = './dataset/IEMOCAP_full_release/'
    test_dataset = iemocap_dataset(preprocessing_pickle(data_path + 'IEMOCAP_test.csv',
                                                        data_path + 'faces_data_627.pkl'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=16, collate_fn=make_batchs_pickle)
    clsNum = len(test_dataset.emoList)
    classes = test_dataset.emoList
    print("classes:", classes, " text_model:", args.text_model)

    model_t = Teacher_model(args.text_model, clsNum)
    model_t.load_state_dict(torch.load(args.teacher_ckpt)); model_t = model_t.cuda().eval()
    audio_s = Student_Audio(args.audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load(args.audio_ckpt)); audio_s = audio_s.cuda().eval()
    video_s = Student_Video(args.video_model, clsNum)
    video_s.load_state_dict(torch.load(args.video_ckpt)); video_s = video_s.cuda().eval()
    video_s_masked = Student_Video(args.video_model, clsNum)
    video_s_masked.load_state_dict(torch.load(args.video_masked_ckpt)); video_s_masked = video_s_masked.cuda().eval()

    # IEMOCAP fusion dims (fusion.py): h=4, beta=0.2, dropout=0.2
    hs, beta, dp, nh = 768, 2e-1, 0.2, 4
    fusion_visual = ASF_visual(clsNum, hs, beta, dp, nh)
    fusion_visual.load_state_dict(torch.load(args.fusion_visual_ckpt)); fusion_visual = fusion_visual.cuda().eval()
    fusion = ASF(clsNum, hs, beta, dp, nh)
    fusion.load_state_dict(torch.load(args.fusion_ckpt)); fusion = fusion.cuda().eval()

    y_true, y_pred = evaluation(model_t, audio_s, video_s, video_s_masked, fusion_visual, fusion, test_loader)

    wf1 = f1_score(y_true, y_pred, average='weighted') * 100
    mf1 = f1_score(y_true, y_pred, average='macro') * 100
    per = f1_score(y_true, y_pred, average=None, labels=list(range(clsNum))) * 100
    print(f"\n===== IEMOCAP Stage 1B (HMF) =====")
    print(f"weighted-F1 = {wf1:.2f}   macro-F1 = {mf1:.2f}   (Table 7 기대값 wF1=68.63)")
    print("per-class F1:")
    for c, f in zip(classes, per):
        print(f"   {c:<12} {f:5.2f}")
    print("\nper-class list:", [round(float(f), 2) for f in per])
    print("\n=== Confusion Matrix (row=true, col=pred) ===")
    print("labels:", classes)
    print(confusion_matrix(y_true, y_pred, labels=list(range(clsNum))))
    print("\n", classification_report(y_true, y_pred, target_names=classes, digits=4))


if __name__ == "__main__":
    main(parse_args())
