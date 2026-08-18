"""
generate_kmind_anchor.py
Generate K-MIND 7-class emotion anchors by averaging token embeddings of emotion words
with a specified (pretrained) text model.

K-MIND label order: 기쁨(joy), 슬픔(sadness), 화남(anger), 놀람(surprise),
                    무서움(fear), 혐오(disgust), 중립(neutral)

Usage:
  # English Ekman anchors aligned with kfdb text encoder
  python generate_kmind_anchor.py \
    --text_model kakaobank/kf-deberta-base \
    --lang eng \
    --out_dir ./K-MIND/emo_anchors/kfdb/

  # Korean anchors
  python generate_kmind_anchor.py \
    --text_model kakaobank/kf-deberta-base \
    --lang kor \
    --out_dir ./K-MIND/emo_anchors/kfdb/
"""
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModel


# K-MIND label order: 0=기쁨, 1=슬픔, 2=화남, 3=놀람, 4=무서움, 5=혐오, 6=중립
KMIND_EMO_ENG = ["joy", "sadness", "anger", "surprise", "fear", "disgust", "neutral"]
KMIND_EMO_KOR = ["기쁨", "슬픔", "화남", "놀람", "무서움", "혐오", "중립"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text_model", required=True, type=str)
    p.add_argument("--lang", default="eng", choices=["eng", "kor"])
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--save_name", default=None, type=str,
                   help="defaults to kmind_emo_{lang}.pt")
    p.add_argument("--pool", default="mean", choices=["mean", "cls"])
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emos = KMIND_EMO_ENG if args.lang == "eng" else KMIND_EMO_KOR
    save_name = args.save_name or f"kmind_emo_{args.lang}.pt"
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, save_name)

    print(f"text_model : {args.text_model}")
    print(f"lang       : {args.lang}  (labels={emos})")
    print(f"pool       : {args.pool}")
    print(f"output     : {out_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = AutoModel.from_pretrained(args.text_model).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for word in emos:
            enc = tokenizer(word, return_tensors="pt").to(device)
            out = model(**enc, return_dict=True)
            hidden = out.last_hidden_state  # [1, T, H]

            if args.pool == "mean":
                # mean over tokens (excluding padding; batch=1 so no padding issue)
                emb = hidden[0].mean(dim=0)  # [H]
            else:
                emb = hidden[0, 0]  # CLS token

            embeddings.append(emb.detach().cpu().unsqueeze(0))

    anchor = torch.cat(embeddings, dim=0)  # [7, H]
    print(f"saved shape: {tuple(anchor.shape)}")
    torch.save(anchor, out_path)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
