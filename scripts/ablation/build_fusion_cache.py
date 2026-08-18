"""Cache the frozen-encoder fusion inputs for IEMOCAP (CMC revision, speedup).

The fusion ablation only trains the final fusion module; the encoders
(text teacher + audio/video students + visual fusion) are FROZEN and their
outputs (th, pvh, ah) do not depend on the fusion variant or seed. So we run the
encoders ONCE and cache (th, pvh, ah, label) per utterance. Afterwards every
variant/seed trains on the cache in seconds via train_fusion_cached.py, instead
of re-running roberta-large + data2vec + 2x timesformer every epoch (~1.5h each).

Output (.pt):
  {"train": {th,pvh,ah,y}, "dev": {th,pvh,ah,y},
   "test": {th,pvh,ah,y,meta}, "clsNum", "emoList"}
"""
import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "IEMOCAP"))

from preprocessing import preprocessing_pickle          # noqa: E402
from dataset import iemocap_dataset                      # noqa: E402
from utils import make_batchs_pickle                     # noqa: E402
from model import Teacher_model, Student_Audio, Student_Video  # noqa: E402


class LegacyASFVisual(torch.nn.Module):
    """Reader for the released paper visual-fusion checkpoint.

    The released checkpoint predates the corrected pairwise-attention module:
    it attends over a single 1536-d concatenated visual token.  Keeping this
    compatibility reader confined to cache construction preserves the released
    encoder features without weakening strict checkpoint loading elsewhere.
    """
    def __init__(self, cls_num, hidden_size=768, beta_shift=0.2,
                 dropout_prob=0.2, num_head=4):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(1536, num_head)
        self.W_hav = torch.nn.Linear(2304, 768)
        self.W_av = torch.nn.Linear(1536, 768)
        self.beta_shift = beta_shift
        self.LayerNorm = torch.nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = torch.nn.LayerNorm(1536)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.W = torch.nn.Linear(768, cls_num)

    def forward(self, text_embedding, *, visual, visual_2):
        nv = torch.cat((visual, visual_2), dim=-1)
        attended = self.multihead_attn(nv, nv, nv)[0] + nv
        av = self.dropout(self.AV_LayerNorm(attended))
        h_m = torch.relu(self.W_hav(torch.cat((av, text_embedding), dim=-1))) * self.W_av(av)
        alpha = torch.minimum(
            (text_embedding.norm(2, dim=-1) / (h_m.norm(2, dim=-1) + 1e-6)) * self.beta_shift,
            torch.ones(text_embedding.shape[0], device=text_embedding.device),
        ).unsqueeze(-1)
        output = self.dropout(self.LayerNorm(alpha * h_m + text_embedding))
        return output, self.W(output)


@dataclass
class Config:
    mask_time_length: int = 3


def reconstruct_meta(sessions, emo_list):
    rows, dlg = [], -1
    for sess in sessions:
        if len(sess) == 1:
            dlg += 1
        t = sess[-1]
        # Mirror make_batchs_pickle: only targets with usable media and a
        # headline label produce cached features.  Metadata must follow the
        # exact same filter to preserve row alignment.
        if t[2] is None or t[6] not in emo_list:
            continue
        rows.append(dict(dialogue_id=f"dlg{dlg:04d}", turn_idx=len(sess) - 1,
                         speaker=int(t[0]), gold_name=str(t[6]),
                         gold=emo_list.index(t[6]) if t[6] in emo_list else -1))
    return rows


@torch.no_grad()
def encode_split(sessions, encoders, device, batch_size, num_workers):
    model_t, audio_s, video_s, video_s_masked, fusion_visual = encoders
    ds = iemocap_dataset(sessions)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=make_batchs_pickle)
    THS, PVH, AH, Y = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        bi, am, ai, vi, vi2, bl = [x.to(device) for x in batch]
        th, _ = model_t(bi, am)
        ah, _ = audio_s(ai)
        vh, _ = video_s(vi)
        vh2, _ = video_s_masked(vi2)
        pvh, _ = fusion_visual(text_embedding=th, visual=vh2, visual_2=vh)
        THS.append(th.cpu()); PVH.append(pvh.cpu()); AH.append(ah.cpu()); Y.append(bl.cpu())
    return dict(th=torch.cat(THS), pvh=torch.cat(PVH), ah=torch.cat(AH), y=torch.cat(Y))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    dp = args.data_path

    train_sessions = preprocessing_pickle(dp + args.train_csv, dp + args.pickle)
    dev_sessions = preprocessing_pickle(dp + args.dev_csv, dp + args.pickle)
    test_sessions = preprocessing_pickle(dp + args.test_csv, dp + args.pickle)
    emo_list = iemocap_dataset(train_sessions).emoList
    clsNum = len(emo_list)

    def frozen(m, ck, strict=True):
        m.load_state_dict(torch.load(ck, map_location="cpu"), strict=strict)
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    encoders = (
        frozen(Teacher_model(args.text_model, clsNum), args.teacher_ckpt, strict=False),
        frozen(Student_Audio(args.audio_model, clsNum, cfg), args.audio_ckpt),
        frozen(Student_Video(args.video_model, clsNum), args.visual_ckpt),
        frozen(Student_Video(args.video_model, clsNum), args.visual_masked_ckpt),
        frozen(LegacyASFVisual(clsNum), args.fusion_visual_ckpt),
    )

    print("[cache] encoding train ...")
    train = encode_split(train_sessions, encoders, device, args.batch_size, args.num_workers)
    train["meta"] = reconstruct_meta(train_sessions, emo_list)
    if len(train["meta"]) != len(train["y"]):
        raise RuntimeError(f"train cache/meta mismatch: {len(train['y'])}/{len(train['meta'])}")
    print(f"[cache] train: {train['th'].shape[0]} utts")
    print("[cache] encoding dev ...")
    dev = encode_split(dev_sessions, encoders, device, args.batch_size, args.num_workers)
    dev["meta"] = reconstruct_meta(dev_sessions, emo_list)
    if len(dev["meta"]) != len(dev["y"]):
        raise RuntimeError(f"dev cache/meta mismatch: {len(dev['y'])}/{len(dev['meta'])}")
    print(f"[cache] dev:   {dev['th'].shape[0]} utts")
    print("[cache] encoding test ...")
    test = encode_split(test_sessions, encoders, device, args.batch_size, args.num_workers)
    test["meta"] = reconstruct_meta(test_sessions, emo_list)
    if len(test["meta"]) != len(test["y"]):
        raise RuntimeError(f"test cache/meta mismatch: {len(test['y'])}/{len(test['meta'])}")
    print(f"[cache] test:  {test['th'].shape[0]} utts (meta={len(test['meta'])})")

    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite immutable cache: {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(train=train, dev=dev, test=test, clsNum=clsNum, emoList=emo_list), args.out)
    print(f"[saved] {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out", default="scripts/analysis/fusion_cache_iemocap.pt")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
