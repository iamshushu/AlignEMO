"""Cache frozen MELD encoder outputs for fair, fast fusion comparisons."""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "MELD"))

from preprocessing import preprocessing_pickle  # noqa: E402
from dataset import meld_dataset  # noqa: E402
from utils import make_batchs_pickle  # noqa: E402
from model import Teacher_model, Student_Audio, Student_Video, ASF_visual  # noqa: E402


@dataclass
class Config:
    mask_time_length: int = 3


def reconstruct_meta(sessions, emo_list):
    rows, dialogue = [], -1
    for session in sessions:
        if len(session) == 1:
            dialogue += 1
        target = session[-1]
        rows.append({
            "dialogue_id": f"dlg{dialogue:04d}",
            "turn_idx": len(session) - 1,
            "speaker": int(target[0]),
            "gold_name": str(target[3]),
            "gold": emo_list.index(target[3]),
        })
    return rows


@torch.no_grad()
def encode_split(sessions, encoders, device, batch_size, num_workers):
    teacher, audio, video, video_masked, fusion_visual = encoders
    loader = DataLoader(
        meld_dataset(sessions), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=make_batchs_pickle,
    )
    collected = {"th": [], "pvh": [], "ah": [], "y": []}
    for batch in loader:
        if batch is None:
            continue
        bi, am, ai, vi, vi2, labels = [item.to(device) for item in batch]
        th, _ = teacher(bi, am)
        ah, _ = audio(ai)
        vh, _ = video(vi)
        vh2, _ = video_masked(vi2)
        pvh, _ = fusion_visual(text_embedding=th, visual=vh2, visual_2=vh)
        for key, value in (("th", th), ("pvh", pvh), ("ah", ah), ("y", labels)):
            collected[key].append(value.cpu())
    if not collected["y"]:
        raise RuntimeError("No usable MELD samples remained after media validation")
    return {key: torch.cat(values) for key, values in collected.items()}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data_path)
    split_files = {
        "train": (args.train_csv, args.train_pickle),
        "dev": (args.dev_csv, args.dev_pickle),
        "test": (args.test_csv, args.test_pickle),
    }
    sessions = {
        split: preprocessing_pickle(str(root / csv_name), str(root / pickle_name))
        for split, (csv_name, pickle_name) in split_files.items()
    }
    emo_list = meld_dataset(sessions["train"]).emoList
    cls_num = len(emo_list)

    def frozen(module, checkpoint, strict=True):
        module.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=strict)
        module = module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
        return module

    config = Config()
    encoders = (
        frozen(Teacher_model(args.text_model, cls_num), args.teacher_ckpt, strict=False),
        frozen(Student_Audio(args.audio_model, cls_num, config), args.audio_ckpt),
        frozen(Student_Video(args.video_model, cls_num), args.visual_ckpt),
        frozen(Student_Video(args.video_model, cls_num), args.visual_masked_ckpt),
        frozen(ASF_visual(cls_num, 768, 0.2, 0.2, 4), args.fusion_visual_ckpt),
    )

    cache = {"clsNum": cls_num, "emoList": emo_list, "dataset": "MELD"}
    for split in ("train", "dev", "test"):
        print(f"[cache] encoding {split} ...", flush=True)
        cache[split] = encode_split(
            sessions[split], encoders, device, args.batch_size, args.num_workers
        )
        print(f"[cache] {split}: {len(cache[split]['y'])} samples", flush=True)
    cache["test"]["meta"] = reconstruct_meta(sessions["test"], emo_list)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    print(f"[saved] {args.out}", flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./dataset/MELD.Raw")
    p.add_argument("--train_csv", default="train_meld_emo.csv")
    p.add_argument("--dev_csv", default="dev_meld_emo.csv")
    p.add_argument("--test_csv", default="test_meld_emo.csv")
    p.add_argument("--train_pickle", default="train_faces_data.pkl")
    p.add_argument("--dev_pickle", default="dev_faces_data.pkl")
    p.add_argument("--test_pickle", default="test_faces_data.pkl")
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
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
