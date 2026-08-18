"""Stage-2 K-MIND cache builder: combine partial cache with visual features.

Reads the partial cache (text/audio from build_cache_1_partial.py), computes the
scene (v1) and focused-facial (v2) visual features with the frozen visual students
(Student_Video_Auto), and writes the final per-row multimodal cache
(text/audio/v1/v2/y + logits + meta) read by ELACL_fusion.py and
stage1b_per_class_eval.py via FinalMultimodalCacheDataset. See README section 3.
"""
import os
import cv2
import gc
import json
import time
import argparse
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from preprocessing import preprocessing_pickle
from dataset_ekman import kimind_dataset
from model import Student_Video_Auto
from mask_variants import VARIANTS as VISUAL_VARIANTS


# =========================================================
# globals
# =========================================================
LABELS = ["기쁨", "슬픔", "화남", "놀람", "무서움", "혐오", "중립"]
label2id = {lab: i for i, lab in enumerate(LABELS)}

VIDEO_PROCESSOR = None

NUM_FRAMES = 8
IMG_SIZE = 224


# =========================================================
# args / seed
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build final multimodal cache (text/audio from partial cache + v1/v2 from fixed visual teachers)"
    )

    parser.add_argument("--video_model", default="facebook/timesformer-base-finetuned-k400", type=str)

    parser.add_argument(
        "--partial_cache_root",
        default="./K-MIND/cache_audio_text_partial_rowwise",
        type=str,
        help="root of rowwise partial cache; must contain train/dev/test subdirs",
    )
    parser.add_argument(
        "--final_cache_root",
        default="./K-MIND/cache_total_multimodal_rowwise",
        type=str,
        help="root to save final multimodal cache; train/dev/test subdirs will be created",
    )

    parser.add_argument(
        "--visual_raw_ckpt",
        required=True,
        type=str,
        help="best checkpoint for raw visual student -> final v1",
    )
    parser.add_argument(
        "--visual_masked_ckpt",
        required=True,
        type=str,
        help="best checkpoint for masked visual student -> final v2",
    )

    parser.add_argument("--train_csv", default="./dataset/KMIND/splits/train_ekman_0210.csv", type=str)
    parser.add_argument("--dev_csv", default="./dataset/KMIND/splits/dev_ekman_0210.csv", type=str)
    parser.add_argument("--test_csv", default="./dataset/KMIND/splits/test_ekman_0210.csv", type=str)

    parser.add_argument("--train_pkl", default="./dataset/KMIND/faces_train_2_0210.pkl", type=str)
    parser.add_argument("--dev_pkl", default="./dataset/KMIND/faces_dev_2_0210.pkl", type=str)
    parser.add_argument("--test_pkl", default="./dataset/KMIND/faces_test_2_0210.pkl", type=str)

    parser.add_argument(
        "--v1_variant",
        default="scene",
        choices=["scene", "focused_facial", "masked_facial"],
        help="visual preprocessing variant for v1 input (default: scene = raw frames)",
    )
    parser.add_argument(
        "--v2_variant",
        default="focused_facial",
        choices=["scene", "focused_facial", "masked_facial"],
        help="visual preprocessing variant for v2 input (default: focused_facial = face bbox only)",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="which splits to build (default: all three)",
    )

    parser.add_argument("--teacher_hidden_size", default=1280, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_dtype", default="float16", choices=["float16", "float32"], type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write_manifest", action="store_true")

    parser.add_argument(
        "--label_col_candidates",
        default="emotion,label,emo,Affective_State,Emotion,Label",
        type=str,
        help="comma-separated CSV label column candidates for fallback",
    )

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# processor init
# =========================================================
def init_processors(video_model_name: str):
    global VIDEO_PROCESSOR
    VIDEO_PROCESSOR = AutoImageProcessor.from_pretrained(video_model_name)


def _check_processors():
    if VIDEO_PROCESSOR is None:
        raise RuntimeError("VIDEO_PROCESSOR is not initialized. Call init_processors(...) first.")


# =========================================================
# utils
# =========================================================
def split_csv_candidates(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() == ""


def normalize_text(x: Any) -> str:
    if is_missing(x):
        return ""
    return str(x).strip()


def row_get_first(row: pd.Series, candidates: List[str], default: Any = "") -> Any:
    for c in candidates:
        if c in row.index and not is_missing(row[c]):
            return row[c]
    return default


def cast_tensor(x: torch.Tensor, save_dtype: str) -> torch.Tensor:
    x = x.detach().cpu().contiguous()
    return x.half() if save_dtype == "float16" else x.float()


# =========================================================
# row key / session helpers
# =========================================================
def build_row_key(row_idx: int) -> str:
    return f"row_{row_idx:08d}"


def parse_last_line(session: Any) -> Dict[str, Any]:
    info = {
        "speaker": "",
        "utt": "",
        "wav_path": "",
        "video_path": "",
        "start_time": "",
        "end_time": "",
        "emotion": "",
        "bbox_list": None,
    }

    if isinstance(session, (list, tuple)) and len(session) > 0:
        last_line = session[-1]
    else:
        last_line = session

    if isinstance(last_line, dict):
        info["speaker"] = last_line.get("speaker", "")
        info["utt"] = last_line.get("utt", last_line.get("text", ""))
        info["wav_path"] = last_line.get("wav_path", last_line.get("audio_path", ""))
        info["video_path"] = last_line.get("video_path", "")
        info["start_time"] = last_line.get("start_time", "")
        info["end_time"] = last_line.get("end_time", "")
        info["emotion"] = last_line.get("emotion", last_line.get("label", ""))
        info["bbox_list"] = last_line.get("bbox_list", None)
        return info

    if isinstance(last_line, (list, tuple)):
        vals = list(last_line)

        if len(vals) >= 8:
            info["speaker"] = vals[0]
            info["utt"] = vals[1]
            info["wav_path"] = vals[2]
            info["video_path"] = vals[3]
            info["start_time"] = vals[4]
            info["end_time"] = vals[5]
            info["emotion"] = vals[6]
            info["bbox_list"] = vals[7]
            return info

        if len(vals) >= 7:
            info["speaker"] = vals[0]
            info["utt"] = vals[1]
            info["wav_path"] = vals[2]
            info["video_path"] = vals[3]
            info["start_time"] = vals[4]
            info["end_time"] = vals[5]
            info["emotion"] = vals[6]
            return info

    return info


# =========================================================
# video helpers
# =========================================================
def load_frames_from_dir(
    jpg_path: str,
    num_frames: int = 8,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[str]]]:
    if not jpg_path or not os.path.isdir(jpg_path):
        return None, None

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    frame_paths = sorted(
        os.path.join(jpg_path, f)
        for f in os.listdir(jpg_path)
        if f.lower().endswith(image_exts)
    )

    frames = []
    valid_paths = []

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        valid_paths.append(frame_path)

    if len(frames) == 0:
        return None, None

    if len(frames) >= num_frames:
        idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        frames = [frames[i] for i in idxs]
        valid_paths = [valid_paths[i] for i in idxs]
    else:
        lack = num_frames - len(frames)
        frames.extend([frames[-1].copy() for _ in range(lack)])
        valid_paths.extend([valid_paths[-1] for _ in range(lack)])

    frame_keys = [os.path.basename(p) for p in valid_paths]
    return frames[:num_frames], frame_keys[:num_frames]


def apply_bbox_mask_to_frames(
    frames: List[np.ndarray],
    frame_keys: List[str],
    bbox: Optional[Dict[str, List[Tuple[int, int, int, int]]]],
) -> List[np.ndarray]:
    if bbox is None:
        return [f.copy() for f in frames]

    masked_frames = []

    for frame, frame_key in zip(frames, frame_keys):
        masked_frame = np.zeros_like(frame)
        has_bbox = False

        if frame_key in bbox:
            H, W = frame.shape[:2]
            for (x, y, w, h) in bbox[frame_key]:
                if w <= 0 or h <= 0:
                    continue

                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(W, int(x + w))
                y2 = min(H, int(y + h))

                if x2 > x1 and y2 > y1:
                    masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    has_bbox = True

        masked_frames.append(masked_frame if has_bbox else frame)

    return masked_frames


def get_video_pixels(
    processor,
    path: str,
    bbox: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None,
    num_frames: int = 8,
    v1_variant: str = "scene",
    v2_variant: str = "focused_facial",
):
    zeros = torch.zeros((num_frames, 3, IMG_SIZE, IMG_SIZE))

    frames, frame_keys = load_frames_from_dir(path, num_frames=num_frames)
    if frames is None:
        return zeros, zeros

    v1_frames = VISUAL_VARIANTS[v1_variant](frames, frame_keys, bbox)
    v2_frames = VISUAL_VARIANTS[v2_variant](frames, frame_keys, bbox)

    try:
        v1_inputs = processor(v1_frames, return_tensors="pt")
        v1_pixels = v1_inputs["pixel_values"]

        v2_inputs = processor(v2_frames, return_tensors="pt")
        v2_pixels = v2_inputs["pixel_values"]

        if v1_pixels.dim() == 5 and v1_pixels.size(0) == 1:
            v1_pixels = v1_pixels[0]
        if v2_pixels.dim() == 5 and v2_pixels.size(0) == 1:
            v2_pixels = v2_pixels[0]

        return v1_pixels, v2_pixels

    except Exception as e:
        print(f"[WARN] video processor failed: {path} ({e})")
        return zeros, zeros


# =========================================================
# dataset
# =========================================================
class IndexedKimindPickleDataset(Dataset):
    """
    csv row index와 kimind_dataset(preprocessing_pickle(...)) index를 같이 사용.
    """
    def __init__(self, csv_path: str, pkl_path: str):
        self.csv_path = csv_path
        self.pkl_path = pkl_path
        self.df = pd.read_csv(csv_path)
        self.base = kimind_dataset(preprocessing_pickle(csv_path, pkl_path))

        if len(self.df) != len(self.base):
            raise RuntimeError(
                f"CSV rows ({len(self.df)}) != dataset size ({len(self.base)}). "
                "이 스크립트는 CSV index와 dataset index가 1:1로 맞는 전제입니다."
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return idx, self.base[idx]


# =========================================================
# row filtering from partial cache
# =========================================================
def get_available_row_indices(cache_split_dir: str, expected_rows: int, split_name: str) -> List[int]:
    valid_rows = []
    missing_count = 0

    for row_idx in range(expected_rows):
        row_key = build_row_key(row_idx)
        cache_path = os.path.join(cache_split_dir, f"{row_key}.pt")
        if os.path.isfile(cache_path):
            valid_rows.append(row_idx)
        else:
            missing_count += 1

    if len(valid_rows) == 0:
        raise RuntimeError(f"[{split_name}] no usable partial cache rows found under {cache_split_dir}")

    print(
        f"[{split_name}] usable partial cache rows: {len(valid_rows)} / {expected_rows} "
        f"(excluded={missing_count})"
    )
    return valid_rows


# =========================================================
# model loading
# =========================================================
def load_visual_model(video_model_name: str, cls_num: int, hidden_size: int, ckpt_path: str, device: torch.device):
    model = Student_Video_Auto(video_model_name, cls_num, kd_hidden_size=hidden_size)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    model.eval()
    return model


# =========================================================
# dump final cache
# =========================================================
@torch.no_grad()
def dump_final_split(
    split_name: str,
    dataset: IndexedKimindPickleDataset,
    partial_cache_dir: str,
    final_cache_dir: str,
    valid_rows: List[int],
    raw_visual_model: torch.nn.Module,
    masked_visual_model: torch.nn.Module,
    label_cols: List[str],
    save_dtype: str,
    overwrite: bool,
    device: torch.device,
    write_manifest: bool,
    v1_variant: str = "scene",
    v2_variant: str = "focused_facial",
):
    os.makedirs(final_cache_dir, exist_ok=True)

    manifest = {
        "split": split_name,
        "partial_cache_dir": partial_cache_dir,
        "final_cache_dir": final_cache_dir,
        "num_valid_rows": len(valid_rows),
        "saved": 0,
        "skipped_existing": 0,
        "failed": [],
    }

    df = dataset.df

    for row_idx in tqdm(valid_rows, desc=f"final-cache:{split_name}"):
        row_key = build_row_key(row_idx)
        partial_path = os.path.join(partial_cache_dir, f"{row_key}.pt")
        final_path = os.path.join(final_cache_dir, f"{row_key}.pt")

        if (not overwrite) and os.path.isfile(final_path):
            manifest["skipped_existing"] += 1
            continue

        try:
            if not os.path.isfile(partial_path):
                raise FileNotFoundError(f"partial cache missing: {partial_path}")

            obj = torch.load(partial_path, map_location="cpu")

            if "text" not in obj:
                raise KeyError(f"'text' missing in partial cache: {partial_path}")
            if "audio" not in obj and "audio_hidden" not in obj:
                raise KeyError(f"'audio'/'audio_hidden' missing in partial cache: {partial_path}")
            if "y" not in obj:
                raise KeyError(f"'y' missing in partial cache: {partial_path}")

            text_hidden = obj["text"]
            audio_hidden = obj["audio"] if "audio" in obj else obj["audio_hidden"]
            label = obj["y"].long()

            _, session = dataset[row_idx]
            row = df.iloc[row_idx]
            session_info = parse_last_line(session)
            video_path = normalize_text(session_info.get("video_path", ""))
            bbox_list = session_info.get("bbox_list", None)

            raw_video, masked_video = get_video_pixels(
                VIDEO_PROCESSOR,
                video_path,
                bbox=bbox_list,
                num_frames=NUM_FRAMES,
                v1_variant=v1_variant,
                v2_variant=v2_variant,
            )

            raw_video = raw_video.unsqueeze(0).to(device, non_blocking=True)
            masked_video = masked_video.unsqueeze(0).to(device, non_blocking=True)

            v1_hidden, v1_logit = raw_visual_model(raw_video)
            v2_hidden, v2_logit = masked_visual_model(masked_video)

            emotion = normalize_text(session_info.get("emotion", ""))
            if not emotion:
                emotion = normalize_text(row_get_first(row, label_cols, default=""))

            final_obj = {
                "id": row_key,
                "row_index": row_idx,
                "text": cast_tensor(text_hidden, save_dtype),
                "audio": cast_tensor(audio_hidden, save_dtype),
                "v1": cast_tensor(v1_hidden[0], save_dtype),
                "v2": cast_tensor(v2_hidden[0], save_dtype),
                "y": label.detach().cpu().long(),
                # optional extras for debugging / ablation
                "text_logit": cast_tensor(obj["text_logit"], save_dtype) if "text_logit" in obj else None,
                "audio_logit": cast_tensor(obj["audio_logit"], save_dtype) if "audio_logit" in obj else None,
                "v1_logit": cast_tensor(v1_logit[0], save_dtype),
                "v2_logit": cast_tensor(v2_logit[0], save_dtype),
                "meta": {
                    **obj.get("meta", {}),
                    "row_key": row_key,
                    "video_path": video_path,
                    "emotion": emotion,
                    "visual_raw_ckpt": args.visual_raw_ckpt if 'args' in globals() else "",
                    "visual_masked_ckpt": args.visual_masked_ckpt if 'args' in globals() else "",
                },
            }

            torch.save(final_obj, final_path)
            manifest["saved"] += 1

        except Exception as e:
            manifest["failed"].append({
                "row_index": row_idx,
                "row_key": row_key,
                "error": repr(e),
            })
            raise RuntimeError(f"[{split_name}] failed at row {row_idx} ({row_key}): {e}") from e

    print(
        f"[{split_name}] final cache done: saved={manifest['saved']} "
        f"skipped_existing={manifest['skipped_existing']} total_valid={len(valid_rows)}"
    )

    if write_manifest:
        with open(os.path.join(final_cache_dir, "_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


# =========================================================
# main
# =========================================================
def main(args):
    seed_everything(args.seed)
    init_processors(args.video_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_partial_dir = os.path.join(args.partial_cache_root, "train")
    dev_partial_dir = os.path.join(args.partial_cache_root, "dev")
    test_partial_dir = os.path.join(args.partial_cache_root, "test")

    for p in [train_partial_dir, dev_partial_dir, test_partial_dir]:
        if not os.path.isdir(p):
            raise RuntimeError(f"partial cache dir not found: {p}")

    train_dataset = IndexedKimindPickleDataset(args.train_csv, args.train_pkl)
    dev_dataset = IndexedKimindPickleDataset(args.dev_csv, args.dev_pkl)
    test_dataset = IndexedKimindPickleDataset(args.test_csv, args.test_pkl)

    train_valid_rows = get_available_row_indices(train_partial_dir, len(train_dataset), "train")
    dev_valid_rows = get_available_row_indices(dev_partial_dir, len(dev_dataset), "dev")
    test_valid_rows = get_available_row_indices(test_partial_dir, len(test_dataset), "test")

    cls_num = len(train_dataset.base.emoList) if hasattr(train_dataset.base, "emoList") else len(LABELS)
    label_cols = split_csv_candidates(args.label_col_candidates)

    print("### Partial Cache Root ###", args.partial_cache_root)
    print("### Final Cache Root ###", args.final_cache_root)
    print("### Visual Raw CKPT ###", args.visual_raw_ckpt)
    print("### Visual Masked CKPT ###", args.visual_masked_ckpt)
    print("### cls_num ###", cls_num)

    raw_visual_model = load_visual_model(
        args.video_model,
        cls_num,
        args.teacher_hidden_size,
        args.visual_raw_ckpt,
        device,
    )
    masked_visual_model = load_visual_model(
        args.video_model,
        cls_num,
        args.teacher_hidden_size,
        args.visual_masked_ckpt,
        device,
    )

    print("### v1_variant ###", args.v1_variant)
    print("### v2_variant ###", args.v2_variant)
    print("### splits ###", args.splits)

    SPLIT_CONFIG = {
        "train": (train_dataset, train_partial_dir, train_valid_rows),
        "dev":   (dev_dataset,   dev_partial_dir,   dev_valid_rows),
        "test":  (test_dataset,  test_partial_dir,  test_valid_rows),
    }

    for split_name in args.splits:
        ds, partial_dir, rows = SPLIT_CONFIG[split_name]
        dump_final_split(
            split_name=split_name,
            dataset=ds,
            partial_cache_dir=partial_dir,
            final_cache_dir=os.path.join(args.final_cache_root, split_name),
            valid_rows=rows,
            raw_visual_model=raw_visual_model,
            masked_visual_model=masked_visual_model,
            label_cols=label_cols,
            save_dtype=args.save_dtype,
            overwrite=args.overwrite,
            device=device,
            write_manifest=args.write_manifest,
            v1_variant=args.v1_variant,
            v2_variant=args.v2_variant,
        )

    print("--------------- Done --------------")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)