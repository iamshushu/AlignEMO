"""Stage-1 K-MIND cache builder: text + Whisper-audio features per utterance.

Runs the frozen text student (Teacher_Text_Auto) and Whisper audio student
(Student_Audio_Whisper) over each split and writes the rowwise *partial* cache
(text / audio hidden + logits) consumed by build_cache_2_final.py. See README section 3.
"""
import os
import gc
import re
import json
import argparse
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoTokenizer

from preprocessing import preprocessing
from dataset_ekman import kimind_dataset
from model import Student_Audio_Whisper, Teacher_Text_Auto


LABELS = ["기쁨", "슬픔", "화남", "놀람", "무서움", "혐오", "중립"]
label2id = {lab: i for i, lab in enumerate(LABELS)}
DEBUG_ROW = 50951

def parse_args():
    parser = argparse.ArgumentParser(description="Build full audio+text partial cache by CSV row")

    parser.add_argument("--text_model", default="klue/roberta-large", type=str)
    parser.add_argument("--audio_model", default="openai/whisper-large-v3", type=str)

    parser.add_argument(
        "--audio_teacher_ckpt",
        default="./K-MIND/save_model/student_audio_2_whisper/total_student_ekman.bin",
        type=str,
    )
    parser.add_argument(
        "--text_student_ckpt",
        default="./K-MIND/save_model/teacher_audio_student_text_2_whisper/total_student_ekman.bin",
        type=str,
    )

    # 46 of 51,093 train rows (0.09%, all in session SET180_M002_S001) have no audio
    # file on disk; dev and test are complete. Off by default so a NEW gap still
    # stops the build instead of silently shrinking the training set.
    parser.add_argument("--allow_missing_audio", action="store_true",
                        help="skip rows whose wav file is absent, and record them in _manifest.json")
    parser.add_argument("--train_csv", default="./dataset/KMIND/splits/train_ekman_0210.csv", type=str)
    parser.add_argument("--dev_csv", default="./dataset/KMIND/splits/dev_ekman_0210.csv", type=str)
    parser.add_argument("--test_csv", default="./dataset/KMIND/splits/test_ekman_0210.csv", type=str)

    parser.add_argument(
        "--cache_root",
        default="./K-MIND/cache_audio_text_partial_rowwise",
        type=str,
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_audio_seconds", default=30.0, type=float)
    parser.add_argument("--max_text_length", default=256, type=int)
    parser.add_argument("--teacher_hidden_size", default=1280, type=int)
    parser.add_argument("--save_dtype", default="float16", choices=["float16", "float32"], type=str)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--id_col_candidates", default="sample_id,segment_id,clip_id,utt_id,id,ID", type=str)
    parser.add_argument("--label_col_candidates", default="emotion,label,emo,Emotion,Label", type=str)
    parser.add_argument("--wav_col_candidates", default="wav_path,audio_path,audio,wav,voice_path,path", type=str)
    parser.add_argument("--text_col_candidates", default="utt,text,transcript,script,sentence,content", type=str)

    parser.add_argument("--path_src_prefix", default="", type=str)
    parser.add_argument("--path_dst_prefix", default="", type=str)

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_csv_candidates(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() == ""


def norm(x: Any) -> str:
    return "" if is_missing(x) else str(x).strip()


def row_get_first(row: pd.Series, candidates: List[str], default: Any = "") -> Any:
    for c in candidates:
        if c in row.index and not is_missing(row[c]):
            return row[c]
    return default


def safe_replace_path(path: str, src_prefix: str, dst_prefix: str) -> str:
    if path and src_prefix and dst_prefix:
        return path.replace(src_prefix, dst_prefix)
    return path


def cast_tensor(x: torch.Tensor, save_dtype: str) -> torch.Tensor:
    x = x.detach().cpu().contiguous()
    return x.half() if save_dtype == "float16" else x.float()


def get_audio_whisper(processor, path: str, max_seconds: float = 30.0):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"audio file not found: {path}")

    audio, _ = librosa.load(path, sr=16000, mono=True)
    if max_seconds is not None:
        max_samples = int(max_seconds * 16000)
        audio = audio[-max_samples:]

    inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
    return inputs["input_features"][0]


def parse_session_last_line(session: Any) -> Dict[str, Any]:
    info = {
        "speaker": "",
        "utt": "",
        "wav_path": "",
        "video_path": "",
        "start_time": "",
        "end_time": "",
        "emotion": "",
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
        return info

    if isinstance(last_line, (list, tuple)):
        vals = list(last_line)
        if len(vals) >= 7:
            info["speaker"] = vals[0]
            info["utt"] = vals[1]
            info["wav_path"] = vals[2]
            info["video_path"] = vals[3]
            info["start_time"] = vals[4]
            info["end_time"] = vals[5]
            info["emotion"] = vals[6]
    return info


def build_meta(row: pd.Series, row_idx: int, session_info: Dict[str, Any], args) -> Dict[str, Any]:
    wav_cols = split_csv_candidates(args.wav_col_candidates)
    text_cols = split_csv_candidates(args.text_col_candidates)
    label_cols = split_csv_candidates(args.label_col_candidates)
    id_cols = split_csv_candidates(args.id_col_candidates)

    meta = {
        "row_index": int(row_idx),
        "row_key": f"row_{row_idx:08d}",
        "csv_id": norm(row_get_first(row, id_cols, default="")),
        "speaker": norm(session_info.get("speaker", "")),
        "utt": norm(row_get_first(row, text_cols, default="")) or norm(session_info.get("utt", "")),
        "wav_path": norm(row_get_first(row, wav_cols, default="")) or norm(session_info.get("wav_path", "")),
        "video_path": norm(session_info.get("video_path", "")),
        "start_time": norm(session_info.get("start_time", "")),
        "end_time": norm(session_info.get("end_time", "")),
        "emotion": norm(row_get_first(row, label_cols, default="")) or norm(session_info.get("emotion", "")),
    }
    return meta


@torch.no_grad()
def build_split_cache(
    split_name: str,
    csv_path: str,
    out_dir: str,
    audio_processor,
    text_tokenizer,
    audio_model,
    text_model,
    args,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    dataset = kimind_dataset(preprocessing(csv_path))

    if len(df) != len(dataset):
        raise RuntimeError(f"[{split_name}] CSV rows ({len(df)}) != dataset size ({len(dataset)})")

    manifest = {
        "split": split_name,
        "csv_path": csv_path,
        "num_rows": len(df),
        "saved": 0,
        "failed_rows": [],
    }

    for row_idx in tqdm(range(len(df)), desc=f"cache:{split_name}"):
        row_key = f"row_{row_idx:08d}"
        save_path = os.path.join(out_dir, f"{row_key}.pt")

        if os.path.isfile(save_path) and not args.overwrite:
            manifest["saved"] += 1
            continue

        row = df.iloc[row_idx]
        session = dataset[row_idx]
        session_info = parse_session_last_line(session)
        meta = build_meta(row, row_idx, session_info, args)

        if row_idx == DEBUG_ROW:
            print("\n" + "=" * 100)
            print(f"[DEBUG] split={split_name}, row_idx={row_idx}, row_key=row_{row_idx:08d}")
            print("[DEBUG] csv row:")
            print(row.to_dict())
            print("[DEBUG] session_info:")
            print(session_info)
            print("[DEBUG] meta:")
            print(meta)

            dbg_wav_path = safe_replace_path(meta["wav_path"], args.path_src_prefix, args.path_dst_prefix)
            dbg_utt = meta["utt"]
            dbg_emotion = meta["emotion"]

            print("[DEBUG] resolved utt:", repr(dbg_utt))
            print("[DEBUG] resolved emotion:", repr(dbg_emotion))
            print("[DEBUG] resolved wav_path:", repr(dbg_wav_path))
            print("[DEBUG] wav exists:", os.path.isfile(dbg_wav_path) if dbg_wav_path else False)
            print("=" * 100 + "\n")

        try:
            wav_path = safe_replace_path(meta["wav_path"], args.path_src_prefix, args.path_dst_prefix)
            utt = meta["utt"]
            emotion = meta["emotion"]

            if not utt:
                raise ValueError(f"text empty at row {row_idx}")
            if emotion not in label2id:
                raise ValueError(f"invalid label at row {row_idx}: {emotion}")
            if not wav_path:
                raise ValueError(f"wav_path empty at row {row_idx}")

            label = label2id[emotion]

            audio_input = get_audio_whisper(
                audio_processor,
                wav_path,
                max_seconds=args.max_audio_seconds,
            ).unsqueeze(0).cuda()

            tok = text_tokenizer(
                [utt],
                padding=True,
                truncation=True,
                max_length=args.max_text_length,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].cuda()
            attention_mask = tok["attention_mask"].cuda()

            audio_hidden, audio_logit = audio_model(audio_input)
            text_hidden, text_logit = text_model(input_ids, attention_mask)

            obj = {
                "id": row_key,                      # lookup key는 무조건 row_key
                "row_index": row_idx,
                "csv_id": meta["csv_id"],          # 참고용
                "text": cast_tensor(text_hidden[0], args.save_dtype),
                "audio": cast_tensor(audio_hidden[0], args.save_dtype),
                "text_logit": cast_tensor(text_logit[0], args.save_dtype),
                "audio_logit": cast_tensor(audio_logit[0], args.save_dtype),
                "y": torch.tensor(label, dtype=torch.long),
                "meta": {
                    **meta,
                    "wav_path": wav_path,
                },
            }
            torch.save(obj, save_path)
            manifest["saved"] += 1

        except Exception as e:
            print("\n" + "!" * 100)
            print(f"[ERROR] split={split_name}, row_idx={row_idx}, row_key={row_key}")
            print("[ERROR] exception:", repr(e))
            try:
                print("[ERROR] csv row:", row.to_dict())
            except Exception:
                pass
            try:
                print("[ERROR] session_info:", session_info)
            except Exception:
                pass
            try:
                print("[ERROR] meta:", meta)
            except Exception:
                pass
            print("!" * 100 + "\n")

            manifest["failed_rows"].append({
                "row_index": row_idx,
                "row_key": row_key,
                "error": repr(e),
                "meta": meta,
            })
            if args.allow_missing_audio and isinstance(e.__cause__, FileNotFoundError):
                manifest.setdefault("skipped_missing_audio", []).append({
                    "row_index": row_idx, "row_key": row_key,
                    "wav_path": meta.get("wav_path", ""),
                })
                print(f"[{split_name}] SKIP row {row_idx} ({row_key}): audio file absent")
                continue
            raise RuntimeError(f"[{split_name}] failed at row {row_idx} ({row_key}): {e}") from e

    n_skipped = len(manifest.get("skipped_missing_audio", []))
    if manifest["saved"] + n_skipped != manifest["num_rows"]:
        raise RuntimeError(
            f"[{split_name}] incomplete cache: saved={manifest['saved']} "
            f"+ skipped={n_skipped} != rows={manifest['num_rows']}"
        )
    if n_skipped:
        pct = 100.0 * n_skipped / max(manifest["num_rows"], 1)
        print(f"[{split_name}] DROPPED {n_skipped} of {manifest['num_rows']} rows "
              f"({pct:.2f}%) for missing audio -- listed under "
              f"'skipped_missing_audio' in _manifest.json. Report this number.")

    with open(os.path.join(out_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[{split_name}] completed: {manifest['saved']} / {manifest['num_rows']}")


def main(args):
    seed_everything(args.seed)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    audio_processor = AutoProcessor.from_pretrained(args.audio_model)

    @dataclass
    class Config:
        mask_time_length: int = 3

    train_dataset = kimind_dataset(preprocessing(args.train_csv))
    cls_num = len(train_dataset.emoList) if hasattr(train_dataset, "emoList") else len(LABELS)

    print("### audio teacher ckpt ###", args.audio_teacher_ckpt)
    print("### text student ckpt ###", args.text_student_ckpt)
    print("### cache root ###", args.cache_root)
    print("### cls_num ###", cls_num)

    # audio teacher
    audio_model = Student_Audio_Whisper(args.audio_model, cls_num, Config())
    audio_model.load_state_dict(torch.load(args.audio_teacher_ckpt, map_location="cpu"), strict=True)
    for p in audio_model.parameters():
        p.requires_grad = False
    audio_model = audio_model.cuda().eval()

    # text student: checkpoint vocab 크기 기준으로 생성
    text_ckpt = torch.load(args.text_student_ckpt, map_location="cpu")
    ckpt_vocab_size = text_ckpt["encoder.embeddings.word_embeddings.weight"].shape[0]
    print("### text ckpt vocab size ###", ckpt_vocab_size)
    print("### current tokenizer vocab size ###", len(text_tokenizer))

    text_model = Teacher_Text_Auto(
        args.text_model,
        cls_num,
        vocab_size=ckpt_vocab_size,
        kd_hidden_size=args.teacher_hidden_size,
    )
    text_model.load_state_dict(text_ckpt, strict=True)
    for p in text_model.parameters():
        p.requires_grad = False
    text_model = text_model.cuda().eval()

    build_split_cache(
        "train",
        args.train_csv,
        os.path.join(args.cache_root, "train"),
        audio_processor,
        text_tokenizer,
        audio_model,
        text_model,
        args,
    )
    # build_split_cache(
    #     "dev",
    #     args.dev_csv,
    #     os.path.join(args.cache_root, "dev"),
    #     audio_processor,
    #     text_tokenizer,
    #     audio_model,
    #     text_model,
    #     args,
    # )
    # build_split_cache(
    #     "test",
    #     args.test_csv,
    #     os.path.join(args.cache_root, "test"),
    #     audio_processor,
    #     text_tokenizer,
    #     audio_model,
    #     text_model,
    #     args,
    # )

    print("--------------- Done --------------")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)