"""Efficiency comparison for AlignEMO vs re-implemented baselines (CMC revision).

Reports, per model, under identical GPU / batch conditions (warmup then timed):
  - #params of each student/sub-encoder and the total model
  - inference latency (ms per utterance)
  - peak GPU memory (MB)

Design
------
Models are registered in ``MODELS`` as (build_fn, input_fn):
  build_fn()  -> {component_name: nn.Module}   (frozen, on the target device)
  input_fn(B) -> tuple fed to a forward_fn, or use the default AlignEMO pipeline.

AlignEMO is wired to the repo's ``IEMOCAP/model.py`` encoders. To add a baseline
(TelME*, MM-DFN*, ...), register its builder the same way — see the commented
stub at the bottom. Only models importable in this environment are measured.

Fairness: same ``--batch_size``, ``--device``, dtype; ``--warmup`` untimed
iterations before ``--iters`` timed iterations; latency = total_time /
(iters * batch_size). Peak memory is reset per model.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# make repo encoders importable (IEMOCAP/model.py etc.)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "IEMOCAP"))


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


# ---------------------------------------------------------------------------
# AlignEMO pipeline (IEMOCAP config: roberta-large + data2vec + timesformer)
# ---------------------------------------------------------------------------
def build_alignemo(device, clsNum=6, text_model="roberta-large",
                   audio_model="facebook/data2vec-audio-base-960h",
                   video_model="facebook/timesformer-base-finetuned-k400"):
    from dataclasses import dataclass
    from model import Teacher_model, Student_Audio, Student_Video, ASF, ASF_visual

    @dataclass
    class Config:
        mask_time_length: int = 3

    comps = {
        "text_teacher": Teacher_model(text_model, clsNum),
        "audio_student": Student_Audio(audio_model, clsNum, Config()),
        "visual_student": Student_Video(video_model, clsNum),
        "visual_masked_student": Student_Video(video_model, clsNum),
        "fusion_visual": ASF_visual(clsNum, 768, 2e-1, 0.2, 4),
        "fusion": ASF(clsNum, 768, 2e-1, 0.2, 4),
    }
    for m in comps.values():
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
    return comps


@torch.no_grad()
def alignemo_forward(comps, batch_size, device, seq_len=64, audio_len=48000, frames=8):
    ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attn = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    audio = torch.randn(batch_size, audio_len, device=device)
    video = torch.randn(batch_size, frames, 3, 224, 224, device=device)
    video2 = torch.randn(batch_size, frames, 3, 224, 224, device=device)

    text_h, _ = comps["text_teacher"](ids, attn)
    audio_h, _ = comps["audio_student"](audio)
    v1_h, _ = comps["visual_student"](video)
    v2_h, _ = comps["visual_masked_student"](video2)
    fv_h, _ = comps["fusion_visual"](text_h, v2_h, v1_h)
    _ = comps["fusion"](text_h, fv_h, audio_h)


# ---------------------------------------------------------------------------
# AlignEMO K-MIND config (klue-roberta-base + whisper-large-v3 + timesformer x2)
# ---------------------------------------------------------------------------
def build_alignemo_kmind(device, clsNum=7,
                         text_model="klue/roberta-base",
                         audio_model="openai/whisper-large-v3",
                         video_model="facebook/timesformer-base-finetuned-k400"):
    import torch.nn as nn
    from transformers import AutoModel, WhisperModel, TimesformerModel

    comps = {
        "text_student": AutoModel.from_pretrained(text_model),
        "audio_teacher": WhisperModel.from_pretrained(audio_model).encoder,  # whisper encoder only
        "visual_scene": TimesformerModel.from_pretrained(video_model),
        "visual_facial": TimesformerModel.from_pretrained(video_model),
        # K-MIND fusion is 1280-dim; small vs the encoders. Represented by an
        # ASF-scale 2-step head for an order-of-magnitude fusion cost.
        "fusion_visual": nn.Sequential(nn.Linear(1280 * 2, 1280), nn.ReLU(), nn.Linear(1280, 1280)),
        "fusion": nn.Sequential(nn.Linear(1280 * 3, 1280), nn.ReLU(), nn.Linear(1280, clsNum)),
    }
    for m in comps.values():
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
    return comps


@torch.no_grad()
def alignemo_kmind_forward(comps, batch_size, device, seq_len=64, mel_bins=128, mel_frames=3000, frames=8):
    ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attn = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    mel = torch.randn(batch_size, mel_bins, mel_frames, device=device)
    video = torch.randn(batch_size, frames, 3, 224, 224, device=device)
    video2 = torch.randn(batch_size, frames, 3, 224, 224, device=device)

    comps["text_student"](input_ids=ids, attention_mask=attn)
    comps["audio_teacher"](input_features=mel)
    comps["visual_scene"](video)
    comps["visual_facial"](video2)
    th = torch.randn(batch_size, 1280, device=device)
    ah = torch.randn(batch_size, 1280, device=device)
    v1 = torch.randn(batch_size, 1280, device=device)
    v2 = torch.randn(batch_size, 1280, device=device)
    pvh = comps["fusion_visual"](torch.cat([v2, v1], dim=-1))
    comps["fusion"](torch.cat([th, pvh, ah], dim=-1))


MODELS = {
    "alignemo": dict(build=build_alignemo, forward=alignemo_forward,
                     students=["audio_student", "visual_student",
                               "visual_masked_student", "text_teacher"]),
    "alignemo_kmind": dict(build=build_alignemo_kmind, forward=alignemo_kmind_forward,
                           students=["text_student", "visual_scene", "visual_facial"]),
    # ---- add baselines here, e.g.: -------------------------------------
    # "telme":  dict(build=build_telme,  forward=telme_forward,  students=[...]),
}


def measure(spec, name, args, device):
    comps = spec["build"](device)
    forward = spec["forward"]

    # ---- params ----
    per_comp = {k: count_params(m) for k, m in comps.items()}
    total = sum(per_comp.values())
    student_total = sum(per_comp.get(s, 0) for s in spec.get("students", []))

    # ---- warmup ----
    for _ in range(args.warmup):
        forward(comps, args.batch_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # ---- timed ----
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        forward(comps, args.batch_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_utt = 1000.0 * elapsed / (args.iters * args.batch_size)
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else float("nan")

    del comps
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return dict(
        model=name,
        total_params=total,
        total_params_h=human(total),
        student_params=student_total,
        student_params_h=human(student_total),
        latency_ms_per_utt=round(ms_per_utt, 3),
        peak_gpu_mem_mb=round(peak_mb, 1),
        per_component={k: human(v) for k, v in per_comp.items()},
    )


def main(args):
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available -> latency/memory are CPU numbers, not comparable to paper.")

    names = args.models if args.models else list(MODELS.keys())
    rows = []
    for name in names:
        if name not in MODELS:
            print(f"[skip] '{name}' not registered (available: {list(MODELS)})")
            continue
        try:
            r = measure(MODELS[name], name, args, device)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        print(f"\n=== {name} ===")
        print(f"  total params   : {r['total_params_h']} ({r['total_params']:,})")
        print(f"  student params : {r['student_params_h']}")
        for k, v in r["per_component"].items():
            print(f"    - {k:<24} {v}")
        print(f"  latency        : {r['latency_ms_per_utt']} ms/utterance "
              f"(bs={args.batch_size}, {args.iters} iters, {args.warmup} warmup)")
        print(f"  peak GPU mem   : {r['peak_gpu_mem_mb']} MB")
        rows.append({k: v for k, v in r.items() if k != "per_component"})

    if rows:
        out = pd.DataFrame(rows)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"\n[saved] {args.out_csv}  ({len(out)} rows)")
        print("\n=== comparison ===")
        print(out[["model", "total_params_h", "student_params_h",
                   "latency_ms_per_utt", "peak_gpu_mem_mb"]].to_string(index=False))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=None,
                   help=f"subset to measure (default all registered: {list(MODELS)})")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_csv", default="scripts/analysis/efficiency_results.csv")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
