#!/usr/bin/env python
"""AlignEMO preflight check.

Verifies that the environment, the pretrained backbones, the dataset paths and
the emotion anchors are all reachable *before* you launch a multi-hour training
run. Nothing here trains or downloads a dataset.

    python check_setup.py                 # check everything it can
    python check_setup.py --dataset MELD  # check one dataset only
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"
_state = {"fail": 0, "warn": 0}


def report(status, msg):
    print(f"{status} {msg}")
    if status is FAIL:
        _state["fail"] += 1
    elif status is WARN:
        _state["warn"] += 1


def section(title):
    print(f"\n=== {title} ===")


def check_python():
    section("Python / packages")
    report(OK, f"python {sys.version.split()[0]} at {sys.executable}")
    for mod, label in [
        ("torch", "torch"), ("transformers", "transformers"), ("sklearn", "scikit-learn"),
        ("librosa", "librosa"), ("cv2", "opencv"), ("av", "av"), ("pandas", "pandas"),
    ]:
        try:
            m = importlib.import_module(mod)
            report(OK, f"{label} {getattr(m, '__version__', '?')}")
        except ImportError as e:
            report(FAIL, f"{label} missing ({e})")


def check_gpu():
    section("GPU")
    try:
        import torch
    except ImportError:
        report(FAIL, "torch missing, cannot check GPU")
        return
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            report(OK, f"cuda:{i} {torch.cuda.get_device_name(i)} "
                       f"({free / 1e9:.1f}/{total / 1e9:.1f} GB free)")
    else:
        report(FAIL, "no CUDA device. The fusion modules allocate with .cuda() "
                     "unconditionally, so training and evaluation both require a GPU.")


def check_backbones(datasets):
    section("Pretrained backbones (HuggingFace cache or network)")
    needed = set()
    if {"IEMOCAP", "MELD"} & set(datasets):
        needed |= {("roberta-large", "AutoModel"),
                   ("facebook/data2vec-audio-base-960h", "AutoModel"),
                   ("facebook/timesformer-base-finetuned-k400", "AutoModel")}
    if "K-MIND" in datasets:
        needed |= {("klue/roberta-base", "AutoModel"),
                   ("openai/whisper-large-v3", "AutoConfig")}
    from transformers import AutoConfig
    for name, _ in sorted(needed):
        try:
            AutoConfig.from_pretrained(name)
            report(OK, name)
        except Exception as e:  # noqa: BLE001 - any failure is actionable the same way
            report(WARN, f"{name} not resolvable ({type(e).__name__}). "
                         f"It will be downloaded on first use if the node has network access.")


DATA_EXPECT = {
    "IEMOCAP": [("dataset/IEMOCAP_full_release", "raw sessions"),
                ("dataset/IEMOCAP_full_release/IEMOCAP_train_2.csv", "train split"),
                ("dataset/IEMOCAP_full_release/IEMOCAP_dev_2.csv", "dev split"),
                ("dataset/IEMOCAP_full_release/IEMOCAP_test_2.csv", "test split")],
    "MELD": [("dataset/MELD.Raw", "raw videos"),
             ("dataset/MELD.Raw/train_sent_emo.csv", "train split"),
             ("dataset/MELD.Raw/dev_sent_emo.csv", "dev split"),
             ("dataset/MELD.Raw/test_sent_emo.csv", "test split")],
    "K-MIND": [("dataset/KMIND", "raw media")],
}

ANCHOR_EXPECT = {
    "IEMOCAP": "IEMOCAP/emo_anchors/roberta-large/iemocap_emo.pt",
    "MELD": "MELD/emo_anchors/roberta-large/meld_emo.pt",
    "K-MIND": "K-MIND/emo_anchors/roberta-base/kmind_emo_eng.pt",
}


def check_datasets(datasets):
    section("Datasets")
    for ds in datasets:
        for rel, label in DATA_EXPECT[ds]:
            p = ROOT / rel
            if p.exists():
                report(OK, f"{ds}: {rel} ({label})")
            else:
                report(WARN, f"{ds}: {rel} missing ({label}) — see README §4")


def check_anchors(datasets):
    section("Emotion anchors")
    for ds in datasets:
        rel = ANCHOR_EXPECT[ds]
        if (ROOT / rel).exists():
            report(OK, f"{ds}: {rel}")
        else:
            report(WARN, f"{ds}: {rel} missing — run the generator in README §5")


def check_fusion_order():
    section("Fusion slot-order consistency")
    import subprocess
    r = subprocess.run([sys.executable, "scripts/ablation/check_fusion_order.py"],
                       cwd=ROOT, capture_output=True, text=True)
    if "RESULT: all call sites agree" in r.stdout:
        report(OK, "all fusion call sites agree on modality slot order")
    else:
        tail = [l for l in r.stdout.splitlines() if l.startswith("  [FAIL]")]
        report(WARN, "slot-order disagreement reported: " + ("; ".join(tail) or "see output"))
        report(WARN, "run `python scripts/ablation/check_fusion_order.py` for detail")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["IEMOCAP", "MELD", "K-MIND"], action="append",
                    help="restrict checks to this dataset (repeatable; default: all)")
    args = ap.parse_args()
    datasets = args.dataset or ["IEMOCAP", "MELD", "K-MIND"]

    print(f"AlignEMO preflight — repo root {ROOT}")
    check_python()
    check_gpu()
    check_backbones(datasets)
    check_datasets(datasets)
    check_anchors(datasets)
    check_fusion_order()

    print(f"\n{_state['fail']} failure(s), {_state['warn']} warning(s).")
    if _state["fail"]:
        print("Failures block training. Warnings are usually just missing data/anchors,")
        print("which you fix by following README §4 and §5.")
    return 1 if _state["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
