"""Train a fusion variant from the cached frozen-encoder features (fast path).

Consumes the cache built by build_fusion_cache.py (th, pvh, ah per utterance) and
trains ONLY the fusion module — no roberta/data2vec/timesformer forward — so each
variant/seed finishes in seconds. Reports test weighted-F1 / macro-F1 and can dump
per-utterance best-model predictions for the emotion-shift subset analysis.

    python scripts/ablation/train_fusion_cached.py --cache <...>.pt --variant shiftaware --seed 42
"""
import argparse
import copy
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_variants import build_fusion                 # noqa: E402


def seed_everything(seed):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def loader_from(split, batch_size, shuffle):
    ds = TensorDataset(split["th"], split["pvh"], split["ah"], split["y"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def validate_cache(cache):
    required_top = {"clsNum", "emoList", "train", "dev", "test"}
    missing = required_top - set(cache)
    if missing:
        raise ValueError(f"cache is missing keys: {sorted(missing)}")
    cls_num = int(cache["clsNum"])
    if cls_num != len(cache["emoList"]):
        raise ValueError("clsNum and emoList length disagree")
    for split_name in ("train", "dev", "test"):
        split = cache[split_name]
        missing = {"th", "pvh", "ah", "y"} - set(split)
        if missing:
            raise ValueError(f"{split_name} is missing keys: {sorted(missing)}")
        lengths = {key: len(split[key]) for key in ("th", "pvh", "ah", "y")}
        if len(set(lengths.values())) != 1 or not lengths["y"]:
            raise ValueError(f"invalid {split_name} lengths: {lengths}")
        for key in ("th", "pvh", "ah"):
            value = split[key]
            if value.ndim != 2 or value.shape[1] != 768:
                raise ValueError(f"{split_name}.{key} must have shape [N,768], got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{split_name}.{key} contains NaN or Inf")
        labels = split["y"]
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError(f"{split_name}.y must be 1-D torch.long, got {labels.shape}/{labels.dtype}")
        if int(labels.min()) < 0 or int(labels.max()) >= cls_num:
            raise ValueError(f"{split_name}.y is outside [0,{cls_num - 1}]")
    if "meta" not in cache["test"] or len(cache["test"]["meta"]) != len(cache["test"]["y"]):
        raise ValueError("test metadata length does not match cached test labels")


@torch.no_grad()
def evaluate(loader, fusion, device):
    fusion.eval()
    preds, golds = [], []
    for th, pvh, ah, y in loader:
        if isinstance(fusion, _HMF_CLASS):
            _, logits = fusion(text_embedding=th.to(device), visual=pvh.to(device), acoustic=ah.to(device))
        else:
            _, logits = fusion(text_embedding=th.to(device), visual=pvh.to(device), acoustic=ah.to(device))
        preds.append(logits.argmax(1).cpu().numpy()); golds.append(y.numpy())
    preds = np.concatenate(preds); golds = np.concatenate(golds)
    return (f1_score(golds, preds, average="weighted", zero_division=0) * 100,
            f1_score(golds, preds, average="macro", zero_division=0) * 100, preds, golds)


def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = torch.load(args.cache, map_location="cpu")
    validate_cache(cache)
    clsNum, emo_list = cache["clsNum"], cache["emoList"]
    dataset_name = cache.get("dataset", args.dataset_name)

    # modality-drop ablation: zero a modality's cached features (train+test) and
    # re-train the fusion without it, to measure each modality's contribution.
    if args.drop != "none":
        key = {"text": "th", "visual": "pvh", "audio": "ah"}[args.drop]
        for sp in ("train", "dev", "test"):
            cache[sp][key] = torch.zeros_like(cache[sp][key])
        print(f"[drop] zeroed '{args.drop}' ({key})")

    tl = loader_from(cache["train"], args.batch_size, True)
    dl = loader_from(cache["dev"], args.batch_size, False)
    test_loader = loader_from(cache["test"], args.batch_size, False)

    fusion = build_fusion(args.variant, clsNum, hidden_size=768,
                          beta_shift=2e-1, dropout_prob=0.2, num_head=4).to(device)
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.learning_rate)
    ce = nn.CrossEntropyLoss()
    global _HMF_CLASS
    _HMF_CLASS = type(fusion) if args.variant == "hmf" else ()

    best_wf1, best_mf1 = float("-inf"), float("-inf")
    best_epoch = None
    best_state = copy.deepcopy(fusion.state_dict())
    for epoch in range(args.epochs):
        fusion.train()
        for th, pvh, ah, y in tl:
            opt.zero_grad()
            if args.variant == "hmf":
                _, logits = fusion(text_embedding=th.to(device), visual=pvh.to(device), acoustic=ah.to(device))
            else:
                _, logits = fusion(text_embedding=th.to(device), visual=pvh.to(device), acoustic=ah.to(device))
            loss = ce(logits, y.to(device))
            loss.backward(); opt.step()
        wf1, mf1, _, _ = evaluate(dl, fusion, device)
        print(f"[{args.variant}][seed {args.seed}] epoch {epoch+1}: validation w-F1={wf1:.2f} macro-F1={mf1:.2f}")
        if wf1 > best_wf1:
            best_wf1, best_mf1 = wf1, mf1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(fusion.state_dict())

    fusion.load_state_dict(best_state)
    test_wf1, test_mf1, preds, golds = evaluate(test_loader, fusion, device)
    print(f"[{args.variant}][seed {args.seed}] validation-best test w-F1={test_wf1:.2f} macro-F1={test_mf1:.2f}")

    if args.out_checkpoint:
        checkpoint_path = Path(args.out_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_validation_weighted_f1": best_wf1,
            "test_weighted_f1": test_wf1,
            "selection_split": "validation",
            "test_evaluated_once_after_selection": True,
        }, checkpoint_path)
        print(f"[checkpoint] {checkpoint_path}")

    if args.dump_csv:
        meta = cache["test"]["meta"]
        n = len(preds)
        # gold MUST come from the cache label y (same space + alignment as pred);
        # meta only supplies dialogue/speaker/turn for shift detection.
        mism = sum(1 for i in range(n) if meta[i]["gold"] != int(golds[i]))
        print(f"[dump] meta-gold vs cache-y mismatch: {mism}/{n} "
              f"({'label-space/alignment bug -> using cache-y' if mism else 'aligned'})")
        rows = []
        for i in range(n):
            m = dict(meta[i])
            m["gold"] = int(golds[i])
            m["gold_name"] = emo_list[int(golds[i])] if 0 <= int(golds[i]) < len(emo_list) else str(int(golds[i]))
            rows.append(dict(dataset=dataset_name, id=f"utt{i:05d}", variant=args.variant,
                             seed=args.seed, **m, pred_stage1=int(preds[i])))
        Path(args.dump_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.dump_csv, index=False)
        print(f"[dumped] {args.dump_csv}  rows={n}")

    row = dict(dataset=dataset_name, variant=args.variant, seed=args.seed, drop=args.drop,
               best_validation_f1=round(best_wf1, 2),
               weighted_f1=round(test_wf1, 2), macro_f1=round(test_mf1, 2),
               selection_split="validation", selection_metric="weighted_f1")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.out_csv, mode="a",
                              header=not os.path.exists(args.out_csv), index=False)
    print(f"[appended] {args.out_csv}: {row}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="scripts/analysis/fusion_cache_iemocap.pt")
    p.add_argument("--dataset_name", default="IEMOCAP")
    p.add_argument("--variant", required=True,
                   choices=["concat", "crossattn", "hmf_nogate", "hmf", "shiftaware"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--out_csv", default="scripts/analysis/fusion_ablation_results.csv")
    p.add_argument("--dump_csv", default=None)
    p.add_argument("--out_checkpoint", default=None,
                   help="optional validation-best fusion checkpoint with args and metrics")
    p.add_argument("--drop", default="none", choices=["none", "text", "audio", "visual"],
                   help="zero this modality's cached features and re-train (modality-drop ablation)")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
