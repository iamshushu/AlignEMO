#!/usr/bin/env python3
"""Train hierarchical K-MIND fusion variants from a projected 768-d cache."""
import argparse
import copy
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))
from fusion_variants import build_fusion  # noqa: E402


def load_kmind_model():
    spec = importlib.util.spec_from_file_location("kmind_model_768", ROOT / "K-MIND" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProjectedCacheDataset(Dataset):
    def __init__(self, split_dir, expected_dim=768):
        self.files = sorted(Path(split_dir).glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"no cache rows under {split_dir}")
        first = torch.load(self.files[0], map_location="cpu")
        for key in ("text", "v2", "v1", "audio"):
            if tuple(first[key].shape) != (expected_dim,):
                raise ValueError(f"{self.files[0]} {key}: expected {expected_dim}, got {tuple(first[key].shape)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        obj = torch.load(self.files[index], map_location="cpu")
        return (
            obj["text"].float(), obj["v2"].float(), obj["v1"].float(),
            obj["audio"].float(), obj["y"].view(-1)[0].long(), obj.get("meta", {}),
        )


def collate(batch):
    text, v2, v1, audio, labels, meta = zip(*batch)
    return (
        torch.stack(text), torch.stack(v2), torch.stack(v1), torch.stack(audio),
        torch.stack(labels), list(meta),
    )


class HierarchicalFusionVariant(nn.Module):
    """K-MIND uses audio as the HMF gate-anchor modality."""
    def __init__(self, variant, num_classes, hidden_size, beta_shift, dropout, heads,
                 visual_beta_shift=None):
        super().__init__()
        kmind = load_kmind_model()
        if visual_beta_shift is None:
            visual_beta_shift = beta_shift
        self.visual_fusion = kmind.ASF_visual(
            num_classes, hidden_size, visual_beta_shift, dropout, heads
        )
        self.final_fusion = build_fusion(
            variant, num_classes, hidden_size, beta_shift, dropout, heads
        )

    def forward(self, *, text, v2, v1, audio):
        visual_hidden, _ = self.visual_fusion(
            text_embedding=audio, visual=v2, visual_2=v1
        )
        return self.final_fusion(
            text_embedding=audio, visual=visual_hidden, acoustic=text
        )


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, collate_fn=collate,
    )


def run(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss()
    predictions, golds, metas = [], [], []
    total_loss = 0.0
    for text, v2, v1, audio, labels, meta in loader:
        text = text.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        v1 = v1.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            _, logits = model(text=text, v2=v2, v1=v1, audio=audio)
            loss = ce(logits, labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        total_loss += float(loss.item())
        predictions.append(logits.argmax(dim=-1).detach().cpu())
        golds.append(labels.detach().cpu())
        metas.extend(meta)
    pred = torch.cat(predictions).numpy()
    gold = torch.cat(golds).numpy()
    return {
        "loss": total_loss / len(loader),
        "weighted_f1": f1_score(gold, pred, average="weighted", zero_division=0) * 100.0,
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0) * 100.0,
        "pred": pred,
        "gold": gold,
        "meta": metas,
    }


def main(args):
    seed_everything(args.seed)
    for path in (args.out_csv, args.out_checkpoint, args.dump_csv):
        if path and path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    splits = ("train", "dev") if args.validation_only else ("train", "dev", "test")
    datasets = {
        split: ProjectedCacheDataset(args.cache_root / split, args.hidden_size)
        for split in splits
    }
    loaders = {
        split: make_loader(dataset, args, split == "train")
        for split, dataset in datasets.items()
    }
    model = HierarchicalFusionVariant(
        args.variant, args.num_classes, args.hidden_size,
        args.beta_shift, args.dropout, args.num_heads, args.visual_beta_shift,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_f1 = float("-inf")
    best_epoch = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_result = run(model, loaders["train"], device, optimizer)
        dev_result = run(model, loaders["dev"], device)
        print(
            f"[{args.variant}][seed {args.seed}] epoch={epoch} "
            f"train_loss={train_result['loss']:.6f} "
            f"validation_wf1={dev_result['weighted_f1']:.4f}"
        )
        if dev_result["weighted_f1"] > best_f1:
            best_f1 = dev_result["weighted_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    checkpoint_record = {
        "model_state_dict": best_state,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_validation_weighted_f1": best_f1,
        "selection_split": "validation",
        "test_evaluated_during_selection": False,
    }
    if args.validation_only:
        torch.save(checkpoint_record, args.out_checkpoint)
        row = {
            "dataset": "K-MIND", "variant": args.variant, "seed": args.seed,
            "best_validation_f1": round(best_f1, 4),
            "weighted_f1": "", "macro_f1": "",
            "selection_split": "validation", "selection_metric": "weighted_f1",
            "test_evaluated": False,
        }
        pd.DataFrame([row]).to_csv(args.out_csv, index=False)
        print("validation-only selection candidate; test split was not loaded or evaluated")
        return
    model.load_state_dict(best_state)
    test_result = run(model, loaders["test"], device)
    print(
        f"validation-best test w-F1={test_result['weighted_f1']:.4f} "
        f"macro-F1={test_result['macro_f1']:.4f}"
    )
    torch.save({
        **checkpoint_record,
        "test_weighted_f1": test_result["weighted_f1"],
        "test_evaluated_once_after_selection": True,
    }, args.out_checkpoint)
    row = {
        "dataset": "K-MIND", "variant": args.variant, "seed": args.seed,
        "best_validation_f1": round(best_f1, 4),
        "weighted_f1": round(test_result["weighted_f1"], 4),
        "macro_f1": round(test_result["macro_f1"], 4),
        "selection_split": "validation", "selection_metric": "weighted_f1",
    }
    pd.DataFrame([row]).to_csv(args.out_csv, index=False)
    if args.dump_csv:
        rows = []
        for index, (gold, pred, meta) in enumerate(zip(
            test_result["gold"], test_result["pred"], test_result["meta"]
        )):
            rows.append({
                "dataset": "K-MIND", "id": f"utt{index:05d}",
                "variant": args.variant, "seed": args.seed,
                **meta, "gold": int(gold), "pred_stage1": int(pred),
            })
        pd.DataFrame(rows).to_csv(args.dump_csv, index=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", required=True, type=Path)
    p.add_argument("--variant", required=True,
                   choices=("concat", "crossattn", "hmf_nogate", "hmf", "shiftaware"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--beta_shift", type=float, default=0.1)
    p.add_argument("--visual_beta_shift", type=float,
                   help="Fix the visual-stage beta while sweeping only final HMF beta.")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num_heads", type=int, default=3)
    p.add_argument("--out_csv", required=True, type=Path)
    p.add_argument("--out_checkpoint", required=True, type=Path)
    p.add_argument("--dump_csv", type=Path)
    p.add_argument("--validation_only", action="store_true",
                   help="Select on dev and save checkpoint/CSV without loading test.")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
