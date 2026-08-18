#!/usr/bin/env python3
"""Learn validation-selected 1280->768 modality projections for K-MIND.

The source cache contains frozen upstream encoder/student representations.  The
paper specifies a trainable linear projection into a shared 768-dimensional
space before HMF.  This script learns only those projections (plus disposable
modality classifiers), selects the checkpoint on validation weighted F1, and
then writes an immutable rowwise 768-d cache.  Test labels are never evaluated
or used for model selection.
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


MODALITIES = ("text", "v2", "v1", "audio")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RowwiseCacheDataset(Dataset):
    def __init__(self, split_dir):
        self.files = sorted(Path(split_dir).glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"no .pt cache rows under {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        obj = torch.load(path, map_location="cpu")
        missing = [key for key in (*MODALITIES, "y") if key not in obj]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        features = tuple(obj[key].float() for key in MODALITIES)
        label = obj["y"].view(-1)[0].long()
        return features, label


def collate(batch):
    features, labels = zip(*batch)
    stacked = tuple(torch.stack(items, dim=0) for items in zip(*features))
    return stacked, torch.stack(labels)


class ModalityProjectionCalibrator(nn.Module):
    def __init__(self, input_dim=1280, output_dim=768, num_classes=7):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Linear(input_dim, output_dim) for name in MODALITIES
        })
        self.classifiers = nn.ModuleDict({
            name: nn.Linear(output_dim, num_classes) for name in MODALITIES
        })

    def forward(self, features):
        projected = {
            name: self.projections[name](feature)
            for name, feature in zip(MODALITIES, features)
        }
        logits = {
            name: self.classifiers[name](projected[name]) for name in MODALITIES
        }
        return projected, logits


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for features, labels in loader:
        features = tuple(item.to(device, non_blocking=True) for item in features)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(features)
        loss = torch.stack([ce(value, labels) for value in logits.values()]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
    return total / len(loader)


@torch.no_grad()
def validation_f1(model, loader, device):
    model.eval()
    predictions, labels_all = [], []
    for features, labels in loader:
        features = tuple(item.to(device, non_blocking=True) for item in features)
        _, logits = model(features)
        ensemble = torch.stack(list(logits.values()), dim=0).mean(dim=0)
        predictions.append(ensemble.argmax(dim=-1).cpu())
        labels_all.append(labels)
    pred = torch.cat(predictions).numpy()
    gold = torch.cat(labels_all).numpy()
    return f1_score(gold, pred, average="weighted", zero_division=0) * 100.0


@torch.no_grad()
def write_projected_cache(model, source_root, output_root, args, device):
    model.eval()
    manifests = {}
    for split in ("train", "dev", "test"):
        source_files = sorted((source_root / split).glob("*.pt"))
        output_dir = output_root / split
        output_dir.mkdir(parents=True, exist_ok=False)
        for start in range(0, len(source_files), args.batch_size):
            paths = source_files[start:start + args.batch_size]
            objects = [torch.load(path, map_location="cpu") for path in paths]
            features = tuple(
                torch.stack([obj[name].float() for obj in objects]).to(
                    device, non_blocking=True
                )
                for name in MODALITIES
            )
            projected, _ = model(features)
            projected = {
                name: value.half().cpu() for name, value in projected.items()
            }
            for index, (path, obj) in enumerate(zip(paths, objects)):
                output = {
                    "id": obj.get("id", path.stem),
                    "row_index": obj.get("row_index"),
                    "text": projected["text"][index],
                    "v2": projected["v2"][index],
                    "v1": projected["v1"][index],
                    "audio": projected["audio"][index],
                    "y": obj["y"].view(-1)[0].long().cpu(),
                    "meta": {
                        **obj.get("meta", {}),
                        "source_cache": str(path),
                        "projection_checkpoint": str(args.checkpoint),
                        "representation_dim": args.output_dim,
                    },
                }
                for name in MODALITIES:
                    value = output[name]
                    if tuple(value.shape) != (args.output_dim,):
                        raise ValueError(
                            f"{path} {name}: unexpected shape {tuple(value.shape)}"
                        )
                    if not torch.isfinite(value).all():
                        raise ValueError(f"{path} {name}: non-finite projected value")
                torch.save(output, output_dir / path.name)
            print(
                f"[{split}] projected {min(start + len(paths), len(source_files))}"
                f"/{len(source_files)}",
                flush=True,
            )
        manifests[split] = {
            "source_dir": str(source_root / split),
            "output_dir": str(output_dir),
            "count": len(source_files),
        }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump({
            "dataset": "K-MIND",
            "source_dimension": args.input_dim,
            "projected_dimension": args.output_dim,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "projection_checkpoint": str(args.checkpoint),
            "projection_checkpoint_sha256": sha256(args.checkpoint),
            "splits": manifests,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(args):
    seed_everything(args.seed)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output cache: {output_root}")
    for path in (args.checkpoint, args.selection_json):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    trainset = RowwiseCacheDataset(source_root / "train")
    devset = RowwiseCacheDataset(source_root / "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    model = ModalityProjectionCalibrator(
        args.input_dim, args.output_dim, args.num_classes
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = make_loader(trainset, args, True)
    dev_loader = make_loader(devset, args, False)
    best_f1 = float("-inf")
    best_epoch = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        dev_f1 = validation_f1(model, dev_loader, device)
        print(f"epoch={epoch} train_loss={loss:.6f} validation_weighted_f1={dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_validation_weighted_f1": best_f1,
        "selection_split": "validation",
        "test_evaluated_during_selection": False,
    }, args.checkpoint)
    with args.selection_json.open("x", encoding="utf-8") as handle:
        json.dump({
            "best_epoch": best_epoch,
            "best_validation_weighted_f1": best_f1,
            "selection_split": "validation",
            "test_evaluated_during_selection": False,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_projected_cache(model, source_root, output_root, args, device)
    print(f"projected cache: {output_root}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_root", required=True, type=Path)
    p.add_argument("--output_root", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--selection_json", required=True, type=Path)
    p.add_argument("--input_dim", type=int, default=1280)
    p.add_argument("--output_dim", type=int, default=768)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
