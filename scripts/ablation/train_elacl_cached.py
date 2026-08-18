"""Run Stage 2 (ELACL) on a frozen-encoder cache instead of re-encoding every epoch.

Why
---
IEMOCAP/MELD ELACL_fusion.py pushes roberta-large + data2vec + 2x TimeSformer through
every batch of every epoch, so one seed costs ~4 h and five seeds are out of reach.
But once the encoders, the visual fusion and the HMF are all frozen, the fused
representation f_M is a fixed function of the input: with the corrected
PairwiseModalityAttention it no longer depends on how the batch happens to be
composed, so it can be computed once and reused exactly. Only CLModel
(map_function, map_function_anchor, predictor) and then the anchors are trained.

This makes each seed a few minutes, which is what puts a five-seed mean +/- sd in
reach for the paper. K-MIND already works this way (`ELACL_fusion.py --cache_root`);
this is the IEMOCAP/MELD equivalent.

    encoder cache (th, pvh, ah)  +  trained fusion ckpt
        -> f_M                        (computed here, seconds)
        -> Stage 1: CLModel under L_anchor,  validation-selected
        -> Stage 2: anchor adaptation,       validation-selected
        -> test read once per stage

NOTE ON VALIDITY: the cache is only an exact stand-in under the corrected attention.
If the fusion checkpoint carries the legacy (B, 2D) attention its output depends on
batch composition, and this script refuses to run -- see the shape check below.

Usage
-----
    python scripts/ablation/train_elacl_cached.py \
        --dataset IEMOCAP --variant hmf_nogate \
        --cache runs/.../frozen_encoder_cache.pt \
        --fusion_ckpt runs/.../hmf_nogate_validation_best.pt \
        --anchor_path ./IEMOCAP/emo_anchors/roberta-large \
        --seed 42 --out_dir runs/.../elacl/hmf_nogate/seed_42
"""
import argparse
import copy
import json
import os
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ablation"))
from fusion_variants import build_fusion  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_paramsgroup(model, lr, ptmlr):
    """Mirrors ELACL_fusion.py:get_paramsgroup (no-decay on bias/LayerNorm)."""
    no_decay = ["bias", "LayerNorm.weight"]
    groups = []
    for name, param in model.named_parameters():
        groups.append({
            "params": param,
            "lr": lr,
            "weight_decay": 0.0 if any(nd in name for nd in no_decay) else 0.01,
        })
    return sorted(groups, key=lambda x: x["lr"])


def assert_corrected_attention(state, what):
    shape = next((tuple(v.shape) for k, v in state.items() if k.endswith("in_proj_weight")), None)
    if shape is not None and shape[1] != 768:
        raise SystemExit(
            f"\n{what} carries a legacy attention projection {shape}.\n"
            "  Under that attention f_M depends on batch composition, so a cached f_M\n"
            "  is not a faithful stand-in and this script would silently train on one\n"
            "  arbitrary batching. Retrain the fusion with PairwiseModalityAttention.\n")


@torch.no_grad()
def build_fm(cache_split, fusion, device, batch_size):
    """f_M for every utterance in a split, from cached (th, pvh, ah)."""
    ds = TensorDataset(cache_split["th"], cache_split["pvh"], cache_split["ah"])
    out = []
    for th, pvh, ah in DataLoader(ds, batch_size=batch_size, shuffle=False):
        fm, _ = fusion(text_embedding=th.to(device), visual=pvh.to(device), acoustic=ah.to(device))
        out.append(fm.cpu())
    return torch.cat(out)


def run_epoch(model, loss_function, loader, device, args, optimizer=None, train=False):
    model.train() if train else model.eval()
    losses, preds, golds, reps, rep_labels = [], [], [], [], []
    for fm, y in loader:
        fm, y = fm.to(device), y.to(device)
        mask = torch.ones(len(fm), dtype=torch.bool, device=device)
        with torch.set_grad_enabled(train):
            log_prob, mapped, _, anchor_scores = model(fm, return_mask_output=True)
            out = loss_function(log_prob, mapped, y, mask, model)
            loss = out.ce_loss * args.ce_loss_weight + (1 - args.ce_loss_weight) * out.cl_loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss.item())
        preds.append(anchor_scores.argmax(-1).detach().cpu())
        golds.append(y.detach().cpu())
        if not train:
            reps.append(mapped.detach().cpu())
            rep_labels.append(y.detach().cpu())
    p = torch.cat(preds).numpy()
    g = torch.cat(golds).numpy()
    return (float(np.mean(losses)),
            round(accuracy_score(g, p) * 100, 2),
            round(f1_score(g, p, average="weighted", zero_division=0) * 100, 2),
            round(f1_score(g, p, average="macro", zero_division=0) * 100, 2))


@torch.no_grad()
def embed(model, loader, device):
    model.eval()
    Z, Y = [], []
    for fm, y in loader:
        _, mapped, _, _ = model(fm.to(device), return_mask_output=True)
        Z.append(mapped.cpu()); Y.append(y)
    return torch.cat(Z), torch.cat(Y)


def retrain_epoch(clf, criterion, loader, device, optimizer=None, train=False):
    clf.train() if train else clf.eval()
    preds, golds = [], []
    for z, y in loader:
        z, y = z.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = clf(z)
            loss = criterion(logits, y)
        if train:
            loss.backward(); optimizer.step(); optimizer.zero_grad()
        preds.append(logits.argmax(-1).detach().cpu()); golds.append(y.detach().cpu())
    p = torch.cat(preds).numpy(); g = torch.cat(golds).numpy()
    return (round(f1_score(g, p, average="weighted", zero_division=0) * 100, 2),
            round(f1_score(g, p, average="macro", zero_division=0) * 100, 2))


def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.cuda = device == "cuda"
    sys.path.insert(0, str(REPO / args.dataset))
    from model import CLModel, Classifier            # noqa: E402
    from ELACL_loss import loss_function             # noqa: E402

    cache = torch.load(args.cache, map_location="cpu")
    clsNum = int(cache["clsNum"])

    # train_fusion_cached.py wraps the weights in a provenance dict whose numpy
    # scalars trip PyTorch 2.6's weights_only=True default, hence the explicit False.
    ckpt = torch.load(args.fusion_ckpt, map_location="cpu", weights_only=False)
    fusion_state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(ckpt, dict) and "selection_split" in ckpt:
        print(f"[fusion] {args.variant} seed {ckpt.get('args', {}).get('seed')}: "
              f"epoch {ckpt.get('best_epoch')} selected on {ckpt['selection_split']}, "
              f"valid w-F1 {ckpt.get('best_validation_weighted_f1')}")
    assert_corrected_attention(fusion_state, f"fusion checkpoint {args.fusion_ckpt}")
    fusion = build_fusion(args.variant, clsNum, hidden_size=768,
                          beta_shift=2e-1, dropout_prob=0.2, num_head=4).to(device)
    fusion.load_state_dict(fusion_state)
    fusion.eval()
    for p in fusion.parameters():
        p.requires_grad = False

    loaders, sizes = {}, {}
    for split in ("train", "dev", "test"):
        fm = build_fm(cache[split], fusion, device, args.batch_size)
        ds = TensorDataset(fm, cache[split]["y"])
        sizes[split] = len(ds)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=(split == "train"))
    print(f"[data] train/dev/test = {sizes['train']}/{sizes['dev']}/{sizes['test']}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------- Stage 1: ELACL representation ----------------
    model = CLModel(args, clsNum).to(device)
    optimizer = AdamW(get_paramsgroup(model, args.lr, args.ptmlr))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_valid, best_epoch, best_state = -1.0, None, None
    print("\n===== Stage 1 (ELACL representation) =====")
    for e in range(args.epochs):
        tr = run_epoch(model, loss_function, loaders["train"], device, args, optimizer, True)
        scheduler.step()
        va = run_epoch(model, loss_function, loaders["dev"], device, args)
        print(f"  epoch {e+1:>2}  train wF1 {tr[2]:.2f}  valid wF1 {va[2]:.2f}  macro {va[3]:.2f}")
        if va[2] > best_valid:
            best_valid, best_epoch = va[2], e + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(best_state, out / "stage1_model.pkl")
    s1 = run_epoch(model, loss_function, loaders["test"], device, args)
    print(f"[Stage1] validation-best test wF1 {s1[2]:.2f}  macro {s1[3]:.2f}")

    # ---------------- Stage 2: anchor adaptation ----------------
    with torch.no_grad():
        anchors = model.map_function_anchor(model.emo_anchor).detach().clone()
    z = {sp: embed(model, loaders[sp], device) for sp in ("train", "dev", "test")}
    s2loaders = {sp: DataLoader(TensorDataset(*z[sp]), batch_size=args.stage2_batch_size,
                                shuffle=(sp == "train")) for sp in z}

    clf = Classifier(args, anchors).to(device)
    opt2 = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    best_valid2, best_epoch2, best_state2 = -1.0, None, None
    print("\n===== Stage 2 (anchor adaptation) =====")
    for e in range(args.stage2_epochs):
        retrain_epoch(clf, criterion, s2loaders["train"], device, opt2, True)
        va = retrain_epoch(clf, criterion, s2loaders["dev"], device)
        print(f"  epoch {e+1:>2}  valid wF1 {va[0]:.2f}  macro {va[1]:.2f}")
        if va[0] > best_valid2:
            best_valid2, best_epoch2 = va[0], e + 1
            best_state2 = copy.deepcopy(clf.state_dict())
    clf.load_state_dict(best_state2)
    torch.save(best_state2, out / "stage2_model.pkl")
    s2 = retrain_epoch(clf, criterion, s2loaders["test"], device)
    print(f"[Stage2] validation-best test wF1 {s2[0]:.2f}  macro {s2[1]:.2f}")

    result = {
        "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "selection_split": "validation", "selection_metric": "weighted_f1",
        "test_evaluated_once_after_selection": True,
        "cache": str(args.cache), "fusion_ckpt": str(args.fusion_ckpt),
        "stage1": {"best_epoch": best_epoch, "best_validation_f1": best_valid,
                   "test_weighted_f1": s1[2], "test_macro_f1": s1[3]},
        "stage2": {"best_epoch": best_epoch2, "best_validation_f1": best_valid2,
                   "test_weighted_f1": s2[0], "test_macro_f1": s2[1]},
    }
    with open(out / "elacl_selection.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[written] {out/'elacl_selection.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["IEMOCAP", "MELD"])
    p.add_argument("--variant", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--fusion_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--anchor_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--ptmlr", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--ce_loss_weight", type=float, default=0.9)
    p.add_argument("--angle_loss_weight", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_nearest_neighbour", type=bool, default=True)
    p.add_argument("--stage_two_lr", type=float, default=1e-4)
    p.add_argument("--stage2_batch_size", type=int, default=64)
    p.add_argument("--stage2_epochs", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--fp16", type=bool, default=False)
    main(p.parse_args())
