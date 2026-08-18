import glob
import os
import json
import copy
import gc
import time
import math
import random
import argparse
import warnings
import sys
import importlib.util
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

KMIND_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (KMIND_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def load_local_module(filename, module_name):
    """Load K-MIND model files without the repository's many `model.py` collisions."""
    spec = importlib.util.spec_from_file_location(module_name, KMIND_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


legacy_models = load_local_module("teacher_audio_model.py", "kmind_legacy_models_1280")
paper_models = load_local_module("model.py", "kmind_paper_models_768")
LegacyASFVisual1280 = legacy_models.ASF_visual
LegacyASF1280 = legacy_models.ASF
LegacyCLModel1280 = legacy_models.CLModel
LegacyClassifier1280 = legacy_models.Classifier
PaperASFVisual768 = paper_models.ASF_visual
PaperASF768 = paper_models.ASF
PaperCLModel768 = paper_models.CLModel
PaperClassifier768 = paper_models.Classifier

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.ablation.fusion_variants import HMFNoGate
from ELACL_loss import loss_function
from ELACL_trainer import retrain


class FinalMultimodalCacheDataset(Dataset):
    def __init__(self, cache_dir: str, expected_dim=None):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in: {cache_dir}")
        self.labels = []
        for fp in self.files:
            obj = torch.load(fp, map_location="cpu")
            required = ["text", "v2", "v1", "audio", "y"]
            missing = [key for key in required if key not in obj]
            if missing:
                raise KeyError(f"Missing keys {missing} in cache file: {fp}")
            if expected_dim is not None:
                for key in ("text", "v2", "v1", "audio"):
                    if tuple(obj[key].shape) != (expected_dim,):
                        raise ValueError(
                            f"{fp} {key}: expected ({expected_dim},), "
                            f"got {tuple(obj[key].shape)}"
                        )
            y = int(obj["y"].view(-1)[0].item())
            self.labels.append(y)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj = torch.load(self.files[idx], map_location="cpu")
        required = ["text", "v2", "v1", "audio", "y"]
        for k in required:
            if k not in obj:
                raise KeyError(f"Missing key '{k}' in cache file: {self.files[idx]}")
        return obj["text"], obj["v2"], obj["v1"], obj["audio"], obj["y"]


def stack_collate(batch):
    text, visual2, visual, audio, y = zip(*batch)
    text = torch.stack(text, 0)
    visual2 = torch.stack(visual2, 0)
    visual = torch.stack(visual, 0)
    audio = torch.stack(audio, 0)
    y = torch.stack([yy.view(-1)[0] for yy in y], 0).long()
    return text, visual2, visual, audio, y


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)

    parser.add_argument("--cache_root", default="./K-MIND/cache_total_multimodal_rowwise", type=str)
    parser.add_argument("--fusion_visual_ckpt", type=str)
    parser.add_argument("--fusion_ckpt", type=str)
    parser.add_argument("--random_init_fusion", action="store_true",
                        help="Train corrected HMF from scratch; do not load legacy fusion checkpoints.")
    parser.add_argument("--representation_dim", choices=[768, 1280], default=1280, type=int,
                        help="Select the paper-aligned 768-d or legacy 1280-d K-MIND model family.")
    parser.add_argument("--final_fusion_variant", choices=["hmf", "hmf_nogate"], default="hmf",
                        help="Final hierarchical fusion used in Stage 1; visual fusion remains ASF_visual.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)

    parser.add_argument("--clsNum", default=7, type=int)
    parser.add_argument("--fv_hidden_size", default=768, type=int)
    parser.add_argument("--fv_beta_shift", default=0.2, type=float)
    parser.add_argument("--fv_dropout_prob", default=0.3, type=float)
    parser.add_argument("--fv_num_head", default=4, type=int)
    parser.add_argument("--f_hidden_size", default=768, type=int)
    parser.add_argument("--f_beta_shift", default=0.1, type=float)
    parser.add_argument("--f_dropout_prob", default=0.2, type=float)
    parser.add_argument("--f_num_head", default=3, type=int)

    parser.add_argument("--dataset_name", default="K-MIND", type=str)
    parser.add_argument("--ce_loss_weight", type=float, default=0.9)
    parser.add_argument("--angle_loss_weight", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=0.5)

    parser.add_argument("--ptmlr", type=float, default=1e-5)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--accumulation_step", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=4e-4)

    parser.add_argument("--use_nearest_neighbour", action="store_true")
    parser.add_argument("--anchor_path", type=str, default="./K-MIND/emo_anchors/roberta-base/")
    parser.add_argument("--disable_emo_anchor", action="store_true")
    parser.add_argument("--supcon_query_mode", default="current",
                        choices=["current", "valid_query", "utterance_only"],
                        help="SupCon query population; anchors always remain supports.")
    parser.add_argument("--disable_two_stage_training", type=bool, default=True)
    # README Table 2: Stage-2 LR is 1e-5 (IEMOCAP) / 5e-7 (MELD) / 4e-4 (K-MIND).
    # This had been left at MELD's 5e-7, i.e. 800x too small -- Stage 2 then moves
    # so little that valid w-F1 is identical to 2 d.p. across all 10 epochs.
    parser.add_argument("--stage_two_lr", default=4e-4, type=float)
    # Stage 1 uses --weight_decay 1e-3; leaking that into the 7xD anchor matrix
    # just shrinks it. Separate knob, off by default.
    parser.add_argument("--stage2_weight_decay", default=0.0, type=float)
    # Classifier logits are (1+cos)/2 / temp. Reusing the SCL temperature is a
    # coincidence of naming; leave None to keep that behaviour.
    parser.add_argument("--stage2_temp", default=None, type=float)
    # Skip Stage-1 training and load this checkpoint instead (Stage-2 sweeps).
    parser.add_argument("--stage1_ckpt", default=None, type=str)
    parser.add_argument("--save_stage_two_cache", type=bool, default=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--stage2_batch_size", type=int, default=32)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_normalize", action="store_true")
    parser.add_argument("--use_stage2_class_weight", action="store_true")
    parser.add_argument("--stage2_validation_only", action="store_true",
                        help="Tune Stage 2 on train/dev without loading or evaluating test.")

    parser.add_argument("--unfreeze_fusion", action="store_true")
    parser.add_argument("--anchor_modality", choices=["audio", "text"], default="audio",
                        help="modality placed in ASF's text_embedding (gate-anchor) slot. "
                             "Must match the convention the fusion checkpoints were "
                             "trained with; see scripts/ablation/check_fusion_order.py.")
    parser.add_argument("--fusion_lr", type=float, default=2e-6)
    parser.add_argument("--use_stage1_class_weight", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--stage1_balance_power", type=float, default=0.5)

    parser.add_argument("--save_threshold", type=float, default=None)
    parser.add_argument("--save_stage1_every_epoch", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")

    # --- NEW: label smoothing & scheduler ---
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CE loss (0.0 = off)")
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="LR scheduler after warmup: linear decay or cosine annealing")

    return parser.parse_args()



def hmf_slots(args, text_hidden, audio_hidden):
    """Which modality anchors the HMF gate, and which fills ASF's `acoustic` slot.

    K-MIND shipped two incompatible conventions: fusion_ekman_grid.py (which TRAINS
    the Stage-1B checkpoints) is text-anchored, while this file (which produced the
    reported numbers) is audio-anchored. A checkpoint trained under one and run under
    the other silently receives permuted features, so the two must agree -- see
    scripts/ablation/check_fusion_order.py.
    """
    if args.anchor_modality == "audio":
        return audio_hidden, text_hidden
    return text_hidden, audio_hidden


def resolve_model_classes(args):
    """Resolve an explicit representation contract without changing legacy runs."""
    if args.representation_dim == 768:
        final_fusion = PaperASF768 if args.final_fusion_variant == "hmf" else HMFNoGate
        return PaperASFVisual768, final_fusion, PaperCLModel768, PaperClassifier768
    if args.final_fusion_variant != "hmf":
        raise ValueError("hmf_nogate is only implemented for the corrected 768-d path")
    return LegacyASFVisual1280, LegacyASF1280, LegacyCLModel1280, LegacyClassifier1280


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_paramsgroup(model, lr: float, weight_decay: float = 0.01, exclude_param_ids=None):
    no_decay = ["bias", "LayerNorm.weight"]
    params = []
    if exclude_param_ids is None:
        exclude_param_ids = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in exclude_param_ids:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        params.append({"params": [param], "lr": lr, "weight_decay": wd})
    return params


def flatten_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return np.concatenate(
            [t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in x],
            axis=0,
        )
    return np.array(x)


def build_stage1_balance_base(class_counts: torch.Tensor, power: float) -> torch.Tensor:
    if power < 0:
        raise ValueError(f"stage1_balance_power must be >= 0, got {power}")
    return 1.0 / torch.pow(class_counts + 1e-8, power)


def compute_ce_loss(log_prob, batch_labels, mask, class_weights, label_smoothing):
    """CE loss with optional class weights and label smoothing."""
    return F.cross_entropy(
        log_prob[mask],
        batch_labels[mask],
        weight=class_weights,
        ignore_index=-1,
        label_smoothing=label_smoothing,
    )


def train_or_eval_model_from_cache(
    fusion_visual,
    fusion,
    stage1_model,
    loss_function_fn,
    dataloader,
    epoch,
    device,
    args,
    optimizer=None,
    scheduler=None,
    train=False,
):
    if train:
        stage1_model.train()
        if args.unfreeze_fusion:
            fusion_visual.train()
            fusion.train()
        else:
            fusion_visual.eval()
            fusion.eval()
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        optimizer.zero_grad(set_to_none=True)
    else:
        stage1_model.eval()
        fusion_visual.eval()
        fusion.eval()

    losses = []
    preds = []
    labels_all = []

    use_amp = bool(args.fp16 and device == "cuda")
    max_cosine_last = None

    for batch_id, batch in enumerate(dataloader):
        text_hidden, video_hidden_2, video_hidden, audio_hidden, batch_labels = batch

        text_hidden = text_hidden.to(device, non_blocking=True).float()
        video_hidden_2 = video_hidden_2.to(device, non_blocking=True).float()
        video_hidden = video_hidden.to(device, non_blocking=True).float()
        audio_hidden = audio_hidden.to(device, non_blocking=True).float()
        batch_labels = batch_labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
                head, tail = hmf_slots(args, text_hidden, audio_hidden)
                pred_video_hidden, _ = fusion_visual(
                    text_embedding=head, visual=video_hidden_2, visual_2=video_hidden
                )
                pred_hidden, _ = fusion(
                    text_embedding=head, visual=pred_video_hidden, acoustic=tail
                )

                log_prob, masked_mapped_output, _, anchor_scores = stage1_model(
                    pred_hidden, return_mask_output=True
                )

                mask = torch.ones(len(pred_hidden), dtype=torch.bool, device=device)

                loss_out = loss_function_fn(
                    log_prob,
                    masked_mapped_output,
                    batch_labels,
                    mask,
                    stage1_model,
                )

                # CE loss: class weight + label smoothing
                ce_loss = compute_ce_loss(
                    log_prob, batch_labels, mask,
                    args.stage1_class_weights,
                    args.label_smoothing,
                )

                loss_val = (
                    ce_loss * stage1_model.args.ce_loss_weight
                    + (1 - stage1_model.args.ce_loss_weight) * loss_out.cl_loss
                )
                max_cosine_last = loss_out.max_cosine

        if train:
            (loss_val / args.accumulation_step).backward()

            if (batch_id + 1) % args.accumulation_step == 0 or (batch_id + 1) == len(dataloader):
                if args.unfreeze_fusion:
                    fusion_param_ids = {id(p) for p in fusion.parameters()}
                    params_to_clip = [p for p in stage1_model.parameters() if id(p) not in fusion_param_ids]
                    params_to_clip += list(fusion.parameters())
                    params_to_clip += list(fusion_visual.parameters())
                else:
                    params_to_clip = list(stage1_model.parameters())

                torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm, norm_type=2)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if args.use_nearest_neighbour:
            batch_pred = torch.argmax(anchor_scores[mask], dim=-1)
        else:
            batch_pred = torch.argmax(log_prob[mask], dim=-1)

        preds.append(batch_pred.detach().cpu())
        labels_all.append(batch_labels[mask].detach().cpu())
        losses.append(loss_val.item())

    if len(preds) == 0:
        return 0.0, 0.0, [], [], 0.0, None, None

    preds_cat = torch.cat(preds, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)

    y_pred = preds_cat.numpy()
    y_true = labels_cat.numpy()

    avg_loss = float(np.mean(losses))
    acc = round(float((y_pred == y_true).mean()) * 100, 2)
    f1 = round(float(precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]) * 100, 2)
    detail_f1 = [
        round(float(x) * 100, 2)
        for x in precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[2]
    ]

    return avg_loss, acc, labels_cat, preds_cat, f1, detail_f1, max_cosine_last


@torch.no_grad()
def build_stage2_embeddings_from_cache(dataloader, fusion_visual, fusion, stage1_model, device, args):
    emb_all = []
    label_all = []

    stage1_model.eval()
    fusion_visual.eval()
    fusion.eval()

    for batch in tqdm(dataloader, desc="build_stage2_embeddings"):
        text_hidden, video_hidden_2, video_hidden, audio_hidden, batch_labels = batch

        text_hidden = text_hidden.to(device, non_blocking=True).float()
        video_hidden_2 = video_hidden_2.to(device, non_blocking=True).float()
        video_hidden = video_hidden.to(device, non_blocking=True).float()
        audio_hidden = audio_hidden.to(device, non_blocking=True).float()
        batch_labels = batch_labels.to(device, non_blocking=True)

        head, tail = hmf_slots(args, text_hidden, audio_hidden)
        pred_video_hidden, _ = fusion_visual(
            text_embedding=head, visual=video_hidden_2, visual_2=video_hidden
        )
        pred_hidden, _ = fusion(
            text_embedding=head, visual=pred_video_hidden, acoustic=tail
        )

        _, masked_mapped_output, _, _ = stage1_model(pred_hidden, return_mask_output=True)

        if torch.isnan(masked_mapped_output).any() or torch.isinf(masked_mapped_output).any():
            raise ValueError("Invalid stage2 embedding detected")

        if args.stage2_normalize:
            masked_mapped_output = F.normalize(masked_mapped_output, p=2, dim=-1)

        emb_all.append(masked_mapped_output.detach().cpu())
        label_all.append(batch_labels.detach().cpu())

    return torch.cat(emb_all, dim=0), torch.cat(label_all, dim=0)


def build_scheduler(args, optimizer, num_warmup_steps, num_training_steps):
    if args.scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def optimizer_training_steps(loader_length, epochs, accumulation_step):
    """Scheduler steps must match optimizer.step(), not micro-batch iterations."""
    if accumulation_step < 1:
        raise ValueError("accumulation_step must be >= 1")
    return math.ceil(loader_length / accumulation_step) * epochs


def main(args):
    seed_everything(args.seed)

    if args.fv_hidden_size != args.representation_dim or args.f_hidden_size != args.representation_dim:
        raise ValueError(
            "fv_hidden_size and f_hidden_size must equal representation_dim; "
            f"got {args.fv_hidden_size}, {args.f_hidden_size}, {args.representation_dim}"
        )
    if not args.random_init_fusion and (not args.fusion_visual_ckpt or not args.fusion_ckpt):
        raise ValueError("fusion checkpoints are required unless --random_init_fusion is set")
    ASFVisualClass, ASFClass, CLModelClass, ClassifierClass = resolve_model_classes(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA가 필요합니다.")

    ensure_dir(args.save_dir)

    stage1_ckpt_path = os.path.join(args.save_dir, f"stage1_{args.save_name}.pkl")
    # CLModel keeps `fusion` as a submodule so it rides along in the state dict,
    # but fusion_visual does not -- and with --unfreeze_fusion it is trained too.
    # Save it beside the Stage-1 checkpoint or Stage-2 cannot be rebuilt later.
    fv_ckpt_path = stage1_ckpt_path.replace(".pkl", ".fusion_visual.pkl")
    stage2_ckpt_path = os.path.join(args.save_dir, f"stage2_{args.save_name}.pkl")

    train_cache_dir = os.path.join(args.cache_root, "train")
    dev_cache_dir = os.path.join(args.cache_root, "dev")
    test_cache_dir = os.path.join(args.cache_root, "test")

    train_dataset = FinalMultimodalCacheDataset(train_cache_dir, args.representation_dim)
    dev_dataset = FinalMultimodalCacheDataset(dev_cache_dir, args.representation_dim)
    test_dataset = None if args.stage2_validation_only else FinalMultimodalCacheDataset(
        test_cache_dir, args.representation_dim
    )

    train_labels_tensor = torch.tensor(train_dataset.labels, dtype=torch.long)
    class_counts = torch.bincount(train_labels_tensor, minlength=args.clsNum).float()

    print("stage1 class_counts:", class_counts.tolist())
    print(f"stage1_balance_power: {args.stage1_balance_power}")

    balance_base = build_stage1_balance_base(class_counts, args.stage1_balance_power)
    print("stage1 balance_base:", balance_base.tolist())

    train_sampler = None
    if args.use_weighted_sampler:
        sample_weights = balance_base[train_labels_tensor]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        print("[Stage1] WeightedRandomSampler enabled")

    if args.use_stage1_class_weight:
        stage1_class_weights = balance_base / balance_base.mean()
        args.stage1_class_weights = stage1_class_weights.to(device)
        print("[Stage1] class weights:", stage1_class_weights.tolist())
    else:
        args.stage1_class_weights = None

    train_loader_stage1 = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=stack_collate,
        persistent_workers=(args.num_workers > 0),
    )
    train_loader_full = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=stack_collate,
        persistent_workers=(args.num_workers > 0),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=stack_collate,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=max(1, args.batch_size * 2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=stack_collate,
            persistent_workers=(args.num_workers > 0),
        )

    print(f"train samples: {len(train_dataset)}")
    print(f"dev samples:   {len(dev_dataset)}")
    print(
        f"test samples:  {len(test_dataset)}" if test_dataset is not None
        else "test samples:  NOT LOADED (--stage2_validation_only)"
    )

    fusion_visual = ASFVisualClass(
        args.clsNum, args.fv_hidden_size, args.fv_beta_shift, args.fv_dropout_prob, args.fv_num_head,
    ).to(device)
    if args.random_init_fusion:
        print("[Stage1] corrected HMF random initialization; legacy fusion checkpoints are not loaded")
    else:
        fusion_visual.load_state_dict(torch.load(args.fusion_visual_ckpt, map_location="cpu"), strict=True)

    fusion = ASFClass(
        args.clsNum, args.f_hidden_size, args.f_beta_shift, args.f_dropout_prob, args.f_num_head,
    ).to(device)
    if not args.random_init_fusion:
        fusion.load_state_dict(torch.load(args.fusion_ckpt, map_location="cpu"), strict=True)

    if args.unfreeze_fusion:
        for p in fusion_visual.parameters():
            p.requires_grad_(True)
        for p in fusion.parameters():
            p.requires_grad_(True)
        fusion_visual.train()
        fusion.train()
        print(f"[Stage1] fusion unfrozen with fusion_lr={args.fusion_lr}")
    else:
        for p in fusion_visual.parameters():
            p.requires_grad_(False)
        for p in fusion.parameters():
            p.requires_grad_(False)
        fusion_visual.eval()
        fusion.eval()
        print("[Stage1] fusion frozen")

    stage1_model = CLModelClass(args, args.clsNum, fusion).to(device)

    fusion_param_ids = set()
    if args.unfreeze_fusion:
        fusion_param_ids = {id(p) for p in fusion.parameters()}

    optim_groups = get_paramsgroup(stage1_model, lr=args.learning_rate, weight_decay=0.01, exclude_param_ids=fusion_param_ids)
    if args.unfreeze_fusion:
        optim_groups += get_paramsgroup(fusion, lr=args.fusion_lr, weight_decay=0.01)
        optim_groups += get_paramsgroup(fusion_visual, lr=args.fusion_lr, weight_decay=0.01)

    optimizer = AdamW(optim_groups)

    num_training_steps = optimizer_training_steps(
        len(train_loader_stage1), args.epochs, args.accumulation_step
    )
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = build_scheduler(args, optimizer, num_warmup_steps, num_training_steps)

    print(f"[Scheduler] {args.scheduler} | warmup_steps={num_warmup_steps} | total_steps={num_training_steps}")
    print(f"[LabelSmoothing] {args.label_smoothing}")

    best_valid_fscore = 0.0
    best_stage1_epoch = None
    best_stage1_test_label = None
    best_stage1_test_pred = None

    if args.stage1_ckpt:
        # Stage-2-only mode: reuse a finished Stage-1 checkpoint so a Stage-2
        # hyperparameter sweep costs minutes instead of retraining Stage 1.
        stage1_ckpt_path = args.stage1_ckpt
        fv_ckpt_path = stage1_ckpt_path.replace(".pkl", ".fusion_visual.pkl")
        if not os.path.isfile(fv_ckpt_path):
            raise SystemExit(
                f"\n{fv_ckpt_path} is missing.\n"
                "  The Stage-1 checkpoint carries CLModel and `fusion` (CLModel keeps it\n"
                "  as a submodule), but NOT fusion_visual -- which --unfreeze_fusion also\n"
                "  trains. Rebuilding Stage-2 embeddings without it would silently use a\n"
                "  randomly initialised fusion_visual and produce meaningless numbers.\n"
                "  Re-run Stage 1 with the current code, which saves this file.\n")
        fusion_visual.load_state_dict(torch.load(fv_ckpt_path, map_location="cpu"))
        fusion_visual = fusion_visual.to(device).eval()
        print(f"[Stage1] skipped; loaded {stage1_ckpt_path} + {fv_ckpt_path}")
    else:
        print("\n===== Stage 1 Training =====")
        for e in tqdm(range(args.epochs), desc="stage1_epochs"):
            start_time = time.time()

            train_loss, train_acc, _, _, train_fscore, _, _ = train_or_eval_model_from_cache(
                fusion_visual, fusion, stage1_model, loss_function,
                train_loader_stage1, e, device, args,
                optimizer=optimizer, scheduler=scheduler, train=True,
            )
            valid_loss, valid_acc, _, _, valid_fscore, _, _ = train_or_eval_model_from_cache(
                fusion_visual, fusion, stage1_model, loss_function,
                dev_loader, e, device, args, train=False,
            )
            print(
                "Epoch: {}, train_loss: {:.6f}, train_acc: {:.6f}, train_fscore: {:.6f}, "
                "valid_loss: {:.6f}, valid_acc: {:.6f}, valid_fscore: {:.6f}, time: {:.2f} sec".format(
                    e + 1,
                    train_loss, train_acc, train_fscore,
                    valid_loss, valid_acc, valid_fscore,
                    time.time() - start_time,
                )
            )

            if args.save_stage1_every_epoch:
                epoch_ckpt_path = os.path.join(args.save_dir, f"stage1_epoch{e + 1}_{args.save_name}.pkl")
                torch.save(stage1_model.state_dict(), epoch_ckpt_path)

            if valid_fscore > best_valid_fscore:
                best_valid_fscore = valid_fscore
                best_stage1_epoch = e + 1
                torch.save(stage1_model.state_dict(), stage1_ckpt_path)
                torch.save(fusion_visual.state_dict(), fv_ckpt_path)
                print(f"[Stage1] saved best-valid model -> {stage1_ckpt_path}")

    print("finish stage 1 training!")

    stage1_model.load_state_dict(torch.load(stage1_ckpt_path, map_location="cpu"))
    stage1_model = stage1_model.to(device).eval()
    # fusion_visual is trained beside Stage 1 but is not a CLModel submodule.
    # Reload its validation-best companion before the one-time test evaluation
    # and before building Stage-2 embeddings; otherwise final-epoch visual
    # weights are silently mixed with the validation-best Stage-1 checkpoint.
    fusion_visual.load_state_dict(torch.load(fv_ckpt_path, map_location="cpu"))
    fusion_visual = fusion_visual.to(device).eval()
    fusion.eval()
    if args.stage2_validation_only:
        best_stage1_test_fscore = None
        print("[Stage1] test not loaded or evaluated during Stage-2 selection")
    else:
        _, _, best_stage1_test_label, best_stage1_test_pred, best_stage1_test_fscore, _, _ = train_or_eval_model_from_cache(
            fusion_visual, fusion, stage1_model, loss_function, test_loader, 0, device, args, train=False,
        )
        print(f"[Stage1] validation-best test F1: {best_stage1_test_fscore:.2f}")
        print(confusion_matrix(flatten_to_numpy(best_stage1_test_label), flatten_to_numpy(best_stage1_test_pred)))
    if not args.stage1_ckpt:
      with open(os.path.join(args.save_dir, f"stage1_selection_{args.save_name}.json"), "x") as handle:
          record = {"checkpoint": stage1_ckpt_path, "selection_split": "validation",
                    "selection_metric": "weighted_f1", "best_epoch": best_stage1_epoch,
                    "best_validation_f1": best_valid_fscore,
                    "test_loaded_or_evaluated": not args.stage2_validation_only}
          if best_stage1_test_fscore is not None:
              record["test_f1"] = best_stage1_test_fscore
              record["test_evaluated_once_after_selection"] = True
          json.dump(record, handle, indent=2)

    if args.skip_stage2:
        print("[Stage2] skipped by --skip_stage2")
        print("---------------Done--------------")
        return

    # --- Stage 2 ---
    print("\n===== Build Stage2 Embeddings =====")
    torch.cuda.empty_cache()

    stage1_model.load_state_dict(torch.load(stage1_ckpt_path, map_location="cpu"))
    stage1_model = stage1_model.to(device)
    stage1_model.eval()
    fusion_visual.eval()
    fusion.eval()

    with torch.no_grad():
        anchors = stage1_model.map_function_anchor(stage1_model.emo_anchor)
        if args.stage2_normalize:
            anchors = F.normalize(anchors, p=2, dim=-1)

    emb_train, label_train = build_stage2_embeddings_from_cache(train_loader_full, fusion_visual, fusion, stage1_model, device, args)
    emb_val, label_val = build_stage2_embeddings_from_cache(dev_loader, fusion_visual, fusion, stage1_model, device, args)
    emb_test = label_test = None
    if test_loader is not None:
        emb_test, label_test = build_stage2_embeddings_from_cache(test_loader, fusion_visual, fusion, stage1_model, device, args)

    trainset = TensorDataset(emb_train, label_train)
    validset = TensorDataset(emb_val, label_val)
    testset = TensorDataset(emb_test, label_test) if emb_test is not None else None

    train_loader_stage2 = DataLoader(trainset, batch_size=args.stage2_batch_size, shuffle=True, pin_memory=True)
    valid_loader_stage2 = DataLoader(validset, batch_size=args.stage2_batch_size, shuffle=False)
    test_loader_stage2 = (
        DataLoader(testset, batch_size=args.stage2_batch_size, shuffle=False)
        if testset is not None else None
    )

    clf_args = args
    if args.stage2_temp is not None:
        clf_args = copy.copy(args)
        clf_args.temp = args.stage2_temp
    clf = ClassifierClass(clf_args, anchors).to(device)
    print(f"[Stage2] lr={args.stage_two_lr} weight_decay={args.stage2_weight_decay} "
          f"temp={clf_args.temp} epochs={args.stage2_epochs}")
    optimizer_stage2 = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr,
                                        weight_decay=args.stage2_weight_decay)

    if args.use_stage2_class_weight:
        s2_counts = torch.bincount(label_train[label_train >= 0], minlength=args.clsNum).float()
        s2_weights = 1.0 / torch.sqrt(s2_counts + 1e-8)
        s2_weights = s2_weights / s2_weights.mean()
        criterion_stage2 = nn.CrossEntropyLoss(ignore_index=-1, weight=s2_weights.to(device))
    else:
        criterion_stage2 = nn.CrossEntropyLoss(ignore_index=-1)

    best_valid_fscore = 0.0
    best_stage2_epoch = None
    all_fscore = []
    best_stage2_test_label = None
    best_stage2_test_pred = None
    best_stage2_detail_f1 = None

    print("\n===== Stage 2 Training =====")
    for e in range(args.stage2_epochs):
        train_loss, train_ce_loss, train_acc, _, _, train_fscore, _ = retrain(
            clf, criterion_stage2, train_loader_stage2, e, device, args, optimizer=optimizer_stage2, train=True
        )
        valid_loss, valid_ce_loss, valid_acc, _, _, valid_fscore, _ = retrain(
            clf, criterion_stage2, valid_loader_stage2, e, device, args, train=False
        )
        print(
            "Epoch: {}, train_loss: {:.6f}, train_acc: {:.6f}, train_fscore: {:.6f}, "
            "valid_loss: {:.6f}, valid_acc: {:.6f}, valid_fscore: {:.6f}".format(
                e + 1, train_loss, train_acc, train_fscore,
                valid_loss, valid_acc, valid_fscore,
            )
        )

        if valid_fscore > best_valid_fscore:
            best_valid_fscore = valid_fscore
            best_stage2_epoch = e + 1
            torch.save(clf.state_dict(), stage2_ckpt_path)
            print(f"[Stage2] saved best model -> {stage2_ckpt_path}")
    if args.stage2_validation_only:
        with open(os.path.join(args.save_dir, f"stage2_selection_{args.save_name}.json"), "x") as handle:
            json.dump({"checkpoint": stage2_ckpt_path, "selection_split": "validation",
                       "selection_metric": "weighted_f1", "best_epoch": best_stage2_epoch,
                       "best_validation_f1": best_valid_fscore,
                       "test_loaded_or_evaluated": False}, handle, indent=2)
        print("[Stage2] validation-only sweep candidate; test was not loaded or evaluated")
        print("---------------Done--------------")
        return
    clf.load_state_dict(torch.load(stage2_ckpt_path, map_location="cpu"))
    _, _, _, best_stage2_test_label, best_stage2_test_pred, best_stage2_test_fscore, _ = retrain(
        clf, criterion_stage2, test_loader_stage2, 0, device, args, train=False
    )
    print(f"[Stage2] validation-best test F1: {best_stage2_test_fscore:.2f}")
    print(confusion_matrix(flatten_to_numpy(best_stage2_test_label), flatten_to_numpy(best_stage2_test_pred)))
    with open(os.path.join(args.save_dir, f"stage2_selection_{args.save_name}.json"), "x") as handle:
        json.dump({"checkpoint": stage2_ckpt_path, "selection_split": "validation",
                   "selection_metric": "weighted_f1", "best_epoch": best_stage2_epoch,
                   "best_validation_f1": best_valid_fscore,
                   "test_f1": best_stage2_test_fscore,
                   "test_evaluated_once_after_selection": True}, handle, indent=2)

    print("---------------Done--------------")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
