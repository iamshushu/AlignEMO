"""Verify that every HMF (ASF / ASF_visual) call site agrees on modality-slot order.

Why this exists
---------------
``ASF.forward(text_embedding, visual, acoustic)`` builds ``cat((visual, acoustic))``
and feeds it to weights that were trained under one specific slot assignment.
Swapping two 768-d (or 1280-d) halves therefore produces a *different model*, not a
cosmetic change -- but it raises no error, so a mismatch between the script that
trained a fusion checkpoint and the script that later loads it is silent.

Two modes:

  static   (default) -- parse every .py under the repo, resolve each fusion(...) /
           fusion_visual(...) call to a (slot0, slot1, slot2) modality triple, and
           report per-dataset whether all call sites agree.

  runtime  -- load a frozen fusion checkpoint and evaluate the Stage-1B classifier
           under *each* candidate order on the test split. The order the checkpoint
           was actually trained with wins by a wide margin, so this settles
           "which convention does this .bin expect?" empirically rather than by
           reading git history.

Usage
-----
    python scripts/ablation/check_fusion_order.py                     # static, whole repo
    python scripts/ablation/check_fusion_order.py --paths IEMOCAP     # static, subset

    # Which order does the paper's IEMOCAP HMF checkpoint expect?
    D=./IEMOCAP/save_model
    python scripts/ablation/check_fusion_order.py --runtime iemocap \
        --teacher_ckpt        "$D/teacher_csv2_lr1e_6.bin" \
        --audio_ckpt          "$D/student_audio_csv/total_student.bin" \
        --visual_ckpt         "$D/student_video_csv/total_student.bin" \
        --visual_masked_ckpt  "$D/student_video_masked_csv_0827/total_student_best.bin" \
        --fusion_visual_ckpt  "$D/IEMOCAP_demo/total_fusion_student_csv_visual_0827_2.bin" \
        --fusion_ckpt         "$D/IEMOCAP_demo/total_fusion_student_csv_visual_all_0827_5.bin"

    # Which order does the K-MIND HMF checkpoint expect?
    python scripts/ablation/check_fusion_order.py --runtime kmind \
        --cache_root ./K-MIND/cache_total_multimodal_rowwise_kfdb \
        --fusion_visual_ckpt ./K-MIND/save_model/0325_fusion_visual_grid/fv_lr1e-5_bs4_dp0.2_h8_b0.2_hs1280_seed42.bin \
        --fusion_ckpt        ./K-MIND/save_model/0421_fusion_kfdb/fusion_lr1e-5_bs4_dp0.1_h8_b0.3_hs1280_seed42.bin

Exit code is 1 if the static scan finds a disagreement, so this can gate CI.
"""
import argparse
import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Ordered longest-first: `video_hidden_2` must win over `video_hidden`.
MODALITY_PATTERNS = [
    (r"^(pred_video_hidden|pred_hidden|pvh)$", "V-fused"),   # output of fusion_visual
    (r"^(video_hidden_2|vh2|v2_h|v2|visual_2)$", "v2-face"),
    (r"^(video_hidden|vh|v1_h|v1|video)$", "v1-scene"),
    (r"^(audio_hidden|audio_h|ah|audio)$", "AUDIO"),
    (r"^(text_hidden|text_h|th|text)$", "TEXT"),
]

# (function name -> parameter names, in positional order)
SIGNATURES = {
    "fusion": ("text_embedding", "visual", "acoustic"),
    "fusion_visual": ("text_embedding", "visual", "visual_2"),
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints"}

# Files containing this marker are skipped by the static scan (see static_check).
EXEMPT_MARKER = "fusion-order-probe: exempt"  # fusion-order-probe: exempt


def load_or_explain(module, ckpt, what):
    """load_state_dict, but turn the HMF-refactor shape clash into a real explanation."""
    import torch
    state = torch.load(ckpt, map_location="cpu")
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as err:
        ckpt_shape = next((tuple(v.shape) for k, v in state.items()
                           if k.endswith("in_proj_weight")), None)
        want_shape = next((tuple(v.shape) for k, v in module.state_dict().items()
                           if k.endswith("in_proj_weight")), None)
        if ckpt_shape and want_shape and ckpt_shape != want_shape:
            raise SystemExit(
                f"\n{'!' * 72}\n"
                f"INCOMPATIBLE CHECKPOINT -- {what}\n"
                f"{'!' * 72}\n"
                f"  {ckpt}\n\n"
                f"  checkpoint in_proj_weight : {ckpt_shape}\n"
                f"  current class expects     : {want_shape}\n\n"
                "  The HMF attention was refactored: MultiheadAttention used to run on\n"
                "  the concatenated (B, 2*D) vector -- which PyTorch reads as an\n"
                "  UNBATCHED (L=B, E) sequence, so utterances attended to each other\n"
                "  across the batch. PairwiseModalityAttention now stacks the two\n"
                "  modalities as 2 tokens of D with batch_first=True, which is correct\n"
                "  but is a different architecture.\n\n"
                "  Every checkpoint trained before that refactor is unloadable, and its\n"
                "  numbers came from the cross-batch-attention model. There is no\n"
                "  conversion -- these stages have to be retrained.\n"
                f"{'!' * 72}\n") from err
        raise
    return module


def classify(name):
    for pattern, label in MODALITY_PATTERNS:
        if re.match(pattern, name):
            return label
    return f"?{name}"


def expr_name(node):
    """Best-effort source name for a call argument."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return expr_name(node.func.value)
    return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"


def scan_file(path):
    """Return [(lineno, fn_name, (slot0, slot1, slot2))] for every fusion call."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        fn = node.func.id
        if fn not in SIGNATURES:
            continue

        params = SIGNATURES[fn]
        slots = [None] * len(params)
        for i, arg in enumerate(node.args):
            if i < len(slots):
                slots[i] = classify(expr_name(arg))
        for kw in node.keywords:
            if kw.arg in params:
                slots[params.index(kw.arg)] = classify(expr_name(kw.value))

        if any(s is None for s in slots):
            continue  # not a full 3-arg call (e.g. a partial/pipe helper)
        found.append((node.lineno, fn, tuple(slots)))
    return found


def group_of(path):
    """Which pipeline a file belongs to, for grouping the report."""
    rel = path.relative_to(REPO)
    top = rel.parts[0]
    if top in ("IEMOCAP", "MELD", "K-MIND"):
        return top
    name = rel.name.lower()
    for ds in ("iemocap", "meld", "kmind"):
        if ds in name:
            return {"iemocap": "IEMOCAP", "meld": "MELD", "kmind": "K-MIND"}[ds]
    return "scripts (dataset-agnostic)"


def static_check(paths):
    files = []
    for p in paths:
        root = REPO / p
        if root.is_file():
            files.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            files.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))

    # Probe scripts deliberately call every order in a loop; they are not call sites
    # a checkpoint has to agree with. They opt out with the marker below.
    files = [f for f in files if EXEMPT_MARKER not in f.read_text(
        encoding="utf-8", errors="replace")]

    # group -> fn -> order -> [call sites]
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(set(files)):
        for lineno, fn, order in scan_file(f):
            site = f"{f.relative_to(REPO)}:{lineno}"
            table[group_of(f)][fn][order].append(site)

    conflicts = 0
    for group in sorted(table):
        print(f"\n{'=' * 72}\n{group}\n{'=' * 72}")
        for fn in sorted(table[group]):
            orders = table[group][fn]
            ok = len(orders) == 1
            if not ok:
                conflicts += 1
            mark = "OK  " if ok else "FAIL"
            print(f"\n  [{mark}] {fn}(...)  -- {len(orders)} distinct order(s)")
            # Most-used order first, so the majority convention reads as the baseline.
            for order, sites in sorted(orders.items(), key=lambda kv: -len(kv[1])):
                slots = SIGNATURES[fn]
                mapping = ", ".join(f"{s}={m}" for s, m in zip(slots, order))
                print(f"        {mapping}")
                for site in sites:
                    print(f"            {site}")

    print(f"\n{'=' * 72}")
    if conflicts:
        print(f"RESULT: {conflicts} group(s) disagree on slot order -- see FAIL above.")
        print("A checkpoint trained under one order and loaded under another silently")
        print("receives permuted features. Pick one convention per dataset and retrain")
        print("or re-map, then re-run this check.")
    else:
        print("RESULT: every call site within each group agrees. "
              "(This does NOT prove the order matches the checkpoints -- "
              "use --runtime for that.)")
    print("=" * 72)
    return conflicts


# ---------------------------------------------------------------------------
# runtime: which order does this .bin actually expect?
# ---------------------------------------------------------------------------
def _wf1(golds, preds):
    from sklearn.metrics import f1_score
    return round(f1_score(golds, preds, average="weighted") * 100, 2)


def runtime_iemocap(args):
    import torch
    from dataclasses import dataclass
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(REPO / "IEMOCAP"))
    from model import Teacher_model, Student_Audio, Student_Video, ASF, ASF_visual
    from preprocessing import preprocessing_pickle
    from dataset import iemocap_dataset
    from utils import make_batchs_pickle

    device = "cuda" if torch.cuda.is_available() else "cpu"

    @dataclass
    class Config:
        mask_time_length: int = 3

    ds = iemocap_dataset(preprocessing_pickle(args.test_csv, args.pickle))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=make_batchs_pickle)
    clsNum = len(ds.emoList)

    def frozen(model, ckpt, what, strict=True):
        if strict:
            load_or_explain(model, ckpt, what)
        else:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        for p in model.parameters():
            p.requires_grad = False
        return model.to(device).eval()

    model_t = frozen(Teacher_model(args.text_model, clsNum), args.teacher_ckpt,
                     "text teacher", strict=False)
    audio_s = frozen(Student_Audio(args.audio_model, clsNum, Config()), args.audio_ckpt, "audio student")
    video_s = frozen(Student_Video(args.video_model, clsNum), args.visual_ckpt, "visual student")
    video_m = frozen(Student_Video(args.video_model, clsNum), args.visual_masked_ckpt, "masked visual student")
    fusion_visual = frozen(ASF_visual(clsNum, 768, 2e-1, 0.2, 4), args.fusion_visual_ckpt, "fusion_visual (ASF_visual)")
    fusion = frozen(ASF(clsNum, 768, 2e-1, 0.2, 4), args.fusion_ckpt, "fusion (ASF)")

    # (label, builds the (visual, acoustic) pair fed to ASF)
    orders = {
        "release (text, visual=VIDEO, acoustic=AUDIO)": lambda v, a: (v, a),
        "paper   (text, visual=AUDIO, acoustic=VIDEO)": lambda v, a: (a, v),
    }
    preds = {k: [] for k in orders}
    golds = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            bi, am, ai, vi, vi2, bl = [x.to(device) for x in batch]
            th, _ = model_t(bi, am)
            vh, _ = video_s(vi)
            vh2, _ = video_m(vi2)
            ah, _ = audio_s(ai)
            pvh, _ = fusion_visual(text_embedding=th, visual=vh2, visual_2=vh)
            for key, pick in orders.items():
                vis, aco = pick(pvh, ah)
                _, logits = fusion(text_embedding=th, visual=vis, acoustic=aco)
                preds[key].extend(logits.argmax(1).cpu().numpy())
            golds.extend(bl.cpu().numpy())

    _report_runtime("IEMOCAP", args.fusion_ckpt, golds, preds)


def runtime_kmind(args):
    import glob
    import torch
    from torch.utils.data import DataLoader, Dataset

    sys.path.insert(0, str(REPO / "K-MIND"))
    from teacher_audio_model import ASF, ASF_visual

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class CacheDS(Dataset):
        def __init__(self, d):
            self.files = sorted(glob.glob(os.path.join(d, "*.pt")))
            if not self.files:
                raise RuntimeError(f"No .pt files in {d}")

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            o = torch.load(self.files[i], map_location="cpu")
            return o["text"], o["v2"], o["v1"], o["audio"], o["y"]

    def collate(b):
        t, v2, v1, a, y = zip(*b)
        return (torch.stack(t, 0), torch.stack(v2, 0), torch.stack(v1, 0),
                torch.stack(a, 0), torch.stack([yy.view(-1)[0] for yy in y], 0).long())

    loader = DataLoader(CacheDS(os.path.join(args.cache_root, "test")),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    def frozen(model, ckpt, what):
        load_or_explain(model, ckpt, what)
        for p in model.parameters():
            p.requires_grad = False
        return model.to(device).eval()

    fusion_visual = frozen(ASF_visual(args.clsNum, args.fv_hidden_size, args.fv_beta_shift,
                                      args.fv_dropout_prob, args.fv_num_head),
                           args.fusion_visual_ckpt, "fusion_visual (ASF_visual)")
    fusion = frozen(ASF(args.clsNum, args.f_hidden_size, args.f_beta_shift,
                        args.f_dropout_prob, args.f_num_head), args.fusion_ckpt,
                   "fusion (ASF)")

    # K-MIND has two live conventions in the tree; test both anchor modalities.
    orders = {
        "grid    (text-anchored: fv(text,v2,v1); f(text, V, AUDIO))": "text",
        "ELACL   (audio-anchored: fv(audio,v2,v1); f(audio, V, TEXT))": "audio",
    }
    preds = {k: [] for k in orders}
    golds = []

    with torch.no_grad():
        for t, v2, v1, a, y in loader:
            t, v2, v1, a = [x.to(device).float() for x in (t, v2, v1, a)]
            for key, anchor in orders.items():
                head, tail = (t, a) if anchor == "text" else (a, t)
                pvh, _ = fusion_visual(text_embedding=head, visual=v2, visual_2=v1)
                _, logits = fusion(text_embedding=head, visual=pvh, acoustic=tail)
                preds[key].extend(logits.argmax(1).cpu().numpy())
            golds.extend(y.numpy())

    _report_runtime("K-MIND", args.fusion_ckpt, golds, preds)


def _report_runtime(dataset, ckpt, golds, preds):
    print(f"\n{'=' * 72}")
    print(f"RUNTIME ORDER PROBE -- {dataset}")
    print(f"  checkpoint: {ckpt}")
    print(f"  test utterances: {len(golds)}")
    print("=" * 72)
    scores = {k: _wf1(golds, v) for k, v in preds.items()}
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6.2f}  w-F1   {k}")
    best, second = sorted(scores.values(), reverse=True)[:2]
    gap = best - second
    print("-" * 72)
    if gap < 1.0:
        print(f"  INCONCLUSIVE: top two orders differ by only {gap:.2f} pp.")
        print("  Either the checkpoint is near-random, or the modalities are")
        print("  interchangeable here. Do not use this to justify an order.")
    else:
        winner = max(scores, key=scores.get)
        print(f"  -> This checkpoint expects:  {winner}")
        print(f"     (beats the runner-up by {gap:.2f} pp w-F1)")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", nargs="*", default=["IEMOCAP", "MELD", "K-MIND", "scripts"],
                   help="repo-relative dirs/files to scan (static mode)")
    p.add_argument("--runtime", choices=["iemocap", "kmind"], default=None,
                   help="also probe a checkpoint empirically")
    # shared
    p.add_argument("--fusion_ckpt")
    p.add_argument("--fusion_visual_ckpt")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    # iemocap runtime
    p.add_argument("--test_csv", default="./dataset/IEMOCAP_full_release/IEMOCAP_test_2.csv")
    p.add_argument("--pickle", default="./dataset/IEMOCAP_full_release/faces_data_627.pkl")
    p.add_argument("--text_model", default="roberta-large")
    p.add_argument("--audio_model", default="facebook/data2vec-audio-base-960h")
    p.add_argument("--video_model", default="facebook/timesformer-base-finetuned-k400")
    p.add_argument("--teacher_ckpt")
    p.add_argument("--audio_ckpt")
    p.add_argument("--visual_ckpt")
    p.add_argument("--visual_masked_ckpt")
    # kmind runtime
    p.add_argument("--cache_root", default="./K-MIND/cache_total_multimodal_rowwise_kfdb")
    p.add_argument("--clsNum", type=int, default=7)
    p.add_argument("--fv_hidden_size", type=int, default=1280)
    p.add_argument("--fv_beta_shift", type=float, default=0.2)
    p.add_argument("--fv_dropout_prob", type=float, default=0.2)
    p.add_argument("--fv_num_head", type=int, default=8)
    p.add_argument("--f_hidden_size", type=int, default=1280)
    p.add_argument("--f_beta_shift", type=float, default=0.3)
    p.add_argument("--f_dropout_prob", type=float, default=0.1)
    p.add_argument("--f_num_head", type=int, default=8)
    args = p.parse_args()

    conflicts = static_check(args.paths)

    if args.runtime == "iemocap":
        runtime_iemocap(args)
    elif args.runtime == "kmind":
        runtime_kmind(args)

    sys.exit(1 if conflicts else 0)


if __name__ == "__main__":
    main()
