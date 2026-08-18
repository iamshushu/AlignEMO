# AlignEMO

Official implementation of **AlignEMO: Teacher-Guided Multimodal Emotion Representation
Learning and Emotion Label Anchored Contrastive Learning for Emotion Recognition in
Conversation**.

AlignEMO is a two-stage multimodal ERC framework over text, audio, and two complementary
visual streams (a **scene** image and a **focused facial** image):

- **Stage 1 — Teacher-Guided Multimodal Emotion Representation Learning**
  - **Stage 1A · CMKD** (Cross-Modal Knowledge Distillation). The strongest unimodal
    modality for each dataset acts as the teacher — *text* on IEMOCAP and MELD, *audio*
    on K-MIND — and distils into the remaining student encoders at both the response and
    the relational-feature level.
  - **Stage 1B · HMF** (Hierarchical Multimodal Fusion). Gated displacement vectors refine
    the teacher representation in two steps: the two visual streams are merged first, then
    the remaining modality is incorporated.
- **Stage 2 — ELACL** (Emotion Label Anchored Contrastive Learning). The fused multimodal
  representation is aligned with emotion-label anchors in a shared embedding space, then
  the anchors themselves are refined.

Reported weighted-F1: **IEMOCAP 70.83 · MELD 67.49 · K-MIND 75.29**.

---

## 1. Quickstart

```bash
git clone <this-repo> AlignEMO && cd AlignEMO

conda env create -f environment.yml && conda activate TelME2

# Tell the code where your datasets live (see §4)
ln -s /path/to/IEMOCAP_full_release dataset/IEMOCAP_full_release

# Verify env, GPU, backbones, data paths and anchors before a multi-hour run
python check_setup.py --dataset IEMOCAP

python generate_anchors.py --text_model roberta-large   # §5
bash scripts/slurm/run_iemocap.sbatch                   # or sbatch, see §6
```

`check_setup.py` is the fastest way to find out whether your machine is ready. It never
trains or downloads a dataset; it just reports what is reachable and what is missing.

> **A GPU is required.** The fusion modules allocate with `.cuda()` unconditionally, so
> both training and evaluation need a CUDA device. All reported experiments used a single
> NVIDIA GeForce RTX 4090.

---

## 2. Repository layout

```
AlignEMO/
├── check_setup.py               preflight: env / GPU / backbones / data / anchors
├── generate_anchors.py          builds IEMOCAP + MELD emotion anchors
├── environment.yml              conda env (TelME2)
├── requirements.txt             pip-only deps on top of conda PyTorch
│
├── IEMOCAP/                     English, 6-way, dyadic — text teacher
│   ├── teacher.py                  Stage 1A teacher (RoBERTa-large)
│   ├── student_audio.py            Stage 1A audio student (data2vec, CMKD)
│   ├── student_visual.py           Stage 1A visual students (TimeSformer, CMKD)
│   ├── fusion_visual.py            Stage 1B step 1 — merge the two visual streams
│   ├── fusion.py                   Stage 1B step 2 — full HMF (+ audio)
│   ├── ELACL_fusion.py             Stage 2 — ELACL
│   ├── eval_stage1b.py             Stage 1 only, per-class F1 + confusion matrix
│   ├── eval_stage12.py             Stage 1 vs Stage 1+2, per-class F1 + confusion matrix
│   ├── model.py                    ASF / ASF_visual / PairwiseModalityAttention
│   ├── iemocap_kd.py               DIST response loss + relational feature loss
│   ├── ELACL_loss.py               scl + ang + anchor-adaptation losses
│   └── dataset.py / preprocessing.py / utils.py / ELACL_trainer.py
│
├── MELD/                        English, 7-way, multi-party — text teacher
│   └── same structure as IEMOCAP (meld_kd.py in place of iemocap_kd.py)
│
├── K-MIND/                      Korean, 7-way Ekman — audio teacher
│   ├── teacher_audio_ekman.py      Stage 1A teacher (Whisper-large-v3)
│   ├── teacher_ekman.py            text encoder (KLUE-RoBERTa-base)
│   ├── student_visual_ekman.py     Stage 1A visual students (CMKD)
│   ├── fusion_visual_ekman_grid.py Stage 1B step 1
│   ├── fusion_ekman_grid.py        Stage 1B step 2
│   ├── ELACL_fusion.py             main entry — Stage 1B (+ Stage 2 ELACL)
│   ├── stage1b_per_class_eval.py   evaluate a frozen Stage 1B checkpoint
│   ├── build_cache_1_partial.py    rebuild the feature cache, part 1 (text + audio)
│   ├── build_cache_2_final.py      rebuild the feature cache, part 2 (+ visual)
│   ├── mask_variants.py            scene / focused-facial / masked-facial variants
│   └── generate_kmind_anchor.py    K-MIND emotion anchors
│
├── scripts/
│   ├── slurm/                   ready-to-adapt batch scripts, one per dataset
│   ├── ablation/                cached-fusion trainer, fusion variants, cache builders
│   ├── analysis/                paper tables, Figure 6, prediction dumps, efficiency
│   └── audit/                   split / media / cache integrity validators
│
├── eda/                         dataset analysis scripts + their outputs
├── figures/                     Figure 5 (t-SNE anchors), Figure 6 (confusion matrices)
└── tests/                       28 regression tests, no GPU or dataset required
```

> **Do not rename the dataset folders.** `IEMOCAP/`, `MELD/` and `K-MIND/` are referenced by
> relative path from the repository root (`./K-MIND/save_model/...`, `./IEMOCAP/emo_anchors/...`),
> and every command below is meant to be run **from the repository root**.

---

## 3. Environment

The code is developed against a conda environment named `TelME2` (Python 3.9, CUDA PyTorch).

### Option A — from the provided YAML

```bash
conda env create -f environment.yml
conda activate TelME2
```

### Option B — manual

```bash
conda create -n TelME2 python=3.9 -y
conda activate TelME2

# CUDA-enabled PyTorch matching your driver
conda install pytorch torchaudio torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg          # audio decoding

pip install -r requirements.txt
```

Then confirm with `python check_setup.py`.

Backbones are pulled from HuggingFace on first use: `roberta-large`,
`facebook/data2vec-audio-base-960h`, `facebook/timesformer-base-finetuned-k400`
(IEMOCAP / MELD) and `klue/roberta-base`, `openai/whisper-large-v3` (K-MIND).
On an offline compute node, pre-populate `HF_HOME` from a networked node first.

---

## 4. Datasets

The datasets are **not redistributed here**; obtain them from their original providers.

| Dataset | Source | Classes |
|---|---|---|
| IEMOCAP | https://sail.usc.edu/iemocap/ | 6: neutral, happy, sad, angry, excited, frustrated |
| MELD | https://affective-meld.github.io/ | 7: neutral, joy, surprise, anger, sadness, disgust, fear |
| K-MIND | Yang et al., LREC 2026 | 7 Ekman: neutral, joy, surprise, anger, sad, disgust, fear |

Make them reachable at the relative paths the scripts expect. Symlinks are the least
invasive way:

```bash
mkdir -p dataset
ln -s /path/to/IEMOCAP_full_release  dataset/IEMOCAP_full_release
ln -s /path/to/MELD.Raw              dataset/MELD.Raw
ln -s /path/to/KMIND                 dataset/KMIND
```

**Expected layout**

| Dataset | Path | Contents |
|---|---|---|
| IEMOCAP | `dataset/IEMOCAP_full_release/` | `Session{1..5}/` plus `IEMOCAP_{train,dev,test}_2.csv` |
| MELD | `dataset/MELD.Raw/` | `train_splits/`, `dev_splits_complete/`, `output_repeated_splits_test/` plus `{train,dev,test}_sent_emo.csv` |
| K-MIND | `dataset/KMIND/` | utterance-level `.wav` and video, plus the split CSVs |

**Splits.** MELD uses the official split. IEMOCAP has no speaker-disjoint validation split
in its standard partition, so we hold out dialogues from the training sessions to form one
and keep Session 5 as the test set — these are the `IEMOCAP_*_2.csv` files
(100 / 20 / 31 dialogues; 4,810 / 1,000 / 1,623 utterances). K-MIND had no public split at
the time of these experiments, so it is partitioned at comparable ratios
(587 / 65 / 164 dialogues). Verify any split with:

```bash
python scripts/audit/validate_splits.py          # dialogue overlap is a hard failure
python scripts/audit/validate_meld_media.py      # missing/empty media directories
```

If the K-MIND CSVs store absolute wav paths from another machine, rewrite them on the fly:

```bash
export KMIND_WAV_OLD_PREFIX=/original/path/to/KMIND/
export KMIND_WAV_NEW_PREFIX=/your/path/to/KMIND/
```

---

## 5. Emotion anchors

Stage 2 needs one anchor tensor per emotion label. Anchor tensors are not committed
(they are `*.pt`, excluded by `.gitignore`); generate them once:

```bash
# IEMOCAP + MELD in one run, using the same text encoder as the teacher
#   → IEMOCAP/emo_anchors/roberta-large/iemocap_emo.pt
#   → MELD/emo_anchors/roberta-large/meld_emo.pt
python generate_anchors.py --text_model roberta-large

# K-MIND → K-MIND/emo_anchors/roberta-base/kmind_emo_eng.pt
python K-MIND/generate_kmind_anchor.py \
    --text_model klue/roberta-base --lang eng \
    --out_dir K-MIND/emo_anchors/roberta-base
```

The Stage 2 scripts read them through `--anchor_path`, whose defaults already point at
these locations.

---

## 6. Running the pipelines

Run every command **from the repository root**. Each stage writes a checkpoint that the
next stage's `argparse` default already points at, so the bare commands chain end to end;
pass a flag only to override a path.

### 6.1 IEMOCAP and MELD (text teacher)

```bash
# Stage 1A — text teacher              → ./IEMOCAP/save_model/teacher.bin
python IEMOCAP/teacher.py --learning_rate 1e-6

# Stage 1A — students, distilled from the frozen teacher
#   audio  → ./IEMOCAP/save_model/student_audio/total_student.bin
#   visual → ./IEMOCAP/save_model/student_video2/total_student.bin
#            ./IEMOCAP/save_model/student_video_masked2/total_student.bin
python IEMOCAP/student_audio.py
python IEMOCAP/student_visual.py

# Stage 1B — merge the two visual streams → ./IEMOCAP/save_model/total_fusion_visual.bin
python IEMOCAP/fusion_visual.py

# Stage 1B — full HMF (+ audio)          → ./IEMOCAP/save_model/total_fusion2.bin
python IEMOCAP/fusion.py

# Stage 2 — ELACL
python IEMOCAP/ELACL_fusion.py
```

Swap `IEMOCAP` for `MELD` to run the MELD pipeline — same scripts, same flags, and each
script's defaults already carry the MELD hyperparameters from §7.

`scripts/slurm/run_iemocap.sbatch` and `run_meld.sbatch` chain all six stages in one job.
The `#SBATCH` partition/gres lines and the `conda activate` path are site-specific.

Useful overrides: `--save_dir`; `--teacher_ckpt` on the students; `--visual_ckpt` /
`--visual_masked_ckpt` on `fusion_visual.py`; `--audio_ckpt` / `--fusion_visual_ckpt` on
`fusion.py`; every upstream `--*_ckpt` plus `--fusion_ckpt` and `--anchor_path` on
`ELACL_fusion.py`. `--epochs --learning_rate --batch_size --seed` are exposed throughout.

### 6.2 K-MIND (audio teacher)

K-MIND runs off a **pre-computed per-utterance feature cache**, because Stage 1B + Stage 2
would otherwise re-encode 71k utterances with Whisper-large-v3 every epoch. Build it once,
in two parts:

```bash
# Stage 1A — audio teacher and the text/visual students
python K-MIND/teacher_audio_ekman.py
python K-MIND/student_visual_ekman.py

# Cache part 1 — text + Whisper audio features
python K-MIND/build_cache_1_partial.py \
    --cache_root         ./K-MIND/cache_audio_text_partial_rowwise \
    --audio_teacher_ckpt ./K-MIND/save_model/student_audio_2_whisper/total_student_ekman.bin \
    --text_student_ckpt  ./K-MIND/save_model/teacher_audio_student_text_2_whisper/total_student_ekman.bin

# Cache part 2 — add scene (v1) and focused-facial (v2) visual features
python K-MIND/build_cache_2_final.py \
    --partial_cache_root ./K-MIND/cache_audio_text_partial_rowwise \
    --final_cache_root   ./K-MIND/cache_total_multimodal_rowwise_kfdb \
    --visual_raw_ckpt    ./K-MIND/save_model/<scene_visual_student>/total_student_ekman.bin \
    --visual_masked_ckpt ./K-MIND/save_model/<masked_visual_student>/total_student_ekman.bin

# Project the cache to the 768-d representation used by the reported configuration.
# This trains a 1280->768 projector, selects it on validation, and writes both the
# projected cache and a provenance record (--checkpoint / --selection_json are outputs
# and must not already exist).
python scripts/ablation/project_kmind_cache_768.py \
    --source_root    ./K-MIND/cache_total_multimodal_rowwise_kfdb \
    --output_root    ./K-MIND/cache_768 \
    --checkpoint     ./K-MIND/projector/projection_validation_best.pt \
    --selection_json ./K-MIND/projector/projection_selection.json \
    --input_dim 1280 --output_dim 768 --num_classes 7 \
    --seed 42 --epochs 10 --batch_size 256 --num_workers 6 \
    --learning_rate 1e-3 --weight_decay 1e-4
```

`mask_variants.py` supplies the scene / focused-facial / masked-facial variants; pass
`--v1_variant` / `--v2_variant` to build the caches for the Table 5 visual ablation.
Validate a finished cache with `python scripts/audit/validate_kmind_cache_splits.py`.

Then train Stage 1B + Stage 2 in one entry point:

```bash
python K-MIND/ELACL_fusion.py \
    --cache_root ./K-MIND/cache_768 \
    --random_init_fusion \
    --representation_dim 768 --final_fusion_variant hmf \
    --save_dir ./K-MIND/save_model/stage12 --save_name kmind_768_hmf_seed42 \
    --learning_rate 3e-5 --batch_size 4 --accumulation_step 4 \
    --epochs 15 --warmup_ratio 0.05 --seed 42 --clsNum 7 \
    --num_workers 4 --dataset_name K-MIND \
    --ce_loss_weight 0.8 --angle_loss_weight 0.3 --temp 0.15 --weight_decay 1e-3 \
    --anchor_path ./K-MIND/emo_anchors/roberta-base/ \
    --fv_hidden_size 768 --fv_beta_shift 0.2 --fv_dropout_prob 0.2 --fv_num_head 8 \
    --f_hidden_size  768 --f_beta_shift  0.3 --f_dropout_prob  0.1 --f_num_head  8 \
    --stage1_balance_power 0.5 --unfreeze_fusion --fusion_lr 2e-6 \
    --label_smoothing 0.05 --scheduler cosine \
    --stage2_epochs 20 --stage_two_lr 4e-4 --stage2_weight_decay 0.0
```

`scripts/slurm/run_kmind.sbatch` submits this across seeds 42 / 13 / 21 / 87 / 100.

For the Stage 1B-only number, evaluate the frozen fusion checkpoint directly:

```bash
python K-MIND/stage1b_per_class_eval.py \
    --fusion_ckpt ./K-MIND/save_model/stage12/kmind_768_hmf_seed42.bin \
    --cache_root  ./K-MIND/cache_768
```

> **Slot order on K-MIND.** Because K-MIND uses an *audio* teacher, the teacher feature
> occupies HMF's `text_embedding` slot and the text feature its `acoustic` slot. This is
> controlled by `--anchor_modality` (default `audio`) and matters when you load a Stage 1B
> checkpoint trained elsewhere: a checkpoint trained under one convention and run under the
> other silently receives permuted features. The command above passes `--random_init_fusion`,
> so it trains its own fusion weights and the question does not arise. If you do load an
> external checkpoint, check the convention first with
> `python scripts/ablation/check_fusion_order.py`, which also has a `--runtime` mode that
> settles it empirically against the test split.

---

## 7. Hyperparameters

These are the values the scripts use; the IEMOCAP and MELD defaults already match this
table, so the bare commands in §6.1 reproduce the reported configuration.

| Stage | Hyperparameter | IEMOCAP | MELD | K-MIND |
|---|---|---|---|---|
| **1A · CMKD** | λ_r (response weight) | 0.1 | 1 | 1 |
| | τ for L_response | 2 | 4 | 4 |
| | τ for L_feature | 1 | 1 | 1 |
| **1B · HMF** | threshold θ_V / θ_S | 0.2 | 0.1 | 0.2 / 0.3 |
| | dropout, Eq. (2) / Eq. (3) | 0.2 | 0.1 | 0.2 / 0.1 |
| | attention heads | 4 | 3 | 8 |
| | representation dim | 768 | 768 | 768 |
| **2 · ELACL** | λ₁ (`--ce_loss_weight`) | 0.9 | 0.1 | 0.8 |
| | λ₂ (`--angle_loss_weight`) | 1.0 | 1.0 | 0.3 |
| | τ for L_scl (`--temp`) | 0.5 | 0.5 | 0.15 |
| | LR, L_anchor / L_ada | 4e-4 / 1e-4 | 4e-4 / 1e-4 | 3e-5 / 4e-4 |
| | batch size, L_anchor / L_ada | 4 / 64 | 4 / 32 | 4×4 / 32 |
| | epochs, L_anchor / L_ada | 8 / 20 | 8 / 10 | 15 / 20 |

K-MIND additionally uses `--stage1_balance_power 0.5` (class-balanced CE, n_c^−1/2),
`--label_smoothing 0.05`, `--scheduler cosine`, `--unfreeze_fusion --fusion_lr 2e-6`, and
`--accumulation_step 4` — so its effective L_anchor batch size is 16.

---

## 8. Evaluation, tables and figures

```bash
# Per-class F1 and confusion matrix, Stage 1 vs Stage 1+2 (Tables 6-8, Figure 6)
python IEMOCAP/eval_stage12.py
python MELD/eval_stage12.py
python K-MIND/stage1b_per_class_eval.py --fusion_ckpt <ckpt> --cache_root <cache>

# Stage 1 only
python IEMOCAP/eval_stage1b.py

# Dump per-utterance predictions, then build the paper's tables and Figure 6
python scripts/analysis/dump_predictions_iemocap.py
python scripts/analysis/dump_predictions_meld.py
python scripts/analysis/dump_predictions_kmind.py
python scripts/analysis/build_paper_result_tables.py
python scripts/analysis/plot_figure6_confusion.py --out figures/all_confusion.pdf

# Parameter count, latency per utterance, peak memory
python scripts/analysis/efficiency.py
```

**Stage 1B fusion ablation (Table 10).** Encoder outputs are frozen during Stage 1B, so the
variants are compared over a cached feature tensor rather than by re-encoding:

```bash
# 1. Cache the frozen encoder outputs once
python scripts/ablation/build_fusion_cache.py       --out scripts/analysis/fusion_cache_iemocap.pt
python scripts/ablation/build_meld_fusion_cache.py  --out scripts/analysis/fusion_cache_meld.pt

# 2. Train each fusion variant over the cache (seconds per variant, not hours)
for v in hmf hmf_nogate concat crossattn; do
  python scripts/ablation/train_fusion_cached.py \
      --cache scripts/analysis/fusion_cache_iemocap.pt --dataset_name IEMOCAP \
      --variant "$v" --seed 42 --out_csv scripts/analysis/fusion_ablation_results.csv
done

# 3. Pick the winner on validation w-F1 alone; the test column is never consulted
python scripts/ablation/pick_variant_by_validation.py scripts/analysis --seed 42
```

**Dataset analysis (`eda/`).** The scripts behind the dataset statistics and the Figure 3
label distributions; outputs are checked in under `eda/outputs/`.

| Script | Purpose |
|---|---|
| `eda_compare_datasets.py` | per-dataset statistics, class distribution, sample utterances |
| `eda_emotion_diversity.py` | distinct emotions per dialogue |
| `eda_emotion_shift.py` | emotion-shift counts per dialogue |
| `eda_sad_density.py` | length-normalised Sad / Neutral / Joy / Anger density |
| `eda_sample_dialogues.py` | high / median / low density sample dialogues |
| `eda_neutral_table.py` | neutral-focal summary table |
| `eda_kmind_sad_context.py` | K-MIND Sad utterance context analysis |
| `eda_sentiment_level_eval.py` | sentiment-level (pos / neg / neutral) evaluation |
| `eda_cm_sentiment_collapse.py` | sentiment-collapsed confusion-matrix analysis |
| `combine_tsne.py` | t-SNE anchor visualisation (Figure 5) |

---

## 9. Tests

28 regression tests covering the loss definitions, the Stage 2 classifier contract, the
fusion call convention, and the split / cache validators. They need neither a GPU nor a
dataset and run in a few seconds:

```bash
python -m unittest discover -s tests -v
```

---

## 10. Citation

```bibtex
@article{hwang2026alignemo,
  title   = {AlignEMO: Teacher-Guided Multimodal Emotion Representation Learning and
             Emotion Label Anchored Contrastive Learning for Emotion Recognition in
             Conversation},
  author  = {Hwang, Sooyeon and Choi, Han-Jun and Kim, Hoemin and Gweon, Gahgene},
  journal = {Computers, Materials \& Continua},
  year    = {2026}
}
```

## 11. Acknowledgement

This codebase builds on the TelME framework (Yun et al., NAACL 2024). The gated
displacement formulation in Stage 1B follows MAG (Rahman et al., ACL 2020), and the
label-anchored contrastive objective in Stage 2 extends EACL (Yu et al., NAACL Findings
2024) to multimodal representations. The Korean dataset K-MIND is from Yang et al.
(LREC 2026).

This work was supported by Institute of Information & Communications Technology Planning
& Evaluation (IITP) grants funded by the Korea government (MSIT), including the Artificial
Intelligence Graduate School Program (Seoul National University), under grants
No. 2022-0-00608, RS-2022-II220608 and RS-2021-II211343.

## 12. License

MIT — see [LICENSE](LICENSE).
