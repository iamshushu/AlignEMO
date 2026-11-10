# AlignEMO

Code for *AlignEMO: Teacher-Guided Multimodal Emotion Representation Learning and Emotion Label Anchored Contrastive Learning for Emotion Recognition in Conversation*.

AlignEMO uses cross-modal distillation and hierarchical fusion to combine text, audio, and two visual streams. It then aligns the fused representations with emotion-label anchors.

| Dataset | w-F1 (%), mean ± SD over five runs |
|---|---:|
| IEMOCAP | 70.70 ± 0.34 |
| MELD | 67.44 ± 0.80 |
| K-MIND | 74.90 ± 0.23 |

## Setup

Training and evaluation require a CUDA GPU. Run the commands from the repository root.

```bash
git clone https://github.com/iamshushu/AlignEMO.git
cd AlignEMO
conda env create -f environment.yml
conda activate TelME2
```

## Datasets

The datasets are **not redistributed here**; obtain them from their original providers.

| Dataset | Source | Classes |
|---|---|---|
| IEMOCAP | https://sail.usc.edu/iemocap/ | 6: neutral, happy, sad, angry, excited, frustrated |
| MELD | https://affective-meld.github.io/ | 7: neutral, joy, surprise, anger, sadness, disgust, fear |
| K-MIND | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71986) | 7 Ekman: neutral, joy, surprise, anger, sad, disgust, fear |

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

After setting up the dataset paths, check the environment and data with
`python check_setup.py --dataset IEMOCAP`.

## Training

Generate the emotion anchors before running Stage 2.

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

### IEMOCAP and MELD

Run these commands in order. Replace `IEMOCAP` with `MELD` for the MELD pipeline.

```bash
python IEMOCAP/teacher.py --learning_rate 1e-6

python IEMOCAP/student_audio.py
python IEMOCAP/student_visual.py

python IEMOCAP/fusion_visual.py

python IEMOCAP/fusion.py

python IEMOCAP/ELACL_fusion.py
```

### K-MIND

Train the Stage 1A models, build the feature cache, and project it to 768 dimensions.
Set `SCENE_CKPT` and `FACIAL_CKPT` to the actual visual checkpoint paths before building the cache.

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
    --visual_raw_ckpt    "$SCENE_CKPT" \
    --visual_masked_ckpt "$FACIAL_CKPT"

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

Then run Stage 1B and Stage 2 with the projected cache.

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

Slurm scripts for all three datasets are in `scripts/slurm/`. Adjust the
cluster settings and environment paths before submitting.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@article{hwang2026alignemo,
  title   = {AlignEMO: Teacher-Guided Multimodal Emotion Representation Learning and
             Emotion Label Anchored Contrastive Learning for Emotion Recognition in
             Conversation},
  author  = {Hwang, Sooyeon and Choi, Han-Jun and Kim, Hoemin and Gweon, Gahgene},
  year    = {2026}
}
```

## Acknowledgement
This work was supported by Institute of Information & Communications Technology Planning
& Evaluation (IITP) grants funded by the Korea government (MSIT), including the Artificial
Intelligence Graduate School Program (Seoul National University), under grants
No. 2022-0-00608, RS-2022-II220608 and RS-2021-II211343.

## License

MIT. See [LICENSE](LICENSE).
