#!/bin/bash
#SBATCH --job-name=alignemo_kmind
#SBATCH --output=logs/kmind-%A_%a.out
#SBATCH --partition=comm
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4

# =========================================================================
# AlignEMO — K-MIND Stage 1B + Stage 2, one array task per seed.
#
# Prerequisite: the 768-d feature cache must already exist (README §6.2):
#   K-MIND/build_cache_1_partial.py -> build_cache_2_final.py
#   -> scripts/ablation/project_kmind_cache_768.py --output_root ./K-MIND/cache_768
#
# Submit from the repo root:  sbatch scripts/slurm/run_kmind.sbatch
# Cluster-specific lines (partition / gres / conda) — adapt to your site.
# =========================================================================
set -euo pipefail
source /home/${USER}/.bashrc
source /opt/anaconda3/bin/activate TelME2

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

SEEDS=(42 13 21 87 100)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
SAVE="./K-MIND/save_model/stage12/seed_${SEED}"
mkdir -p "$SAVE"

python K-MIND/ELACL_fusion.py \
  --cache_root ./K-MIND/cache_768 \
  --random_init_fusion \
  --representation_dim 768 --final_fusion_variant hmf \
  --save_dir "$SAVE" --save_name "kmind_768_hmf_seed${SEED}" \
  --learning_rate 3e-5 --batch_size 4 --accumulation_step 4 \
  --epochs 15 --warmup_ratio 0.05 --seed "$SEED" --clsNum 7 \
  --num_workers 4 --dataset_name K-MIND \
  --ce_loss_weight 0.8 --angle_loss_weight 0.3 --temp 0.15 --weight_decay 1e-3 \
  --anchor_path ./K-MIND/emo_anchors/roberta-base/ \
  --fv_hidden_size 768 --fv_beta_shift 0.2 --fv_dropout_prob 0.2 --fv_num_head 8 \
  --f_hidden_size  768 --f_beta_shift  0.3 --f_dropout_prob  0.1 --f_num_head  8 \
  --stage1_balance_power 0.5 --unfreeze_fusion --fusion_lr 2e-6 \
  --label_smoothing 0.05 --scheduler cosine \
  --stage2_epochs 20 --stage_two_lr 4e-4 --stage2_weight_decay 0.0

echo "---Done (seed ${SEED})---"
