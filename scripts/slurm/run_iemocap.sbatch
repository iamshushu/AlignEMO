#!/bin/bash
#SBATCH --job-name=alignemo_iemocap
#SBATCH --output=%x_%A_%N.out
#SBATCH --partition=comm
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1

# =========================================================================
# AlignEMO — IEMOCAP full pipeline (text-teacher), run sequentially.
# Stage 1A teacher -> 1A students -> 1B visual fusion -> 1B HMF -> Stage 2 ELACL.
# Each script's checkpoint flags default to ./IEMOCAP/save_model/, so the bare
# commands chain end-to-end. Submit from the repo root:  sbatch scripts/run_iemocap.sbatch
# Cluster-specific lines (partition / gres / conda) — adapt to your site.
# =========================================================================
set -euo pipefail
source /home/${USER}/.bashrc
source /opt/anaconda3/bin/activate TelME2
export CUDA_VISIBLE_DEVICES=0

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== [1A] text teacher ==="
python IEMOCAP/teacher.py --learning_rate 1e-6

echo "=== [1A] audio student (CMKD) ==="
python IEMOCAP/student_audio.py

echo "=== [1A] visual students (CMKD; scene + masked-facial) ==="
python IEMOCAP/student_visual.py

echo "=== [1B] visual-text fusion ==="
python IEMOCAP/fusion_visual.py

echo "=== [1B] full HMF (+audio) ==="
python IEMOCAP/fusion.py

echo "=== [2] ELACL ==="
python IEMOCAP/ELACL_fusion.py

echo "---Done---"
