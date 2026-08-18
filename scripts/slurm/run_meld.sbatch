#!/bin/bash
#SBATCH --job-name=alignemo_meld
#SBATCH --output=%x_%A_%N.out
#SBATCH --partition=comm
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1

# =========================================================================
# AlignEMO — MELD full pipeline (text-teacher), run sequentially.
# Stage 1A teacher -> 1A students -> 1B visual fusion -> 1B HMF -> Stage 2 ELACL.
# Checkpoint flags default to ./MELD/save_model/, so the bare commands chain.
# Submit from the repo root:  sbatch scripts/run_meld.sbatch
# Cluster-specific lines (partition / gres / conda) — adapt to your site.
# =========================================================================
set -euo pipefail
source /home/${USER}/.bashrc
source /opt/anaconda3/bin/activate TelME2
export CUDA_VISIBLE_DEVICES=0

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== [1A] text teacher ==="
python MELD/teacher.py --learning_rate 1e-6

echo "=== [1A] audio student (CMKD) ==="
python MELD/student_audio.py

echo "=== [1A] visual students (CMKD; scene + masked-facial) ==="
python MELD/student_visual.py

echo "=== [1B] visual-text fusion ==="
python MELD/fusion_visual.py

echo "=== [1B] full HMF (+audio) ==="
python MELD/fusion.py

echo "=== [2] ELACL ==="
python MELD/ELACL_fusion.py

echo "---Done---"
