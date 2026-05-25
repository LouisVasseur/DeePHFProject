#!/bin/bash
#SBATCH --job-name=deephf-train
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/train_all_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "${VENV:-.venv}/bin/activate"
mkdir -p logs results figures tables

# Main grid (seed 43)
python scripts/train_grid.py --seed 43 --resume

# Ablations
python scripts/ablate_edge_bias.py
python scripts/ablate_edge_features.py --dataset alkanes
python scripts/test_invariance.py
python scripts/k_sweep.py

# Figures + tables
python scripts/make_artifacts.py
