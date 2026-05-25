#!/bin/bash
#SBATCH --job-name=deephf-prep
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/prepare_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "${VENV:-.venv}/bin/activate"
mkdir -p logs cache_electronic

for s in water alkanes qm7b_T gdb13_T; do
    echo "=== $s: graph features ==="
    python prepare/prepare_mobml.py --subset "$s"
    echo "=== $s: SOAP + electronic ==="
    python prepare/compute_descriptors.py --subset "$s"
done
