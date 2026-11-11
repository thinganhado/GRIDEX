#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRIDEX_ROOT="${GRIDEX_ROOT:-$(cd "${SLURM_DIR}/../.." && pwd)}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
bash "${GRIDEX_ROOT}/training/scripts/sft_multiturn.sh"
