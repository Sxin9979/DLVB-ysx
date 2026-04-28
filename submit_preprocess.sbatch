#!/bin/bash
#SBATCH -J e3vb_preprocess
#SBATCH -p pc
#SBATCH -w pc001
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.slurm.out
#SBATCH -e logs/%x_%j.slurm.err

set -euo pipefail

PROJECT_ROOT="/pool1/home/ysxin/E3nn/E3VB"
CONDA_ROOT="/pool1/home/ysxin/miniconda3"
ENV_NAME="e3vb_jax"
CONFIG_PATH="${PROJECT_ROOT}/config_e2e.yaml"
JOB_ID="${SLURM_JOB_ID:-manual}"
PREPROCESS_LOG="${PROJECT_ROOT}/logs/preprocess_${JOB_ID}.log"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/processed"

source ~/.bashrc
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_NAME=${ENV_NAME}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PREPROCESS_LOG=${PREPROCESS_LOG}"

python -u main.py \
  --mode preprocess \
  --config "${CONFIG_PATH}" \
  --log_path "${PREPROCESS_LOG}" \
  --quiet_stdout
