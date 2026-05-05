#!/bin/bash
#SBATCH -J e3vb_preprocess
#SBATCH -p pc
#SBATCH -w pc001
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /pool1/home/ysxin/E3nn/E3VB/logs/%x_%j.slurm.out
#SBATCH -e /pool1/home/ysxin/E3nn/E3VB/logs/%x_%j.slurm.err

set -euo pipefail

PROJECT_ROOT="/pool1/home/ysxin/E3nn/E3VB"
CONDA_ROOT="/pool1/home/ysxin/miniconda3"
ENV_NAME="e3vb_jax"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config_e2e.yaml}"
JOB_ID="${SLURM_JOB_ID:-manual}"
PREPROCESS_LOG="${PROJECT_ROOT}/logs/preprocess_${JOB_ID}.log"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/processed"

source ~/.bashrc
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"

readarray -t CACHE_INFO < <(
python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
data = payload["data"]
print(data["processed_cache_path"])
print(data["packed_cache_path"])
PY
)
PROCESSED_CACHE="${CACHE_INFO[0]}"
PACKED_CACHE="${CACHE_INFO[1]}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_NAME=${ENV_NAME}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PREPROCESS_LOG=${PREPROCESS_LOG}"
echo "PROCESSED_CACHE=${PROCESSED_CACHE}"
echo "PACKED_CACHE=${PACKED_CACHE}"
echo "FORCE_REBUILD=${FORCE_REBUILD}"

PREPROCESS_ARGS=(
  --mode preprocess
  --config "${CONFIG_PATH}"
  --log_path "${PREPROCESS_LOG}"
  --quiet_stdout
)
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  PREPROCESS_ARGS+=(--force_preprocess)
fi

python -u main.py "${PREPROCESS_ARGS[@]}"
