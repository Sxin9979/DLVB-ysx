#!/bin/bash
#SBATCH -J e3vb_train
#SBATCH -p pc
#SBATCH -w pc001
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o /pool1/home/ysxin/E3nn/E3VB/logs/%x_%j.slurm.out
#SBATCH -e /pool1/home/ysxin/E3nn/E3VB/logs/%x_%j.slurm.err
#SBATCH --signal=B:TERM@300

set -euo pipefail

PROJECT_ROOT="/pool1/home/ysxin/E3nn/E3VB"
CONDA_ROOT="/pool1/home/ysxin/miniconda3"
ENV_NAME="e3vb_e2e"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config_e2e.yaml}"
RUN_MODE="${RUN_MODE:-formal}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_TAG="${RUN_MODE}_${JOB_ID}"
TRAIN_LOG="${PROJECT_ROOT}/logs/${RUN_TAG}.log"
GPU_LOG="${PROJECT_ROOT}/logs/${RUN_TAG}.gpu.csv"
CHECKPOINT_PATH="${PROJECT_ROOT}/checkpoints/${RUN_TAG}_best.pkl"
PROCESSED_CACHE="${PROJECT_ROOT}/processed/processed_dataset_cache.pkl"
PACKED_CACHE="${PROJECT_ROOT}/processed/packed_dataset_cache.pkl"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/checkpoints"

source ~/.bashrc
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"

# pc001 shells may inherit a broken localhost proxy; disable it inside the job.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_NAME=${ENV_NAME}"
echo "RUN_MODE=${RUN_MODE}"
echo "TRAIN_LOG=${TRAIN_LOG}"
echo "GPU_LOG=${GPU_LOG}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "PROCESSED_CACHE=${PROCESSED_CACHE}"
echo "PACKED_CACHE=${PACKED_CACHE}"
echo "XLA_FLAGS=${XLA_FLAGS}"
nvidia-smi
python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__, 'backend', jax.default_backend(), 'devices', jax.devices())"

echo "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,pstate,power.draw" > "${GPU_LOG}"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,pstate,power.draw \
      --format=csv,noheader,nounits >> "${GPU_LOG}"
    sleep 30
  done
) &
GPU_MONITOR_PID=$!
cleanup() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ ! -f "${PROCESSED_CACHE}" ]]; then
  echo "Missing processed cache: ${PROCESSED_CACHE}" >&2
  echo "Run offline preprocess first: python -u main.py --mode preprocess --config ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PACKED_CACHE}" ]]; then
  echo "Missing packed cache: ${PACKED_CACHE}" >&2
  echo "Run offline preprocess first: python -u main.py --mode preprocess --config ${CONFIG_PATH}" >&2
  exit 1
fi

COMMON_ARGS=(
  --mode train
  --config "${CONFIG_PATH}"
  --log_path "${TRAIN_LOG}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --quiet_stdout
)

if [[ "${RUN_MODE}" == "smoke" ]]; then
  python -u main.py \
    "${COMMON_ARGS[@]}" \
    --smoke \
    --smoke_epochs 1 \
    --smoke_max_batches 1 \
    --sample_limit_per_split 32
else
  python -u main.py "${COMMON_ARGS[@]}"
fi
