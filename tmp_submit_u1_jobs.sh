#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/pool1/home/ysxin/E3nn/E3VB"

CONFIG_2DS="${PROJECT_ROOT}/config_e2e_checked_vb_unified_topmass_lowmem_u1_macrofwmae.yaml"
CONFIG_3DS="${PROJECT_ROOT}/config_e2e_checked_batch2_checked_vb_unified_topmass_lowmem_u1_macrofwmae.yaml"

PRE2=$(
  CONFIG_PATH="${CONFIG_2DS}" FORCE_REBUILD=1 \
  sbatch --parsable -p pc -w pc001 --mem=48G "${PROJECT_ROOT}/submit_preprocess.sbatch"
)
PRE3=$(
  CONFIG_PATH="${CONFIG_3DS}" FORCE_REBUILD=1 \
  sbatch --parsable -p pc -w pc001 --mem=48G "${PROJECT_ROOT}/submit_preprocess.sbatch"
)

TRAIN2=$(
  CONFIG_PATH="${CONFIG_2DS}" RUN_MODE="u1_checked_vb_2ds" \
  sbatch --parsable --dependency="afterok:${PRE2}" -p pc -w pc001 --mem=48G "${PROJECT_ROOT}/submit_train.sbatch"
)
TRAIN3=$(
  CONFIG_PATH="${CONFIG_3DS}" RUN_MODE="u1_checked_checkedb2_vb_3ds" \
  sbatch --parsable --dependency="afterok:${PRE3}" -p pc -w pc001 --mem=48G "${PROJECT_ROOT}/submit_train.sbatch"
)

echo "preprocess_2ds=${PRE2}"
echo "train_2ds=${TRAIN2}"
echo "preprocess_3ds=${PRE3}"
echo "train_3ds=${TRAIN3}"
