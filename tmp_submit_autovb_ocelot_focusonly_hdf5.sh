#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/pool1/home/ysxin/E3nn/E3VB"
CONFIG_PATH="${PROJECT_ROOT}/config_e2e_autovb_ocelot_unified_topmass_lowmem_u1_macrofwmae_focusonly_hdf5.yaml"
RUN_MODE="autovb_ocelot_focusonly_hdf5"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "RUN_MODE=${RUN_MODE}"
echo "Submitting with 48G memory"

sbatch \
  --parsable \
  -p pc \
  -w pc001 \
  --mem=48G \
  --export=ALL,CONFIG_PATH="${CONFIG_PATH}",RUN_MODE="${RUN_MODE}" \
  "${PROJECT_ROOT}/submit_train.sbatch"
