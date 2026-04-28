#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd /pool1/home/ysxin/E3nn/E3VB
source /pool1/home/ysxin/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate e3vb_jax >/dev/null 2>&1

python -u main.py \
  --mode train \
  --config config_e2e.yaml \
  --log_path "${TRAIN_LOG}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --quiet_stdout
