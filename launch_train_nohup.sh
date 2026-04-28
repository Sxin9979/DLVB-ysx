#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd /pool1/home/ysxin/E3nn/E3VB
source /pool1/home/ysxin/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate e3vb_jax >/dev/null 2>&1

run_tag="nohup_2000ep_epochonly_$(date +%Y%m%d_%H%M%S)"
train_log="/pool1/home/ysxin/E3nn/E3VB/logs/${run_tag}.log"
launch_log="/pool1/home/ysxin/E3nn/E3VB/logs/${run_tag}.launch.out"
checkpoint_path="/pool1/home/ysxin/E3nn/E3VB/checkpoints/${run_tag}_best.pkl"
pid_file="/pool1/home/ysxin/E3nn/E3VB/logs/${run_tag}.pid"

nohup env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_per_fusion_autotune_cache_dir=/pool1/home/ysxin/E3nn/E3VB/processed/xla_autotune_cache" \
  python -u main.py \
    --mode train \
    --config config_e2e.yaml \
    --log_path "${train_log}" \
    --checkpoint_path "${checkpoint_path}" \
    --quiet_stdout \
  > "${launch_log}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${pid_file}"

printf 'PID=%s\nTRAIN_LOG=%s\nLAUNCH_LOG=%s\nCHECKPOINT_PATH=%s\nPID_FILE=%s\n' \
  "${pid}" "${train_log}" "${launch_log}" "${checkpoint_path}" "${pid_file}"

sleep 5
ps -fp "${pid}" || true
tail -n 20 "${train_log}" 2>/dev/null || true
