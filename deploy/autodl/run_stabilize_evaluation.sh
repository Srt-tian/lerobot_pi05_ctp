#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/code/lerobot_pi05_ctp_stabilize
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 WANDB_MODE=online
export NO_PROXY="${NO_PROXY:-},api.wandb.ai,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
set -a
source /root/.ssh/ctp_wandb.env
set +a
test -z "$(git status --porcelain)"
exec /root/autodl-tmp/code/lerobot_pi05_ctp/.venv/bin/python deploy/autodl/watch_stabilize.py "$@"
