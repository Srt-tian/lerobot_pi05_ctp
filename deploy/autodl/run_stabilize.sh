#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=/root/autodl-tmp/code/lerobot_pi05_ctp_stabilize
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src"
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 WANDB_DIR=/root/autodl-tmp/logs WANDB_MODE=online
export NO_PROXY="${NO_PROXY:-},api.wandb.ai,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
set -a
source /root/.ssh/ctp_wandb.env
set +a
mode=${1:?Pass debug or resume}
shift
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"
echo "CODE_COMMIT=$(git rev-parse HEAD) RUN_MODE=$mode"
python=/root/autodl-tmp/code/lerobot_pi05_ctp/.venv/bin/torchrun
if [[ "$mode" == debug ]]; then
  exec "$python" --standalone --nproc_per_node=4 -m lerobot.scripts.lerobot_train \
    --config_path=deploy/autodl/configs/stabilize.json --stop_after_step=100 "$@"
elif [[ "$mode" == resume ]]; then
  config=${1:?Checkpoint train_config.json required}
  shift
  exec "$python" --standalone --nproc_per_node=4 -m lerobot.scripts.lerobot_train \
    --config_path="$config" --resume=true --stop_after_step=0 "$@"
else
  echo "Unknown run mode: $mode" >&2
  exit 2
fi
