#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=/root/autodl-tmp/code/lerobot_pi05_ctp
cd "$REPO_DIR"
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export WANDB_DIR=/root/autodl-tmp/logs
export WANDB_MODE=online
export NO_PROXY="${NO_PROXY:-},api.wandb.ai,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
set -a
source /root/.ssh/ctp_wandb.env
set +a
mode=${1:?Pass native_probe, debug, formal, or resume plus train_config.json}
shift
if [[ "$mode" == resume ]]; then
  config=${1:?Checkpoint train_config.json required}
  shift
  exec .venv/bin/torchrun --standalone --nproc_per_node=4 -m lerobot.scripts.lerobot_train \
    --config_path="$config" --resume=true "$@"
fi
case "$mode" in
  native_probe|debug|formal) ;;
  *) echo "Unknown run mode" >&2; exit 2 ;;
esac
if [[ "$mode" != native_probe ]]; then
  test -s /root/autodl-tmp/models/pi05_base_pens168_init/model.safetensors
fi
if [[ "$mode" == formal ]]; then
  test -z "$(git status --porcelain)"
  test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"
fi
echo "RUN_MODE=$mode CODE_COMMIT=$(git rev-parse HEAD)"
exec .venv/bin/torchrun --standalone --nproc_per_node=4 -m lerobot.scripts.lerobot_train \
  --config_path="deploy/autodl/configs/$mode.json" "$@"
