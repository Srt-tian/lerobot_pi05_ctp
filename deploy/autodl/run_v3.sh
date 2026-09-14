#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
RUNTIME_ROOT=${CTP_RUNTIME_ROOT:-/root/autodl-tmp/ctp_pens_v3}
PYTHON=${CTP_PYTHON:-/root/autodl-tmp/code/lerobot_pi05_ctp/.venv/bin/python}
# Task-scoped vGPU compatibility, validated by probe_nccl.py.
NCCL_LIB="$RUNTIME_ROOT/lib/libnccl.so.2.28.9"
test -f "$NCCL_LIB"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$NCCL_LIB"
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 PYTHONFAULTHANDLER=1
mode=${1:?Usage: run_v3.sh debug|resume105|resume|train /absolute/config.json}
config=${2:?A resolved config or checkpoint train_config.json is required}
case "$mode" in
  debug) stop_after=100 ;;
  resume105) stop_after=105 ;;
  train|resume) stop_after=0 ;;
  *) echo "Unsupported run mode" >&2; exit 2 ;;
esac
cd "$REPO_DIR"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"
[[ "$config" = /* && -f "$config" ]]
credential_file=${CTP_WANDB_ENV:-/root/.ssh/ctp_wandb.env}
if [[ -f "$credential_file" ]]; then
  set -a
  source "$credential_file"
  set +a
fi
: "${WANDB_API_KEY:?W&B credentials required}"
: "${WANDB_ENTITY:?W&B account required}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 WANDB_MODE=online WANDB_DIR="$RUNTIME_ROOT/logs/wandb"
export NO_PROXY="${NO_PROXY:-},api.wandb.ai,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
mkdir -p "$WANDB_DIR"
"$PYTHON" - "$config" "$mode" "$RUNTIME_ROOT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

config_path, mode, runtime = sys.argv[1:]
cfg = json.loads(Path(config_path).read_text())
assert cfg['steps'] == cfg['policy']['scheduler_decay_steps'] == 20000
assert cfg['policy']['scheduler_warmup_steps'] == 500
assert cfg['batch_size'] == 8 and cfg['parallelism']['dp_shard'] == 4
assert cfg['parallelism']['dp_replicate'] == 1
assert cfg['accelerator']['gradient_accumulation']['steps'] == 1
assert cfg['accelerator']['mixed_precision'] == 'bf16' and cfg['policy']['dtype'] == 'float32'
assert cfg['save_freq'] == 2000 and cfg['checkpoint_keep_last'] == 1
assert cfg['checkpoint_min_free_gb'] >= 10
assert cfg['policy']['ctp_enabled'] and cfg['policy']['ctp_split_head']
assert cfg['policy']['ctp_detach_width_context']
assert not cfg['policy']['freeze_vision_encoder'] and not cfg['policy']['train_expert_only']
assert cfg['ctp_color_jitter'] == .1 and cfg['eval_steps'] == 250
assert cfg['wandb']['enable'] and cfg['wandb']['disable_artifact']
assert cfg['wandb']['project'] == 'pi05-ctp-pens'
assert torch.cuda.device_count() == 4
assert all(torch.cuda.get_device_properties(i).total_memory >= 31 * 1024**3 for i in range(4))
root = Path(cfg['dataset']['root'])
manifest_path = root / 'preparation_manifest.json'
manifest = json.loads(manifest_path.read_text())
validation = json.loads((Path(runtime) / 'validation.json').read_text())
assert validation['all_mapped_values_exact'] and validation['train_only_quantiles_verified']
assert validation['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert validation['stats_sha256'] == hashlib.sha256((root / 'meta/stats.json').read_bytes()).hexdigest()
assert manifest['action_source'] == 'real_action' and manifest['state_source'] == 'observation.qpos'
assert cfg['dataset']['eval_split'] == manifest.get('eval_split', 0.1)
assert manifest['task'] == 'Put the pens into the pen holder.'
assert manifest['episodes'] == 168
assert manifest['train_episodes'] == list(range(151))
assert manifest['validation_episodes'] == list(range(151, 168))
base_report = json.loads((Path(runtime) / 'base_verified.json').read_text())
base = Path(base_report['path'])
assert base_report['sha256'] == '0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'
assert base.stat().st_size == base_report['bytes'] == 14467165872
assert base.stat().st_mtime_ns == base_report['mtime_ns']
if mode in {'debug', 'train'}:
    assert (Path(cfg['policy']['pretrained_path']) / 'model.safetensors').samefile(base)
    assert not Path(cfg['output_dir']).exists(), 'Fresh BASE launch must not overwrite a previous run'
    # Account for latest+new recovery points, older best weights and conservative write guard.
    required = 42652000000 + 16627000000 + 49756000000 + 10 * 1024**3
else:
    checkpoint = Path(config_path).parents[1]
    marker = json.loads((checkpoint / 'checkpoint_complete.json').read_text())
    assert marker['step'] > 0 and cfg['wandb']['run_id']
    # Native save_checkpoint performs its full write-size guard before each save.
    required = 49756000000 + 10 * 1024**3
assert shutil.disk_usage(runtime).free >= required, 'Insufficient space for safe checkpoint rotation'
print('AUTODL_V3_INPUTS_VERIFIED', json.dumps({'task': manifest['task'], 'episodes': manifest['episodes'],
      'steps': 20000, 'global_batch': 32, 'save_freq': 2000, 'keep_latest_full': 1,
      'free_bytes': shutil.disk_usage(runtime).free}), flush=True)
PY
printf 'CODE_COMMIT=%s RUN_MODE=%s\n' "$(git rev-parse HEAD)" "$mode"
args=(--config_path="$config" --stop_after_step="$stop_after")
if [[ "$mode" = resume || "$mode" = resume105 ]]; then args+=(--resume=true); fi
if [[ "$mode" = resume105 ]]; then args+=(--full_eval_steps=0); fi
if [[ "$mode" = resume ]]; then args+=(--full_eval_steps=1000); fi
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m lerobot.scripts.lerobot_train "${args[@]}"
