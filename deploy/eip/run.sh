#!/usr/bin/env bash
# Launch only a separately reviewed, fingerprinted configuration.
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
RUNTIME_ROOT=/pfs/user/data/ctp_pi05_eip
PYTHON="$RUNTIME_ROOT/venv/bin/python"
mode=${1:?Usage: run.sh debug|train|resume /absolute/config.json}
config=${2:?A resolved training configuration is required}
case "$mode" in
  debug) workers=4; stop_after=100 ;;
  train) workers=8; stop_after=0 ;;
  resume) workers=${RESUME_GPU_COUNT:?Set the original checkpoint GPU count}; stop_after=${RESUME_STOP_AFTER_STEP:-0} ;;
  *) echo 'Mode must be debug, train or resume' >&2; exit 2 ;;
esac
[[ "$config" = /* && -f "$config" ]]
[[ "$workers" = 4 || "$workers" = 8 ]]
: "${EXPECTED_CODE_COMMIT:?Pin the reviewed execution commit}"
: "${EXPECTED_CONFIG_SHA256:?Pin the reviewed configuration}"
: "${WANDB_API_KEY:?Inject WANDB_API_KEY through the EIP environment}"
: "${WANDB_ENTITY:?Inject WANDB_ENTITY through the EIP environment}"
cd "$REPO_DIR"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_CODE_COMMIT"
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_CODE_COMMIT"
test "$(sha256sum "$config" | cut -d' ' -f1)" = "$EXPECTED_CONFIG_SHA256"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME_ROOT/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 WANDB_MODE=online WANDB_DIR="$RUNTIME_ROOT/logs/wandb"
mkdir -p "$WANDB_DIR"
"$PYTHON" - "$config" "$workers" "$mode" "$RUNTIME_ROOT" <<'PY'
import json
from pathlib import Path
import shutil
import sys
import torch

config_path, workers, mode, root = sys.argv[1:]
cfg = json.loads(Path(config_path).read_text())
workers = int(workers)
root = Path(root)
assert cfg['steps'] == cfg['policy']['scheduler_decay_steps'] == 20000
assert cfg['policy']['scheduler_warmup_steps'] == 500
assert cfg['batch_size'] * workers == 32
assert cfg['parallelism']['dp_shard'] == workers and cfg['parallelism']['dp_replicate'] == 1
assert cfg['accelerator']['gradient_accumulation']['steps'] == 1
assert cfg['accelerator']['mixed_precision'] == 'bf16' and cfg['policy']['dtype'] == 'float32'
assert cfg['policy']['ctp_enabled'] and cfg['policy']['chunk_size'] == 50
assert cfg['policy']['ctp_split_head'] and cfg['policy']['ctp_detach_width_context']
assert cfg['policy']['ctp_optimizer_vlm_lr_scale'] == .25
assert cfg['policy']['ctp_optimizer_head_lr_scale'] == 5
assert cfg['policy']['ctp_optimizer_logits_lr_scale'] == 1
assert cfg['policy']['ctp_optimizer_width_lr_scale'] == .5
assert cfg['save_freq'] == 2000 and cfg['checkpoint_keep_last'] == 0
assert cfg['ctp_color_jitter'] == .1 and cfg['eval_steps'] == 250
assert not cfg['policy']['freeze_vision_encoder'] and not cfg['policy']['train_expert_only']
assert cfg['wandb']['enable'] and cfg['wandb']['project'] == 'pi05-ctp-pen168'
if mode != 'resume':
    assert not cfg.get('resume', False)
assert torch.cuda.device_count() == workers
assert all('A800' in torch.cuda.get_device_name(i) for i in range(workers))
required_gib = 500 if mode == 'train' else 120
assert shutil.disk_usage(root).free >= required_gib * 1024**3, 'Insufficient checkpoint headroom'
dataset = Path(cfg['dataset']['root'])
manifest = json.loads((dataset / 'preparation_manifest.json').read_text())
assert manifest['episodes'] == 168 and manifest['train_episodes'] == list(range(151))
assert manifest['validation_episodes'] == list(range(151, 168))
validation = json.loads((root / 'logs/data_validation.json').read_text())
assert validation['all_mapped_values_exact'] and validation['train_only_quantiles_verified']
base = root / 'models/pi05_base'
provenance = json.loads((base / 'source_provenance.json').read_text())
assert provenance['revision'] == 'b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba'
assert provenance['sha256'] == '0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'
assert (base / 'model.safetensors').stat().st_size == 14467165872
if mode != 'resume':
    initial = Path(cfg['policy']['pretrained_path']) / 'model.safetensors'
    assert initial.samefile(base / 'model.safetensors'), 'Initialization must hardlink the verified BASE'
print('EIP_INPUT_PREFLIGHT_OK', json.dumps({'gpus': workers, 'global_batch': 32, 'steps': 20000}), flush=True)
PY
args=(--config_path="$config" --stop_after_step="$stop_after")
if [[ "$mode" = resume ]]; then args+=(--resume=true); fi
if [[ "${DEBUG_RESUME_SKIP_FULL_EVAL:-0}" = 1 ]]; then
  [[ "$mode" = resume && "$workers" = 4 && "$stop_after" = 105 ]]
  args+=(--full_eval_steps=0)
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$workers" \
  -m lerobot.scripts.lerobot_train "${args[@]}"
