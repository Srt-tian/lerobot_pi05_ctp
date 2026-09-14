#!/usr/bin/env bash
# Four-GPU debug preserves the formal schedule, then tests a fresh-process resume.
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
RUNTIME_ROOT=/pfs/user/data/ctp_pi05_eip
mkdir -p "$RUNTIME_ROOT/logs"
log_file="$RUNTIME_ROOT/logs/debug_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; echo "DEBUG_ENTRYPOINT_FAILED exit=$rc line=$LINENO" >&2' ERR
printf 'DEBUG_ENTRYPOINT_STARTED log=%s\n' "$log_file"
PYTHON="$RUNTIME_ROOT/venv/bin/python"
config="$RUNTIME_ROOT/configs/v3/debug.json"
output="$RUNTIME_ROOT/outputs/pi05_ctp_v3_debug_4xa800"
[[ ! -e "$output" ]] || { echo "Debug output already exists; inspect before relaunch" >&2; exit 2; }
bash "$REPO_DIR/deploy/eip/run.sh" debug "$config"
"$PYTHON" "$REPO_DIR/deploy/eip/verify_debug.py" "$output" 100
resume_config="$output/checkpoints/000100/pretrained_model/train_config.json"
export EXPECTED_CONFIG_SHA256
EXPECTED_CONFIG_SHA256=$(sha256sum "$resume_config" | cut -d' ' -f1)
export RESUME_GPU_COUNT=4 RESUME_STOP_AFTER_STEP=105
# The first phase already tests full coverage; repeat only the fixed anchors on resume.
export DEBUG_RESUME_SKIP_FULL_EVAL=1
bash "$REPO_DIR/deploy/eip/run.sh" resume "$resume_config"
"$PYTHON" "$REPO_DIR/deploy/eip/verify_debug.py" "$output" 105
