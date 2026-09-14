# EIP IDC PI05 CTP V3

The authoritative checkout is `/pfs/user/code/lerobot_pi05_ctp_eip_a800`, branch
`deploy/pi05-ctp-eip-a800`, canonical Git `/pfs/user/code/lerobot_pi05_ctp.git`.
See `PLANS.md` for the experiment contract and latest user decisions.

Image (one published version):
`mfyz.icompify.com:5000/magiclab/lerobot_pi05_ctp@sha256:3b5308fe55c71682fff4feb40cc5dc70b97d9d4658d9d135c2d79e3c335e1a10`.
The image provides OS/ffmpeg; exact-lock Python3.12.13 environment is mounted from
`/pfs/user/data/ctp_pi05_eip/venv`. Code/config edits do not require image uploads.

`prepare_v3.py` produces native debug/formal JSON and task processors; raw BASE
weights are hardlinked, not duplicated. Debug JSON and formal JSON differ only
in GPU topology/local batch, output/run identity and operational stop. Both
use global batch32 and the full20000-step schedule. W&B uses the user's online
project and injected API key, with model-artifact uploads disabled.

After reviewing the fully resolved EIP payload and obtaining user confirmation:

```
bash /pfs/user/code/lerobot_pi05_ctp_eip_a800/deploy/eip/debug.sh
```

Required EIP variables: `EXPECTED_CODE_COMMIT`, `EXPECTED_CONFIG_SHA256`,
`WANDB_API_KEY`, `WANDB_ENTITY`. Never put secret values in Git or ledgers.
The script verifies clean/published code, config fingerprint, dataset/BASE
identity, four A800s and free storage. It trains100 steps, runs fixed/full
validation, writes a complete checkpoint, then resumes to105 in a fresh process.
Resume skips a duplicate full validation; fixed anchors and full scheduler remain.
`verify_debug.py` checks artifacts, step, W&B continuity and validation coverage.

Formal command (separately confirmed after debug passes and8-GPU resources resolve):

```
bash /pfs/user/code/lerobot_pi05_ctp_eip_a800/deploy/eip/run.sh train /pfs/user/data/ctp_pi05_eip/configs/v3/formal.json
```

Formal saves every2000 steps and retains all ten recovery checkpoints. Fixed
validation every250/full every1000; best among saved checkpoints uses
`.7*normalized_prefix10_MSE + .3*normalized_full50_MSE`. No V1 reference dependency.
Before every save, check expected checkpoint bytes plus10GiB reserve; publication
of completion markers is independent of retention. Never edit this checkout
while an EIP task uses it.

Validation:26 focused tests passed; native configs/processors generated; Ruff
and shell syntax passed. Three missing Torch libraries were restored from the
exact-version uv cache after RECORD SHA256 checks; subsequent PFS requests stalled,
then actual tests/config generation completed. Filesystem root cause remains
unverified. Real FSDP GPU training/save/resume are pending, not implied by CPU tests.
