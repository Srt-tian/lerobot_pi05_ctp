# AutoDL pi0.5 CTP pen168

The execution checkout is `/root/autodl-tmp/code/lerobot_pi05_ctp`, branch
`pi05-ctp-pen168-full`. AutoDL has four 49,140 MiB RTX 4090-labelled vGPUs, 80
assigned CPU cores, 384 GB assigned RAM, and a 150 GB data disk. The system disk is
not used for datasets, models, environments or checkpoints.

## Environment and inputs

Use Python 3.12 and the baseline `uv.lock`: `uv sync --frozen --extra pi --extra
training --extra test`. Preparation also used `sentencepiece==0.2.1`; lint used
`ruff==0.14.10`. Runtime uses the already prepared `.venv` without dependency sync.
FFmpeg 4.4 is installed for PyAV decoding.

- Raw data: `/root/autodl-tmp/data/pens_task_1841_teleop_20260807`.
- Curated data: `/root/autodl-tmp/data/pens168_v30`.
- Immutable base: `/root/autodl-tmp/models/pi05_base`.
- Task processor/init bundle: `/root/autodl-tmp/models/pi05_base_pens168_init`.
- Local tokenizer: `/root/autodl-tmp/models/paligemma_tokenizer`.
- Logs: `/root/autodl-tmp/logs`.
- Formal output: `/root/autodl-tmp/outputs/pi05_ctp_pens168_full_20k`.

`prepare_pens168.py`, `validate_data.py`, `prepare_tokenizer.py`, and
`prepare_configs.py` document and reproduce the preparation. The init bundle is a
task-specific processor/config view of the unmodified base weights, not a trained
CTP checkpoint. The CTP head alone starts from fresh initialization.

## Launch and resume

The checked-in entrypoint loads `WANDB_API_KEY` and `WANDB_ENTITY` from the user's
protected environment file, selects online mode, and prevents accidental Hub
downloads. Project: `pi05-ctp-pen168`. Never enable shell tracing around secrets.

```bash
cd /root/autodl-tmp/code/lerobot_pi05_ctp
bash deploy/autodl/run.sh debug
bash deploy/autodl/run.sh formal
bash deploy/autodl/run.sh resume \
  /root/autodl-tmp/outputs/pi05_ctp_pens168_full_20k/checkpoints/last/pretrained_model/train_config.json
```

Use a detached shell/nohup with output redirected under the logs directory for the
formal run. Do not change the execution checkout while a training process uses it.
The formal entrypoint requires a clean working tree and HEAD matching its tracked
remote branch. Record the exact PID, W&B run ID and full commit after launch.

## Storage and export

A native DCP checkpoint includes the model, Adam moments, scheduler, RNG, training
step and processor statistics. It is resumable with the entrypoint above; it is not
a single inference safetensors file. Convert DCP with the baseline's
`lerobot-convert-dcp` command when an inference export is needed, budgeting extra
disk space before conversion.

Only directories carrying `checkpoint_complete.json` in the same run's checkpoint
parent are eligible for automatic rotation. New saves require enough free space
for the estimated model plus two Adam moments and 10 GiB headroom. Raw data and
the immutable initialization weights are never rotation targets. Remove only this
task's expendable random-init checkpoints after the recovery check; preserve the
verification logs and JSON reports. Debug checkpoints must be retired deliberately
before formal training so two runs do not compete for the checkpoint budget.

The native random-init infrastructure run is separately named and must never be
reported as a pretrained training result or used as formal initialization.

## Verified formal batch

Use per-GPU batch 8 (global 32). The full pretrained 100-step debug and 100→105
recovery completed with this batch. Steady training steps took about 3.7 seconds;
plan roughly 21 hours for 20,000 steps including checkpoint and validation overhead,
subject to runtime variation. There is no LoRA or encoder freezing.
