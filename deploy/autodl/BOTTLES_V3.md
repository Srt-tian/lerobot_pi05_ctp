# AutoDL bottle-task V3

Separate from the IDC pen run. Use the existing accepted V3 implementation with
4x48GB, batch8 per GPU, full PI05 BASE initialization,20k schedule. Save every2000,
latest1 full recovery point plus best model. Run fixed validation every250 and full
heldout every1000 synchronously; best selection=.7*prefix10+.3*full50 normalized MSE.

Preparation is parameterized: prepare_task_data.py merges recording batches with
per-batch tail episode holdout, maps qpos/state and real_action/action, verifies all
video packet counts and recomputes train-only statistics. validate_task_data.py
independently compares every command/state row and statistics, then tests native
video loading. prepare_autodl_v3.py creates task-specific processors and a hardlink
to the verified BASE. Never reuse pen normalization or task text.

Runtime root: /root/autodl-tmp/ctp_bottles_v3. Existing Python:
/root/autodl-tmp/code/lerobot_pi05_ctp/.venv/bin/python. W&B credentials remain in
/root/.ssh/ctp_wandb.env. Source and execution commits must match canonical Git.

Use run_v3.sh debug CONFIG for a100-step BASE debug; preserve the20k schedule.
Use run_v3.sh resume105 CHECKPOINT_TRAIN_CONFIG for fresh-process recovery to105;
full validation is skipped only for this five-step probe. Use run_v3.sh resume
CHECKPOINT_TRAIN_CONFIG to continue the same run to20000; full validation is
explicitly restored to1000. Each terminal debug/probe save is an intentional
exception to the2000-step formal interval. Validate checkpoint completeness,
model/Adam/scheduler/RNG, all five learning-rate groups, W&B identity, data coverage
and restored training progress before formal continuation.

Data disk is150GiB. V2 early pen checkpoints were explicitly authorized for
cleanup. Check actual free space and conservative next-write size before starting
and before every save. Do not delete the sole complete recovery point to make
room for an unfinished save. Best hardlinks survive old checkpoint pruning.
Dynamic process IDs and resource/dataset evidence live in the external run ledger.
