# CTP-V3 pen insertion on AutoDL

User requests training pens into the pen holder on the specified AutoDL host,
using the current bottle-task V3 settings. Dedicated branch:
train/pi05-ctp-pens-v3-autodl. Deployment checkout:
/root/autodl-tmp/code/lerobot_pi05_ctp_pens_v3.
Canonical repository: IDC /pfs/user/code/lerobot_pi05_ctp.git.

Fresh official BASE initialization, 20000 steps, global batch32, four GPUs,
FP32 master/BF16 FSDP shard4. Preserve V3 split head, detached width context,
five learning rates, masked loss, T20 to1 over1500 steps and color jitter0.1.
Use existing pens168 data: train0..150, heldout151..167, train-only statistics,
absolute real_action targets and observation.qpos states. Verify independently.
Rebuild task processors from pen statistics. Never use bottle processors.
Evaluate every250, full heldout every1000, save every2000; latest1 full plus best.
Reserve10GiB with pre-save write guards. A debug100/resume105 sequence verifies
four-card memory fit and checkpoint recovery before continuing to20000.
Do not modify other runs or delete their artifacts. Freeze this checkout in use.
Credentials stay in the existing protected W&B environment file, never in Git.
Run records and process state belong outside this file.
