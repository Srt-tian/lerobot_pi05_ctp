# Complete held-out evaluation

The live training checkout and optimizer remain unchanged. Its legacy online
validation metric uses the first 128 frames of episode 151 only; it must not be
interpreted as full held-out validation.

This separate branch evaluates complete checkpoints on episodes 151–167. Steps
1000, 5000, 10000, 15000 and 20000 cover all 12,883 frames. Other 1000-step checkpoints
use 64 evenly spaced frames per episode (1,088 total), covering both endpoints and
all four relative-time quarters. This limits interference with the shared training GPU. Each observation is a ground-truth reset; output is a
top-1 50-step absolute-command chunk. This is offline prediction evaluation, not
closed-loop robot success. Invalid future actions at episode ends are excluded
from new error metrics. The legacy loss retains the original padding semantics
for comparison. Saved checkpoint processors retain the train-only statistics.

Metrics include top-1 and oracle best-of-K normalized MSE, first-action and
horizon-specific MSE, per-joint raw-unit MAE/RMSE, hold-current-state baseline,
per-episode and per-quarter-of-episode aggregates, frame-error percentiles,
mode selections, probabilities and responsibilities. Joint units are not mixed.

The sidecar loads DCP model shards strictly on CPU, uses BF16 Linear/Embedding
weights with FP32 remaining parameters and metrics, and BF16 autocast. This
matches the main compute precision, but is a separate single-GPU evaluation path.
A per-process allocator cap of 32% leaves headroom for the live training rank.
The 34-frame pilot covered every episode's first and last frame and used 8.04 GiB.
Five metric and checkpoint tests cover episode selection, padding, top-1/oracle distinction and
valid-action-weighted aggregation. A processor roundtrip is checked on real data.

The watcher serially hardlinks only completed model files, never optimizer
states. It removes its own snapshot only after evaluation and W&B synchronization
succeed. Existing training files are never edited or removed. If evaluation lags
checkpoint rotation, unavailable steps may be skipped; actual evaluated steps
are recorded. Failure stops the watcher with a log instead of automatic retries.

Run `bash deploy/autodl/run_evaluation.sh`. Outputs are under
`/root/autodl-tmp/evaluations/pens168/stepXXXXXX/full/`. The W&B run is
`afcg8d7v-heldout17-full` for exhaustive evaluation and
`afcg8d7v-heldout17-strat64` for the fixed sample; both use `checkpoint_step` as the curve x-axis, linked to
training run `afcg8d7v`. The per-frame JSONL and per-episode summaries are preserved.

The initial exhaustive run measured 8.33 GiB peak allocated GPU memory at batch 4.
Sharing GPU 0 temporarily increased training step time from about 3.8 to 6.2 seconds.
The two evaluation schedules use separate W&B runs to avoid mixing sample populations.
The first 128 frames reproduced legacy loss within 0.0034 absolute (0.0031 versus -0.0002);
single-GPU batch/kernel and BF16 weight-storage paths need not be bit-identical to FSDP2.
