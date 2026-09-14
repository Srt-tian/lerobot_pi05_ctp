"""Full held-out evaluation at v2 milestones; serial model-only snapshots."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from watch_checkpoints import process_running, snapshot_checkpoint


def main():
    """Watch a named v2 run without confusing its metrics with the original run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--last-step", type=int, default=8000)
    args = parser.parse_args()
    assert args.interval > 0 and args.last_step > 0
    args.output.mkdir(parents=True, exist_ok=True)
    while True:
        pending = []
        for snapshot in sorted(args.output.glob("step*/model_snapshot")):
            if (snapshot.parent / "full/evaluation_complete.json").exists():
                shutil.rmtree(snapshot)
            else:
                pending.append(snapshot)
        if not pending:
            for checkpoint in sorted(args.checkpoints.iterdir()):
                if not checkpoint.name.isdecimal() or not (checkpoint / "checkpoint_complete.json").exists():
                    continue
                step = int(checkpoint.name)
                if step % args.interval and step != args.last_step:
                    continue
                destination = args.output / f"step{step:06d}"
                if (destination / "full/evaluation_complete.json").exists():
                    continue
                try:
                    snapshot_checkpoint(checkpoint, destination / "model_snapshot")
                except FileNotFoundError:
                    continue
                pending.append(destination / "model_snapshot")
                break
        if pending:
            snapshot = pending[0]
            with (snapshot.parent / "full.log").open("a") as stream:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("evaluate_checkpoint.py")),
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(snapshot.parent / "full"),
                    "--per-episode",
                    "0",
                    "--batch-size",
                    "4",
                    "--wandb",
                    "--training-run-id",
                    args.training_run_id,
                    "--release-snapshot-after-load",
                ]
                print("START_FULL_EVALUATION", snapshot.parent.name, flush=True)
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads((snapshot.parent / "full/evaluation_complete.json").read_text())
            assert result["frames"] == 12883 and result["episodes"] == 17
            if snapshot.exists():
                shutil.rmtree(snapshot)
            print("COMPLETE_FULL_EVALUATION", result, flush=True)
            if result["checkpoint_step"] >= args.last_step:
                return
            continue
        if not process_running(args.training_pid):
            print("TRAINING_TERMINAL_NO_PENDING_FULL_EVALUATION", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
