"""Evaluate each available completed checkpoint serially, without touching training."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def snapshot_checkpoint(source, destination):
    """Pin only immutable model shards via hardlinks; never pin optimizer states."""
    marker = json.loads((source / "checkpoint_complete.json").read_text())
    assert marker["step"] == int(source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        saved = json.loads((destination.parent / "source_checkpoint_complete.json").read_text())
        assert saved["step"] == marker["step"]
        return
    temporary = destination.with_name("model_snapshot.partial")
    if temporary.exists():
        raise RuntimeError(f"Incomplete prior snapshot needs inspection: {temporary}")
    try:
        shutil.copytree(source / "pretrained_model", temporary, copy_function=os.link)
    except FileNotFoundError:
        # A newer save may have pruned the source during the short hardlink operation.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    shutil.copy2(source / "checkpoint_complete.json", destination.parent / "source_checkpoint_complete.json")
    temporary.rename(destination)


def evaluation_protocol(step):
    """Use full milestones and deterministic coverage of every episode otherwise."""
    full = step == 1000 or step % 5000 == 0 or step >= 20000
    return ("full", 0, 12883) if full else ("stratified64", 64, 1088)


def process_running(pid):
    """Treat exited unreaped children as terminal in container environments."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    """Watch the existing formal run, keeping only one temporary model snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", type=int, default=220955)
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    root = Path("/root/autodl-tmp/evaluations/pens168")
    checkpoints = Path("/root/autodl-tmp/outputs/pi05_ctp_pens168_full_20k/checkpoints")
    if args.wait_for_pid:
        print(f"WAITING_FOR_EXISTING_EVALUATION {args.wait_for_pid}", flush=True)
        while process_running(args.wait_for_pid):
            time.sleep(15)
        assert (root / "step001000/full/evaluation_complete.json").is_file(), "Initial full evaluation failed"
    while True:
        pending = []
        # Include an already pinned model even if the training run has rotated it away.
        for p in root.glob("step*/model_snapshot"):
            step = int(p.parent.name.removeprefix("step"))
            protocol, _, expected = evaluation_protocol(step)
            complete = p.parent / protocol / "evaluation_complete.json"
            if complete.exists():
                assert json.loads(complete.read_text())["frames"] == expected
                assert p.name == "model_snapshot" and p.parent.parent == root
                shutil.rmtree(p)
                continue
            if step >= args.start_step:
                pending.append((step, p))
        if not pending:
            for source in sorted(checkpoints.iterdir()):
                if not source.name.isdigit() or not (source / "checkpoint_complete.json").is_file():
                    continue
                step = int(source.name)
                protocol, _, _ = evaluation_protocol(step)
                target = root / f"step{step:06d}" / "model_snapshot"
                if step < args.start_step or (target.parent / protocol / "evaluation_complete.json").exists():
                    continue
                try:
                    snapshot_checkpoint(source, target)
                except FileNotFoundError:
                    continue
                pending.append((step, target))
                break
        if pending:
            step, snapshot = sorted(pending)[0]
            protocol, per_episode, expected = evaluation_protocol(step)
            output = snapshot.parent / protocol
            with (snapshot.parent / f"{protocol}.log").open("a") as log:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("evaluate_checkpoint.py")),
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(output),
                    "--batch-size",
                    str(args.batch_size),
                    "--per-episode",
                    str(per_episode),
                    "--wandb",
                ]
                print(f"EVALUATING_CHECKPOINT {step}", flush=True)
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"Evaluation failed at step {step}; inspect {snapshot.parent / (protocol + '.log')}"
                )
            summary = json.loads((output / "summary.json").read_text())
            assert summary["overall"]["frames"] == expected and len(summary["per_episode"]) == 17
            # The pinned shards belong only to this evaluator; training files remain intact.
            assert snapshot.name == "model_snapshot" and snapshot.parent.parent == root
            shutil.rmtree(snapshot)
            summary["model_snapshot_released"] = True
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(
                f"COMPLETED_CHECKPOINT {step} protocol={protocol} frames={expected} episodes=17", flush=True
            )
            if args.once or step >= 20000:
                return
            continue
        if not process_running(args.training_pid):
            print("TRAINING_PROCESS_TERMINAL_NO_PENDING_CHECKPOINT", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
