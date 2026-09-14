"""Bounded checkpoint retention; only completed checkpoints from this run are removed."""

import json
import os
import shutil
from pathlib import Path

MARKER = "checkpoint_complete.json"


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def check_space(parent: Path, expected_bytes: int, reserve_gb: float) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    needed = expected_bytes + int(reserve_gb * 1024**3)
    if free < needed:
        raise RuntimeError(
            f"Checkpoint needs {needed / 1024**3:.1f} GiB including reserve; {free / 1024**3:.1f} GiB free. Previous checkpoint is preserved."
        )


def mark_complete_and_prune(checkpoint: Path, keep: int) -> None:
    """Call only after every rank has finished saving, and last has been updated."""
    if keep < 0:
        raise ValueError("keep must be nonnegative; zero retains every checkpoint")
    checkpoint = Path(checkpoint)
    if not checkpoint.name.isdecimal() or checkpoint.is_symlink():
        raise ValueError("Expected a numeric checkpoint directory")
    if not (checkpoint / "pretrained_model/config.json").is_file():
        raise RuntimeError("Refusing to mark checkpoint complete without policy config")
    if not (checkpoint / "training_state/training_step.json").is_file():
        raise RuntimeError("Refusing to mark checkpoint complete without training metadata")
    marker = checkpoint / MARKER
    temporary = checkpoint / (MARKER + ".tmp")
    temporary.write_text(json.dumps({"step": int(checkpoint.name), "bytes": directory_bytes(checkpoint)}))
    os.replace(temporary, marker)
    if keep == 0:
        return
    completed = sorted(
        (
            p
            for p in checkpoint.parent.iterdir()
            if p.name.isdecimal() and p.is_dir() and not p.is_symlink() and (p / MARKER).is_file()
        ),
        key=lambda p: int(p.name),
        reverse=True,
    )
    for old in completed[keep:]:
        if old != checkpoint:
            shutil.rmtree(old)
