import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.common.checkpoint_retention import MARKER, check_space, mark_complete_and_prune


def checkpoint(root: Path, step: int, complete: bool = True):
    path = root / str(step)
    (path / "pretrained_model").mkdir(parents=True)
    (path / "training_state").mkdir()
    (path / "pretrained_model/config.json").write_text("{}")
    (path / "training_state/training_step.json").write_text(json.dumps({"step": step}))
    if complete:
        mark_complete_and_prune(path, keep=99)
    return path


def test_only_completed_numeric_checkpoints_are_rotated(tmp_path):
    old = checkpoint(tmp_path, 50)
    partial = checkpoint(tmp_path, 75, complete=False)
    newest = checkpoint(tmp_path, 100)
    unrelated = tmp_path / "notes"
    unrelated.mkdir()
    (tmp_path / "last").symlink_to(newest.name)
    mark_complete_and_prune(newest, keep=1)
    assert not old.exists()
    assert partial.exists() and unrelated.exists()
    assert (newest / MARKER).is_file()
    assert (tmp_path / "last").resolve() == newest


def test_low_space_preserves_existing_checkpoint(tmp_path, monkeypatch):
    existing = checkpoint(tmp_path, 50)
    monkeypatch.setattr(
        "lerobot.common.checkpoint_retention.shutil.disk_usage", lambda _: SimpleNamespace(free=100)
    )
    with pytest.raises(RuntimeError, match="Previous checkpoint is preserved"):
        check_space(tmp_path, expected_bytes=101, reserve_gb=0)
    assert (existing / MARKER).is_file()


def test_incomplete_new_save_does_not_delete_previous(tmp_path):
    old = checkpoint(tmp_path, 50)
    new = tmp_path / "100"
    new.mkdir()
    with pytest.raises(RuntimeError, match="without policy config"):
        mark_complete_and_prune(new, keep=1)
    assert old.exists() and not (new / MARKER).exists()
