import pytest

from lerobot.common.train_utils import update_last_checkpoint


def test_last_link_replaced_with_relative_completed_checkpoint(tmp_path):
    old = tmp_path / "50"
    new = tmp_path / "100"
    old.mkdir()
    new.mkdir()
    (tmp_path / "last").symlink_to(old.name)
    update_last_checkpoint(new)
    assert (tmp_path / "last").readlink() == new.relative_to(tmp_path)
    assert old.is_dir()


def test_failed_atomic_replace_preserves_previous_last(tmp_path, monkeypatch):
    old = tmp_path / "50"
    new = tmp_path / "100"
    old.mkdir()
    new.mkdir()
    (tmp_path / "last").symlink_to(old.name)

    def fail(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr("lerobot.common.train_utils.os.replace", fail)
    with pytest.raises(OSError, match="simulated interruption"):
        update_last_checkpoint(new)
    assert (tmp_path / "last").resolve() == old
    assert not list(tmp_path.glob(".last.*.tmp"))
