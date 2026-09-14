"""Check sidecar checkpoint pinning and resource-aware evaluation cadence."""

import json

from watch_checkpoints import evaluation_protocol, snapshot_checkpoint


def test_full_milestones_and_stratified_intervals():
    """Keep exhaustive and sampled curves separate and evaluate the final model fully."""
    for step in [1000, 5000, 10000, 15000, 20000]:
        assert evaluation_protocol(step) == ("full", 0, 12883)
    for step in [2000, 3000, 4000, 6000, 19000]:
        assert evaluation_protocol(step) == ("stratified64", 64, 1088)


def test_snapshot_survives_source_rotation_without_pinning_optimizer(tmp_path):
    """Model-only hardlinks preserve immutable data after source files are unlinked."""
    source = tmp_path / "001000"
    model = source / "pretrained_model"
    model.mkdir(parents=True)
    (model / "weights").write_bytes(b"model-only")
    state = source / "training_state"
    state.mkdir()
    (state / "optimizer").write_bytes(b"optimizer")
    (source / "checkpoint_complete.json").write_text(json.dumps({"step": 1000}))
    destination = tmp_path / "evaluation" / "model_snapshot"
    snapshot_checkpoint(source, destination)
    assert (model / "weights").stat().st_ino == (destination / "weights").stat().st_ino
    (model / "weights").unlink()
    assert (destination / "weights").read_bytes() == b"model-only"
    assert not (destination / "training_state").exists()
    assert json.loads((destination.parent / "source_checkpoint_complete.json").read_text())["step"] == 1000
