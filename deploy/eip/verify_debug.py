"""Verify a completed real-data debug checkpoint and its recovery provenance."""

import json
import math
import sys
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader


def main():
    """Check complete artifacts, validation coverage, step and run continuity."""
    output, step = Path(sys.argv[1]), int(sys.argv[2])
    checkpoint = output / f"checkpoints/{step:06d}"
    marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
    state_dir = checkpoint / "training_state"
    state = json.loads((state_dir / "training_step.json").read_text())
    cfg = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
    assert marker["step"] == state["step"] == step
    assert cfg["wandb"]["run_id"]
    assert cfg["steps"] == cfg["policy"]["scheduler_decay_steps"] == 20000
    assert cfg["policy"]["ctp_split_head"] and cfg["batch_size"] == 8
    assert cfg["save_freq"] == 2000 and cfg["checkpoint_keep_last"] == 0
    for rank in range(4):
        assert (state_dir / f"rng_rank_{rank}/rng_state.safetensors").is_file()
    assert (state_dir / "scheduler_state.json").is_file()
    model_metadata = FileSystemReader(checkpoint / "pretrained_model/pytorch_model_fsdp_0").read_metadata()
    optim_metadata = FileSystemReader(state_dir / "optimizer_0").read_metadata()
    for branch in ("centers", "logits", "widths"):
        assert any(f"ctp_parameter_head.{branch}." in key for key in model_metadata.state_dict_metadata)
        assert any(f"ctp_parameter_head.{branch}." in key for key in optim_metadata.state_dict_metadata)
    values = [json.loads(line) for line in (output / "validation_history.jsonl").read_text().splitlines()]
    result = next(value for value in values if value["step"] == step)
    assert result["samples"] == 1088
    assert all(math.isfinite(v) for v in result.values() if isinstance(v, (float, int)))
    full = [json.loads(line) for line in (output / "full_validation_history.jsonl").read_text().splitlines()]
    assert any(v["step"] == 100 and v["samples"] == 12883 for v in full)
    if step == 105:
        before = json.loads((output / "checkpoints/000100/pretrained_model/train_config.json").read_text())
        assert cfg["wandb"]["run_id"] == before["wandb"]["run_id"]
    report = {
        "step": step,
        "complete": True,
        "run_id": cfg["wandb"]["run_id"],
        "fixed_validation_samples": 1088,
        "full_validation_samples": 12883,
        "optimizer_metadata_keys": len(optim_metadata.state_dict_metadata),
        "next_step_temperature": 20 * (0.05 ** (step / 1500)),
        "resume_exercised": step == 105,
    }
    (output / f"debug_verified_{step}.json").write_text(json.dumps(report, indent=2))
    print("V3_DEBUG_VERIFIED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
