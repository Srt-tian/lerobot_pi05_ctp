"""Independently check converted values, train-only statistics and native video loading."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.pi05.configuration_pi05 import PI05Config

root = Path("/root/autodl-tmp/data/pens168_v30")
raw = Path("/root/autodl-tmp/data/pens_task_1841_teleop_20260807")
cfg = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="local/pens168_ctp", root=str(root), eval_split=0.1, video_backend="pyav"),
    policy=PI05Config(ctp_enabled=True, ctp_belief_cumulative_action_dims=0),
    tolerance_s=0.01,
)
train, val = make_train_eval_datasets(cfg)
assert train.episodes == list(range(151)) and val.episodes == list(range(151, 168))
assert len(train) == 134204 and len(val) == 12883
converted = pq.read_table(sorted((root / "data").rglob("*.parquet"))).to_pandas()
states, actions = [], []
for ep, path in enumerate(sorted((raw / "data").rglob("*.parquet"))):
    source = pq.read_table(path)
    s = np.asarray(source["observation.qpos"].to_pylist(), dtype=np.float32)
    a = np.asarray(source["real_action"].to_pylist(), dtype=np.float32)
    target = converted[converted.episode_index == ep]
    np.testing.assert_array_equal(np.stack(target["observation.state"]), s)
    np.testing.assert_array_equal(np.stack(target["action"]), a)
    if ep < 151:
        states.append(s)
        actions.append(a)
for key, arrays in [("observation.state", states), ("action", actions)]:
    x = np.concatenate(arrays).astype(np.float64)
    for stat, q in [("q01", 0.01), ("q99", 0.99)]:
        np.testing.assert_allclose(
            train.meta.stats[key][stat], np.quantile(x, q, axis=0), rtol=1e-6, atol=1e-6
        )
for ds in (train, val):
    for index in (0, len(ds) // 2, len(ds) - 1):
        sample = ds[index]
        assert sample["action"].shape == (50, 14)
        assert sample["observation.state"].shape == (14,)
        for key in ds.meta.camera_keys:
            assert sample[key].shape[0] == 3 and sample[key].numel() > 0
        assert sample["task"] == "Put the pens into the pen holder."
report = {
    "episodes": 168,
    "train_episodes": 151,
    "validation_episodes": 17,
    "train_frames": len(train),
    "validation_frames": len(val),
    "all_mapped_values_exact": True,
    "train_only_quantiles_verified": True,
    "video_samples_decoded": 18,
}
Path("/root/autodl-tmp/logs/data_validation.json").write_text(json.dumps(report, indent=2))
print("DATA_VALIDATED", json.dumps(report), flush=True)
