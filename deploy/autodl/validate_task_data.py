"""Independent raw-command and native-loader acceptance for a prepared task."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def main():
    """Run the explicit task preparation or validation command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    manifest_path = root / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=manifest["repo_id"],
            root=str(root),
            eval_split=manifest["eval_split"],
            video_backend="pyav",
        ),
        policy=PI05Config(ctp_enabled=True, ctp_belief_cumulative_action_dims=0),
        tolerance_s=0.01,
    )
    train, val = make_train_eval_datasets(cfg)
    assert train.episodes == manifest["train_episodes"]
    assert val.episodes == manifest["validation_episodes"]
    assert len(train) == manifest["train_frames"] and len(val) == manifest["validation_frames"]
    converted = pq.read_table(sorted((root / "data").rglob("*.parquet"))).to_pandas()
    states, actions = [], []
    for mapping in manifest["source_mapping"]:
        source = pq.read_table(mapping["parquet"])
        state = np.asarray(source["observation.qpos"].to_pylist(), dtype=np.float32)
        action = np.asarray(source["real_action"].to_pylist(), dtype=np.float32)
        target = converted[converted.episode_index == mapping["episode_index"]].sort_values("frame_index")
        np.testing.assert_array_equal(np.stack(target["observation.state"]), state)
        np.testing.assert_array_equal(np.stack(target["action"]), action)
        np.testing.assert_array_equal(target["frame_index"].to_numpy(), np.arange(len(state)))
        if mapping["split"] == "train":
            states.append(state)
            actions.append(action)
    for key, arrays in [("observation.state", states), ("action", actions)]:
        x = np.concatenate(arrays).astype(np.float64)
        for stat, quantile in [("q01", 0.01), ("q99", 0.99)]:
            np.testing.assert_allclose(
                train.meta.stats[key][stat], np.quantile(x, quantile, axis=0), rtol=1e-6, atol=1e-6
            )
        np.testing.assert_allclose(train.meta.stats[key]["mean"], x.mean(axis=0), rtol=1e-6, atol=1e-6)
    decoded = 0
    for dataset in (train, val):
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[index]
            assert sample["action"].shape == (50, 14)
            assert sample["observation.state"].shape == (14,)
            assert sample["task"] == manifest["task"]
            for key in dataset.meta.camera_keys:
                assert sample[key].shape == (3, 480, 640)
                decoded += 1
    report = {
        "episodes": manifest["episodes"],
        "train_episodes": len(train.episodes),
        "validation_episodes": len(val.episodes),
        "train_frames": len(train),
        "validation_frames": len(val),
        "all_mapped_values_exact": True,
        "train_only_quantiles_verified": True,
        "native_video_samples_decoded": decoded,
        "source_video_packet_counts_verified": manifest["source_video_packet_counts_verified"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "stats_sha256": hashlib.sha256((root / "meta/stats.json").read_bytes()).hexdigest(),
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("TASK_DATA_VALIDATED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
