"""Prepare the user-approved state/command mapping and train-only statistics."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
        "count": [len(x)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    src = args.source
    out = args.output
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite {out}")
    info = json.loads((src / "meta/info.json").read_text())
    assert info["total_episodes"] == 168 and info["fps"] == 30
    raw_files = sorted((src / "data").rglob("*.parquet"))
    assert len(raw_files) == 168
    task = "Put the pens into the pen holder."
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    assert len(cameras) == 3
    features = {k: info["features"][k] for k in cameras}
    for camera in cameras:
        video = src / "videos/chunk-000" / camera / "episode_000000.mp4"
        meta = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        features[camera]["shape"] = [meta["height"], meta["width"], 3]
        features[camera]["names"] = ["height", "width", "channels"]
        assert meta["r_frame_rate"] == "30/1", meta
    joint_names = [
        f"{side}_joint{j}" if j != 7 else f"{side}_gripper" for side in ("left", "right") for j in range(1, 8)
    ]
    features["observation.state"] = {"dtype": "float32", "shape": [14], "names": joint_names}
    features["action"] = {"dtype": "float32", "shape": [14], "names": joint_names}
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[k] = info["features"][k]
    (out / "meta").mkdir(parents=True)
    (out / "data/chunk-000").mkdir(parents=True)
    episodes = []
    episode_stats = []
    train_states = []
    train_actions = []
    frames = 0
    for ep, path in enumerate(raw_files):
        t = pq.read_table(path)
        state = np.asarray(t["observation.qpos"].to_pylist(), dtype=np.float32)
        action = np.asarray(t["real_action"].to_pylist(), dtype=np.float32)
        assert state.shape == action.shape and state.shape[1] == 14
        assert np.isfinite(state).all() and np.isfinite(action).all(), ep
        n = len(state)
        values = {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / 30),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
            "index": pa.array(np.arange(frames, frames + n, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
        }
        pq.write_table(pa.table(values), out / "data/chunk-000" / f"episode_{ep:06d}.parquet")
        episodes.append({"episode_index": ep, "tasks": [task], "length": n})
        episode_stats.append(
            {"episode_index": ep, "stats": {"observation.state": stats(state), "action": stats(action)}}
        )
        for camera in cameras:
            relative = Path("videos/chunk-000") / camera / f"episode_{ep:06d}.mp4"
            dest = out / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(src / relative, dest)
        if ep < 151:
            train_states.append(state)
            train_actions.append(action)
        frames += n
    assert frames == 147087, frames
    info["features"] = features
    info["splits"] = {"train": "0:151", "validation": "151:168"}
    (out / "meta/info.json").write_text(json.dumps(info, indent=2))
    (out / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    for filename, rows in [("episodes.jsonl", episodes), ("episodes_stats.jsonl", episode_stats)]:
        (out / "meta" / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    train_stats = {
        "observation.state": stats(np.concatenate(train_states)),
        "action": stats(np.concatenate(train_actions)),
    }
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    convert_dataset(repo_id="local/pens168_ctp", root=out, push_to_hub=False, force_conversion=True)
    # The upstream converter aggregates all episodes; replace with the approved train-only stats.
    (out / "meta/stats.json").write_text(json.dumps(train_stats, indent=2))
    manifest = {
        "source": str(src),
        "output": str(out),
        "state_source": "observation.qpos",
        "action_source": "real_action",
        "task": task,
        "episodes": 168,
        "frames": frames,
        "train_episodes": list(range(151)),
        "validation_episodes": list(range(151, 168)),
        "train_frames": sum(len(x) for x in train_states),
        "split_policy": "Native LeRobot tail ceil(0.1*168) episode holdout; fixed, no frame mixing",
        "normalization": "q01/q99 and moments from train episodes only; absolute joint actions; no action differencing",
    }
    (out / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "PREPARATION_COMPLETE",
        json.dumps({k: v for k, v in manifest.items() if not k.endswith("_episodes")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
