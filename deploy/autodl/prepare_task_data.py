"""Merge recorded batches with episode holdout and train-only command statistics."""

import argparse
import copy
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_pens168 import stats


def plan_episodes(sources, fraction):
    """Keep each recording batch represented in the held-out episode suffix."""
    train, validation = [], []
    for source in sources:
        info = json.loads((source / "meta/info.json").read_text())
        paths = sorted((source / "data").rglob("*.parquet"))
        if len(paths) != info["total_episodes"] or len(paths) < 2:
            raise ValueError(f"Incomplete or insufficient source episodes: {source}")
        n_val = max(1, math.ceil(len(paths) * fraction))
        if n_val >= len(paths):
            raise ValueError("Validation would consume a complete recording batch")
        rows = [
            {"source": str(source), "source_episode": int(p.stem.split("_")[-1]), "parquet": str(p)}
            for p in paths
        ]
        train.extend(rows[:-n_val])
        validation.extend(rows[-n_val:])
    return train, validation


def video_metadata(path):
    """Check MP4 packet counts without decoding every video frame."""
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(result)["streams"][0]


def main():
    """Run the explicit task preparation or validation command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    train, val = plan_episodes(args.source, args.eval_fraction)
    rows = train + val
    infos = {str(p): json.loads((p / "meta/info.json").read_text()) for p in args.source}
    info = copy.deepcopy(infos[str(args.source[0])])
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    if len(cameras) != 3 or any(v["fps"] != 30 for v in infos.values()):
        raise ValueError("This PI05 deployment expects three cameras at 30 FPS")
    features = {k: copy.deepcopy(info["features"][k]) for k in cameras}
    names = [
        f"{side}_joint{j}" if j < 7 else f"{side}_gripper" for side in ("left", "right") for j in range(1, 8)
    ]
    for key in ("observation.state", "action"):
        features[key] = {"dtype": "float32", "shape": [14], "names": names}
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = info["features"][key]
    (out / "meta").mkdir(parents=True)
    (out / "data/chunk-000").mkdir(parents=True)
    episodes, episode_stats, train_states, train_actions = [], [], [], []
    total_frames = 0
    for episode, row in enumerate(rows):
        table = pq.read_table(row["parquet"])
        state = np.asarray(table["observation.qpos"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["real_action"].to_pylist(), dtype=np.float32)
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
            raise ValueError(f"Bad state/action shape: {row}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Nonfinite state/action: {row}")
        n = len(state)
        old_indices = np.asarray(table["frame_index"].to_pylist())
        np.testing.assert_array_equal(old_indices, np.arange(n))
        source_info = infos[row["source"]]
        source_episode = row["source_episode"]
        for camera in cameras:
            relative = source_info["video_path"].format(
                episode_chunk=source_episode // source_info["chunks_size"],
                video_key=camera,
                episode_index=source_episode,
            )
            video = Path(row["source"]) / relative
            metadata = video_metadata(video)
            if metadata["r_frame_rate"] != "30/1" or int(metadata["nb_read_packets"]) != n:
                raise ValueError(f"Video/frame alignment mismatch: {video}: {metadata}, rows={n}")
            if metadata.get("nb_frames", "N/A") != "N/A" and int(metadata["nb_frames"]) != n:
                raise ValueError(f"Video declared frame count mismatch: {video}")
            shape = [metadata["height"], metadata["width"], 3]
            if episode and features[camera]["shape"] != shape:
                raise ValueError(f"Camera shape changed: {video}")
            features[camera]["shape"] = shape
            features[camera]["names"] = ["height", "width", "channels"]
            dest = out / "videos/chunk-000" / camera / f"episode_{episode:06d}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(video, dest)
        values = {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / 30),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, episode, dtype=np.int64)),
            "index": pa.array(np.arange(total_frames, total_frames + n, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
        }
        pq.write_table(pa.table(values), out / "data/chunk-000" / f"episode_{episode:06d}.parquet")
        episodes.append({"episode_index": episode, "tasks": [args.task], "length": n})
        episode_stats.append(
            {"episode_index": episode, "stats": {"observation.state": stats(state), "action": stats(action)}}
        )
        row.update(episode_index=episode, frames=n, split="train" if episode < len(train) else "validation")
        if episode < len(train):
            train_states.append(state)
            train_actions.append(action)
        total_frames += n
        if (episode + 1) % 20 == 0:
            print("SOURCE_EPISODES_CHECKED", episode + 1, len(rows), flush=True)
    info.update(
        features=features,
        total_episodes=len(rows),
        total_frames=total_frames,
        total_tasks=1,
        total_videos=len(rows) * len(cameras),
        total_chunks=1,
        chunks_size=1000,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        splits={"train": f"0:{len(train)}", "validation": f"{len(train)}:{len(rows)}"},
    )
    (out / "meta/info.json").write_text(json.dumps(info, indent=2))
    (out / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": args.task}) + "\n")
    for name, content in [("episodes.jsonl", episodes), ("episodes_stats.jsonl", episode_stats)]:
        (out / "meta" / name).write_text("".join(json.dumps(v) + "\n" for v in content))
    train_stats = {
        "observation.state": stats(np.concatenate(train_states)),
        "action": stats(np.concatenate(train_actions)),
    }
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    # Retain one video per episode: cross-batch HEVC streams have incompatible DTS boundaries.
    convert_dataset(
        repo_id=args.repo_id, root=out, push_to_hub=False, force_conversion=True, preserve_episode_videos=True
    )
    (out / "meta/stats.json").write_text(json.dumps(train_stats, indent=2))
    manifest = {
        "sources": [str(p) for p in args.source],
        "output": str(out),
        "repo_id": args.repo_id,
        "task": args.task,
        "episodes": len(rows),
        "frames": total_frames,
        "train_episodes": list(range(len(train))),
        "validation_episodes": list(range(len(train), len(rows))),
        "train_frames": sum(len(v) for v in train_states),
        "validation_frames": sum(v["frames"] for v in val),
        "eval_split": (len(val) - 0.5) / len(rows),
        "source_mapping": rows,
        "state_source": "observation.qpos",
        "action_source": "real_action",
        "split_policy": "last ceil(10%) whole episodes of each source batch, regrouped as held-out suffix",
        "timestamp_policy": "frame-index / 30 aligned to verified 30 FPS video; no row/image removal",
        "normalization": "train-only q01/q99 and moments; absolute joint commands",
        "source_video_packet_counts_verified": len(rows) * len(cameras),
    }
    (out / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "TASK_DATA_PREPARED",
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in {"source_mapping", "train_episodes", "validation_episodes"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
